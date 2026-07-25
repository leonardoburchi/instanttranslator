"""API di regia: gestione canali, device, avvio/arresto, monitor, livelli.

Tutte le rotte richiedono il token della regia (vedi ``app/security.py``):
la landing è pubblica, la regia no.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter, Depends, HTTPException, Response, WebSocket, WebSocketDisconnect,
)

from ..audio.devices import host_api_names, list_input_devices
from ..context import AppContext
from ..deps import get_admin_ctx
from ..languages import exists as lang_exists
from ..models import ChannelCreate, ChannelInfo, ChannelUpdate, DeviceInfo, LanguageInfo
from ..state import topic_transcript

log = logging.getLogger("instanttranslator.api.admin")
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/info")
async def info(ctx: AppContext = Depends(get_admin_ctx)) -> dict:
    return {
        "engine_mode": ctx.cfg.engine.mode,
        "samplerate": ctx.engines.tts.samplerate,
        "hls_enabled": ctx.hls_enabled,
        "ws_audio": ctx.ws_audio_allowed,
        "audio": {
            "samplerate": ctx.cfg.audio.samplerate,
            "device_samplerate": ctx.cfg.audio.device_samplerate,
            "host_apis": host_api_names(),
        },
        "languages": [l.model_dump() for l in ctx.all_language_infos()],
        "target_languages": [l.model_dump() for l in ctx.target_language_infos()],
    }


@router.get("/devices", response_model=list[DeviceInfo])
async def devices(ctx: AppContext = Depends(get_admin_ctx)) -> list[DeviceInfo]:
    return list_input_devices()


@router.get("/languages", response_model=list[LanguageInfo])
async def languages(ctx: AppContext = Depends(get_admin_ctx)) -> list[LanguageInfo]:
    return ctx.all_language_infos()


@router.get("/levels")
async def levels(ctx: AppContext = Depends(get_admin_ctx)) -> dict:
    """Livelli e stato VAD per canale: serve a verificare il routing del mixer.

    Batti su un microfono e guarda quale canale si muove: è il modo più rapido
    di mappare i ``channel_index`` di un XR18 senza tirare a indovinare.
    """
    out: dict[str, dict] = {}
    for key, meter in ctx.orchestrator.levels().items():
        out[key] = dict(meter)
    for cid in ctx.orchestrator.running_channels():
        stats = ctx.orchestrator.source_stats(cid)
        if stats:
            out.setdefault(cid, {}).update(stats)
    return {"levels": out, "capture": ctx.orchestrator.capture_health()}


@router.get("/channels", response_model=list[ChannelInfo])
async def list_channels(ctx: AppContext = Depends(get_admin_ctx)) -> list[ChannelInfo]:
    return ctx.channel_infos()


@router.post("/channels", response_model=ChannelInfo, status_code=201)
async def create_channel(
    data: ChannelCreate, ctx: AppContext = Depends(get_admin_ctx)
) -> ChannelInfo:
    if not lang_exists(data.source_language):
        raise HTTPException(400, f"lingua sorgente sconosciuta: {data.source_language}")
    try:
        ctx.registry.create(data)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return ctx.channel_info(data.id)  # type: ignore[return-value]


@router.patch("/channels/{channel_id}", response_model=ChannelInfo)
async def update_channel(
    channel_id: str, data: ChannelUpdate, ctx: AppContext = Depends(get_admin_ctx)
) -> ChannelInfo:
    if data.source_language is not None and not lang_exists(data.source_language):
        raise HTTPException(400, f"lingua sorgente sconosciuta: {data.source_language}")
    try:
        ctx.registry.update(channel_id, data)
    except KeyError:
        raise HTTPException(404, "canale non trovato")

    # Se cambia la lingua sorgente, il routing o le lingue HLS, riavvia il canale.
    if ctx.orchestrator.is_running(channel_id):
        restart = any(
            getattr(data, f) is not None
            for f in ("source_language", "input_device", "channel_index",
                      "gain_db", "floor_channel_index", "broadcast_languages")
        )
        if restart:
            ctx.orchestrator.stop_channel(channel_id)
            try:
                ctx.orchestrator.start_channel(channel_id)
            except Exception as exc:
                log.exception("riavvio canale dopo modifica fallito")
                raise HTTPException(400, f"canale fermato, riavvio fallito: {exc}")
    return ctx.channel_info(channel_id)  # type: ignore[return-value]


@router.delete("/channels/{channel_id}")
async def delete_channel(
    channel_id: str, ctx: AppContext = Depends(get_admin_ctx)
) -> Response:
    if ctx.orchestrator.is_running(channel_id):
        ctx.orchestrator.stop_channel(channel_id)
    try:
        ctx.registry.delete(channel_id)
    except KeyError:
        raise HTTPException(404, "canale non trovato")
    return Response(status_code=204)


@router.post("/channels/{channel_id}/start", response_model=ChannelInfo)
async def start_channel(
    channel_id: str, ctx: AppContext = Depends(get_admin_ctx)
) -> ChannelInfo:
    if not ctx.registry.exists(channel_id):
        raise HTTPException(404, "canale non trovato")
    try:
        ctx.orchestrator.start_channel(channel_id)
    except Exception as exc:
        log.exception("start canale fallito")
        raise HTTPException(400, f"avvio fallito: {exc}")
    return ctx.channel_info(channel_id)  # type: ignore[return-value]


@router.post("/channels/{channel_id}/stop", response_model=ChannelInfo)
async def stop_channel(
    channel_id: str, ctx: AppContext = Depends(get_admin_ctx)
) -> ChannelInfo:
    if not ctx.registry.exists(channel_id):
        raise HTTPException(404, "canale non trovato")
    ctx.orchestrator.stop_channel(channel_id)
    return ctx.channel_info(channel_id)  # type: ignore[return-value]


@router.websocket("/monitor/{channel_id}")
async def monitor(websocket: WebSocket, channel_id: str) -> None:
    """Stream del testo sorgente trascritto, per il monitor di regia."""
    ctx: AppContext = websocket.app.state.ctx
    # I WebSocket non passano dalle dependencies HTTP: token via ?token=
    if not ctx.auth.check(websocket.query_params.get("token")):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    sub = ctx.hub.subscribe(topic_transcript(channel_id))
    try:
        while True:
            msg = await sub.queue.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception:  # pragma: no cover - socket chiuso lato client
        pass
    finally:
        ctx.hub.unsubscribe(sub)
