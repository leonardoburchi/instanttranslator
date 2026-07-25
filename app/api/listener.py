"""API ascoltatore: info per la landing + WebSocket sottotitoli/audio.

Nota di scala: questo modulo gira **nello stesso processo** che fa ASR/MT/TTS.
Tutto ciò che è per-ascoltatore va tenuto fuori dall'origine: per le grandi
platee la consegna è HLS (file statici, cacheabili). Il WebSocket resta per la
regia, gli interpreti e le sale piccole, con un tetto sulle connessioni per non
far mancare CPU alla pipeline.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..context import AppContext
from ..deps import get_ctx
from ..languages import exists as lang_exists
from ..state import topic_audio, topic_subtitle

log = logging.getLogger("instanttranslator.api.listener")
router = APIRouter(prefix="/api", tags=["listener"])

# Codici di chiusura WebSocket applicativi.
WS_NOT_FOUND = 4404
WS_BUSY = 4429


@router.get("/info")
async def info(ctx: AppContext = Depends(get_ctx)) -> JSONResponse:
    """Tutto ciò che serve alla landing per costruirsi: canali + lingue.

    Risposta *cacheabile* qualche secondo: all'inizio dell'evento la chiedono
    tutti insieme e non deve arrivare all'origine 2500 volte.
    """
    channels = []
    for c in ctx.channel_infos():
        if not c.enabled:
            continue
        d = c.model_dump()
        d["hls_languages"] = ctx.hls_languages(c.id)
        d["hls_floor"] = ctx.hls_floor(c.id)
        d["languages"] = [l.model_dump() for l in ctx.audience_language_infos(c.id)]
        channels.append(d)
    payload = {
        "engine_mode": ctx.cfg.engine.mode,
        "samplerate": ctx.engines.tts.samplerate,
        "hls_enabled": ctx.hls_enabled,
        "ws_audio": ctx.ws_audio_allowed,
        "subtitle_poll_seconds": ctx.cfg.delivery.subtitle_poll_seconds,
        "hls_live_sync": ctx.cfg.delivery.hls_live_sync,
        "channels": channels,
        # Lista globale (retrocompatibile): la landing usa channel.languages.
        "target_languages": [l.model_dump() for l in ctx.target_language_infos()],
    }
    ttl = max(int(ctx.cfg.delivery.info_cache_seconds), 0)
    return JSONResponse(payload, headers={
        "Cache-Control": f"public, max-age={ttl}" if ttl else "no-store",
    })


@router.websocket("/listen/{channel_id}/{lang}")
async def listen(websocket: WebSocket, channel_id: str, lang: str) -> None:
    ctx: AppContext = websocket.app.state.ctx

    if not ctx.registry.exists(channel_id) or not lang_exists(lang):
        await websocket.close(code=WS_NOT_FOUND)
        return

    # Tetto di sicurezza: l'origine non deve reggere una piazza su WebSocket.
    limit = ctx.cfg.delivery.max_ws_listeners
    if limit and ctx.hub.total_listeners() >= limit:
        log.warning("connessione WS rifiutata: raggiunto il tetto di %d ascoltatori", limit)
        await websocket.close(code=WS_BUSY)
        return

    await websocket.accept()
    ctx.hub.add_listener(channel_id, lang)
    sub = ctx.hub.subscribe(topic_subtitle(channel_id, lang))
    aud = ctx.hub.subscribe(topic_audio(channel_id, lang)) if ctx.ws_audio_allowed else None

    await websocket.send_json({
        "type": "hello",
        "channel": channel_id,
        "lang": lang,
        "samplerate": ctx.engines.tts.samplerate,
        "audio": aud is not None,
        "running": ctx.orchestrator.is_running(channel_id),
    })

    async def pump_subtitles() -> None:
        while True:
            msg = await sub.queue.get()
            await websocket.send_json(msg)

    async def pump_audio() -> None:
        assert aud is not None
        while True:
            chunk = await aud.queue.get()
            await websocket.send_bytes(chunk.pcm)

    async def watch_disconnect() -> None:
        # Rileva la chiusura lato client (e ignora eventuali messaggi in arrivo).
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
        except (WebSocketDisconnect, RuntimeError):
            return

    tasks = [
        asyncio.create_task(pump_subtitles()),
        asyncio.create_task(watch_disconnect()),
    ]
    if aud is not None:
        tasks.append(asyncio.create_task(pump_audio()))
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        ctx.hub.unsubscribe(sub)
        if aud is not None:
            ctx.hub.unsubscribe(aud)
        ctx.hub.remove_listener(channel_id, lang)
        log.debug("ascoltatore disconnesso canale=%s lang=%s", channel_id, lang)
