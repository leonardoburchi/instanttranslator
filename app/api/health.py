"""Health-check per monitoraggio e failover automatico (LB/keepalived).

``/healthz`` ritorna 200 quando il servizio è sano o solo degradato (continua a
servire), 503 quando è **critico** (consegna totalmente giù) così un load
balancer può togliere il nodo e passare allo standby.

Distinzione importante per un evento: "degradato" significa che qualcosa non va
ma il pubblico sta ancora sentendo qualcosa (magari solo l'audio originale);
"critico" significa che dal telefono non esce più niente.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..context import AppContext
from ..deps import get_ctx

router = APIRouter(tags=["health"])

# Un segmento HLS dovrebbe comparire ~ogni segment_time secondi; oltre questa
# soglia lo stream è considerato "fermo".
_STALE_AFTER = 6.0


@router.get("/healthz")
async def healthz(ctx: AppContext = Depends(get_ctx)) -> JSONResponse:
    running = sorted(ctx.orchestrator.running_channels())
    warnings: list[str] = []

    gpu = None
    if ctx.cfg.engine.mode == "real":
        gpu = {"cuda": False, "devices": 0}
        try:
            import ctranslate2
            n = ctranslate2.get_cuda_device_count()
            gpu = {"cuda": n > 0, "devices": n}
        except Exception:
            pass

    streams = ctx.hls.health() if ctx.hls is not None else []
    alive = sum(1 for s in streams if s["alive"])
    stale = [
        s for s in streams
        if not s["alive"]
        or s["last_segment_age"] is None
        or s["last_segment_age"] > _STALE_AFTER
    ]
    backlogged = [s for s in streams if s.get("backlog_s", 0) >= ctx.cfg.hls.max_backlog]

    status = "ok"
    if ctx.hls is not None and streams:
        if alive == 0:
            status = "critical"          # consegna HLS totalmente giù
        elif stale:
            status = "degraded"
            warnings.append(f"{len(stale)} stream HLS fermi o senza segmenti freschi")
    if backlogged:
        warnings.append(
            f"{len(backlogged)} stream con audio TTS in arretrato: la voce tradotta "
            f"sta accumulando ritardo (valuta tts.length_scale < 1 o meno lingue)"
        )

    capture = ctx.orchestrator.capture_health()
    if ctx.cfg.engine.mode == "real" and running:
        if not ctx.orchestrator.capture_running():
            status = "degraded" if status == "ok" else status
            warnings.append("canali in esecuzione ma cattura audio non attiva")
        elif capture.get("stalled"):
            status = "degraded" if status == "ok" else status
            warnings.append("la scheda audio non consegna più blocchi (cavo/driver?)")

    missing_voices = [
        l for l in ctx.cfg.target_languages if not ctx.engines.tts.has_voice(l)
    ]
    if missing_voices:
        warnings.append("voci TTS mancanti: " + ", ".join(missing_voices))

    payload = {
        "status": status,
        "warnings": warnings,
        "engine_mode": ctx.cfg.engine.mode,
        "missing_voices": missing_voices,
        "uptime_s": round(time.time() - ctx.started_at, 1),
        "gpu": gpu,
        "capture_running": ctx.orchestrator.capture_running(),
        "capture": capture,
        "channels": {
            "running": running,
            "count": len(running),
            "listeners": {cid: ctx.hub.listener_count(cid) for cid in running},
            "ws_listeners_total": ctx.hub.total_listeners(),
        },
        "hls": {
            "enabled": ctx.hls is not None,
            "streams_total": len(streams),
            "streams_alive": alive,
            "streams": streams,
        },
    }
    code = 503 if status == "critical" else 200
    return JSONResponse(payload, status_code=code,
                        headers={"Cache-Control": "no-store"})
