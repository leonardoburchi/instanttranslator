"""Applicazione FastAPI: monta API, WebSocket e frontend statico."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import admin, health, hls as hls_api, listener
from .config import load_config
from .context import AppContext
from .pipeline.factory import build_engines
from .pipeline.orchestrator import Orchestrator
from .registry import ChannelRegistry
from .security import AdminAuth
from .state import Hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("instanttranslator")

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()

    hub = Hub()
    hub.bind_loop(asyncio.get_running_loop())
    state_dir = Path(cfg.state_dir)
    registry = ChannelRegistry(cfg.channels, store_path=state_dir / "channels.json")
    auth = AdminAuth(cfg.admin)
    auth.announce()

    # In modalità reale questo carica i modelli (può richiedere tempo/GPU).
    engines = build_engines(cfg)

    hls_manager = None
    if cfg.hls.enabled:
        from .hls import HlsManager
        hls_manager = HlsManager(cfg.hls, input_samplerate=engines.tts.samplerate)
        log.info("HLS abilitato (output: %s, ffmpeg: %s)",
                 cfg.hls.output_dir, cfg.hls.ffmpeg)
    # Pre-volo: una lingua senza voce TTS non può essere trasmessa. Meglio
    # accorgersene all'avvio che dal pubblico che sceglie un canale muto.
    missing = [l for l in cfg.target_languages if not engines.tts.has_voice(l)]
    if missing:
        log.warning(
            "voci TTS mancanti per: %s — quelle lingue non verranno offerte "
            "(python scripts/download_models.py --languages %s)",
            ", ".join(missing), " ".join(missing),
        )
    if not cfg.ws_audio_allowed:
        log.info(
            "audio PCM su WebSocket disabilitato (consegna via HLS): "
            "delivery.ws_audio_with_hls=true per riattivarlo"
        )

    orchestrator = Orchestrator(cfg, registry, hub, engines, hls=hls_manager)

    ctx = AppContext(
        cfg=cfg, hub=hub, registry=registry,
        engines=engines, orchestrator=orchestrator, hls=hls_manager, auth=auth,
    )
    app.state.ctx = ctx
    log.info("avvio in modalità '%s' su %s:%d",
             cfg.engine.mode, cfg.server.host, cfg.server.port)

    # Preparazione pesante (misura delle voci installate) PRIMA di aprire i
    # canali: farla a cattura attiva significa perdere blocchi audio.
    engines.prepare()

    # Warmup dei kernel GPU in background: la prima inferenza è molto più lenta.
    import threading
    threading.Thread(target=engines.warmup, name="warmup", daemon=True).start()

    # Avvia i canali abilitati (la traduzione/TTS resta lazy sugli ascoltatori).
    for ch in registry.list():
        if ch.enabled:
            try:
                orchestrator.start_channel(ch.id)
            except Exception:
                log.exception("impossibile avviare il canale '%s'", ch.id)

    try:
        yield
    finally:
        log.info("arresto orchestratore…")
        orchestrator.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Instant Translator", version="0.2.0", lifespan=lifespan)

    app.include_router(admin.router)
    app.include_router(listener.router)
    app.include_router(hls_api.router)
    app.include_router(health.router)

    # Frontend statico. In produzione conviene farlo servire da nginx
    # direttamente da disco (vedi deploy/nginx.conf): 2500 telefoni che
    # scaricano pagina e JS non devono passare dal processo della pipeline.
    static_dir = WEB_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # La pagina è identica per tutti: qualche secondo di cache assorbe il picco
    # iniziale ("aprite il link adesso" → migliaia di richieste in pochi secondi).
    page_headers = {"Cache-Control": "public, max-age=30"}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"), headers=page_headers)

    @app.get("/admin")
    async def admin_page() -> FileResponse:
        return FileResponse(str(WEB_DIR / "admin.html"), headers={"Cache-Control": "no-store"})

    return app


app = create_app()
