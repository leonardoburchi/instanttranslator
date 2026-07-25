"""Contesto applicativo condiviso, esposto su ``app.state.ctx``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import AppConfig
from .languages import LANGUAGES, Language, all_languages
from .models import ChannelInfo, LanguageInfo
from .pipeline.base import Engines
from .pipeline.orchestrator import Orchestrator
from .registry import ChannelRegistry
from .state import Hub


@dataclass
class AppContext:
    cfg: AppConfig
    hub: Hub
    registry: ChannelRegistry
    engines: Engines
    orchestrator: Orchestrator
    hls: object | None = None  # HlsManager | None
    auth: object | None = None  # AdminAuth
    started_at: float = field(default_factory=time.time)

    @property
    def hls_enabled(self) -> bool:
        return self.hls is not None

    @property
    def ws_audio_allowed(self) -> bool:
        return self.cfg.ws_audio_allowed

    def audience_language_infos(self, channel_id: str) -> list[LanguageInfo]:
        """Lingue che l'ascoltatore può scegliere su questo canale, con il modo.

        Tre livelli di servizio, in ordine di costo:

        * ``hls``  – audio tradotto in onda (un TTS e un ffmpeg per lingua);
        * ``ws``   – audio a bassa latenza su WebSocket (~350 kbps e una
          connessione persistente per telefono: regia e sale piccole);
        * ``text`` – soli sottotitoli, serviti dalla stessa cache degli altri:
          costano una traduzione e scalano come l'audio.

        Così si possono offrire *tutte* le lingue senza moltiplicare per dodici
        il costo dell'evento: audio per quelle principali, testo per le altre.
        """
        broadcast = set(self.hls_languages(channel_id)) if self.hls is not None else set()
        show_all = self.cfg.delivery.audience_languages == "all" or self.hls is None
        out: list[LanguageInfo] = []
        for lang in self.target_languages():
            if lang.code in broadcast:
                mode = "hls"
            elif not show_all:
                continue
            elif self.hls is not None:
                mode = "text"          # niente audio: sottotitoli cacheabili
            else:
                mode = "ws"            # senza HLS si passa dal WebSocket
            out.append(LanguageInfo(
                code=lang.code, name=lang.name, english_name=lang.english_name,
                flag=lang.flag, mode=mode,
            ))
        return out

    def text_languages(self, channel_id: str) -> list[str]:
        """Lingue offerte come soli sottotitoli (traduzione sì, TTS no)."""
        return [
            l.code for l in self.audience_language_infos(channel_id) if l.mode == "text"
        ]

    def hls_languages(self, channel_id: str) -> list[str]:
        """Lingue effettivamente trasmesse in HLS per il canale (vuoto se off).

        Se il canale è in onda vale ciò che gli stream stanno realmente
        producendo; altrimenti la configurazione, filtrata sulle voci TTS
        disponibili (una lingua senza voce non va offerta).
        """
        if self.hls is None:
            return []
        ch = self.registry.get(channel_id)
        if ch is None:
            return []
        targets = self.cfg.target_languages
        if ch.broadcast_languages is not None:
            candidates = list(ch.broadcast_languages)
        else:
            # Come nell'orchestratore: la lingua del canale non si ritrasmette
            # col TTS, l'originale è già sullo stream FLOOR.
            candidates = [l for l in targets if l != ch.source_language]
        configured = [
            l for l in candidates
            if l in LANGUAGES and self.engines.tts.has_voice(l)
            and (not targets or l in targets)
        ]
        if self.orchestrator.is_running(channel_id):
            live = self.hls.languages(channel_id)  # type: ignore[attr-defined]
            return [l for l in configured if l in live]
        return configured

    def hls_floor(self, channel_id: str) -> bool:
        """True se il canale espone lo stream FLOOR (audio originale)."""
        return self.hls is not None and self.cfg.hls.floor

    # -- helper di presentazione --------------------------------------------
    def target_languages(self) -> list[Language]:
        codes = self.cfg.target_languages or list(LANGUAGES)
        return [LANGUAGES[c] for c in codes if c in LANGUAGES]

    def target_language_infos(self) -> list[LanguageInfo]:
        return [
            LanguageInfo(code=l.code, name=l.name, english_name=l.english_name, flag=l.flag)
            for l in self.target_languages()
        ]

    def all_language_infos(self) -> list[LanguageInfo]:
        return [
            LanguageInfo(code=l.code, name=l.name, english_name=l.english_name, flag=l.flag)
            for l in all_languages()
        ]

    def channel_info(self, channel_id: str) -> ChannelInfo | None:
        ch = self.registry.get(channel_id)
        if ch is None:
            return None
        return ChannelInfo(
            **ch.model_dump(),
            running=self.orchestrator.is_running(ch.id),
            listeners=self.hub.listener_count(ch.id),
        )

    def channel_infos(self) -> list[ChannelInfo]:
        out = []
        for ch in self.registry.list():
            out.append(ChannelInfo(
                **ch.model_dump(),
                running=self.orchestrator.is_running(ch.id),
                listeners=self.hub.listener_count(ch.id),
            ))
        return out
