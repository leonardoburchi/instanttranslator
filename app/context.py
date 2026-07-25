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
        """Lingue che l'ascoltatore può scegliere su questo canale.

        Con HLS attivo, per default si offrono solo le lingue realmente
        trasmesse: una lingua fuori broadcast ricadrebbe sull'audio PCM via
        WebSocket, cioè ~350 kbps e una connessione persistente sull'origine
        per ogni telefono. Con 2500 persone non è un'opzione.
        """
        codes = [l.code for l in self.target_languages()]
        if self.hls is not None and self.cfg.delivery.audience_languages == "broadcast":
            broadcast = set(self.hls_languages(channel_id))
            codes = [c for c in codes if c in broadcast]
        return [
            LanguageInfo(code=c, name=LANGUAGES[c].name,
                         english_name=LANGUAGES[c].english_name, flag=LANGUAGES[c].flag)
            for c in codes if c in LANGUAGES
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
        configured = [
            l for l in (ch.broadcast_languages or targets)
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
