"""Ciclo di vita degli stream HLS + store sottotitoli a rotazione.

I sottotitoli HLS non viaggiano su WebSocket (non scalerebbe a migliaia di
ascoltatori): vengono esposti come piccolo JSON cacheabile. Ogni cue porta un
``start``/``end`` in epoch assoluti, allineati all'air-time dell'audio, così il
client li mostra confrontandoli con ``hls.playingDate``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from ..config import HlsConfig
from .stream import HlsStream

# Pseudo-lingua dello stream FLOOR (audio originale, senza traduzione).
FLOOR_LANG = "orig"


@dataclass
class Cue:
    seq: int
    start: float   # epoch (s) in cui il testo inizia a sentirsi
    end: float
    text: str


class SubtitleStore:
    """Cue recenti per (canale, lingua), con finestra temporale limitata."""

    def __init__(self, window: float) -> None:
        self._window = window
        self._cues: dict[tuple[str, str], deque[Cue]] = {}
        self._lock = threading.Lock()

    def add(self, channel_id: str, lang: str, cue: Cue) -> None:
        key = (channel_id, lang)
        with self._lock:
            dq = self._cues.get(key)
            if dq is None:
                dq = deque(maxlen=200)
                self._cues[key] = dq
            dq.append(cue)

    def recent(self, channel_id: str, lang: str) -> list[Cue]:
        cutoff = time.time() - self._window
        with self._lock:
            dq = self._cues.get((channel_id, lang))
            if not dq:
                return []
            return [c for c in dq if c.end >= cutoff]

    def clear(self, channel_id: str, lang: str) -> None:
        with self._lock:
            self._cues.pop((channel_id, lang), None)


class HlsManager:
    def __init__(self, cfg: HlsConfig, input_samplerate: int) -> None:
        self.cfg = cfg
        self._input_sr = input_samplerate
        self._out_dir = Path(cfg.output_dir)
        self._streams: dict[tuple[str, str], HlsStream] = {}
        self._channel_langs: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self.subtitles = SubtitleStore(cfg.subtitle_window)

    def _make_stream(self, channel_id: str, lang: str, samplerate: int) -> HlsStream:
        return HlsStream(
            channel_id, lang,
            input_samplerate=samplerate,
            output_dir=self._out_dir,
            ffmpeg=self.cfg.ffmpeg,
            segment_time=self.cfg.segment_time,
            list_size=self.cfg.list_size,
            delete_threshold=self.cfg.delete_threshold,
            codec=self.cfg.audio_codec,
            bitrate=self.cfg.audio_bitrate,
            output_samplerate=self.cfg.output_samplerate,
            tick=self.cfg.pacer_tick,
            max_backlog=self.cfg.max_backlog,
        )

    # -- ciclo di vita per canale -------------------------------------------
    def start_channel(
        self, channel_id: str, languages: list[str],
        *, floor: bool = False, floor_samplerate: int = 16000,
    ) -> None:
        with self._lock:
            all_langs = set(languages)
            # Lo stream FLOOR (audio originale) gira a parte, alimentato dal tap
            # della cattura, non dal TTS: serve un sample rate diverso.
            if floor and self.cfg.floor:
                key = (channel_id, FLOOR_LANG)
                if key not in self._streams:
                    s = self._make_stream(channel_id, FLOOR_LANG, floor_samplerate)
                    s.start()
                    self._streams[key] = s
                all_langs.add(FLOOR_LANG)

            self._channel_langs[channel_id] = all_langs
            for lang in languages:
                key = (channel_id, lang)
                if key in self._streams:
                    continue
                stream = self._make_stream(channel_id, lang, self._input_sr)
                stream.start()
                self._streams[key] = stream

    def stop_channel(self, channel_id: str) -> None:
        with self._lock:
            langs = self._channel_langs.pop(channel_id, set())
            for lang in langs:
                stream = self._streams.pop((channel_id, lang), None)
                if stream:
                    stream.stop()
                self.subtitles.clear(channel_id, lang)

    def shutdown(self) -> None:
        for cid in list(self._channel_langs):
            self.stop_channel(cid)

    # -- query / feed --------------------------------------------------------
    def languages(self, channel_id: str) -> set[str]:
        """Lingue tradotte in broadcast (esclude lo stream FLOOR)."""
        return {l for l in self._channel_langs.get(channel_id, set()) if l != FLOOR_LANG}

    def has_floor(self, channel_id: str) -> bool:
        return (channel_id, FLOOR_LANG) in self._streams

    def get(self, channel_id: str, lang: str) -> HlsStream | None:
        return self._streams.get((channel_id, lang))

    def feed(self, channel_id: str, lang: str, seq: int, pcm: bytes, text: str) -> None:
        """Alimenta l'audio HLS e registra il sottotitolo all'air-time corretto."""
        stream = self._streams.get((channel_id, lang))
        if stream is None:
            return
        air, duration = stream.feed(pcm)
        air += self.cfg.subtitle_offset
        self.subtitles.add(channel_id, lang, Cue(
            seq=seq, start=air, end=air + duration, text=text,
        ))

    def feed_floor(self, channel_id: str, pcm: bytes) -> None:
        """Alimenta lo stream FLOOR con l'audio originale (dal tap della cattura)."""
        stream = self._streams.get((channel_id, FLOOR_LANG))
        if stream is not None:
            stream.feed(pcm)

    def add_floor_subtitle(
        self, channel_id: str, seq: int, text: str, *, start: float, end: float
    ) -> None:
        """Sottotitolo (testo sorgente) sullo stream FLOOR.

        ``start``/``end`` sono gli epoch dell'audio *originale* da cui il testo
        proviene: quell'audio è già passato nella timeline HLS mentre l'ASR
        trascriveva, quindi ancoriamo il cue lì e non all'istante corrente
        (altrimenti apparirebbe qualche secondo dopo la voce).
        """
        stream = self._streams.get((channel_id, FLOOR_LANG))
        if stream is None:
            return
        # L'audio originale entra in onda con il ritardo del buffer del pacer.
        air = start + stream.backlog_seconds() + self.cfg.subtitle_offset
        self.subtitles.add(channel_id, FLOOR_LANG, Cue(
            seq=seq, start=air, end=max(end + self.cfg.subtitle_offset, air + 1.5),
            text=text,
        ))

    # -- salute --------------------------------------------------------------
    def health(self) -> list[dict]:
        out = []
        for (cid, lang), s in self._streams.items():
            out.append({
                "channel": cid, "lang": lang,
                "alive": s.is_alive(), "restarts": s.restarts,
                "last_segment_age": s.last_segment_age(),
                "backlog_s": round(s.backlog_seconds(), 2),
                "dropped_s": round(s.dropped_seconds, 1),
                "fed_s": round(s.fed_seconds, 1),
            })
        return out
