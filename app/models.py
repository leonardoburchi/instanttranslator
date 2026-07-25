"""Schemi Pydantic per l'API e i messaggi runtime.

Convenzione: i modelli ``*Info`` sono risposte API verso il frontend; i
modelli ``*Create``/``*Update`` sono richieste della regia; le dataclass
in fondo sono messaggi interni della pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator

from .naming import normalize_id


# --------------------------------------------------------------------------- #
# API: lingue                                                                  #
# --------------------------------------------------------------------------- #
class LanguageInfo(BaseModel):
    code: str
    name: str
    english_name: str
    flag: str


# --------------------------------------------------------------------------- #
# API: device audio                                                            #
# --------------------------------------------------------------------------- #
class DeviceInfo(BaseModel):
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi: str


# --------------------------------------------------------------------------- #
# API: canali                                                                  #
# --------------------------------------------------------------------------- #
class ChannelInfo(BaseModel):
    id: str
    name: str
    description: str
    source_language: str
    input_device: int | str | None
    channel_index: int
    gain_db: float = 0.0
    floor_channel_index: int | None = None
    enabled: bool
    broadcast_languages: list[str] | None = None
    running: bool = False
    listeners: int = 0


class ChannelCreate(BaseModel):
    # Normalizzato in ingresso: l'id diventa parte dell'URL HLS e del path su
    # disco, quindi "Test 1" viene accettato ma salvato come "test-1".
    id: str
    name: str = "Canale"
    description: str = ""
    source_language: str = "it"
    input_device: int | str | None = None
    channel_index: int = 0
    gain_db: float = 0.0
    floor_channel_index: int | None = None
    enabled: bool = True
    broadcast_languages: list[str] | None = None

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return normalize_id(value)


class ChannelUpdate(BaseModel):
    """Tutti i campi opzionali: PATCH parziale dalla regia."""

    name: str | None = None
    description: str | None = None
    source_language: str | None = None
    input_device: int | str | None = None
    channel_index: int | None = None
    gain_db: float | None = None
    floor_channel_index: int | None = None
    enabled: bool | None = None
    broadcast_languages: list[str] | None = None


# --------------------------------------------------------------------------- #
# Messaggi pipeline (interni)                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class SourceSegment:
    """Trascrizione (parziale o finale) nella lingua sorgente di un canale."""

    channel_id: str
    seq: int                 # numero progressivo del segmento
    text: str
    is_final: bool
    source_language: str
    # Epoch (time.time()) di inizio/fine dell'audio da cui viene il testo: serve
    # a sincronizzare i sottotitoli con l'audio già andato in onda (stream FLOOR).
    t_start: float = 0.0
    t_end: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class TranslatedSegment:
    """Segmento tradotto verso una lingua target, pronto per sottotitolo/TTS."""

    channel_id: str
    seq: int
    target_language: str
    text: str
    is_final: bool
    source_text: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class AudioChunk:
    """Audio TTS PCM int16 mono per una coppia (canale, lingua)."""

    channel_id: str
    target_language: str
    seq: int
    samplerate: int
    pcm: bytes               # int16 little-endian, mono
