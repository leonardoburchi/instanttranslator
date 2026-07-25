"""Astrazioni della pipeline: sorgente di canale, traduttore, TTS.

Separiamo tre responsabilità:

* ``ChannelSource``  produce :class:`SourceSegment` (testo nella lingua
  sorgente del canale). Nel mondo reale = cattura audio + VAD + ASR; nel
  mock = generatore di frasi a tempo. Gira su un thread dedicato per canale.
* ``Translator``     traduce testo da una lingua a un'altra (stateless).
* ``TTSEngine``      sintetizza testo in PCM int16 mono.

I motori reali sono pesanti (GPU): vengono istanziati una sola volta e
condivisi tra tutti i canali, con accesso serializzato a monte
nell'orchestratore quando serve.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

from ..models import SourceSegment

EmitFn = Callable[[SourceSegment], None]


class ChannelSource(ABC):
    """Sorgente di trascrizione per un singolo canale."""

    def __init__(self, channel_id: str, source_language: str) -> None:
        self.channel_id = channel_id
        self.source_language = source_language
        # Sostituita dall'orchestratore: i parziali costano un'inferenza ASR,
        # generarli quando nessuno li legge è GPU buttata.
        self.wants_partials: Callable[[], bool] = lambda: True

    def stats(self) -> dict:
        """Stato interno utile alla regia (livelli, VAD…). Vuoto di default."""
        return {}

    @abstractmethod
    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        """Loop bloccante: chiama ``emit`` per ogni segmento finché ``stop``."""

    def feed_audio(self, mono_block) -> None:  # noqa: ANN001
        """Riceve audio dalla cattura (solo sorgenti reali). No-op di default."""

    def speaker_profile(self):
        """Profilo vocale stimato del parlante (``SpeakerProfile`` o None)."""
        return None


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Traduce ``text`` da ``source`` a ``target`` (codici brevi: it, en...)."""


class TTSEngine(ABC):
    samplerate: int

    @abstractmethod
    def synthesize(
        self, text: str, lang: str, *, speaker=None, target_duration: float | None = None
    ) -> bytes:
        """Ritorna PCM int16 little-endian mono al sample rate ``self.samplerate``.

        ``speaker`` (profilo vocale del canale) e ``target_duration`` (durata
        dell'originale) servono ad avvicinare la voce sintetica a quella del
        microfono; un motore che non li supporta può ignorarli.
        """

    def has_voice(self, lang: str) -> bool:
        """Se esiste una voce per questa lingua (controllo pre-volo)."""
        return True

    def close(self) -> None:  # pragma: no cover - opzionale
        pass


class Engines:
    """Contenitore dei motori condivisi + factory delle sorgenti di canale."""

    def __init__(
        self,
        translator: Translator,
        tts: TTSEngine,
        source_factory: Callable[[str, str], ChannelSource],
        warmup: Callable[[], None] | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> None:
        self.translator = translator
        self.tts = tts
        self._source_factory = source_factory
        self._warmup = warmup
        self._prepare = prepare

    def prepare(self) -> None:
        """Lavoro pesante da fare *prima* di mettere in onda i canali.

        Misurare le voci installate satura la CPU per qualche secondo: farlo
        mentre la cattura è già attiva significa perdere blocchi audio. Il
        risultato resta in cache su disco, quindi si paga solo al primo avvio.
        """
        if self._prepare is not None:
            self._prepare()

    def create_source(self, channel_id: str, source_language: str) -> ChannelSource:
        return self._source_factory(channel_id, source_language)

    def warmup(self) -> None:
        """Pre-compila i kernel GPU con inferenze fittizie (no-op se non serve)."""
        if self._warmup is not None:
            self._warmup()
