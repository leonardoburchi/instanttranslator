"""Profilo vocale del parlante: altezza (F0) e ritmo.

Serve ad avvicinare la voce sintetica della traduzione a quella che entra dal
microfono: scegliere la voce Piper più vicina per registro, correggere il
residuo di altezza e far durare la traduzione più o meno quanto l'originale.

Non è clonazione: la voce resta quella del modello TTS. È il grado di
somiglianza che si ottiene senza aggiungere un modello (e senza torch a
runtime): stesso registro, stessa altezza indicativa, stesso passo. In pratica
la differenza tra "una voce a caso" e "una voce plausibile per quella persona".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("instanttranslator.voice")

# Intervallo di F0 considerato: copre voci maschili molto basse e femminili alte.
F0_MIN = 70.0
F0_MAX = 400.0


def estimate_f0(
    audio: np.ndarray, samplerate: int, *, max_seconds: float = 2.0
) -> float | None:
    """F0 mediana dei frame sonori (autocorrelazione via FFT).

    Ritorna ``None`` se non c'è abbastanza voce: meglio nessuna stima che una
    stima presa dal rumore di fondo.
    """
    if audio is None or len(audio) == 0:
        return None
    audio = np.asarray(audio, dtype=np.float32)
    # Su enunciati lunghi bastano i secondi centrali: costo limitato e stima
    # più stabile (inizio e fine sono spesso attacchi o code smorzate).
    max_samples = int(max_seconds * samplerate)
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]

    frame = int(0.04 * samplerate)
    hop = int(0.02 * samplerate)
    if frame < 64 or len(audio) < frame:
        return None
    lag_min = max(int(samplerate / F0_MAX), 2)
    lag_max = min(int(samplerate / F0_MIN), frame - 1)
    if lag_max <= lag_min:
        return None

    n_fft = 1 << (2 * frame - 1).bit_length()
    values: list[float] = []
    for start in range(0, len(audio) - frame, hop):
        x = audio[start:start + frame]
        x = x - float(np.mean(x))
        if float(np.sqrt(np.mean(x * x))) < 0.003:
            continue                      # frame troppo debole: non è voce
        spec = np.fft.rfft(x, n_fft)
        ac = np.fft.irfft(spec * np.conj(spec), n_fft)[:frame]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        window = ac[lag_min:lag_max]
        if len(window) == 0:
            continue
        lag = int(np.argmax(window)) + lag_min
        if ac[lag] < 0.25:
            continue                      # picco debole: frame non sonoro
        values.append(samplerate / lag)

    if len(values) < 3:
        return None
    return float(np.median(values))


@dataclass
class SpeakerProfile:
    """Stima progressiva della voce di un canale.

    Si aggiorna a media mobile su più enunciati: la voce del TTS non deve
    cambiare altezza a ogni frase perché una stima è andata storta.
    """

    f0: float = 0.0
    seconds_per_char: float = 0.0     # ritmo del parlato (durata / caratteri)
    samples: int = 0
    _f0_history: list[float] = field(default_factory=list, repr=False)

    # Prima di questo numero di enunciati la stima non viene usata.
    MIN_SAMPLES = 3

    @property
    def ready(self) -> bool:
        return self.samples >= self.MIN_SAMPLES and self.f0 > 0

    def update(self, f0: float | None, duration: float = 0.0, chars: int = 0) -> None:
        if f0 is not None and F0_MIN <= f0 <= F0_MAX:
            # Mediana sulle ultime stime: robusta agli errori di ottava, che
            # sono l'errore tipico di qualunque stimatore di F0.
            self._f0_history.append(f0)
            del self._f0_history[:-9]
            self.f0 = float(np.median(self._f0_history))
            self.samples += 1
        if duration > 0 and chars > 0:
            spc = duration / chars
            self.seconds_per_char = (
                spc if self.seconds_per_char <= 0
                else 0.7 * self.seconds_per_char + 0.3 * spc
            )

    def as_dict(self) -> dict:
        return {
            "f0_hz": round(self.f0, 1) if self.f0 else None,
            "voice_samples": self.samples,
            "ready": self.ready,
        }


def voice_language(voice_name: str) -> str:
    """``it_IT-riccardo-x_low`` -> ``it`` (le voci Piper si chiamano così)."""
    return voice_name.split("_", 1)[0].split("-", 1)[0].lower()
