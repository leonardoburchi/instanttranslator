"""Implementazioni mock: nessun modello, nessuna GPU, nessun audio reale.

Servono a far girare e provare *tutto* il sistema (regia, landing, fan-out,
sottotitoli, player audio) senza scaricare modelli. La sorgente mock genera
frasi multilingue a tempo; il traduttore mock usa un dizionario delle stesse
frasi (così i sottotitoli sono traduzioni reali nel demo); il TTS mock produce
un tono con cadenza "parlata" così l'audio è udibile.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from ..models import SourceSegment
from .base import ChannelSource, EmitFn, Translator, TTSEngine

# Frasi di esempio, ognuna tradotta in più lingue. La sorgente emette la
# variante nella lingua del canale; il traduttore mock ritrova la riga e
# restituisce la variante nella lingua target.
MOCK_PHRASES: list[dict[str, str]] = [
    {
        "it": "Buongiorno a tutti e benvenuti a questo evento.",
        "en": "Good morning everyone and welcome to this event.",
        "es": "Buenos días a todos y bienvenidos a este evento.",
        "fr": "Bonjour à tous et bienvenue à cet événement.",
        "de": "Guten Morgen allerseits und willkommen zu dieser Veranstaltung.",
    },
    {
        "it": "Oggi parleremo di traduzione in tempo reale.",
        "en": "Today we will talk about real-time translation.",
        "es": "Hoy hablaremos sobre la traducción en tiempo real.",
        "fr": "Aujourd'hui, nous parlerons de traduction en temps réel.",
        "de": "Heute sprechen wir über Echtzeit-Übersetzung.",
    },
    {
        "it": "Il sistema separa i canali audio e traduce ogni flusso.",
        "en": "The system splits the audio channels and translates each stream.",
        "es": "El sistema separa los canales de audio y traduce cada flujo.",
        "fr": "Le système sépare les canaux audio et traduit chaque flux.",
        "de": "Das System trennt die Audiokanäle und übersetzt jeden Stream.",
    },
    {
        "it": "Potete scegliere la vostra lingua dal vostro telefono.",
        "en": "You can choose your language from your phone.",
        "es": "Pueden elegir su idioma desde su teléfono.",
        "fr": "Vous pouvez choisir votre langue depuis votre téléphone.",
        "de": "Sie können Ihre Sprache auf Ihrem Telefon auswählen.",
    },
    {
        "it": "Grazie per l'attenzione e buon proseguimento.",
        "en": "Thank you for your attention and enjoy the rest.",
        "es": "Gracias por su atención y que disfruten lo que sigue.",
        "fr": "Merci de votre attention et bonne continuation.",
        "de": "Vielen Dank für Ihre Aufmerksamkeit und alles Gute.",
    },
]


def _phrase_for(lang: str, index: int) -> str:
    row = MOCK_PHRASES[index % len(MOCK_PHRASES)]
    return row.get(lang) or row.get("en") or next(iter(row.values()))


class MockChannelSource(ChannelSource):
    """Genera frasi a tempo, con qualche parziale prima del finale."""

    def __init__(self, channel_id: str, source_language: str) -> None:
        super().__init__(channel_id, source_language)
        self._seq = 0

    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        index = 0
        while not stop.is_set():
            phrase = _phrase_for(self.source_language, index)
            self._seq += 1
            words = phrase.split()

            # Parziali: accumula parole simulando l'ASR in corso.
            partial = ""
            for w in words:
                if stop.is_set():
                    return
                partial = (partial + " " + w).strip()
                emit(SourceSegment(
                    channel_id=self.channel_id,
                    seq=self._seq,
                    text=partial,
                    is_final=False,
                    source_language=self.source_language,
                ))
                if stop.wait(0.35):
                    return

            # Finale.
            emit(SourceSegment(
                channel_id=self.channel_id,
                seq=self._seq,
                text=phrase,
                is_final=True,
                source_language=self.source_language,
            ))
            index += 1
            # Pausa tra una frase e l'altra.
            if stop.wait(2.5):
                return


class MockTranslator(Translator):
    def translate(self, text: str, source: str, target: str) -> str:
        if source == target:
            return text
        # Ritrova la frase nel dizionario e ritorna la variante target.
        for row in MOCK_PHRASES:
            if row.get(source) == text and target in row:
                return row[target]
        # Fallback (es. parziali o frasi non in dizionario).
        return f"[{target}] {text}"


class MockTTS(TTSEngine):
    """Genera un tono con cadenza 'parlata' di durata proporzionale al testo."""

    def __init__(self, samplerate: int = 22050) -> None:
        self.samplerate = samplerate

    def synthesize(
        self, text: str, lang: str, *, speaker=None, target_duration: float | None = None
    ) -> bytes:
        n_chars = max(len(text), 1)
        duration = min(max(n_chars * 0.055, 0.4), 8.0)
        if target_duration and target_duration > 0:
            duration = min(max(target_duration, 0.4), 8.0)
        n = int(duration * self.samplerate)
        t = np.arange(n, dtype=np.float32) / self.samplerate

        # Pitch leggermente diverso per lingua, così i canali si distinguono.
        # Se conosciamo l'altezza del parlante, il tono la segue: rende visibile
        # l'adattamento della voce anche in demo, senza modelli.
        base = 150.0 + (hash(lang) % 60)
        if speaker is not None and getattr(speaker, "ready", False):
            base = float(speaker.f0)
        carrier = np.sin(2 * math.pi * base * t)
        # Inviluppo a ~4.5 sillabe/sec per evocare il parlato.
        syllable = 0.5 + 0.5 * np.sin(2 * math.pi * 4.5 * t)
        # Fade in/out per evitare click.
        fade = np.minimum(np.minimum(t * 8, (duration - t) * 8), 1.0)
        fade = np.clip(fade, 0.0, 1.0)
        signal = 0.25 * carrier * syllable * fade

        pcm = (signal * 32767).astype("<i2").tobytes()
        return pcm
