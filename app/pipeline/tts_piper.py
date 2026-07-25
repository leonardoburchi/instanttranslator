"""TTS reale con Piper (ONNX), adattato alla voce del parlante.

Una voce per lingua viene caricata pigramente da ``voices_dir`` (file
``{nome}.onnx`` + ``{nome}.onnx.json``). L'output viene ricampionato al
sample rate configurato così il browser usa una frequenza unica per tutti.

Adattamento al parlante (``tts.match_speaker``), in tre passi:

1. **scelta della voce**: tra quelle installate per la lingua target si prende
   quella con l'altezza più vicina al parlante (in pratica: uomo → voce
   maschile, donna → voce femminile, senza doverlo configurare);
2. **correzione dell'altezza residua**: si sintetizza più lento del dovuto e si
   ricampiona, ottenendo uno spostamento di pitch a durata invariata. Lo
   spostamento è limitato (``max_pitch_shift``) perché tirare una voce fuori
   dal suo registro suona peggio di una voce che non somiglia;
3. **durata**: la traduzione dura circa quanto l'originale. Serve alla
   somiglianza, ma soprattutto evita che la voce tradotta accumuli ritardo per
   tutta la serata (le lingue non hanno la stessa lunghezza).

Non è clonazione della voce: per quella serve un modello zero-shot (vedi
README). Questo è il massimo grado di somiglianza a costo ~zero.
"""

from __future__ import annotations

import json
import logging
import threading
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

from ..config import TTSConfig
from ..languages import get as get_lang
from ..voice import SpeakerProfile, estimate_f0, voice_language
from .base import TTSEngine

log = logging.getLogger("instanttranslator.tts")

# Ripiego se una lingua non ha una frase di prova nel registro: meglio di
# niente, ma la stima è meno affidabile (vedi il commento in languages.py).
_PROBE_FALLBACK = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15."

# Per cambiare la voce di default serve un guadagno misurabile, non enorme:
# ~2% di altezza. Serve solo a non far ballare la scelta tra due voci
# praticamente equivalenti, non a impedire una scelta migliore.
_SWITCH_MARGIN = 0.03

# Quante sintesi reali usare per affinare l'altezza di una voce.
_F0_SAMPLES = 8

# Piper allunga il parlato quasi 1:1 con length_scale, ma lo comprime solo per
# metà (le durate minime dei fonemi fanno da pavimento): misurato su più voci.
# Per accorciare davvero bisogna quindi chiedere il doppio.
_COMPRESSION_GAIN = 2.0


def _resample_int16(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    if src_sr == dst_sr or not pcm:
        return pcm
    data = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = int(round(len(data) * dst_sr / src_sr))
    if n_out <= 0:
        return b""
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, data)
    return resampled.astype("<i2").tobytes()


def _resample_ratio(pcm: bytes, ratio: float) -> bytes:
    """Ricampiona di un fattore: cambia insieme durata e altezza."""
    if not pcm or abs(ratio - 1.0) < 1e-3:
        return pcm
    data = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = max(int(round(len(data) / ratio)), 1)
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype("<i2").tobytes()


class PiperTTS(TTSEngine):
    def __init__(self, cfg: TTSConfig, state_dir: str | Path = "data") -> None:
        self.samplerate = cfg.samplerate
        self._cfg = cfg
        self._voices_dir = Path(cfg.voices_dir)
        self._use_cuda = cfg.use_cuda
        self._length_scale = cfg.length_scale
        self._voices: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._load_lock = threading.Lock()
        self._syn_config: dict[float, object] = {}
        # Altezza misurata di ogni voce installata (cache su disco: misurarla
        # costa una sintesi, e all'avvio dell'evento non si aspetta).
        self._catalog_path = Path(state_dir) / "voice_profiles.json"
        self._catalog: dict[str, dict] = self._load_catalog()
        self._catalog_lock = threading.Lock()

    # -- catalogo voci -------------------------------------------------------
    def _load_catalog(self) -> dict[str, dict]:
        try:
            return json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_catalog(self) -> None:
        try:
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            self._catalog_path.write_text(
                json.dumps(self._catalog, indent=2), encoding="utf-8")
        except Exception:
            log.debug("cache profili voce non salvata", exc_info=True)

    def installed_voices(self, lang: str) -> list[str]:
        """Voci Piper presenti in ``voices_dir`` per quella lingua."""
        try:
            names = sorted(p.stem for p in self._voices_dir.glob("*.onnx"))
        except Exception:
            return []
        return [n for n in names if voice_language(n) == lang]

    def _probe_text(self, name: str) -> str:
        try:
            probe = get_lang(voice_language(name)).probe
        except KeyError:
            probe = ""
        return probe or _PROBE_FALLBACK

    def voice_f0(self, name: str) -> float | None:
        """Altezza media della voce (misurata una volta e messa in cache)."""
        cached = self._catalog.get(name)
        if cached is not None:
            return cached.get("f0")
        try:
            voice = self._load_voice(name)
            with self._lock_for(name):
                pcm, sr = self._raw_synthesize(
                    voice, self._probe_text(name), self._length_scale)
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            f0 = estimate_f0(audio, sr)
        except Exception as exc:
            log.warning("impossibile misurare la voce '%s': %s", name, exc)
            f0 = None
        with self._catalog_lock:
            entry = self._catalog.setdefault(name, {})
            entry["f0"] = f0
            entry["f0_samples"] = 1 if f0 else 0
            self._save_catalog()
        if f0:
            log.info("voce '%s': altezza misurata %.0f Hz", name, f0)
        return f0

    def _learn_voice_f0(self, name: str, pcm: bytes, samplerate: int) -> None:
        """Affina l'altezza della voce sulle sintesi vere.

        La frase di prova (cifre) ha un'intonazione da elenco e dà una stima
        più bassa del parlato reale. Misurando anche le frasi vere, l'altezza
        della voce e quella del parlante finiscono sulla stessa scala, che è
        ciò che serve per confrontarle.
        """
        entry = self._catalog.get(name) or {}
        if entry.get("f0_samples", 0) >= _F0_SAMPLES or not pcm:
            return
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        f0 = estimate_f0(audio, samplerate)
        if not f0:
            return
        with self._catalog_lock:
            entry = self._catalog.setdefault(name, {})
            n = entry.get("f0_samples", 0)
            old = entry.get("f0")
            entry["f0"] = f0 if not old else (old * n + f0) / (n + 1)
            entry["f0_samples"] = n + 1
            self._save_catalog()

    def prepare_catalog(self, languages) -> None:
        """Misura in anticipo le voci delle lingue in uso (chiamata dal warmup).

        Misurare richiede di caricare la voce: lo facciamo qui, in background,
        e non alla prima frase dell'evento. Le voci caricate solo per la misura
        vengono poi rilasciate (la stima resta in cache su disco: ai riavvii
        successivi non si carica più niente per scegliere).
        """
        if not self._cfg.match_speaker:
            return
        already = set(self._voices)
        for lang in languages:
            for name in self.installed_voices(lang):
                self.voice_f0(name)
        with self._load_lock:
            for name in list(self._voices):
                if name not in already:
                    self._voices.pop(name, None)

    def select_voice(self, lang: str, speaker: SpeakerProfile | None) -> str:
        """Voce da usare: la più vicina al parlante, o quella di default."""
        try:
            default = get_lang(lang).piper
        except KeyError:
            default = ""
        if not (self._cfg.match_speaker and speaker is not None and speaker.ready):
            return default

        def distance(name: str) -> float | None:
            f0 = self.voice_f0(name)
            # Distanza in ottave: la percezione dell'altezza è logaritmica.
            return abs(float(np.log2(f0 / speaker.f0))) if f0 else None

        default_dist = distance(default) if default else None
        best, best_dist = default, default_dist
        for name in self.installed_voices(lang):
            dist = distance(name)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = name, dist
        # Si abbandona la voce di default solo se il guadagno è netto.
        if best != default and default_dist is not None and best_dist is not None:
            if best_dist > default_dist - _SWITCH_MARGIN:
                return default
        return best

    # -- caricamento voci ----------------------------------------------------
    def has_voice(self, lang: str) -> bool:
        """Il file della voce esiste? Serve a non offrire lingue mute al pubblico."""
        try:
            name = get_lang(lang).piper
        except KeyError:
            return False
        if name in self._voices:
            return True
        return (self._voices_dir / f"{name}.onnx").exists() or bool(
            self.installed_voices(lang))

    def missing_voices(self, langs) -> list[str]:
        return [l for l in langs if not self.has_voice(l)]

    def _lock_for(self, name: str) -> threading.Lock:
        with self._load_lock:
            lock = self._locks.get(name)
            if lock is None:
                lock = self._locks[name] = threading.Lock()
            return lock

    def _load_voice(self, name: str):
        if name in self._voices:
            return self._voices[name]
        with self._load_lock:
            if name in self._voices:
                return self._voices[name]
            from piper import PiperVoice

            model = self._voices_dir / f"{name}.onnx"
            config = self._voices_dir / f"{name}.onnx.json"
            if not model.exists():
                raise FileNotFoundError(f"voce Piper mancante: {model}")
            log.info("carico voce Piper '%s' (cuda=%s)", name, self._use_cuda)
            self._voices[name] = PiperVoice.load(
                str(model), str(config), use_cuda=self._use_cuda)
            return self._voices[name]

    # -- sintesi -------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        lang: str,
        *,
        speaker: SpeakerProfile | None = None,
        target_duration: float | None = None,
    ) -> bytes:
        if not text.strip():
            return b""
        name = self.select_voice(lang, speaker)
        voice = self._load_voice(name)

        pitch = self._pitch_ratio(name, speaker)
        rate = self._rate_scale(name, text, target_duration)
        # Sintesi più lenta di `pitch`: il ricampionamento successivo la
        # riporta alla durata giusta alzando/abbassando l'altezza.
        length_scale = self._length_scale * rate * pitch

        with self._lock_for(name):
            pcm, voice_sr = self._raw_synthesize(voice, text, length_scale)
            self._learn_duration(name, text, len(pcm) / 2 / voice_sr, length_scale)
            # Misurata prima del ricampionamento: è l'altezza naturale della voce.
            self._learn_voice_f0(name, pcm, voice_sr)

        if abs(pitch - 1.0) > 1e-3:
            pcm = _resample_ratio(pcm, pitch)
        return _resample_int16(pcm, voice_sr, self.samplerate)

    def _pitch_ratio(self, name: str, speaker: SpeakerProfile | None) -> float:
        if not (self._cfg.match_speaker and speaker is not None and speaker.ready):
            return 1.0
        voice_f0 = self.voice_f0(name)
        if not voice_f0:
            return 1.0
        ratio = speaker.f0 / voice_f0
        limit = 1.0 + max(self._cfg.max_pitch_shift, 0.0)
        return float(min(max(ratio, 1.0 / limit), limit))

    def _rate_scale(
        self, name: str, text: str, target_duration: float | None
    ) -> float:
        """Quanto rallentare/accelerare per durare come l'originale."""
        if not (self._cfg.match_duration and target_duration and target_duration > 0):
            return 1.0
        spc = (self._catalog.get(name) or {}).get("spc")
        if not spc:
            return 1.0                      # non abbiamo ancora una stima
        expected = len(text) * spc          # durata prevista a length_scale=1
        if expected <= 0:
            return 1.0
        scale = target_duration / expected
        limit = 1.0 + max(self._cfg.max_rate_shift, 0.0)
        scale = min(max(scale, 1.0 / limit), limit)
        if scale < 1.0:
            # Comprimere costa il doppio di quanto si chiede (vedi costante).
            scale = max(1.0 - (1.0 - scale) * _COMPRESSION_GAIN, 0.7)
        return float(scale)

    def _learn_duration(
        self, name: str, text: str, duration: float, length_scale: float
    ) -> None:
        """Impara quanto dura un carattere con questa voce (media mobile).

        Solo dalle sintesi a velocità naturale: la durata non è proporzionale a
        ``length_scale`` (vedi ``_COMPRESSION_GAIN``), quindi imparare da una
        sintesi già compressa falserebbe la stima e innescherebbe un
        inseguimento tra le due correzioni.
        """
        base = self._length_scale or 1.0
        if duration <= 0 or not text or length_scale <= 0:
            return
        if abs(length_scale / base - 1.0) > 0.05:
            return
        spc = duration / len(text) / length_scale
        with self._catalog_lock:
            entry = self._catalog.setdefault(name, {})
            old = entry.get("spc")
            entry["spc"] = spc if not old else 0.8 * old + 0.2 * spc

    def _synthesis_config(self, length_scale: float):
        """Costruisce (e riusa) la SynthesisConfig di piper 1.4.x."""
        key = round(length_scale, 3)
        if key in self._syn_config:
            return self._syn_config[key]
        try:
            from piper import SynthesisConfig
            cfg = SynthesisConfig(length_scale=key)
        except Exception:
            cfg = False                     # versione senza SynthesisConfig
        self._syn_config[key] = cfg
        return cfg

    def _voice_samplerate(self, voice) -> int:
        cfg = getattr(voice, "config", None)
        return int(getattr(cfg, "sample_rate", self.samplerate))

    def _raw_synthesize(
        self, voice, text: str, length_scale: float
    ) -> tuple[bytes, int]:
        """Compatibilità tra versioni di piper-tts."""
        # piper-tts 1.4.x: synthesize() -> Iterable[AudioChunk].
        if hasattr(voice, "synthesize"):
            try:
                syn = self._synthesis_config(length_scale)
                chunks = list(
                    voice.synthesize(text, syn_config=syn) if syn
                    else voice.synthesize(text)
                )
                if chunks and hasattr(chunks[0], "audio_int16_bytes"):
                    pcm = b"".join(c.audio_int16_bytes for c in chunks)
                    sr = getattr(chunks[0], "sample_rate", None) or self._voice_samplerate(voice)
                    return pcm, int(sr)
            except TypeError:
                pass  # firma diversa (versione vecchia): usa i fallback sotto

        # API a stream (piper-tts 1.2/1.3): yield di bytes int16 mono.
        if hasattr(voice, "synthesize_stream_raw"):
            chunks = list(voice.synthesize_stream_raw(text))
            return b"".join(chunks), self._voice_samplerate(voice)

        # API a WAV: scrivi in un buffer e rileggi i sample.
        buf = BytesIO()
        with wave.open(buf, "wb") as wav:
            voice.synthesize(text, wav)
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            sr = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        return frames, int(sr)
