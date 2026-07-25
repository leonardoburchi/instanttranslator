"""ASR reale con faster-whisper + segmentazione del parlato.

``WhisperASR`` è il backend condiviso (un modello su GPU, accesso serializzato
da un lock: con una sola GPU i canali si mettono in coda sull'inferenza) e
filtra le **allucinazioni**: su audio senza parlato Whisper produce frasi
inventate (ringraziamenti, sigle da sottotitoli) che in una piazza finirebbero
sullo schermo di 2500 telefoni.

``RealChannelSource`` riceve l'audio dalla cattura (``feed_audio``), individua
gli enunciati con un VAD a energia **adattivo** e chiama il backend per
trascrivere, emettendo parziali durante il parlato e un finale alla pausa.

Perché adattivo: in una piazza il fondo non è mai zero (PA, applausi, bleed tra
microfoni aperti sul palco) e cambia durante l'evento. Una soglia fissa o non
apre mai (mic a basso guadagno) o non chiude mai (fondo alto, enunciati tagliati
al limite dei secondi massimi). La soglia segue quindi il rumore di fondo del
singolo canale, con isteresi in chiusura e un *pre-roll* che recupera l'attacco
della prima sillaba.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from ..config import ASRConfig, StreamingConfig, VadConfig
from ..languages import get as get_lang
from ..models import SourceSegment
from ..voice import SpeakerProfile, estimate_f0
from .base import ChannelSource, EmitFn

log = logging.getLogger("instanttranslator.asr")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Frasi che Whisper "allucina" tipicamente sul non-parlato: vengono dai
# sottotitoli su cui è stato addestrato. Confronto su testo normalizzato.
_HALLUCINATIONS = (
    "sottotitoli e revisione a cura di",
    "sottotitoli a cura di",
    "sottotitoli creati dalla comunità amara.org",
    "amara.org",
    "grazie per aver guardato",
    "grazie per l'ascolto",
    "iscrivetevi al canale",
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "subtitles by",
    "transcription by",
    "gracias por ver",
    "suscríbete al canal",
    "merci d'avoir regardé",
    "abonnez-vous",
    "untertitel im auftrag des zdf",
    "untertitelung aufgrund der amara.org",
    "vielen dank für",
    "www.",
    "http",
)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_BRACKETED = re.compile(r"^[\(\[\{♪*].*[\)\]\}♪*]$")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def _strip_overlap(previous_tail: list[str], text: str, max_words: int = 3) -> str:
    """Toglie dall'inizio di ``text`` le parole già mandate in onda.

    Serve col commit incrementale: il buffer viene tagliato un po' *prima*
    dell'ultima parola committata (per non perdere l'attacco di quella dopo),
    quindi la passata successiva può ritrascrivere qualche parola già detta.
    """
    tokens = text.split()
    if not tokens or not previous_tail:
        return text
    norm = [_normalize(t) for t in tokens]
    for k in range(min(max_words, len(previous_tail), len(norm)), 0, -1):
        if previous_tail[-k:] == norm[:k]:
            return " ".join(tokens[k:]).strip()
    return text


def _looks_hallucinated(text: str) -> bool:
    """Euristiche su un singolo segmento trascritto."""
    stripped = text.strip()
    if not stripped:
        return True
    # Solo annotazioni non verbali: "(musica)", "[Applausi]", "♪ ... ♪".
    if _BRACKETED.match(stripped):
        return True
    norm = _normalize(stripped)
    if not norm:
        return True
    if any(marker in norm for marker in _HALLUCINATIONS):
        return True
    # Loop del decoder: la stessa parola che occupa quasi tutto il segmento.
    words = norm.split()
    if len(words) >= 8:
        most = Counter(words).most_common(1)[0][1]
        if most / len(words) > 0.6:
            return True
    return False


@dataclass
class Chunk:
    """Segmento trascritto con i tempi relativi all'inizio dell'audio passato."""

    start: float
    end: float
    text: str


# Fine frase: se un pezzo committato finisce così, è già una unità sensata da
# tradurre anche se corta.
_SENTENCE_END = (".", "!", "?", "…", ";", ":")


class WhisperASR:
    """Wrapper condiviso su un modello faster-whisper."""

    def __init__(self, cfg: ASRConfig) -> None:
        from faster_whisper import WhisperModel  # import pesante, lazy

        device = _resolve_device(cfg.device)
        compute_type = cfg.compute_type if device == "cuda" else "int8"
        log.info("carico Whisper '%s' su %s (%s)", cfg.model, device, compute_type)
        self._model = WhisperModel(cfg.model, device=device, compute_type=compute_type)
        self._cfg = cfg
        self._beam_size = cfg.beam_size
        self._lock = threading.Lock()

    def transcribe_segments(
        self, audio: np.ndarray, language: str, *, vad: bool = True
    ) -> list[Chunk]:
        """Trascrive e ritorna i segmenti **con i tempi**, già ripuliti.

        I segmenti scartati dai filtri restano nella lista con testo vuoto: i
        loro tempi servono comunque a far avanzare il buffer in streaming.
        """
        whisper_lang = get_lang(language).whisper if _known(language) else language
        # Serializza l'accesso alla GPU tra i canali.
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=whisper_lang,
                beam_size=self._beam_size,
                vad_filter=vad,
                condition_on_previous_text=False,
                no_speech_threshold=self._cfg.no_speech_threshold,
                log_prob_threshold=self._cfg.logprob_threshold,
            )
            out: list[Chunk] = []
            for seg in segments:
                text = seg.text.strip()
                if getattr(seg, "no_speech_prob", 0.0) > self._cfg.no_speech_threshold \
                        or getattr(seg, "avg_logprob", 0.0) < self._cfg.logprob_threshold \
                        or _looks_hallucinated(text):
                    log.debug("segmento scartato (allucinazione?): %r", text)
                    text = ""
                out.append(Chunk(
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    text=text,
                ))
        return out

    def transcribe_words(self, audio: np.ndarray, language: str) -> list[Chunk]:
        """Trascrive con i tempi **per parola** (per il commit incrementale).

        I segmenti di Whisper sono lunghi quanto una frase e si estendono fino
        alla fine dell'audio disponibile: troppo grossolani per decidere cosa è
        già definitivo. Le parole no.
        """
        whisper_lang = get_lang(language).whisper if _known(language) else language
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=whisper_lang,
                beam_size=self._beam_size,
                vad_filter=False,
                # Niente initial_prompt col testo già committato: misurato,
                # rallenta la decodifica e su audio incerto invita il modello a
                # "continuare la frase" inventando. Meglio nessun contesto.
                condition_on_previous_text=False,
                word_timestamps=True,
                no_speech_threshold=self._cfg.no_speech_threshold,
                log_prob_threshold=self._cfg.logprob_threshold,
            )
            words: list[Chunk] = []
            for seg in segments:
                if getattr(seg, "no_speech_prob", 0.0) > self._cfg.no_speech_threshold \
                        or getattr(seg, "avg_logprob", 0.0) < self._cfg.logprob_threshold \
                        or _looks_hallucinated(seg.text.strip()):
                    continue
                for w in (getattr(seg, "words", None) or []):
                    words.append(Chunk(
                        start=float(w.start), end=float(w.end), text=str(w.word),
                    ))
        return words

    def transcribe(self, audio: np.ndarray, language: str, *, vad: bool = True) -> str:
        chunks = self.transcribe_segments(audio, language, vad=vad)
        return " ".join(c.text for c in chunks if c.text).strip()


def _known(code: str) -> bool:
    from ..languages import exists
    return exists(code)


class RealChannelSource(ChannelSource):
    """Cattura -> VAD adattivo -> ASR, per un canale."""

    def __init__(
        self,
        channel_id: str,
        source_language: str,
        asr: WhisperASR,
        samplerate: int,
        vad: VadConfig | None = None,
        streaming: StreamingConfig | None = None,
        *,
        drop_repeats: bool = True,
    ) -> None:
        super().__init__(channel_id, source_language)
        self._asr = asr
        self._sr = samplerate
        self._vad = vad or VadConfig()
        self._streaming = streaming or StreamingConfig()
        self._drop_repeats = drop_repeats
        # ~25 s di margine: se l'ASR è in coda sulla GPU non perdiamo audio.
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._min_samples = int(self._vad.min_utterance_seconds * samplerate)
        self._max_samples = int(self._vad.max_utterance_seconds * samplerate)
        self._preroll_samples = int(self._vad.preroll_seconds * samplerate)
        self._seq = 0
        self._last_final = ""
        self._dropped_blocks = 0
        # Profilo vocale del parlante: alimenta la scelta e l'adattamento della
        # voce del TTS (vedi app/voice.py).
        self._profile = SpeakerProfile()
        # stato osservabile dalla regia
        self._noise = 0.0
        self._speaking = False

    def speaker_profile(self) -> SpeakerProfile:
        return self._profile

    def _learn_voice(self, audio: np.ndarray, text: str, duration: float) -> None:
        """Aggiorna il profilo vocale con l'audio appena mandato in onda."""
        if not text or audio is None or len(audio) == 0:
            return
        self._profile.update(
            estimate_f0(audio, self._sr), duration=duration, chars=len(text))

    # -- ingresso audio ------------------------------------------------------
    def feed_audio(self, mono_block: np.ndarray) -> None:
        try:
            self._q.put_nowait(mono_block)
        except queue.Full:
            self._dropped_blocks += 1
            log.debug("backlog audio canale %s: blocco scartato", self.channel_id)

    def stats(self) -> dict:
        def dbfs(v: float) -> float:
            return round(20.0 * float(np.log10(max(v, 1e-6))), 1)
        return {
            "speaking": self._speaking,
            "noise_dbfs": dbfs(self._noise),
            "threshold_dbfs": dbfs(self._open_threshold()),
            "dropped_blocks": self._dropped_blocks,
            **self._profile.as_dict(),
        }

    # -- VAD -----------------------------------------------------------------
    def _open_threshold(self) -> float:
        v = self._vad
        if v.mode == "fixed":
            return v.energy_threshold
        return max(self._noise * (10.0 ** (v.noise_margin_db / 20.0)), v.absolute_floor)

    # -- commit incrementale -------------------------------------------------
    def _stable_prefix(
        self, words: list[Chunk], previous: list[Chunk], buffer_seconds: float,
        *, relax: bool,
    ) -> int:
        """Quante parole iniziali si possono considerare definitive.

        Regola *local agreement*: una parola è definitiva se la passata
        precedente aveva **la stessa parola nella stessa posizione** e se dopo
        di lei c'è già abbastanza audio da escludere che Whisper la stia ancora
        rivedendo. Con ``relax`` la seconda condizione cade: serve quando il
        relatore non fa pause e il buffer cresce troppo.
        """
        margin = 0.0 if relax else self._streaming.stability_margin
        n = 0
        for i, word in enumerate(words):
            if word.end > buffer_seconds - margin:
                break
            if i >= len(previous) or _normalize(previous[i].text) != _normalize(word.text):
                break
            n = i + 1
        return n

    def _cuttable(self, words: list[Chunk], i: int) -> bool:
        """Si può tagliare l'audio dopo la parola ``i``?

        Solo se dopo di lei c'è una pausa vera: i tempi di Whisper sono
        approssimati e un taglio dentro la parola successiva la spezza in due
        ("Stasera" → "Stas" + "sera") o la fa sparire.
        """
        if i >= len(words) - 1:
            return True                     # nessuna parola dopo in questo buffer
        return words[i + 1].start - words[i].end >= self._streaming.min_cut_gap

    def _commit_point(self, words: list[Chunk], stable: int, *, relax: bool) -> int:
        """Dove tagliare dentro il prefisso stabile.

        Meglio chiudere su una punteggiatura: la traduzione di una frase intera
        è molto migliore di quella di mezza. Se non ce n'è, si taglia comunque
        dopo abbastanza parole (e sempre in una pausa), altrimenti
        aspetteremmo la fine del discorso.
        """
        for i in range(stable - 1, -1, -1):
            if words[i].text.strip().endswith(_SENTENCE_END) and self._cuttable(words, i):
                return i + 1
        if stable < self._streaming.min_commit_words and not relax:
            return 0
        for i in range(stable - 1, -1, -1):
            if self._cuttable(words, i):
                return i + 1
        return 0

    def run(self, emit: EmitFn, stop: threading.Event) -> None:
        v = self._vad
        st = self._streaming
        buffer: list[np.ndarray] = []
        buffered = 0
        buf_epoch = 0.0          # epoch del primo campione nel buffer
        preroll: deque[np.ndarray] = deque()
        preroll_len = 0
        in_speech = False
        candidate = 0
        silence_run = 0.0
        last_pass = 0.0
        t_start = 0.0
        t_last_voice = 0.0
        previous: list[Chunk] = []
        last_tail: list[str] = []      # ultime parole già mandate in onda

        min_rms = float("inf")

        def publish(text: str, t0: float, t1: float, *, final: bool) -> None:
            nonlocal last_tail
            if not text:
                return
            if final:
                text = _strip_overlap(last_tail, text)
                if not text:
                    return
                if self._drop_repeats and text == self._last_final:
                    return
                self._last_final = text
                last_tail = [_normalize(w) for w in text.split()][-4:]
                self._seq += 1
            emit(SourceSegment(
                channel_id=self.channel_id, seq=self._seq + (0 if final else 1),
                text=text, is_final=final, source_language=self.source_language,
                t_start=t0, t_end=t1,
            ))

        def stream_pass() -> None:
            """Ri-trascrive il buffer e manda in onda la parte stabilizzata."""
            nonlocal buffer, buffered, buf_epoch, previous
            audio = np.concatenate(buffer)
            buffer_seconds = len(audio) / self._sr
            try:
                words = self._asr.transcribe_words(audio, self.source_language)
            except Exception:
                log.exception("ASR incrementale fallito sul canale %s", self.channel_id)
                return
            if not words:
                previous = []
                return

            relax = buffer_seconds >= st.max_pending_seconds
            stable = self._stable_prefix(words, previous, buffer_seconds, relax=relax)
            n = self._commit_point(words, stable, relax=relax)
            if n == 0:
                previous = words
                # Testo ancora in movimento: al monitor di regia interessa lo
                # stesso (è l'unico consumatore dei parziali).
                if self.wants_partials():
                    publish("".join(w.text for w in words).strip(),
                            buf_epoch, time.time(), final=False)
                return

            text = "".join(words[i].text for i in range(n)).strip()
            text_end = words[n - 1].end
            self._learn_voice(
                audio[int(words[0].start * self._sr):int(text_end * self._sr)],
                text, text_end - words[0].start,
            )
            # Dove tagliare l'audio: **nella pausa** tra l'ultima parola
            # committata e la successiva. Tagliare a metà parola lascia un
            # attacco monco all'inizio del buffer e la passata dopo trascrive
            # sillabe inventate. Se non c'è una parola dopo, si arretra un po'
            # per non mangiare l'attacco di quella che deve ancora arrivare.
            if n < len(words):
                nxt = words[n].start
                cut = (text_end + nxt) / 2.0 if nxt > text_end else nxt
            else:
                cut = max(text_end - st.trim_back_seconds, 0.0)
            publish(text, buf_epoch + words[0].start, buf_epoch + text_end, final=True)

            # L'audio committato esce dal buffer: le passate successive restano
            # corte (e la GPU non ri-trascrive ogni volta tutta la frase).
            cut_samples = min(max(int(cut * self._sr), 0), len(audio))
            remaining = audio[cut_samples:]
            buffer = [remaining] if len(remaining) else []
            buffered = len(remaining)
            buf_epoch += cut_samples / self._sr
            # Le parole non committate restano "già viste", con i tempi
            # ribasati sul nuovo buffer: la prossima passata può committarle
            # subito invece di ricominciare da capo la verifica di stabilità.
            offset = cut_samples / self._sr
            previous = [
                Chunk(start=w.start - offset, end=w.end - offset, text=w.text)
                for w in words[n:]
            ]
            if self.wants_partials():
                publish("".join(w.text for w in words[n:]).strip(),
                        buf_epoch, time.time(), final=False)

        def finalize(*, split: bool = False) -> None:
            """Chiude l'enunciato: trascrive ed emette quel che resta."""
            nonlocal buffer, buffered, in_speech, silence_run, candidate, previous
            nonlocal last_tail
            if buffered >= self._min_samples:
                audio = np.concatenate(buffer)
                try:
                    text = self._asr.transcribe(audio, self.source_language)
                except Exception:
                    log.exception("ASR fallito sul canale %s", self.channel_id)
                    text = ""
                self._learn_voice(audio, text, len(audio) / self._sr)
                publish(text, buf_epoch, t_last_voice or time.time(), final=True)
            buffer = []
            buffered = 0
            silence_run = 0.0
            previous = []
            if not split:   # con split il parlato continua: resta "in speech"
                # Enunciato chiuso: la prossima frase può ricominciare con le
                # stesse parole senza che vengano scambiate per una ripetizione.
                last_tail = []
                in_speech = False
                candidate = 0
                self._speaking = False

        while not stop.is_set():
            try:
                block = self._q.get(timeout=0.2)
            except queue.Empty:
                # Nessun blocco: la cattura è ferma. Chiudi l'enunciato in corso.
                if in_speech:
                    finalize()
                continue

            now = time.time()
            block_seconds = len(block) / self._sr
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) + 1e-9)
            open_th = self._open_threshold()
            close_th = open_th * (10.0 ** (-v.hysteresis_db / 20.0))

            if not in_speech:
                # Stima del rumore di fondo solo fuori dal parlato.
                if rms < open_th:
                    if self._noise <= 0.0:
                        self._noise = rms
                    else:
                        alpha = 1.0 - 0.5 ** (block_seconds / max(v.noise_halflife, 0.1))
                        self._noise += alpha * (rms - self._noise)

                preroll.append(block)
                preroll_len += len(block)
                while preroll and preroll_len - len(preroll[0]) >= self._preroll_samples:
                    preroll_len -= len(preroll.popleft())

                if rms >= open_th:
                    candidate += 1
                    # Due blocchi sopra soglia: un click non apre un enunciato.
                    if candidate >= 2:
                        in_speech = True
                        self._speaking = True
                        buffer = list(preroll)
                        buffered = sum(len(b) for b in buffer)
                        t_start = now - buffered / self._sr
                        buf_epoch = t_start
                        t_last_voice = now
                        last_pass = time.monotonic()
                        previous = []
                        min_rms = rms
                        preroll.clear()
                        preroll_len = 0
                else:
                    candidate = 0
                continue

            # --- dentro l'enunciato ---
            buffer.append(block)
            buffered += len(block)
            min_rms = min(min_rms, rms)
            if rms >= close_th:
                silence_run = 0.0
                t_last_voice = now
            else:
                silence_run += block_seconds

            if silence_run >= v.silence_seconds:
                finalize()
                continue
            if buffered >= self._max_samples:
                # Taglio per durata massima senza mai una pausa: o il relatore
                # non respira, o il fondo del canale è stabilmente sopra soglia
                # (PA che rientra nel microfono, applausi lunghi) e il gate è
                # rimasto aperto. Il livello *minimo* osservato in questa
                # finestra è una buona stima del fondo: alzando la stima la
                # soglia sale e il gate torna a chiudere da solo.
                if v.mode == "adaptive" and min_rms > self._noise:
                    log.info(
                        "canale %s: ricalibro il rumore di fondo a %.1f dBFS "
                        "(gate rimasto aperto per %.0f s)",
                        self.channel_id, 20 * np.log10(max(min_rms, 1e-6)),
                        v.max_utterance_seconds,
                    )
                    self._noise = min_rms
                finalize(split=True)
                min_rms = rms
                t_start = now
                buf_epoch = now
                last_pass = time.monotonic()
                continue

            mono = time.monotonic()
            if buffered < self._min_samples:
                continue

            if st.enabled:
                # Traduzione incrementale: non aspettiamo la fine della frase.
                if (mono - last_pass) >= st.interval:
                    stream_pass()
                    # Cadenza misurata *dopo* la passata: se la GPU è in coda,
                    # rallentiamo invece di accumulare lavoro arretrato.
                    last_pass = time.monotonic()
            elif self.wants_partials() and (mono - last_pass) >= v.partial_interval:
                last_pass = mono
                audio = np.concatenate(buffer)
                try:
                    text = self._asr.transcribe(audio, self.source_language, vad=False)
                except Exception:
                    log.exception("ASR parziale fallito sul canale %s", self.channel_id)
                    text = ""
                publish(text, t_start, now, final=False)

        if in_speech:
            finalize()
