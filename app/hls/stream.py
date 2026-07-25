"""Uno stream HLS per (canale, lingua): ffmpeg + pacer real-time + watchdog.

ffmpeg legge PCM int16 mono dallo stdin e produce una playlist HLS rolling
(segmenti AAC/TS) con ``EXT-X-PROGRAM-DATE-TIME``, che permette al player di
mappare la posizione di riproduzione su un orario assoluto (``hls.playingDate``
o ``getStartDate()`` su Safari) e quindi sincronizzare i sottotitoli.

Il TTS produce audio a raffiche (un enunciato, poi silenzio). HLS però vuole una
timeline continua: il **pacer** scrive verso ffmpeg a velocità reale, prendendo
dal buffer quando c'è parlato e riempiendo di silenzio altrimenti.

Per la tolleranza ai guasti un **supervisore** sorveglia il processo ffmpeg: se
muore (crash, pipe rotta) lo rilancia automaticamente, così lo stream si
ripristina da solo senza intervento.

Due dettagli che contano dietro una CDN:

* i **nomi dei segmenti non vengono mai riusati** (prefisso univoco per ogni
  avvio di ffmpeg): un ``seg_000000.ts`` rigenerato dopo un riavvio avrebbe
  contenuto diverso ma stesso URL, e la cache continuerebbe a servire l'audio
  vecchio per tutta la durata della sua validità;
* il **numero di media sequence non torna indietro** dopo un riavvio, altrimenti
  i player considerano la playlist non valida.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("instanttranslator.hls")


class HlsStream:
    def __init__(
        self,
        channel_id: str,
        lang: str,
        *,
        input_samplerate: int,
        output_dir: Path,
        ffmpeg: str = "ffmpeg",
        segment_time: float = 1.0,
        list_size: int = 10,
        delete_threshold: int = 30,
        codec: str = "aac",
        bitrate: str = "64k",
        output_samplerate: int = 44100,
        tick: float = 0.1,
        max_backlog: float = 4.0,
    ) -> None:
        self.channel_id = channel_id
        self.lang = lang
        self._sr = input_samplerate
        self._bytes_per_sec = input_samplerate * 2  # int16 mono
        self.out_dir = output_dir / channel_id / lang
        self._ffmpeg = ffmpeg
        self._segment_time = segment_time
        self._list_size = list_size
        self._delete_threshold = max(delete_threshold, 1)
        self._codec = codec
        self._bitrate = bitrate
        self._out_sr = output_samplerate
        self._tick = tick
        self._max_backlog_bytes = int(max(max_backlog, 0.5) * self._bytes_per_sec)

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._pacer: threading.Thread | None = None
        self._pacer_gen = 0
        self._supervisor: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self.restarts = 0
        self.dropped_seconds = 0.0
        # Audio reale ricevuto: distingue "stream vivo ma muto" (audio che non
        # arriva) da "stream vivo che sta trasmettendo". In diagnostica serve.
        self.fed_seconds = 0.0

    @property
    def playlist(self) -> Path:
        return self.out_dir / "audio.m3u8"

    # -- ciclo di vita -------------------------------------------------------
    def start(self) -> None:
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir, ignore_errors=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._started_at = time.time()
        self._spawn()
        self._supervisor = threading.Thread(
            target=self._supervise, name=f"hls-sup-{self.channel_id}-{self.lang}", daemon=True,
        )
        self._supervisor.start()
        log.info("HLS avviato %s/%s -> %s", self.channel_id, self.lang, self.out_dir)

    def _segment_pattern(self, gen: int) -> str:
        # Prefisso univoco per avvio: nessun URL viene mai riusato con contenuto
        # diverso, così le cache (nginx/CDN) non servono audio vecchio.
        run = f"{int(self._started_at) % 100000:05d}{gen:02d}"
        return str(self.out_dir / f"seg_{run}_%06d.ts")

    def _start_number(self, gen: int) -> int:
        """Media sequence iniziale, monotona crescente tra i riavvii."""
        elapsed = max(time.time() - self._started_at, 0.0)
        return int(elapsed / max(self._segment_time, 0.1)) + 1000 * gen

    def _build_cmd(self, gen: int) -> list[str]:
        flags = [
            "delete_segments", "program_date_time", "independent_segments",
            "omit_endlist",
        ]
        if gen > 1:
            # Riavvio: segnala ai player che la timeline ha uno stacco.
            flags.append("discont_start")
        return [
            self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "s16le", "-ar", str(self._sr), "-ac", "1", "-i", "pipe:0",
            "-c:a", self._codec, "-b:a", self._bitrate, "-ar", str(self._out_sr), "-ac", "1",
            "-f", "hls",
            "-hls_time", str(self._segment_time),
            "-hls_list_size", str(self._list_size),
            "-hls_delete_threshold", str(self._delete_threshold),
            "-hls_flags", "+".join(flags),
            "-hls_segment_type", "mpegts",
            "-hls_start_number_source", "generic",
            "-start_number", str(self._start_number(gen)),
            "-hls_segment_filename", self._segment_pattern(gen),
            str(self.playlist),
        ]

    def _spawn(self) -> None:
        """Avvia (o riavvia) ffmpeg + un nuovo pacer."""
        with self._proc_lock:
            self._pacer_gen += 1
            gen = self._pacer_gen
            self._proc = subprocess.Popen(
                self._build_cmd(gen), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self._pacer = threading.Thread(
            target=self._pace, args=(gen,),
            name=f"hls-pace-{self.channel_id}-{self.lang}", daemon=True,
        )
        self._pacer.start()

    def _supervise(self) -> None:
        """Rilancia ffmpeg se muore inaspettatamente e ripulisce i vecchi segmenti."""
        backoff = 0.5
        last_cleanup = time.time()
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if self._stop.is_set():
                break
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                self.restarts += 1
                log.warning(
                    "ffmpeg %s/%s morto (rc=%s), riavvio #%d",
                    self.channel_id, self.lang, proc.returncode, self.restarts,
                )
                self._stop.wait(backoff)
                if self._stop.is_set():
                    break
                try:
                    self._spawn()
                    backoff = 0.5
                except Exception:
                    log.exception("respawn ffmpeg %s/%s fallito", self.channel_id, self.lang)
                    backoff = min(backoff * 2, 5.0)
            if time.time() - last_cleanup > 30.0:
                last_cleanup = time.time()
                self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        """Rimuove i segmenti dei run precedenti (ffmpeg cancella solo i propri)."""
        keep = max(60.0, self._delete_threshold * self._segment_time * 3)
        cutoff = time.time() - keep
        try:
            for seg in self.out_dir.glob("seg_*.ts"):
                try:
                    if seg.stat().st_mtime < cutoff:
                        seg.unlink()
                except OSError:
                    pass
        except Exception:  # pragma: no cover
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._supervisor:
            self._supervisor.join(timeout=2.0)
        if self._pacer:
            self._pacer.join(timeout=2.0)
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
        shutil.rmtree(self.out_dir, ignore_errors=True)
        log.info("HLS fermato %s/%s", self.channel_id, self.lang)

    # -- salute --------------------------------------------------------------
    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def last_segment_age(self) -> float | None:
        """Secondi dall'ultimo aggiornamento della playlist (None se assente).

        Usiamo la playlist (un solo ``stat``) e non un glob dei segmenti: viene
        riscritta a ogni nuovo segmento e /healthz può essere interrogato spesso.
        """
        try:
            return time.time() - self.playlist.stat().st_mtime
        except OSError:
            return None

    def backlog_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / self._bytes_per_sec

    # -- alimentazione audio -------------------------------------------------
    def feed(self, pcm: bytes) -> tuple[float, float]:
        """Accoda PCM. Ritorna (air_epoch, durata): quando inizierà a sentirsi.

        Il buffer è limitato: la voce sintetizzata è spesso più lunga
        dell'originale e senza un tetto il ritardo crescerebbe per tutto
        l'evento. Oltre il limite si scarta l'audio più vecchio (uno stacco
        ora è meglio di due minuti di ritardo alla fine).
        """
        duration = len(pcm) / self._bytes_per_sec
        with self._lock:
            self.fed_seconds += duration
            self._buf.extend(pcm)
            excess = len(self._buf) - self._max_backlog_bytes
            if excess > 0:
                del self._buf[:excess]
                self.dropped_seconds += excess / self._bytes_per_sec
                log.warning(
                    "backlog HLS %s/%s oltre il limite: scartati %.1f s "
                    "(il TTS produce più audio del tempo reale)",
                    self.channel_id, self.lang, excess / self._bytes_per_sec,
                )
            ahead = (len(self._buf) - len(pcm)) / self._bytes_per_sec
        return time.time() + max(ahead, 0.0), duration

    # -- pacer ---------------------------------------------------------------
    def _pace(self, gen: int) -> None:
        """Scrive verso ffmpeg in tempo reale, allineandosi all'orologio."""
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.stdin is None:
            return
        stdin = proc.stdin
        t0 = time.monotonic()
        written = 0
        while not self._stop.is_set() and gen == self._pacer_gen:
            now = time.monotonic()
            target = int((now - t0) * self._bytes_per_sec)
            target -= target % 2  # confine di campione
            need = target - written
            if need > 0:
                with self._lock:
                    take = min(need, len(self._buf))
                    chunk = bytes(self._buf[:take])
                    del self._buf[:take]
                if take < need:
                    chunk += b"\x00" * (need - take)  # riempi di silenzio
                try:
                    stdin.write(chunk)
                    stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    # ffmpeg caduto: esce; il supervisore rilancia.
                    return
                written += need
            time.sleep(self._tick)
