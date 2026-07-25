"""Orchestratore: collega cattura -> sorgente -> traduzione -> TTS -> Hub.

Per ogni canale "in esecuzione" girano due thread:
  1. la :class:`ChannelSource` (ASR reale o generatore mock) che emette
     :class:`SourceSegment`;
  2. un worker di *fan-out* che, per ogni lingua con ascoltatori attivi,
     traduce il testo e (sui finali) sintetizza l'audio TTS, pubblicando
     tutto sull'Hub.

Ottimizzazioni per GPU singola + tanti ascoltatori:
  - ASR una sola volta per canale;
  - traduzione/TTS solo per le lingue con ascoltatori o in broadcast HLS;
  - parziali generati solo se qualcuno li sta guardando (costano ASR);
  - TTS generato una volta e riusato da WebSocket e HLS.

La cattura audio è **condivisa**: uno stream per device, riconfigurato a caldo.
Avviare o fermare un canale non interrompe l'audio degli altri.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from ..config import AppConfig
from ..models import AudioChunk, SourceSegment
from ..registry import ChannelRegistry
from ..state import Hub, topic_audio, topic_subtitle, topic_transcript
from .base import ChannelSource, Engines

log = logging.getLogger("instanttranslator.orchestrator")


@dataclass
class _ChannelRuntime:
    source: ChannelSource
    stop: threading.Event
    seg_queue: "queue.Queue[SourceSegment]"
    source_thread: threading.Thread
    fanout_thread: threading.Thread


class Orchestrator:
    def __init__(
        self,
        cfg: AppConfig,
        registry: ChannelRegistry,
        hub: Hub,
        engines: Engines,
        hls=None,  # HlsManager | None
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.hub = hub
        self.engines = engines
        self.hls = hls
        self._runtimes: dict[str, _ChannelRuntime] = {}
        self._lock = threading.RLock()
        self._uses_audio = cfg.engine.mode == "real"
        self._capture = None  # CaptureManager, solo in modalità reale

    def _broadcast_languages(self, channel_id: str) -> list[str]:
        """Lingue trasmesse in HLS: solo quelle davvero sintetizzabili.

        Trasmettere una lingua senza voce TTS installata significa mandare in
        onda uno stream muto che il pubblico sceglie e non sente: meglio non
        offrirla affatto.
        """
        ch = self.registry.get(channel_id)
        if ch is None:
            return []
        from ..languages import exists

        langs = ch.broadcast_languages or self.cfg.target_languages
        wanted = [l for l in langs if exists(l)]
        # Trasmettere una lingua fuori da target_languages costa un TTS e un
        # ffmpeg per canale senza che nessuno possa selezionarla.
        targets = self.cfg.target_languages
        if targets:
            off_target = [l for l in wanted if l not in targets]
            if off_target:
                log.warning(
                    "canale '%s': %s non è in target_languages, non verrà "
                    "trasmessa (aggiungila alle target o togliela da "
                    "broadcast_languages)", channel_id, ", ".join(off_target),
                )
            wanted = [l for l in wanted if l in targets]
        usable = [l for l in wanted if self.engines.tts.has_voice(l)]
        missing = [l for l in wanted if l not in usable]
        if missing:
            log.warning(
                "canale '%s': lingue senza voce TTS installata, non verranno "
                "trasmesse: %s (scaricale con scripts/download_models.py)",
                channel_id, ", ".join(missing),
            )
        return usable

    # -- query ---------------------------------------------------------------
    def is_running(self, channel_id: str) -> bool:
        return channel_id in self._runtimes

    def running_channels(self) -> set[str]:
        return set(self._runtimes)

    def capture_running(self) -> bool:
        return self._capture is not None and self._capture.running

    def capture_health(self) -> dict:
        if self._capture is None:
            return {"running": False}
        health = self._capture.health()
        health["running"] = self._capture.running
        health["stalled"] = self._capture.stalled()
        return health

    def levels(self) -> dict[str, dict]:
        """Livelli per canale (regia): serve a verificare il routing del mixer."""
        return self._capture.levels() if self._capture is not None else {}

    def source_stats(self, channel_id: str) -> dict:
        """Stato del VAD/ASR del canale (rumore di fondo, soglia, parlato)."""
        rt = self._runtimes.get(channel_id)
        return rt.source.stats() if rt is not None else {}

    # -- start / stop --------------------------------------------------------
    def start_channel(self, channel_id: str) -> None:
        with self._lock:
            if channel_id in self._runtimes:
                return
            ch = self.registry.get(channel_id)
            if ch is None:
                raise KeyError(channel_id)

            stop = threading.Event()
            seg_queue: queue.Queue[SourceSegment] = queue.Queue(maxsize=128)
            source = self.engines.create_source(ch.id, ch.source_language)
            # I parziali costano un'inferenza ASR sull'intero buffer: generali
            # solo se c'è davvero un monitor o un ascoltatore WS che li legge.
            source.wants_partials = self._make_partials_predicate(channel_id)

            def emit(seg: SourceSegment) -> None:
                try:
                    seg_queue.put_nowait(seg)
                except queue.Full:
                    try:  # scarta il più vecchio, mantieni il flusso
                        seg_queue.get_nowait()
                        seg_queue.put_nowait(seg)
                    except (queue.Empty, queue.Full):
                        pass

            source_thread = threading.Thread(
                target=self._run_source, args=(source, emit, stop),
                name=f"src-{channel_id}", daemon=True,
            )
            fanout_thread = threading.Thread(
                target=self._run_fanout, args=(channel_id, seg_queue, stop),
                name=f"fan-{channel_id}", daemon=True,
            )

            self._runtimes[channel_id] = _ChannelRuntime(
                source=source, stop=stop, seg_queue=seg_queue,
                source_thread=source_thread, fanout_thread=fanout_thread,
            )

            # Se la cattura o l'HLS non partono, il canale NON resta registrato
            # a metà: si annulla tutto e l'errore arriva alla regia.
            try:
                # Prima l'HLS, poi la cattura: le rotte di cattura decidono se
                # alimentare lo stream FLOOR guardando se esiste, quindi deve
                # esistere già (altrimenti l'audio originale resta muto).
                if self.hls is not None:
                    self.hls.start_channel(
                        channel_id, self._broadcast_languages(channel_id),
                        floor=self.cfg.hls.floor,
                        floor_samplerate=self.cfg.audio.samplerate,
                    )
                if self._uses_audio:
                    self._sync_capture()
            except Exception:
                self._runtimes.pop(channel_id, None)
                stop.set()
                if self._uses_audio:
                    try:
                        self._sync_capture()
                    except Exception:
                        log.exception("rollback cattura fallito")
                if self.hls is not None:
                    self.hls.stop_channel(channel_id)
                raise

            source_thread.start()
            fanout_thread.start()
            log.info("canale '%s' avviato", channel_id)

    def stop_channel(self, channel_id: str) -> None:
        with self._lock:
            rt = self._runtimes.pop(channel_id, None)
            if rt is None:
                return
            rt.stop.set()
            if self._uses_audio:
                try:
                    self._sync_capture()
                except Exception:
                    log.exception("riconfigurazione cattura dopo stop fallita")
            if self.hls is not None:
                self.hls.stop_channel(channel_id)
        # join fuori dal lock per non bloccare altre operazioni
        rt.source_thread.join(timeout=3.0)
        rt.fanout_thread.join(timeout=3.0)
        log.info("canale '%s' fermato", channel_id)

    def shutdown(self) -> None:
        for cid in list(self._runtimes):
            self.stop_channel(cid)
        if self._capture is not None:
            self._capture.stop()
        if self.hls is not None:
            self.hls.shutdown()

    # -- thread di lavoro ----------------------------------------------------
    def _make_partials_predicate(self, channel_id: str):
        hub = self.hub

        def wants() -> bool:
            if not self.cfg.engine.vad.partials:
                return False
            if hub.has_subscribers(topic_transcript(channel_id)):
                return True
            return any(
                hub.has_subscribers(topic_subtitle(channel_id, lang))
                for lang in hub.active_target_languages(channel_id)
            )

        return wants

    def _run_source(self, source: ChannelSource, emit, stop: threading.Event) -> None:
        try:
            source.run(emit, stop)
        except Exception:  # pragma: no cover
            log.exception("sorgente canale '%s' terminata con errore", source.channel_id)

    def _run_fanout(
        self, channel_id: str, seg_queue: "queue.Queue[SourceSegment]",
        stop: threading.Event,
    ) -> None:
        translator = self.engines.translator
        tts = self.engines.tts
        while not stop.is_set():
            try:
                seg = seg_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Se siamo in ritardo, i parziali arretrati non servono più a nessuno:
            # meglio recuperare e restare vicini al parlato.
            if not seg.is_final and seg_queue.qsize() > 2:
                continue

            # Monitor regia: testo sorgente.
            self.hub.publish_threadsafe(topic_transcript(channel_id), {
                "type": "transcript", "channel": channel_id, "seq": seg.seq,
                "text": seg.text, "final": seg.is_final,
                "lang": seg.source_language, "ts": seg.ts,
            })

            # Sottotitolo sullo stream FLOOR (audio originale): testo sorgente.
            # Ancorato all'inizio dell'enunciato: l'audio originale è già andato
            # in onda mentre l'ASR trascriveva, quindi il cue non va marcato
            # "adesso" (arriverebbe qualche secondo dopo la voce).
            if self.hls is not None and seg.is_final and self.hls.has_floor(channel_id):
                self.hls.add_floor_subtitle(
                    channel_id, seg.seq, seg.text,
                    start=seg.t_start or seg.ts,
                    end=seg.t_end or seg.ts,
                )

            # Lingue da elaborare = ascoltatori WS attivi + lingue in broadcast HLS.
            ws_langs = self.hub.active_target_languages(channel_id)
            hls_langs = self.hls.languages(channel_id) if self.hls else set()
            rt = self._runtimes.get(channel_id)
            speaker = rt.source.speaker_profile() if rt is not None else None

            for lang in ws_langs | hls_langs:
                sub_topic = topic_subtitle(channel_id, lang)
                audio_topic = topic_audio(channel_id, lang)
                is_final = seg.is_final
                want_sub_ws = self.hub.has_subscribers(sub_topic)
                want_audio_ws = is_final and self.hub.has_subscribers(audio_topic)
                hls_stream = self.hls.get(channel_id, lang) if (self.hls and is_final) else None

                # Niente da fare per questa lingua? salta (risparmio traduzione/TTS).
                if not (want_sub_ws or want_audio_ws or hls_stream):
                    continue

                try:
                    text = translator.translate(seg.text, seg.source_language, lang)
                except Exception:
                    log.exception("errore traduzione %s->%s", seg.source_language, lang)
                    continue

                if want_sub_ws:
                    self.hub.publish_threadsafe(sub_topic, {
                        "type": "subtitle", "channel": channel_id, "lang": lang,
                        "seq": seg.seq, "text": text, "final": is_final,
                        "ts": seg.ts,
                    })

                # TTS una sola volta, riusato sia per WS sia per HLS.
                if want_audio_ws or hls_stream:
                    try:
                        # Il profilo vocale del canale avvicina la voce
                        # sintetica a quella del microfono; la durata
                        # dell'originale evita che la traduzione si allunghi.
                        pcm = tts.synthesize(
                            text, lang, speaker=speaker,
                            target_duration=(seg.t_end - seg.t_start) or None,
                        )
                    except Exception:
                        log.exception("errore TTS lingua %s", lang)
                        continue
                    if not pcm:
                        continue
                    if want_audio_ws:
                        self.hub.publish_threadsafe(audio_topic, AudioChunk(
                            channel_id=channel_id, target_language=lang,
                            seq=seg.seq, samplerate=tts.samplerate, pcm=pcm,
                        ))
                    if hls_stream is not None:
                        self.hls.feed(channel_id, lang, seg.seq, pcm, text)

    # -- cattura audio (solo reale) ------------------------------------------
    def _sync_capture(self) -> None:
        """Allinea le rotte di cattura ai canali in esecuzione.

        Non chiude gli stream già aperti se il layout non cambia: durante un
        evento far ripartire la scheda per aggiungere un canale significa un
        buco d'audio su *tutti* i canali.
        """
        from ..audio.capture import CaptureManager, Route
        from ..audio.devices import resolve_device

        if self._capture is None:
            self._capture = CaptureManager(
                samplerate=self.cfg.audio.samplerate,
                blocksize=self.cfg.audio.blocksize,
                device_samplerate=self.cfg.audio.device_samplerate,
                latency=self.cfg.audio.latency,
                wasapi_exclusive=self.cfg.audio.wasapi_exclusive,
            )

        routes: list[Route] = []
        for cid, rt in self._runtimes.items():
            ch = self.registry.get(cid)
            if ch is None:
                continue
            spec = ch.input_device if ch.input_device is not None \
                else self.cfg.audio.default_input_device
            device = resolve_device(spec, self.cfg.audio.host_api)
            gain = 10.0 ** (ch.gain_db / 20.0) if ch.gain_db else 1.0

            floor_idx = ch.floor_channel_index
            has_floor = self.hls is not None and self.hls.has_floor(cid)
            # Il FLOOR può venire da un canale diverso (es. il mix del mixer):
            # in quel caso è una rotta a sé, altrimenti si aggancia all'ASR.
            same_floor = has_floor and (floor_idx is None or floor_idx == ch.channel_index)

            routes.append(Route(
                key=cid, device=device, channel_index=ch.channel_index,
                callback=self._make_tap(cid, rt, floor=same_floor), gain=gain,
            ))
            if has_floor and not same_floor:
                routes.append(Route(
                    key=f"{cid}:orig", device=device, channel_index=int(floor_idx),
                    callback=self._make_floor_tap(cid), gain=gain,
                ))

        self._capture.set_routes(routes)

    def _make_tap(self, channel_id: str, rt: "_ChannelRuntime", *, floor: bool):
        """Callback di cattura: alimenta l'ASR e (se richiesto) lo stream FLOOR."""
        feed_asr = rt.source.feed_audio
        if not floor:
            return feed_asr

        feed_floor = self._make_floor_tap(channel_id)

        def tap(block) -> None:  # noqa: ANN001
            feed_asr(block)
            feed_floor(block)

        return tap

    def _make_floor_tap(self, channel_id: str):
        import numpy as np

        hls = self.hls

        def tap(block) -> None:  # noqa: ANN001
            # float32 [-1,1] -> int16 PCM per il passthrough HLS originale.
            pcm = (np.clip(block, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            hls.feed_floor(channel_id, pcm)

        return tap
