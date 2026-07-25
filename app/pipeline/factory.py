"""Costruzione dei motori in base a ``engine.mode`` della config."""

from __future__ import annotations

import logging

from ..config import AppConfig
from .base import ChannelSource, Engines

log = logging.getLogger("instanttranslator.pipeline")


def build_engines(cfg: AppConfig) -> Engines:
    mode = cfg.engine.mode
    log.info("inizializzo motori in modalità '%s'", mode)

    if mode == "mock":
        from .mock import MockChannelSource, MockTranslator, MockTTS

        translator = MockTranslator()
        tts = MockTTS(samplerate=cfg.engine.tts.samplerate)

        def factory(channel_id: str, source_language: str) -> ChannelSource:
            return MockChannelSource(channel_id, source_language)

        return Engines(translator=translator, tts=tts, source_factory=factory)

    # --- modalità reale ---
    from .asr_whisper import RealChannelSource, WhisperASR
    from .translate_nllb import NLLBTranslator
    from .tts_piper import PiperTTS

    asr = WhisperASR(cfg.engine.asr)
    translator = NLLBTranslator(cfg.engine.translate)
    tts = PiperTTS(cfg.engine.tts, state_dir=cfg.state_dir)
    samplerate = cfg.audio.samplerate

    def factory(channel_id: str, source_language: str) -> ChannelSource:
        return RealChannelSource(
            channel_id, source_language, asr, samplerate, cfg.engine.vad,
            cfg.engine.streaming, drop_repeats=cfg.engine.asr.drop_repeats,
        )

    def warmup() -> None:
        """Inferenze fittizie per compilare i kernel CUDA (prima inferenza lenta)."""
        import time
        import numpy as np

        targets = cfg.target_languages or ["en"]
        try:
            t0 = time.time()
            noise = (np.random.randn(samplerate) * 0.05).astype("float32")
            asr.transcribe(noise, "en", vad=False)
            translator.translate("hello world", "en", targets[0])
            log.info("warmup motori completato in %.1fs", time.time() - t0)
        except Exception:
            log.exception("warmup motori fallito (continuo comunque)")

    def prepare() -> None:
        """Misura l'altezza delle voci installate (scelta della voce simile)."""
        import time

        try:
            t0 = time.time()
            tts.prepare_catalog(cfg.target_languages or ["en"])
            log.info("catalogo voci pronto in %.1fs", time.time() - t0)
        except Exception:
            log.exception("misura delle voci fallita (continuo con i default)")

    return Engines(translator=translator, tts=tts, source_factory=factory,
                   warmup=warmup, prepare=prepare)
