"""Traduzione reale con NLLB-200 via CTranslate2.

Il modello va convertito in formato CTranslate2 (vedi
``scripts/download_models.py``). Usiamo il tokenizer HF originale per
encode/decode e CTranslate2 per l'inferenza (veloce su GPU).

L'accesso è serializzato da un lock: una sola GPU, richieste in coda.
"""

from __future__ import annotations

import logging
import threading

from ..config import TranslateConfig
from ..languages import get as get_lang
from .asr_whisper import _resolve_device
from .base import Translator

log = logging.getLogger("instanttranslator.translate")


class NLLBTranslator(Translator):
    def __init__(self, cfg: TranslateConfig) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        device = _resolve_device(cfg.device)
        compute_type = cfg.compute_type if device == "cuda" else "int8"
        log.info("carico NLLB da %s su %s", cfg.model_dir, device)
        self._translator = ctranslate2.Translator(
            cfg.model_dir, device=device, compute_type=compute_type
        )
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self._beam_size = cfg.beam_size
        self._max_decoding_length = cfg.max_decoding_length
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], str] = {}

    def translate(self, text: str, source: str, target: str) -> str:
        if source == target or not text.strip():
            return text

        key = (source, target, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        src_code = get_lang(source).nllb
        tgt_code = get_lang(target).nllb

        with self._lock:
            self._tokenizer.src_lang = src_code
            tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text)
            )
            results = self._translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_code]],
                beam_size=self._beam_size,
                max_decoding_length=self._max_decoding_length,
            )
            out_tokens = results[0].hypotheses[0]
            if out_tokens and out_tokens[0] == tgt_code:
                out_tokens = out_tokens[1:]
            ids = self._tokenizer.convert_tokens_to_ids(out_tokens)
            translated = self._tokenizer.decode(ids, skip_special_tokens=True).strip()

        # Cache limitata per i parziali ricorrenti.
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = translated
        return translated
