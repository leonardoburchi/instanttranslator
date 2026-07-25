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
        return self.translate_many(text, source, [target]).get(target, text)

    def translate_many(self, text: str, source: str, targets: list[str]) -> dict[str, str]:
        """Traduce lo stesso testo verso più lingue in **una sola** chiamata.

        Il decoder lavora sull'intero batch in parallelo: dodici lingue costano
        molto meno di dodici traduzioni in fila, e la GPU resta libera per
        l'ASR incrementale.
        """
        out: dict[str, str] = {}
        pending: list[str] = []
        for target in targets:
            if source == target or not text.strip():
                out[target] = text
                continue
            cached = self._cache.get((source, target, text))
            if cached is not None:
                out[target] = cached
            else:
                pending.append(target)
        if not pending:
            return out

        src_code = get_lang(source).nllb
        tgt_codes = [get_lang(t).nllb for t in pending]

        with self._lock:
            self._tokenizer.src_lang = src_code
            tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text)
            )
            results = self._translator.translate_batch(
                [tokens] * len(pending),
                target_prefix=[[code] for code in tgt_codes],
                beam_size=self._beam_size,
                max_decoding_length=self._max_decoding_length,
            )
            for target, code, result in zip(pending, tgt_codes, results):
                out_tokens = result.hypotheses[0]
                if out_tokens and out_tokens[0] == code:
                    out_tokens = out_tokens[1:]
                ids = self._tokenizer.convert_tokens_to_ids(out_tokens)
                out[target] = self._tokenizer.decode(
                    ids, skip_special_tokens=True).strip()

        # Cache limitata per i parziali ricorrenti.
        if len(self._cache) > 4000:
            self._cache.clear()
        for target in pending:
            self._cache[(source, target, text)] = out[target]
        return out
