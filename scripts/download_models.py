"""Scarica e prepara i modelli per la modalità reale.

  python scripts/download_models.py                 # NLLB + voci delle lingue di default
  python scripts/download_models.py --languages it en es
  python scripts/download_models.py --skip-nllb     # solo voci Piper
  python scripts/download_models.py --skip-voices   # solo NLLB
  python scripts/download_models.py --pairs         # voce maschile E femminile per lingua
  python scripts/download_models.py --voices en_US-ryan-high de_DE-eva_k-x_low

Con ``--pairs`` (consigliato) si scarica per ogni lingua anche una voce
dell'altro registro: l'app misura l'altezza delle voci installate e sceglie da
sola quella più vicina al parlante del canale (uomo -> voce maschile e così
via). Con una sola voce per lingua non c'è niente da scegliere.

Richiede: pip install -r requirements-ml.txt
La conversione di NLLB in CTranslate2 carica il modello via transformers e
quindi richiede torch (solo per questo passo, NON a runtime):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Permette di importare il registro lingue del progetto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.languages import LANGUAGES  # noqa: E402

PIPER_REPO = "rhasspy/piper-voices"

# Voce "dell'altro registro" rispetto al default di app/languages.py, così ogni
# lingua ha sia una voce maschile sia una femminile e l'app può scegliere.
# Se un nome non esiste più nel repo, lo si vede nel log e si prosegue.
COMPANION_VOICES: dict[str, list[str]] = {
    "it": ["it_IT-paola-medium"],
    "en": ["en_US-ryan-high", "en_US-amy-medium"],
    "es": ["es_ES-sharvard-medium"],
    "fr": ["fr_FR-gilles-low"],
    "de": ["de_DE-eva_k-x_low"],
    "pt": ["pt_PT-tugão-medium"],
    "nl": ["nl_BE-nathalie-medium"],
    "pl": ["pl_PL-gosia-medium"],
    "ru": ["ru_RU-dmitri-medium"],
    "uk": ["uk_UA-lada-x_low"],
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_OUT = Path("models/nllb-200-distilled-600M-ct2")
VOICES_DIR = Path("models/piper")


def convert_nllb() -> None:
    if NLLB_OUT.exists():
        print(f"[nllb] già presente in {NLLB_OUT}, salto")
        return
    print(f"[nllb] conversione {NLLB_MODEL} -> {NLLB_OUT} (CTranslate2, float16)")
    NLLB_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", NLLB_MODEL,
            "--output_dir", str(NLLB_OUT),
            "--quantization", "float16",
        ],
        check=True,
    )


def _piper_repo_path(name: str) -> str:
    """it_IT-riccardo-x_low -> it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx"""
    locale, speaker, quality = name.split("-", 2)
    lang2 = locale.split("_")[0]
    return f"{lang2}/{locale}/{speaker}/{quality}/{name}.onnx"


def download_voice(name: str) -> None:
    from huggingface_hub import hf_hub_download

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    dst_onnx = VOICES_DIR / f"{name}.onnx"
    if dst_onnx.exists():
        print(f"[piper] {name} già presente, salto")
        return

    base = _piper_repo_path(name)
    print(f"[piper] scarico {name}")
    for suffix in ("", ".json"):
        remote = base + suffix
        cached = hf_hub_download(repo_id=PIPER_REPO, filename=remote)
        shutil.copy(cached, VOICES_DIR / f"{name}.onnx{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica i modelli per la modalità reale")
    parser.add_argument("--languages", nargs="*", default=["it", "en", "es", "fr", "de"])
    parser.add_argument("--skip-nllb", action="store_true")
    parser.add_argument("--skip-voices", action="store_true")
    parser.add_argument("--pairs", action="store_true",
                        help="scarica anche una voce dell'altro registro per lingua")
    parser.add_argument("--voices", nargs="*", default=[],
                        help="voci Piper aggiuntive per nome (es. en_US-ryan-high)")
    args = parser.parse_args()

    if not args.skip_nllb:
        convert_nllb()

    if not args.skip_voices:
        wanted: list[str] = []
        for code in args.languages:
            lang = LANGUAGES.get(code)
            if lang is None:
                print(f"[piper] lingua sconosciuta '{code}', salto")
                continue
            wanted.append(lang.piper)
            if args.pairs:
                wanted.extend(COMPANION_VOICES.get(code, []))
        wanted.extend(args.voices)

        for name in dict.fromkeys(wanted):       # dedup, ordine preservato
            try:
                download_voice(name)
            except Exception as exc:  # pragma: no cover
                print(f"[piper] errore su {name}: {exc}")

    print("Fatto. Imposta engine.mode: real in config.yaml e riavvia.")


if __name__ == "__main__":
    main()
