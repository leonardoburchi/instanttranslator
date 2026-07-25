"""Registro delle lingue supportate.

Ogni lingua mappa i codici usati dai diversi motori:
  - ``whisper``: codice ISO usato da faster-whisper per l'ASR
  - ``nllb``: codice FLORES-200 usato da NLLB per la traduzione
  - ``piper``: nome del modello vocale Piper per il TTS

Aggiungere una lingua qui la rende disponibile sia come lingua sorgente
(assegnabile a un canale dalla regia) sia come lingua target (scelta
dall'ascoltatore). I modelli Piper vanno scaricati separatamente: vedi
``scripts/download_models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str          # codice interno breve (it, en, ...)
    name: str          # nome nella lingua stessa (per la UI dell'ascoltatore)
    english_name: str  # nome in inglese (per la regia)
    flag: str          # emoji bandiera (solo estetica)
    whisper: str       # codice faster-whisper
    nllb: str          # codice FLORES-200 per NLLB
    piper: str         # nome modello voce Piper
    # Frase usata per misurare l'altezza delle voci di questa lingua. Serve
    # una frase *dichiarativa vera*: su un elenco di numeri l'intonazione
    # discendente abbassa l'F0 mediana fino al 20% e il confronto tra voci
    # diventa inaffidabile. Conta la fonazione, non la grammatica perfetta.
    probe: str = ""


# Set di default. Ampliabile liberamente.
_LANGUAGES: list[Language] = [
    Language("it", "Italiano",  "Italian",    "🇮🇹", "it", "ita_Latn", "it_IT-riccardo-x_low",
             "Buonasera a tutti, benvenuti a questo incontro."),
    Language("en", "English",   "English",    "🇬🇧", "en", "eng_Latn", "en_US-lessac-medium",
             "Good evening everyone, welcome to this meeting."),
    Language("es", "Español",   "Spanish",    "🇪🇸", "es", "spa_Latn", "es_ES-davefx-medium",
             "Buenas tardes a todos, bienvenidos a esta reunión."),
    Language("fr", "Français",  "French",     "🇫🇷", "fr", "fra_Latn", "fr_FR-siwis-medium",
             "Bonsoir à tous, bienvenue à cette rencontre."),
    Language("de", "Deutsch",   "German",     "🇩🇪", "de", "deu_Latn", "de_DE-thorsten-medium",
             "Guten Abend allen, willkommen zu diesem Treffen."),
    Language("pt", "Português", "Portuguese", "🇵🇹", "pt", "por_Latn", "pt_BR-faber-medium",
             "Boa noite a todos, bem-vindos a este encontro."),
    Language("nl", "Nederlands","Dutch",      "🇳🇱", "nl", "nld_Latn", "nl_NL-mls-medium",
             "Goedenavond allemaal, welkom bij deze bijeenkomst."),
    Language("pl", "Polski",    "Polish",     "🇵🇱", "pl", "pol_Latn", "pl_PL-darkman-medium",
             "Dobry wieczór wszystkim, witamy na tym spotkaniu."),
    Language("ru", "Русский",   "Russian",    "🇷🇺", "ru", "rus_Cyrl", "ru_RU-irina-medium",
             "Добрый вечер всем, добро пожаловать на эту встречу."),
    Language("zh", "中文",       "Chinese",    "🇨🇳", "zh", "zho_Hans", "zh_CN-huayan-medium",
             "大家晚上好，欢迎参加这次会议。"),
    Language("ar", "العربية",    "Arabic",     "🇸🇦", "ar", "arb_Arab", "ar_JO-kareem-medium",
             "مساء الخير للجميع، مرحبا بكم في هذا الاجتماع."),
    Language("uk", "Українська","Ukrainian",  "🇺🇦", "uk", "ukr_Cyrl", "uk_UA-ukrainian_tts-medium",
             "Добрий вечір усім, вітаємо на цій зустрічі."),
]

LANGUAGES: dict[str, Language] = {lang.code: lang for lang in _LANGUAGES}


def get(code: str) -> Language:
    """Ritorna la lingua per codice, sollevando KeyError se assente."""
    return LANGUAGES[code]


def exists(code: str) -> bool:
    return code in LANGUAGES


def all_languages() -> list[Language]:
    return list(_LANGUAGES)
