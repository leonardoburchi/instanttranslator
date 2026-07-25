"""Normalizzazione degli identificativi di canale.

L'``id`` di un canale non è solo una chiave interna: finisce in un **URL**
(``/hls/{id}/{lang}/audio.m3u8``), in un **path su disco**
(``data/hls/{id}/{lang}/``) e nelle **chiavi di cache** di nginx/CDN. Un id con
spazi o accenti si rompe in almeno uno di quei tre posti (tipicamente: la
playlist arriva come ``/hls/Test%201/...`` e l'origine la rifiuta), quindi lo
normalizziamo una volta sola, all'ingresso.

Tutto minuscolo di proposito: Windows non distingue le maiuscole nei path, gli
URL e la cache di nginx sì. Con origine su Windows e cache su Linux, ``Test``
e ``test`` diventerebbero due cose diverse a metà strada.
"""

from __future__ import annotations

import re
import unicodedata

# Caratteri ammessi in un id (stessa classe accettata dall'endpoint HLS).
SAFE_ID = re.compile(r"^[a-z0-9._-]+$")
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def normalize_id(value: str) -> str:
    """Rende un id sicuro per URL, filesystem e cache.

    ``"Sala Plenaria"`` → ``"sala-plenaria"``, ``"Città 2"`` → ``"citta-2"``.
    Solleva ``ValueError`` se non resta nulla di utilizzabile.
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = _UNSAFE.sub("-", text).strip("-._")
    text = re.sub(r"-{2,}", "-", text)
    if not text:
        raise ValueError(
            "id canale non valido: usa lettere, numeri, '-' o '_' "
            "(es. 'sala-plenaria')"
        )
    return text
