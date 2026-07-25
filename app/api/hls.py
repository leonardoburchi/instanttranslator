"""Serve i file HLS (playlist + segmenti) e i sottotitoli cacheabili.

Pensato per stare dietro a nginx/CDN: i segmenti ``.ts`` sono immutabili
(cache lunghissima), la playlist ``.m3u8`` e ``subs.json`` hanno TTL breve.
L'origine genera ogni stream una sola volta: la cache fa il fan-out a migliaia
di ascoltatori.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

from ..context import AppContext
from ..deps import get_ctx

log = logging.getLogger("instanttranslator.api.hls")
router = APIRouter(tags=["hls"])

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
_CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".m4s": "video/iso.segment",
    ".mp4": "video/mp4",
}


@router.get("/hls/{channel_id}/{lang}/{filename}")
async def hls_file(
    channel_id: str, lang: str, filename: str, ctx: AppContext = Depends(get_ctx)
):
    if ctx.hls is None:
        raise HTTPException(404, "HLS non abilitato")
    if not (_SAFE.match(channel_id) and _SAFE.match(lang) and _SAFE.match(filename)):
        # Capita con id di canale contenenti spazi o accenti: l'id viene
        # normalizzato in ingresso, ma un client con una pagina vecchia in cache
        # può ancora chiedere il vecchio URL.
        log.warning(
            "richiesta HLS rifiutata, nome non ammesso: canale=%r lingua=%r file=%r",
            channel_id, lang, filename,
        )
        raise HTTPException(400, "nome non valido (usa id senza spazi né accenti)")

    ext = filename[filename.rfind("."):]
    if ext not in _CONTENT_TYPES:
        raise HTTPException(400, "tipo file non ammesso")

    base = (Path(ctx.cfg.hls.output_dir) / channel_id / lang).resolve()
    file_path = (base / filename).resolve()
    if base not in file_path.parents or not file_path.is_file():
        raise HTTPException(404, "file non trovato")

    # I segmenti sono immutabili -> cache lunga; la playlist deve aggiornarsi.
    if ext == ".m3u8":
        cache = "no-cache, max-age=1"
    else:
        cache = "public, max-age=31536000, immutable"

    return FileResponse(
        str(file_path),
        media_type=_CONTENT_TYPES[ext],
        headers={"Cache-Control": cache, "Access-Control-Allow-Origin": "*"},
    )


@router.get("/api/hls/{channel_id}/{lang}/subs.json")
async def hls_subtitles(
    channel_id: str, lang: str, ctx: AppContext = Depends(get_ctx)
) -> Response:
    if ctx.hls is None:
        raise HTTPException(404, "HLS non abilitato")
    cues = ctx.hls.subtitles.recent(channel_id, lang)
    payload = {
        "server_time": time.time(),
        "cues": [
            {"seq": c.seq, "start": c.start, "end": c.end, "text": c.text}
            for c in cues
        ],
    }
    return JSONResponse(
        payload,
        headers={"Cache-Control": "no-cache, max-age=1",
                 "Access-Control-Allow-Origin": "*"},
    )
