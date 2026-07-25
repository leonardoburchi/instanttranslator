"""Consegna HLS scalabile (CDN-ready) per grandi platee.

Per ogni coppia (canale, lingua) un :class:`HlsStream` mantiene un processo
ffmpeg che trasforma un flusso PCM continuo in segmenti HLS cacheabili. Il
:class:`HlsManager` ne gestisce il ciclo di vita e conserva i sottotitoli a
rotazione per la sincronizzazione lato client.
"""

from .manager import HlsManager
from .stream import HlsStream

__all__ = ["HlsManager", "HlsStream"]
