"""Stato runtime: registry dei canali + Hub pub/sub per il fan-out.

L'Hub disaccoppia i produttori (worker della pipeline, su thread separati)
dai consumatori (connessioni WebSocket asyncio). I produttori pubblicano con
``publish_threadsafe`` perché girano fuori dal loop asyncio.

Topic:
  - ``transcript:{channel}``      sorgente trascritta (monitor regia)
  - ``sub:{channel}:{lang}``      sottotitolo tradotto per l'ascoltatore
  - ``audio:{channel}:{lang}``    audio TTS (AudioChunk) per l'ascoltatore
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger("instanttranslator.hub")


def topic_transcript(channel_id: str) -> str:
    return f"transcript:{channel_id}"


def topic_subtitle(channel_id: str, lang: str) -> str:
    return f"sub:{channel_id}:{lang}"


def topic_audio(channel_id: str, lang: str) -> str:
    return f"audio:{channel_id}:{lang}"


@dataclass
class Subscription:
    topic: str
    queue: asyncio.Queue
    _hub: "Hub"

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc) -> None:
        self._hub.unsubscribe(self)


class Hub:
    """Pub/sub asyncio thread-safe con code bounded (drop su overflow)."""

    def __init__(self, queue_maxsize: int = 256) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._queue_maxsize = queue_maxsize
        self._loop: asyncio.AbstractEventLoop | None = None
        # Conteggio ascoltatori per (channel_id, lang) -> quali lingue tradurre.
        self._listeners: dict[tuple[str, str], int] = defaultdict(int)

    # -- ciclo di vita -------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- subscribe / publish -------------------------------------------------
    def subscribe(self, topic: str) -> Subscription:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subs[topic].add(q)
        return Subscription(topic=topic, queue=q, _hub=self)

    def unsubscribe(self, sub: Subscription) -> None:
        subs = self._subs.get(sub.topic)
        if subs:
            subs.discard(sub.queue)
            if not subs:
                self._subs.pop(sub.topic, None)

    def publish(self, topic: str, message) -> None:
        """Pubblica dal loop asyncio. Droppa il messaggio se una coda è piena."""
        # Copia: un consumatore potrebbe disiscriversi mentre iteriamo.
        for q in list(self._subs.get(topic, ())):
            self._offer(q, topic, message)

    def publish_threadsafe(self, topic: str, message) -> None:
        """Pubblica da un thread esterno al loop asyncio."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self.publish, topic, message)

    def _offer(self, q: asyncio.Queue, topic: str, message) -> None:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            # Real-time: scarta il più vecchio e inserisce il nuovo.
            try:
                q.get_nowait()
                q.put_nowait(message)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                log.debug("drop su topic %s (coda piena)", topic)

    def has_subscribers(self, topic: str) -> bool:
        return bool(self._subs.get(topic))

    # -- registry ascoltatori ------------------------------------------------
    def add_listener(self, channel_id: str, lang: str) -> None:
        self._listeners[(channel_id, lang)] += 1

    def remove_listener(self, channel_id: str, lang: str) -> None:
        key = (channel_id, lang)
        if self._listeners.get(key):
            self._listeners[key] -= 1
            if self._listeners[key] <= 0:
                self._listeners.pop(key, None)

    def active_target_languages(self, channel_id: str) -> set[str]:
        """Lingue con almeno un ascoltatore su quel canale (lazy translate/TTS)."""
        return {
            lang for (cid, lang), n in self._listeners.items()
            if cid == channel_id and n > 0
        }

    def listener_count(self, channel_id: str) -> int:
        return sum(
            n for (cid, _), n in self._listeners.items() if cid == channel_id
        )

    def total_listeners(self) -> int:
        """Ascoltatori WebSocket su tutti i canali (per il tetto di sicurezza)."""
        return sum(self._listeners.values())
