"""Registry dei canali configurati dalla regia, con persistenza su disco.

I canali creati o modificati durante l'evento vanno salvati: se il processo
riparte (crash, riavvio del PC, failover sul nodo di riserva) la regia non deve
essere riconfigurata a mano davanti a 2500 persone.

Il file di stato (``data/channels.json``) ha la precedenza sui ``channels`` di
``config.yaml``: la configurazione YAML fa da seme al primo avvio, poi comanda
quello che si vede in regia. Per ripartire dal seme basta cancellare il file.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .config import ChannelConfig
from .models import ChannelCreate, ChannelUpdate

log = logging.getLogger("instanttranslator.registry")


class ChannelRegistry:
    """CRUD thread-safe dei canali. Lo stato 'running' vive nell'orchestratore."""

    def __init__(
        self,
        channels: list[ChannelConfig] | None = None,
        store_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._channels: dict[str, ChannelConfig] = {}
        self._store = Path(store_path) if store_path else None

        loaded = self._load()
        for ch in loaded if loaded is not None else (channels or []):
            if ch.id in self._channels:
                log.warning(
                    "due canali con lo stesso id '%s' (dopo la normalizzazione): "
                    "tengo il primo", ch.id,
                )
                continue
            self._channels[ch.id] = ch
        if loaded is not None:
            log.info("canali ripristinati da %s (%d)", self._store, len(loaded))

    # -- persistenza ---------------------------------------------------------
    def _load(self) -> list[ChannelConfig] | None:
        if self._store is None or not self._store.exists():
            return None
        try:
            data = json.loads(self._store.read_text(encoding="utf-8"))
        except Exception:
            log.exception("stato canali illeggibile (%s): uso la config", self._store)
            return None
        # Un canale malformato non deve far perdere tutti gli altri: durante un
        # evento è la differenza tra un canale mancante e la regia da rifare.
        out: list[ChannelConfig] = []
        for item in data.get("channels", []):
            try:
                out.append(ChannelConfig(**item))
            except Exception as exc:
                log.error("canale salvato non valido, lo salto: %s (%s)", item, exc)
        return out

    def _save_locked(self) -> None:
        if self._store is None:
            return
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
            payload = {"channels": [c.model_dump() for c in self._channels.values()]}
            tmp = self._store.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._store)
        except Exception:
            log.exception("salvataggio canali fallito (%s)", self._store)

    # -- CRUD ----------------------------------------------------------------
    def list(self) -> list[ChannelConfig]:
        with self._lock:
            return list(self._channels.values())

    def get(self, channel_id: str) -> ChannelConfig | None:
        with self._lock:
            return self._channels.get(channel_id)

    def exists(self, channel_id: str) -> bool:
        with self._lock:
            return channel_id in self._channels

    def create(self, data: ChannelCreate) -> ChannelConfig:
        with self._lock:
            if data.id in self._channels:
                raise ValueError(f"canale '{data.id}' già esistente")
            ch = ChannelConfig(**data.model_dump())
            self._channels[ch.id] = ch
            self._save_locked()
            return ch

    def update(self, channel_id: str, data: ChannelUpdate) -> ChannelConfig:
        with self._lock:
            ch = self._channels.get(channel_id)
            if ch is None:
                raise KeyError(channel_id)
            patch = data.model_dump(exclude_none=True)
            updated = ch.model_copy(update=patch)
            self._channels[channel_id] = updated
            self._save_locked()
            return updated

    def delete(self, channel_id: str) -> None:
        with self._lock:
            if channel_id not in self._channels:
                raise KeyError(channel_id)
            del self._channels[channel_id]
            self._save_locked()
