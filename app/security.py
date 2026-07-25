"""Autenticazione della regia.

La landing dell'ascoltatore è pubblica per definizione (in una piazza il link
gira su volantini e QR code), ma le stesse persone raggiungono anche
``/admin``: senza protezione chiunque potrebbe fermare i canali o cambiare la
lingua sorgente mentre l'evento è in corso.

Il modello è volutamente minimo: un **token condiviso** passato come header
``X-Admin-Token`` (o ``?token=`` per il WebSocket del monitor). Se non è
configurato viene generato all'avvio e scritto nel log, così il sistema non è
mai accidentalmente aperto.
"""

from __future__ import annotations

import logging
import secrets

from starlette.requests import HTTPConnection

from .config import AdminConfig

log = logging.getLogger("instanttranslator.security")

HEADER = "X-Admin-Token"


class AdminAuth:
    def __init__(self, cfg: AdminConfig) -> None:
        self.required = cfg.require_auth
        self.generated = False
        self.token: str | None = cfg.token
        if self.required and not self.token:
            self.token = secrets.token_urlsafe(9)
            self.generated = True

    def announce(self) -> None:
        if not self.required:
            log.warning(
                "regia SENZA autenticazione (admin.require_auth=false): "
                "chiunque raggiunga la rete può fermare i canali"
            )
        elif self.generated:
            log.warning(
                "token regia generato per questa sessione: %s   "
                "(fissalo in config.yaml -> admin.token per non cambiarlo a ogni riavvio)",
                self.token,
            )
        else:
            log.info("regia protetta da token (admin.token da configurazione)")

    def check(self, token: str | None) -> bool:
        if not self.required:
            return True
        if not token or not self.token:
            return False
        return secrets.compare_digest(token, self.token)


def token_from_request(conn: HTTPConnection) -> str | None:
    """Estrae il token da header, query string o cookie (HTTP e WebSocket)."""
    return (
        conn.headers.get(HEADER)
        or conn.query_params.get("token")
        or conn.cookies.get("admin_token")
    )
