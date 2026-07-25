"""Enumerazione e risoluzione dei device audio di input via PortAudio.

L'import di ``sounddevice`` è protetto: in mock mode (o se PortAudio non è
disponibile) la lista device è semplicemente vuota e l'app resta usabile.

Nota di campo (mixer digitali tipo Behringer XR18): gli **indici** PortAudio
cambiano quando cambia l'ordine di enumerazione dei device USB (riavvio,
cavo in un'altra porta, driver aggiornato). Per questo un canale può indicare
il device anche per **nome parziale** (es. ``"XR18"``): più lungo da scrivere,
ma non ti si sposta sotto i piedi la mattina dell'evento.
"""

from __future__ import annotations

import logging

from ..models import DeviceInfo

log = logging.getLogger("instanttranslator.audio")

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception as exc:  # pragma: no cover - dipende dall'ambiente
    sd = None  # type: ignore
    _SD_AVAILABLE = False
    log.warning("sounddevice non disponibile: %s", exc)


def available() -> bool:
    return _SD_AVAILABLE


def host_api_names() -> list[str]:
    """Host API compilate in questo build di PortAudio (MME, WASAPI, ASIO…)."""
    if not _SD_AVAILABLE:
        return []
    try:
        return [h.get("name", "") for h in sd.query_hostapis()]
    except Exception:  # pragma: no cover
        return []


def list_input_devices() -> list[DeviceInfo]:
    """Tutti i device con almeno un canale di input."""
    if not _SD_AVAILABLE:
        return []
    out: list[DeviceInfo] = []
    try:
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            host_idx = dev.get("hostapi", 0)
            host_name = ""
            if 0 <= host_idx < len(hostapis):
                host_name = hostapis[host_idx].get("name", "")
            out.append(
                DeviceInfo(
                    index=idx,
                    name=dev.get("name", f"device {idx}"),
                    max_input_channels=int(dev.get("max_input_channels", 0)),
                    default_samplerate=float(dev.get("default_samplerate", 0.0)),
                    hostapi=host_name,
                )
            )
    except Exception as exc:  # pragma: no cover
        log.error("errore enumerazione device: %s", exc)
    return out


# Preferenza di host API a parità di nome e numero di canali. ASIO espone tutti
# i canali dei mixer (quando PortAudio è compilato con l'SDK); WASAPI e WDM-KS
# lavorano al rate nativo della scheda; MME e DirectSound ricampionano nel
# kernel (dichiarano 44100 su una scheda a 48 kHz) e aggiungono latenza.
_HOST_API_RANK = ("ASIO", "WASAPI", "WDM-KS", "DIRECTSOUND", "MME")

# Ultima risoluzione per spec, per non ripetere lo stesso log a ogni riconfigurazione.
_resolved: dict[str, int] = {}


def _host_rank(name: str) -> int:
    upper = name.upper()
    for i, key in enumerate(_HOST_API_RANK):
        if key in upper:
            return i
    return len(_HOST_API_RANK)


def resolve_device(device: int | str | None, host_api: str | None = None) -> int | None:
    """Normalizza un device (indice, nome parziale o None) in un indice PortAudio.

    Col nome si preferisce il device con **più canali di input** e, a pari
    canali, la host API migliore: una scheda multicanale compare più volte (una
    per host API) e a noi serve quella che espone tutti gli ingressi al rate
    nativo, non la versione stereo ricampionata.
    """
    if device is None or not _SD_AVAILABLE:
        return device if isinstance(device, int) else None
    if isinstance(device, int):
        return device
    needle = device.strip().lower()
    if not needle:
        return None
    if needle.isdigit():
        return int(needle)

    candidates = [d for d in list_input_devices() if needle in d.name.lower()]
    if host_api:
        filtered = [d for d in candidates if host_api.upper() in d.hostapi.upper()]
        if filtered:
            candidates = filtered
    if not candidates:
        raise ValueError(f"nessun device di input con nome contenente '{device}'")
    best = min(candidates, key=lambda d: (-d.max_input_channels, _host_rank(d.hostapi)))
    # La risoluzione avviene a ogni riconfigurazione: logghiamo a INFO solo
    # quando cambia davvero (un indice che si sposta è un'informazione utile).
    if _resolved.get(device) != best.index:
        _resolved[device] = best.index
        log.info(
            "device '%s' risolto in [%d] %s (%dch, %s)",
            device, best.index, best.name, best.max_input_channels, best.hostapi,
        )
    return best.index


def device_info(device: int | None) -> dict:
    """Info PortAudio del device di input (dict vuoto se non disponibile)."""
    if not _SD_AVAILABLE:
        return {}
    try:
        return dict(sd.query_devices(device, "input"))
    except Exception as exc:
        log.warning("device %s non interrogabile: %s", device, exc)
        return {}


def device_channel_count(device: int | None) -> int:
    """Numero di canali di input del device (per validare channel_index)."""
    return int(device_info(device).get("max_input_channels", 0) or 0)


def device_samplerate(device: int | None, fallback: int = 48000) -> int:
    """Sample rate nativo del device (i mixer digitali stanno a 48 kHz)."""
    sr = device_info(device).get("default_samplerate") or 0
    return int(sr) if sr else fallback


def device_label(device: int | None) -> str:
    info = device_info(device)
    if not info:
        return f"device {device}"
    return f"[{device}] {info.get('name', '?')} ({info.get('max_input_channels', 0)}ch)"
