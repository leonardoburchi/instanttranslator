"""Cattura audio multicanale, ricampionamento e demux per-canale.

Un evento ha più canali sorgente che condividono la stessa interfaccia
multicanale (tipicamente un mixer digitale: Behringer XR18 = 18 canali su
USB). Apriamo quindi **uno** stream PortAudio per device fisico, ne separiamo
le colonne e instradiamo ogni canale (``channel_index``) alle sue callback.

Due punti che in campo fanno la differenza:

* **Sample rate.** I mixer digitali girano a 48 kHz e non ricampionano: se
  chiedi 16 kHz a PortAudio (WDM-KS/ASIO/WASAPI esclusivo) lo stream non si
  apre affatto. Apriamo quindi al rate **nativo** del device e ricampioniamo
  noi a 16 kHz per Whisper, con filtro anti-alias.
* **Riconfigurazione a caldo.** Avviare o fermare un canale non deve chiudere
  lo stream degli altri: le rotte si aggiornano a caldo e lo stream viene
  riaperto solo se cambia il device o il numero di colonne richieste.

La callback PortAudio gira su un thread real-time: copia, ricampiona e
accoda: mai lavoro pesante (ASR ecc.).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger("instanttranslator.audio")

_com_ready: set[int] = set()


def _ensure_com() -> None:
    """Inizializza COM sul thread corrente (solo Windows).

    Le host API Windows di PortAudio (WASAPI, WDM-KS) sono basate su COM: se lo
    stream viene aperto da un thread dove COM non è inizializzato, l'apertura
    falla con errori host oscuri (``WdmSyncIoctl ... GLE = 0x492``,
    ``AUDCLNT_E_UNSUPPORTED_FORMAT``). Succede in pratica perché i canali si
    avviano anche da una richiesta HTTP della regia, non solo dal thread
    principale all'avvio.
    """
    if not sys.platform.startswith("win"):
        return
    tid = threading.get_ident()
    if tid in _com_ready:
        return
    try:
        import ctypes
        # COINIT_APARTMENTTHREADED. RPC_E_CHANGED_MODE (0x80010106) significa
        # che COM è già inizializzato in un'altra modalità: va bene comunque.
        ctypes.windll.ole32.CoInitializeEx(None, 0x2)
    except Exception as exc:  # pragma: no cover
        log.debug("CoInitializeEx non riuscita: %s", exc)
    _com_ready.add(tid)

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:  # pragma: no cover
    sd = None  # type: ignore
    _SD_AVAILABLE = False

# callback(mono_block: np.ndarray float32 shape (frames,)) al rate di elaborazione
BlockCallback = Callable[[np.ndarray], None]


# --------------------------------------------------------------------------- #
# Ricampionamento                                                              #
# --------------------------------------------------------------------------- #
def _lowpass_taps(cutoff_ratio: float, ntaps: int = 63) -> np.ndarray:
    """FIR passa-basso a fase lineare (sinc finestrato con Hamming).

    ``cutoff_ratio`` è la frequenza di taglio come frazione della frequenza di
    Nyquist del segnale in ingresso.
    """
    n = np.arange(ntaps, dtype=np.float64) - (ntaps - 1) / 2.0
    h = np.sinc(cutoff_ratio * n) * cutoff_ratio
    h *= np.hamming(ntaps)
    h /= h.sum()
    return h.astype(np.float32)


class Resampler:
    """Ricampionatore mono *stateful* (blocchi contigui, nessun click ai bordi).

    Decimando (48 kHz → 16 kHz) senza filtro, tutto ciò che sta sopra gli 8 kHz
    si ripiega nella banda utile e sporca l'ASR: prima filtriamo, poi
    interpoliamo linearmente alla posizione frazionaria esatta.
    """

    def __init__(self, src_sr: int, dst_sr: int) -> None:
        self.src_sr = int(src_sr)
        self.dst_sr = int(dst_sr)
        self._step = self.src_sr / self.dst_sr
        self._taps: np.ndarray | None = None
        if self.dst_sr < self.src_sr:
            # 0.9 di margine sotto la nuova Nyquist: transizione morbida.
            self._taps = _lowpass_taps(0.9 * self.dst_sr / self.src_sr)
        self.reset()

    @property
    def passthrough(self) -> bool:
        return self.src_sr == self.dst_sr

    def reset(self) -> None:
        ntaps = 0 if self._taps is None else len(self._taps)
        self._filter_state = np.zeros(max(ntaps - 1, 0), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return block
        x = np.asarray(block, dtype=np.float32)
        if self._taps is not None:
            padded = np.concatenate((self._filter_state, x))
            ntaps = len(self._taps)
            if len(padded) < ntaps:
                self._filter_state = padded
                return np.zeros(0, dtype=np.float32)
            filtered = np.convolve(padded, self._taps, mode="valid")
            self._filter_state = padded[-(ntaps - 1):] if ntaps > 1 else padded[:0]
        else:
            filtered = x

        buf = np.concatenate((self._pending, filtered)) if len(self._pending) else filtered
        if len(buf) < 2:
            self._pending = buf
            return np.zeros(0, dtype=np.float32)

        # Campioni di uscita alle posizioni pos, pos+step, ... entro il buffer.
        n_out = int(np.floor((len(buf) - 1 - self._pos) / self._step)) + 1
        if n_out <= 0:
            self._pending = buf
            return np.zeros(0, dtype=np.float32)
        idx = self._pos + self._step * np.arange(n_out, dtype=np.float64)
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac

        consumed = int(np.floor(self._pos + n_out * self._step))
        consumed = min(consumed, len(buf) - 1)
        self._pending = buf[consumed:].copy()
        self._pos = self._pos + n_out * self._step - consumed
        return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Rotte e metering                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Route:
    """Un consumatore di un canale fisico."""

    key: str                     # identificativo per i meter (es. "ch1")
    device: int | None           # indice PortAudio già risolto
    channel_index: int
    callback: BlockCallback
    gain: float = 1.0            # lineare (dal gain_db della config)


@dataclass
class Meter:
    """Livelli recenti di un canale, per la verifica in regia."""

    rms: float = 0.0
    peak: float = 0.0
    updated: float = 0.0
    clipped: int = 0

    def as_dict(self) -> dict:
        def dbfs(v: float) -> float:
            return round(20.0 * np.log10(max(v, 1e-6)), 1)
        return {
            "rms_dbfs": dbfs(self.rms),
            "peak_dbfs": dbfs(self.peak),
            "clipped": self.clipped,
            "age": round(max(time.time() - self.updated, 0.0), 2) if self.updated else None,
        }


@dataclass
class _Plan:
    """Piano di demux di un device: colonne dello stream -> consumatori."""

    columns: list[int]                       # colonna fisica per ogni colonna del buffer
    entries: list[tuple[int, Route, Resampler, Meter]] = field(default_factory=list)


class CaptureManager:
    """Stream di input aperti per device, con demux e ricampionamento.

    Le rotte si possono aggiornare a caldo con :meth:`set_routes`: lo stream di
    un device viene riaperto solo se cambia l'insieme delle colonne richieste.
    """

    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        *,
        device_samplerate: int | None = None,
        latency: float | str = "high",
        wasapi_exclusive: bool = False,
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._forced_device_sr = device_samplerate
        self._latency = latency
        self._wasapi_exclusive = wasapi_exclusive

        self._lock = threading.RLock()
        self._routes: list[Route] = []
        self._streams: dict[int | None, object] = {}
        self._plans: dict[int | None, _Plan] = {}
        self._meters: dict[str, Meter] = {}
        self._last_block: dict[int | None, float] = {}
        self._overflows: dict[int | None, int] = {}
        self._device_sr: dict[int | None, int] = {}

    # -- stato ---------------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._streams)

    def levels(self) -> dict[str, dict]:
        with self._lock:
            return {k: m.as_dict() for k, m in self._meters.items()}

    def health(self) -> dict:
        now = time.time()
        with self._lock:
            return {
                "devices": [
                    {
                        "device": dev,
                        "samplerate": self._device_sr.get(dev),
                        "channels": len(self._plans[dev].columns) if dev in self._plans else 0,
                        "last_block_age": (
                            round(now - self._last_block[dev], 2)
                            if self._last_block.get(dev) else None
                        ),
                        "overflows": self._overflows.get(dev, 0),
                    }
                    for dev in self._streams
                ],
                "levels": {k: m.as_dict() for k, m in self._meters.items()},
            }

    def stalled(self, max_age: float = 2.0) -> bool:
        """True se un device aperto non consegna più blocchi (cavo staccato…)."""
        now = time.time()
        with self._lock:
            for dev in self._streams:
                last = self._last_block.get(dev)
                if last is None or now - last > max_age:
                    return True
        return False

    # -- configurazione ------------------------------------------------------
    def set_routes(self, routes: list[Route]) -> None:
        """Applica l'insieme di rotte desiderato, riaprendo solo il necessario.

        Solleva ``ValueError`` se una rotta chiede un canale che il device non
        ha (tipico: ``channel_index`` oltre gli ingressi disponibili) e
        ``RuntimeError`` se PortAudio non riesce ad aprire il device.
        """
        if not _SD_AVAILABLE and routes:
            raise RuntimeError("sounddevice/PortAudio non disponibile")
        _ensure_com()

        with self._lock:
            wanted: dict[int | None, list[Route]] = {}
            for r in routes:
                wanted.setdefault(r.device, []).append(r)

            # Device non più usati: chiudi.
            for dev in list(self._streams):
                if dev not in wanted:
                    self._close_device(dev)

            for dev, dev_routes in wanted.items():
                columns = sorted({r.channel_index for r in dev_routes})
                self._validate(dev, columns)
                plan = self._plans.get(dev)
                # Si riapre solo se lo stream aperto non contiene già le colonne
                # richieste: aggiungere un canale durante l'evento non deve
                # togliere l'audio a quelli che stanno andando in onda.
                if dev not in self._streams or plan is None \
                        or not set(columns) <= set(plan.columns):
                    self._close_device(dev)
                    self._open_device(dev, columns, dev_routes)
                else:
                    self._plans[dev] = self._build_plan(
                        plan.columns, dev_routes, self._device_sr[dev]
                    )

            self._routes = list(routes)
            # Meter dei canali non più instradati: via.
            keys = {r.key for r in routes}
            for k in list(self._meters):
                if k not in keys:
                    self._meters.pop(k, None)

    def stop(self) -> None:
        with self._lock:
            for dev in list(self._streams):
                self._close_device(dev)
            self._routes = []
            self._meters.clear()

    # -- interno -------------------------------------------------------------
    def _validate(self, device: int | None, columns: list[int]) -> None:
        from .devices import device_channel_count, device_label

        available = device_channel_count(device)
        needed = max(columns) + 1
        if available and needed > available:
            raise ValueError(
                f"il device {device_label(device)} espone {available} canali di "
                f"input, ma è richiesto il canale {max(columns)} "
                f"(channel_index parte da 0). Se ti aspettavi più canali: "
                f"verifica il driver/host API della scheda."
            )

    def _use_asio_selectors(self, device: int | None) -> bool:
        """Su ASIO possiamo aprire *solo* le colonne che ci servono."""
        if not _SD_AVAILABLE or not hasattr(sd, "AsioSettings"):
            return False
        from .devices import device_info
        try:
            info = device_info(device)
            hostapis = sd.query_hostapis()
            name = hostapis[int(info.get("hostapi", 0))].get("name", "")
        except Exception:
            return False
        return "ASIO" in name.upper()

    def _is_wasapi(self, device: int | None) -> bool:
        if not _SD_AVAILABLE:
            return False
        from .devices import device_info
        try:
            info = device_info(device)
            hostapis = sd.query_hostapis()
            return "WASAPI" in hostapis[int(info.get("hostapi", 0))].get("name", "").upper()
        except Exception:
            return False

    def _layout_candidates(self, device: int | None, columns: list[int]) -> list[list[int]]:
        """Layout da provare, dal migliore al più permissivo.

        Non tutte le host API accettano un numero *parziale* di canali: WASAPI
        (condiviso) pretende il formato del device, cioè tutti gli ingressi, e
        rifiuta l'apertura con "Invalid number of channels". Apriamo quindi tutto
        il device e facciamo il demux noi; il ripiego a ``max+1`` serve alle API
        che invece accettano un sottoinsieme.
        """
        from .devices import device_channel_count

        candidates: list[list[int]] = []
        if self._use_asio_selectors(device):
            candidates.append(list(columns))        # ASIO: apertura selettiva
        total = device_channel_count(device)
        if total:
            candidates.append(list(range(total)))
        candidates.append(list(range(max(columns) + 1)))
        out: list[list[int]] = []
        for c in candidates:
            if c and c not in out:
                out.append(c)
        return out

    def _build_plan(
        self, layout: list[int], routes: list[Route], device_sr: int
    ) -> _Plan:
        entries: list[tuple[int, Route, Resampler, Meter]] = []
        for r in routes:
            try:
                col = layout.index(r.channel_index)
            except ValueError:
                log.error("canale %s non presente nel layout del device", r.channel_index)
                continue
            meter = self._meters.setdefault(r.key, Meter())
            entries.append((col, r, Resampler(device_sr, self.samplerate), meter))
        return _Plan(columns=list(layout), entries=entries)

    def _open_device(
        self, device: int | None, columns: list[int], routes: list[Route]
    ) -> None:
        from .devices import device_label, device_samplerate

        device_sr = self._forced_device_sr or device_samplerate(device)
        # blocksize espresso nel rate del device (il nostro è al rate di lavoro)
        blocksize = max(int(round(self.blocksize * device_sr / self.samplerate)), 0)
        self._device_sr[device] = device_sr

        # Le schede USB a volte rifiutano la prima apertura (driver che sta
        # ancora rilasciando lo stream precedente): un secondo tentativo dopo
        # una pausa breve evita di far fallire l'avvio di un canale per niente.
        errors: list[str] = []
        candidates = [
            layout for layout in self._layout_candidates(device, columns)
            for _ in range(2)
        ]
        for attempt, layout in enumerate(candidates):
            if attempt and errors:
                time.sleep(0.3)
            extra = None
            if self._use_asio_selectors(device) and layout == list(columns):
                try:
                    extra = sd.AsioSettings(channel_selectors=list(layout))
                except Exception as exc:  # pragma: no cover
                    log.warning("AsioSettings non applicabile: %s", exc)
                    continue
            elif self._wasapi_exclusive and self._is_wasapi(device):
                try:
                    extra = sd.WasapiSettings(exclusive=True)
                except Exception as exc:  # pragma: no cover
                    log.warning("WASAPI esclusivo non applicabile: %s", exc)

            stream = None
            try:
                stream = sd.InputStream(
                    device=device,
                    channels=len(layout),
                    samplerate=device_sr,
                    blocksize=blocksize,
                    dtype="float32",
                    latency=self._latency,
                    extra_settings=extra,
                    callback=self._make_callback(device),
                )
                stream.start()
            except Exception as exc:
                # Fondamentale rilasciare il device: uno stream aperto e non
                # avviato lo tiene occupato e fa fallire ogni tentativo
                # successivo con errori host fuorvianti.
                if stream is not None:
                    try:
                        stream.abort(ignore_errors=True)
                        stream.close(ignore_errors=True)
                    except Exception:
                        pass
                errors.append(f"{len(layout)} canali: {exc}")
                continue

            self._streams[device] = stream
            self._plans[device] = self._build_plan(layout, routes, device_sr)
            self._last_block[device] = time.time()
            self._overflows.setdefault(device, 0)
            log.info(
                "cattura avviata su %s: %d canali @ %d Hz (blocchi da %d) -> %d Hz",
                device_label(device), len(layout), device_sr, blocksize, self.samplerate,
            )
            self._warn_if_suboptimal(device)
            return

        raise RuntimeError(
            f"impossibile aprire {device_label(device)} a {device_sr} Hz "
            f"({'; '.join(errors)})"
        )

    def _warn_if_suboptimal(self, device: int | None) -> None:
        """Avvisa se la scheda è stata aperta su una host API che ricampiona.

        MME e DirectSound passano dal mixer del kernel: dichiarano 44100 su una
        scheda che lavora a 48000, aggiungono latenza e un ricampionamento in
        più. Con un mixer digitale è quasi sempre un indice scelto a mano che
        punta al device sbagliato, e senza un avviso non te ne accorgi.
        """
        from .devices import device_info, list_input_devices

        info = device_info(device)
        if not info:
            return
        try:
            host = sd.query_hostapis()[int(info.get("hostapi", 0))].get("name", "")
        except Exception:
            return
        if not any(k in host.upper() for k in ("MME", "DIRECTSOUND")):
            return
        name = str(info.get("name", ""))
        better = [
            d for d in list_input_devices()
            if d.name.split("(")[0].strip() == name.split("(")[0].strip()
            and any(k in d.hostapi.upper() for k in ("ASIO", "WASAPI", "WDM-KS"))
            and d.max_input_channels >= int(info.get("max_input_channels", 0))
        ]
        hint = (
            f" Usa invece [{better[0].index}] {better[0].name} ({better[0].hostapi})"
            f", oppure indica il device per nome."
            if better else ""
        )
        log.warning(
            "il device è aperto via %s, che ricampiona nel kernel (qualità e "
            "latenza peggiori).%s", host, hint,
        )

    def _close_device(self, device: int | None) -> None:
        stream = self._streams.pop(device, None)
        self._plans.pop(device, None)
        self._last_block.pop(device, None)
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # pragma: no cover
            log.warning("errore chiusura stream device %s: %s", device, exc)

    def _make_callback(self, device: int | None):
        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                # input overflow = blocchi persi: se cresce, la macchina è satura
                # o il cavo/driver perde colpi. Lo esponiamo in /healthz.
                self._overflows[device] = self._overflows.get(device, 0) + 1
                log.debug("PortAudio status device %s: %s", device, status)
            self._last_block[device] = time.time()
            plan = self._plans.get(device)
            if plan is None:
                return
            try:
                for col, route, resampler, meter in plan.entries:
                    if col >= indata.shape[1]:
                        continue
                    mono = np.ascontiguousarray(indata[:, col])
                    if route.gain != 1.0:
                        mono = mono * route.gain
                    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
                    block = resampler.process(mono)
                    if len(block) == 0:
                        continue
                    meter.rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                    meter.peak = peak
                    meter.updated = time.time()
                    if peak >= 0.999:
                        meter.clipped += 1
                    route.callback(block)
            except Exception:  # pragma: no cover - la callback non deve morire
                log.exception("errore nella callback di cattura (device %s)", device)

        return _callback
