"""Diagnostica della scheda audio: quali device, quanti canali, quali livelli.

Serve a due cose che in un evento si fanno sempre, e sempre di corsa:

1. **capire se il mixer espone davvero tutti i canali** (un XR18 deve mostrare
   18 ingressi: se ne vedi 2 stai guardando il device stereo/WDM sbagliato);
2. **mappare i channel_index**: apri il probe, parla in un microfono e guarda
   quale colonna si accende. Niente tentativi alla cieca in regia.

Esempi:

    python scripts/audio_probe.py                     # elenca i device
    python scripts/audio_probe.py --device 12         # meter su tutti i canali
    python scripts/audio_probe.py --device XR18 -s 30 # per nome, 30 secondi
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    print(f"sounddevice non disponibile: {exc}")
    sys.exit(1)


def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("Host API compilate in PortAudio:")
    for h in hostapis:
        print(f"  · {h['name']}")
    if not any("ASIO" in h["name"].upper() for h in hostapis):
        print(
            "  (nessun ASIO: su Windows i canali multipli devono arrivare dal\n"
            "   driver WDM/WASAPI della scheda — vedi README, sezione XR18)"
        )
    print()
    print(f"{'idx':>4}  {'in':>3}  {'sr':>6}  host API              nome")
    print("-" * 78)
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        host = hostapis[dev["hostapi"]]["name"]
        mark = "  <-- multicanale" if dev["max_input_channels"] > 2 else ""
        print(
            f"{idx:>4}  {dev['max_input_channels']:>3}  "
            f"{int(dev['default_samplerate']):>6}  {host:<20}  "
            f"{dev['name']}{mark}"
        )


def resolve(spec: str) -> int:
    if spec.isdigit():
        return int(spec)
    needle = spec.lower()
    best = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            if best is None or dev["max_input_channels"] > best[0]:
                best = (dev["max_input_channels"], idx)
    if best is None:
        raise SystemExit(f"nessun device di input con nome contenente '{spec}'")
    return best[1]


def meter(device: int, seconds: float, samplerate: int | None, channels: int | None) -> None:
    info = sd.query_devices(device, "input")
    n = channels or int(info["max_input_channels"])
    sr = samplerate or int(info["default_samplerate"])
    print(f"Device [{device}] {info['name']}")
    print(f"Apro {n} canali @ {sr} Hz — parla nei microfoni, Ctrl+C per uscire\n")

    peaks = np.zeros(n, dtype=np.float32)

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"  ! PortAudio: {status}")
        np.maximum(peaks, np.abs(indata).max(axis=0), out=peaks)

    try:
        with sd.InputStream(
            device=device, channels=n, samplerate=sr, dtype="float32",
            blocksize=0, latency="high", callback=callback,
        ):
            t0 = time.time()
            while time.time() - t0 < seconds:
                time.sleep(0.15)
                lines = []
                for i in range(n):
                    db = 20 * np.log10(max(float(peaks[i]), 1e-6))
                    filled = int(max(0.0, min(1.0, (db + 60) / 60)) * 24)
                    lines.append(f"ch{i:<2} {'█' * filled}{'·' * (24 - filled)} {db:6.1f} dB")
                peaks[:] = 0.0
                sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\ninterrotto")
    except Exception as exc:
        raise SystemExit(
            f"\nimpossibile aprire il device a {sr} Hz con {n} canali: {exc}\n"
            f"Prova: --samplerate {int(info['default_samplerate'])} oppure un "
            f"--channels più basso, o un altro device/host API."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", help="indice PortAudio o parte del nome (es. XR18)")
    ap.add_argument("--seconds", "-s", type=float, default=60.0)
    ap.add_argument("--samplerate", type=int, default=None,
                    help="default: rate nativo del device (48000 sui mixer)")
    ap.add_argument("--channels", type=int, default=None,
                    help="default: tutti i canali di input del device")
    args = ap.parse_args()

    if not args.device:
        list_devices()
        print("\nPer i livelli:  python scripts/audio_probe.py --device <idx|nome>")
        return
    meter(resolve(args.device), args.seconds, args.samplerate, args.channels)


if __name__ == "__main__":
    main()
