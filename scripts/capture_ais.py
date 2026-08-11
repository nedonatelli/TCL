#!/usr/bin/env python3
"""One-time capture of a live AIS NMEA stream for the validation fixture.

Connects to the Norwegian Coastal Administration's (Kystverket) open AIS
relay and writes every received NMEA line, one per line, stamped with the
receiver's own Unix time: ``<unix_time>\\t<nmea_sentence>``. The timestamp is
this script's wall-clock time on receipt, not any timestamp embedded in the
stream (Kystverket prefixes each sentence with a ``\\s:...,c:...*hh\\`` tag
block that carries its own ``c:`` epoch; that field is left untouched as part
of the raw sentence and is not treated as the receiver time here).

Never run by tests or CI; kept for provenance and reproducibility, like
``scripts/fetch_tle_history.py``. See ``tests/fixtures/ais/SOURCES.md``.

Usage
-----
    python scripts/capture_ais.py --duration 300 --output out.nmea
"""

import argparse
import pathlib
import socket
import sys
import time

HOST = "153.44.253.27"
PORT = 5631
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 30.0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="capture duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="path to write captured lines to",
    )
    parser.add_argument(
        "--host", default=HOST, help=f"AIS stream host (default: {HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"AIS stream port (default: {PORT})"
    )
    return parser.parse_args(argv)


def _connect(host: str, port: int) -> socket.socket:
    try:
        sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S)
    except OSError as e:
        raise SystemExit(
            f"FATAL: could not connect to AIS stream {host}:{port} ({e}). "
            "If the endpoint moved, check kystverket.no for the current one."
        ) from e
    sock.settimeout(READ_TIMEOUT_S)
    return sock


def capture(host: str, port: int, duration_s: float, output: pathlib.Path) -> int:
    sock = _connect(host, port)
    reader = sock.makefile("r", encoding="ascii", errors="replace", newline="\n")

    n_lines = 0
    start = time.time()
    deadline = start + duration_s
    try:
        with output.open("w", encoding="ascii", errors="replace") as out:
            while True:
                now = time.time()
                if now >= deadline:
                    break
                try:
                    line = reader.readline()
                except socket.timeout as e:
                    raise SystemExit(
                        f"FATAL: no data received for {READ_TIMEOUT_S}s from "
                        f"{host}:{port} -- stream stalled after {n_lines} line(s)."
                    ) from e
                if not line:
                    raise SystemExit(
                        f"FATAL: {host}:{port} closed the connection after "
                        f"{n_lines} line(s) ({now - start:.1f}s in)."
                    )
                line = line.rstrip("\r\n")
                if not line:
                    continue
                out.write(f"{now}\t{line}\n")
                n_lines += 1
    finally:
        sock.close()

    elapsed = time.time() - start
    print(f"captured {n_lines} line(s) over {elapsed:.1f}s -> {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return capture(args.host, args.port, args.duration, args.output)


if __name__ == "__main__":
    sys.exit(main())
