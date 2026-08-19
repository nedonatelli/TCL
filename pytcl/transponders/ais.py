"""AIS NMEA decoding via pyais, with position-report extraction.

pyais is an optional dependency (the ``ais`` extra). It is imported lazily
inside `_import_pyais`, so importing this module never requires pyais to be
installed; calling `decode_ais` or `ais_position_reports` without it raises
:class:`~pytcl.core.exceptions.DependencyError`. This mirrors the guard
pattern in :mod:`pytcl.io.dataframes` / :mod:`pytcl.io.readers`.

This is the Python port's counterpart to the MATLAB TCL's
``Transponders/decodeAISString``, which wraps libais; here pyais plays that
role.

Sentinel handling (ITU-R M.1371) is applied in `ais_position_reports` rather
than left to pyais: empirically, pyais 3.2.1 does not normalize "not
available" sentinels itself -- it returns them unchanged (lat 91 deg, lon
181 deg, speed-over-ground 102.3 kn (1023 decideci-knots), course-over-ground
360.0 deg (3600 decidegrees), heading 511) -- so this module detects the raw
sentinel values coming back from pyais and converts each to NaN.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import DISTRIBUTION_NAME
from pytcl.diagnostics import diagnostics_enabled, logger

__all__ = [
    "nmea_checksum",
    "AISMessage",
    "PositionReports",
    "decode_ais",
    "ais_position_reports",
]

#: msg_type values that carry a Class A or Class B position report
#: (ITU-R M.1371 types 1, 2, 3, 18, 19).
_POSITION_REPORT_TYPES = frozenset({1, 2, 3, 18, 19})

# ITU-R M.1371 "not available" sentinels, as returned unmodified by pyais
# 3.2.1 (see module docstring for how this was verified).
_LAT_SENTINEL = 91.0
_LON_SENTINEL = 181.0
_SOG_SENTINEL = 102.3  # 1023 / 10 knots
_COG_SENTINEL = 360.0  # 3600 / 10 degrees
_HEADING_SENTINEL = 511

_KNOTS_TO_MPS = 0.514444


def _dependency_error() -> DependencyError:
    """Build the DependencyError raised when pyais is unavailable."""
    return DependencyError(
        "pyais is required to decode AIS NMEA sentences.",
        package="pyais",
        feature="AIS decoding",
        install_command=f"pip install {DISTRIBUTION_NAME}[ais]",
    )


def _import_pyais() -> Any:
    """Import and return the ``pyais`` module, or raise `DependencyError`.

    Returns
    -------
    module
        The imported ``pyais`` module.

    Raises
    ------
    DependencyError
        If pyais is not installed.
    """
    try:
        import pyais
    except ImportError as e:
        raise _dependency_error() from e
    return pyais


def _raise_missing() -> Any:
    """Unconditionally raise `DependencyError`.

    Same signature as `_import_pyais`; tests monkeypatch `_import_pyais` to
    this function to simulate pyais being absent without actually
    uninstalling it.
    """
    raise _dependency_error()


class AISMessage(NamedTuple):
    """One decoded AIS message.

    Attributes
    ----------
    msg_type : int
        ITU-R M.1371 message type (1-27).
    mmsi : int
        Maritime Mobile Service Identity of the transmitting station.
    fields : dict
        The pyais payload, normalized via its own ``asdict()`` -- every
        field the message type carries, including type-specific ones
        (e.g. ``shipname`` for type 5) not surfaced by `PositionReports`.
    """

    msg_type: int
    mmsi: int
    fields: dict[str, Any]


class PositionReports(NamedTuple):
    """Position reports (msg types 1, 2, 3, 18, 19) as parallel arrays.

    Attributes
    ----------
    mmsi : ndarray of int64, shape (n,)
    t : ndarray of float64, shape (n,)
        Receiver timestamp per report, from `times` when given to
        `ais_position_reports`; NaN otherwise.
    lat : ndarray of float64, shape (n,)
        Latitude, radians. NaN where pyais reports the ITU-R M.1371
        "not available" sentinel (91 deg).
    lon : ndarray of float64, shape (n,)
        Longitude, radians. NaN where pyais reports the sentinel (181 deg).
    sog : ndarray of float64, shape (n,)
        Speed over ground, m/s (pyais reports knots; converted here).
        NaN where pyais reports the sentinel (102.3 kn).
    cog : ndarray of float64, shape (n,)
        Course over ground, radians. NaN where pyais reports the sentinel
        (360.0 deg).
    heading : ndarray of float64, shape (n,)
        True heading, radians. NaN where pyais reports the sentinel (511).
    """

    mmsi: NDArray[np.int64]
    t: NDArray[np.float64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    sog: NDArray[np.float64]
    cog: NDArray[np.float64]
    heading: NDArray[np.float64]


def nmea_checksum(sentence: str) -> str:
    """Compute an NMEA sentence's ``*hh`` checksum.

    The checksum is the XOR of every character strictly between the leading
    ``!``/``$`` and the trailing ``*``, rendered as two uppercase hex digits.

    Parameters
    ----------
    sentence : str
        A full sentence, with or without its trailing ``*hh``.

    Returns
    -------
    str
        Two uppercase hex digits.

    Examples
    --------
    >>> from pytcl.transponders.ais import nmea_checksum
    >>> nmea_checksum("!AIVDM,1,1,,B,15M67FC000G?ufbE`FepT@3n00Sa,0*5C")
    '5C'
    """
    body = sentence.strip()
    # A line may carry an NMEA 4.10 TAG block (``\s:...,c:...*hh\``) and/or a
    # receiver timestamp before the sentence itself. The sentence checksum
    # covers only the characters between its own leading marker and its
    # trailing ``*``, so start at the LAST marker rather than position 0.
    # ``!`` and ``$`` cannot occur inside a six-bit-armoured AIS payload, so
    # the last one is always the sentence start.
    marker = max(body.rfind("!"), body.rfind("$"))
    if marker != -1:
        body = body[marker + 1 :]
    star = body.rfind("*")
    if star != -1:
        body = body[:star]
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return format(checksum, "02X")


def _checksum_is_valid(sentence: str) -> bool:
    """True if the sentence carries a ``*hh`` that matches its body.

    A sentence with no ``*hh`` at all is treated as invalid: an AIS sentence
    is required to carry one, and silently accepting a truncated line is the
    failure mode this guard exists to prevent.
    """
    stripped = sentence.strip()
    marker = max(stripped.rfind("!"), stripped.rfind("$"))
    if marker == -1:
        return False
    star = stripped.rfind("*")
    if star < marker or len(stripped) - star < 3:
        return False
    return stripped[star + 1 : star + 3].upper() == nmea_checksum(stripped)


def decode_ais(nmea_text: str, validate_checksum: bool = True) -> list[AISMessage]:
    """Decode AIS NMEA sentences (one or more, newline-separated) to messages.

    Multipart messages (e.g. type 5, split across two ``!AIVDM`` sentences
    with the same sequence id) are reassembled automatically -- pyais's own
    `~pyais.stream.IterMessages` groups fragments by sequence id, channel,
    talker and fragment count before assembling and decoding them, so a
    fragment is only turned into an `AISMessage` once every part of it has
    arrived.

    Lines that are not valid AIS sentences, or whose payload pyais cannot
    decode (e.g. an unsupported message type), are skipped rather than
    raising -- this is a batch decode over potentially noisy logs, not a
    single-message parse. The number skipped is logged at DEBUG (site
    ``"transponders"``) when diagnostics are enabled.

    Parameters
    ----------
    nmea_text : str
        One or more ``!AIVDM``/``!AIVDO`` sentences, one per line.
    validate_checksum : bool, optional
        Reject sentences whose trailing ``*hh`` does not match the XOR of
        their body, and sentences carrying no ``*hh`` at all. Default True.
        Rejected lines are skipped and counted like any other undecodable
        line. Set False only to ingest a feed known to carry bad checksums,
        accepting that a corrupted position report may decode to a plausible
        but wrong latitude and longitude.

    Returns
    -------
    list of AISMessage
        One entry per successfully decoded (and, where applicable,
        reassembled) message, in the order completed.

    Raises
    ------
    DependencyError
        If pyais is not installed.

    Examples
    --------
    >>> from pytcl.transponders.ais import decode_ais
    >>> vdm = "!AIVDM,1,1,,B,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
    >>> msgs = decode_ais(vdm)
    >>> msgs[0].msg_type
    1
    >>> msgs[0].mmsi
    366053209
    """
    _import_pyais()
    from pyais.exceptions import AISBaseException
    from pyais.stream import IterMessages

    lines = [line for line in nmea_text.splitlines() if line.strip()]

    if validate_checksum:
        kept = [line for line in lines if _checksum_is_valid(line)]
        n_bad = len(lines) - len(kept)
        if n_bad and diagnostics_enabled():
            logger.bind(site="transponders").debug(
                "decode_ais: rejected {} of {} line(s) on checksum",
                n_bad,
                len(lines),
            )
        lines = kept

    messages: list[AISMessage] = []
    consumed_lines = 0
    for sentence in IterMessages.from_strings(lines):
        try:
            decoded = sentence.decode()
        except AISBaseException:
            continue
        consumed_lines += sentence.fragment_count
        messages.append(
            AISMessage(
                msg_type=int(decoded.msg_type),
                mmsi=int(decoded.mmsi),
                fields=decoded.asdict(),
            )
        )

    n_skipped = len(lines) - consumed_lines
    if diagnostics_enabled() and n_skipped > 0:
        logger.bind(site="transponders").debug(
            "decode_ais: skipped {} of {} line(s) that did not decode",
            n_skipped,
            len(lines),
        )

    return messages


def _normalize(value: float, sentinel: float) -> float:
    """Map pyais's raw sentinel value to NaN; pass everything else through."""
    return float("nan") if value == sentinel else value


def ais_position_reports(
    nmea_text_or_messages: Union[str, Sequence[AISMessage]],
    times: Sequence[float] | None = None,
    validate_checksum: bool = True,
) -> PositionReports:
    """Extract position reports (types 1, 2, 3, 18, 19) as parallel arrays.

    Parameters
    ----------
    nmea_text_or_messages : str or sequence of AISMessage
        Either raw NMEA text (decoded internally via `decode_ais`) or an
        already-decoded message list, e.g. from a prior `decode_ais` call.
    times : sequence of float, optional
        One receiver timestamp per entry of `nmea_text_or_messages` (after
        decoding, if text was given) -- ``times[i]`` is the timestamp for
        the *i*-th decoded message, not the *i*-th position report. When
        given, its length must equal the number of decoded messages.
        Entries whose message is not a position report are dropped along
        with that message. When omitted, `PositionReports.t` is all NaN.
    validate_checksum : bool, optional
        Forwarded to `decode_ais` when raw text is given; ignored when an
        already-decoded message list is passed. Default True.

    Returns
    -------
    PositionReports
        One row per position-report message, in decode order. Units:
        lat/lon/cog/heading radians, sog m/s.

    Raises
    ------
    ValueError
        If `times` is given and its length does not match the number of
        decoded messages.
    DependencyError
        If pyais is not installed.

    Examples
    --------
    >>> from pytcl.transponders.ais import ais_position_reports
    >>> vdm = "!AIVDM,1,1,,B,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
    >>> rep = ais_position_reports(vdm)
    >>> rep.mmsi[0]
    366053209
    >>> bool(rep.lat[0] > 0)  # northern hemisphere
    True
    """
    _import_pyais()

    if isinstance(nmea_text_or_messages, str):
        messages: Sequence[AISMessage] = decode_ais(
            nmea_text_or_messages, validate_checksum=validate_checksum
        )
    else:
        messages = nmea_text_or_messages

    if times is not None and len(times) != len(messages):
        raise ValueError(
            f"times has length {len(times)}, expected {len(messages)} "
            "(one per decoded message)"
        )

    mmsi: list[int] = []
    t: list[float] = []
    lat: list[float] = []
    lon: list[float] = []
    sog: list[float] = []
    cog: list[float] = []
    heading: list[float] = []

    for i, msg in enumerate(messages):
        if msg.msg_type not in _POSITION_REPORT_TYPES:
            continue
        f = msg.fields
        mmsi.append(msg.mmsi)
        t.append(float(times[i]) if times is not None else float("nan"))
        lat.append(np.radians(_normalize(f["lat"], _LAT_SENTINEL)))
        lon.append(np.radians(_normalize(f["lon"], _LON_SENTINEL)))
        sog.append(_normalize(f["speed"], _SOG_SENTINEL) * _KNOTS_TO_MPS)
        cog.append(np.radians(_normalize(f["course"], _COG_SENTINEL)))
        heading.append(np.radians(_normalize(f["heading"], _HEADING_SENTINEL)))

    return PositionReports(
        mmsi=np.array(mmsi, dtype=np.int64),
        t=np.array(t, dtype=np.float64),
        lat=np.array(lat, dtype=np.float64),
        lon=np.array(lon, dtype=np.float64),
        sog=np.array(sog, dtype=np.float64),
        cog=np.array(cog, dtype=np.float64),
        heading=np.array(heading, dtype=np.float64),
    )
