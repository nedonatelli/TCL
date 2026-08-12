"""Transponder message decoding: AIS (Automatic Identification System).

Named after the MATLAB TCL's ``Transponders/`` directory, whose
``decodeAISString`` wraps libais; :mod:`pytcl.transponders.ais` plays the
same role here over pyais (the ``ais`` extra). Importing this package never
requires pyais to be installed -- pyais is imported lazily inside
`decode_ais` / `ais_position_reports`, raising
:class:`~pytcl.core.exceptions.DependencyError` if it is missing when one of
them is actually called.

Examples
--------
>>> import pytcl.transponders as transponders
>>> vdm = "!AIVDM,1,1,,A,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
>>> msgs = transponders.decode_ais(vdm)
>>> msgs[0].msg_type
1
"""

from pytcl.transponders.ais import (
    AISMessage,
    PositionReports,
    ais_position_reports,
    decode_ais,
)

__all__ = [
    "AISMessage",
    "PositionReports",
    "decode_ais",
    "ais_position_reports",
]
