"""Tests for AIS decoding (`pytcl.transponders.ais`).

Vector provenance
------------------
``VDM_TYPE1`` is the widely published NMEA test sentence for a type-1
position report (San Francisco Bay AISDeco/gpsd sample); it is cross-checked
against pyais itself below rather than trusted blind.

``VDM_TYPE5_1`` / ``VDM_TYPE5_2`` are NOT copied from a web page: the
installed pyais 3.2.1 wheel ships no ``tests/`` directory (verified via
``find .venv/lib/python*/site-packages/pyais`` -- only the library modules
are present), so per the task brief's fallback this pair was constructed with
pyais's own encoder (``pyais.encode.encode_dict``) for a type-5 static/voyage
message and round-tripped through ``pyais.decode`` to confirm it reassembles
and decodes correctly before being pasted here -- see
``.superpowers/sdd/2026-08-11-results-io/task-4-report.md`` for the exact
construction script and its output.
"""

import numpy as np
import pytest

pytest.importorskip("pyais")

from pytcl.transponders.ais import ais_position_reports, decode_ais

# Standard type-1 position report test sentence (widely published vector).
VDM_TYPE1 = "!AIVDM,1,1,,A,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
# Two-part type-5 static/voyage message, constructed with pyais.encode and
# verified via pyais.decode (see module docstring above).
VDM_TYPE5_1 = (
    "!AIVDM,2,1,0,A,51mg=5@2:N2T48<@000EHE:0LUHDp00000000000<PN::5Wd0ODSm51DQ0C@,0*27"
)
VDM_TYPE5_2 = "!AIVDM,2,2,0,A,00000000000,2*24"


class TestDecode:
    def test_type1_position(self):
        msgs = decode_ais(VDM_TYPE1)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 1
        assert msgs[0].mmsi > 0

    def test_multipart_assembly(self):
        msgs = decode_ais(VDM_TYPE5_1 + "\n" + VDM_TYPE5_2)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 5

    def test_garbage_lines_skipped_not_fatal(self):
        msgs = decode_ais("not an nmea line\n" + VDM_TYPE1 + "\nanother bad line")
        assert len(msgs) == 1


class TestPositionReports:
    def test_arrays_and_units(self):
        rep = ais_position_reports(VDM_TYPE1)
        assert rep.lat.dtype == np.float64
        assert abs(rep.lat[0]) < np.pi / 2 + 1e-9  # radians, not degrees
        assert abs(rep.lon[0]) < np.pi + 1e-9
        # cross-check against pyais directly: same sentence, degrees->radians
        import pyais

        decoded = pyais.decode(VDM_TYPE1)
        assert rep.lat[0] == pytest.approx(np.radians(decoded.lat), abs=0)
        assert rep.sog[0] == pytest.approx(decoded.speed * 0.514444, rel=1e-9)

    def test_non_position_messages_excluded(self):
        rep = ais_position_reports(VDM_TYPE5_1 + "\n" + VDM_TYPE5_2)
        assert len(rep.mmsi) == 0


class TestSentinels:
    """ITU-R M.1371 sentinels -> NaN.

    pyais 3.2.1 does not normalize these itself -- it returns the raw
    sentinel values unchanged (verified empirically: encoding a type-1
    message with lat=91.0, lon=181.0, speed=102.3 (1023 decideci-knots),
    course=360.0 (3600 decidegrees) and heading=511 and decoding it back
    with ``pyais.decode`` returns exactly those values, not ``None`` or
    ``nan``). `ais_position_reports` must do the normalization itself.
    """

    def test_all_sentinels_become_nan(self):
        import pyais
        from pyais.encode import encode_dict

        sentences = encode_dict(
            {
                "type": 1,
                "mmsi": 123456789,
                "lat": 91.0,
                "lon": 181.0,
                "speed": 102.3,
                "course": 360.0,
                "heading": 511,
            },
            talker_id="AI",
            sentence_type="VDM",
        )
        nmea_text = "\n".join(sentences)

        # Confirm pyais itself does not normalize these (documents the
        # empirical behavior this module normalizes away).
        raw = pyais.decode(*sentences)
        assert raw.lat == 91.0
        assert raw.lon == 181.0
        assert raw.speed == 102.3
        assert raw.course == 360.0
        assert raw.heading == 511

        rep = ais_position_reports(nmea_text)
        assert len(rep.mmsi) == 1
        assert np.isnan(rep.lat[0])
        assert np.isnan(rep.lon[0])
        assert np.isnan(rep.sog[0])
        assert np.isnan(rep.cog[0])
        assert np.isnan(rep.heading[0])

    def test_no_times_given_yields_nan_t(self):
        rep = ais_position_reports(VDM_TYPE1)
        assert len(rep.t) == 1
        assert np.isnan(rep.t[0])

    def test_times_given_align_with_position_reports(self):
        msgs = decode_ais(VDM_TYPE1 + "\n" + VDM_TYPE1)
        rep = ais_position_reports(msgs, times=[10.0, 20.0])
        assert rep.t.tolist() == [10.0, 20.0]
