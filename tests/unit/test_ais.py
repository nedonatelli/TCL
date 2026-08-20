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
and decodes correctly before being pasted here. Regenerate an equivalent pair
with ``pyais.encode.encode_dict({"msg_type": 5, ...})`` if these ever need
replacing.
"""

import numpy as np
import pytest

pytest.importorskip("pyais")

from pytcl.transponders.ais import (
    ais_position_reports,
    decode_ais,
    nmea_checksum,
)

# Standard type-1 position report test sentence (widely published vector).
VDM_TYPE1 = "!AIVDM,1,1,,B,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
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


class TestClassBPositionReports:
    """Type-18 (Class B) and type-19 (Class B extended) position reports.

    Carried over from Task 4's review: the reviewer verified units and
    sentinel handling for these two message types by hand, ad hoc, against
    sentences built with ``pyais.encode.encode_dict`` -- this class turns
    that empirical proof into a permanent regression test. Sentences are
    built here (not pasted from a web page) for the same reason as
    ``VDM_TYPE5_1``/``VDM_TYPE5_2`` above: the installed pyais wheel ships
    no ``tests/`` directory to source published vectors from.
    """

    def test_type18_units(self):
        from pyais.encode import encode_dict

        sentences = encode_dict(
            {
                "type": 18,
                "mmsi": 111222333,
                "lat": 59.913,
                "lon": 10.752,
                "speed": 12.3,
                "course": 87.4,
                "heading": 90,
            },
            talker_id="AI",
            sentence_type="VDM",
        )
        nmea_text = "\n".join(sentences)

        msgs = decode_ais(nmea_text)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 18
        assert msgs[0].mmsi == 111222333

        import pyais

        decoded = pyais.decode(*sentences)
        rep = ais_position_reports(nmea_text)
        assert len(rep.mmsi) == 1
        assert rep.mmsi[0] == 111222333
        assert rep.lat[0] == pytest.approx(np.radians(decoded.lat), abs=0)
        assert rep.lon[0] == pytest.approx(np.radians(decoded.lon), abs=0)
        assert rep.sog[0] == pytest.approx(decoded.speed * 0.514444, rel=1e-9)
        assert rep.cog[0] == pytest.approx(np.radians(decoded.course), abs=0)
        assert rep.heading[0] == pytest.approx(np.radians(decoded.heading), abs=0)

    def test_type19_units(self):
        from pyais.encode import encode_dict

        sentences = encode_dict(
            {
                "type": 19,
                "mmsi": 444555666,
                "lat": 60.391,
                "lon": 5.322,
                "speed": 8.7,
                "course": 210.5,
                "heading": 205,
            },
            talker_id="AI",
            sentence_type="VDM",
        )
        nmea_text = "\n".join(sentences)

        msgs = decode_ais(nmea_text)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 19
        assert msgs[0].mmsi == 444555666

        import pyais

        decoded = pyais.decode(*sentences)
        rep = ais_position_reports(nmea_text)
        assert len(rep.mmsi) == 1
        assert rep.mmsi[0] == 444555666
        assert rep.lat[0] == pytest.approx(np.radians(decoded.lat), abs=0)
        assert rep.lon[0] == pytest.approx(np.radians(decoded.lon), abs=0)
        assert rep.sog[0] == pytest.approx(decoded.speed * 0.514444, rel=1e-9)
        assert rep.cog[0] == pytest.approx(np.radians(decoded.course), abs=0)
        assert rep.heading[0] == pytest.approx(np.radians(decoded.heading), abs=0)

    def test_type18_and_type19_sentinels_become_nan(self):
        from pyais.encode import encode_dict

        for msg_type, mmsi in ((18, 111222333), (19, 444555666)):
            sentences = encode_dict(
                {
                    "type": msg_type,
                    "mmsi": mmsi,
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

            rep = ais_position_reports(nmea_text)
            assert len(rep.mmsi) == 1
            assert np.isnan(rep.lat[0])
            assert np.isnan(rep.lon[0])
            assert np.isnan(rep.sog[0])
            assert np.isnan(rep.cog[0])
            assert np.isnan(rep.heading[0])


class TestDependencyError:
    def test_dependency_error_without_pyais(self, monkeypatch):
        import pytcl.transponders.ais as mod

        monkeypatch.setattr(mod, "_import_pyais", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="ais"):
            decode_ais(VDM_TYPE1)


class TestChecksumValidation:
    """``decode_ais`` validates the trailing ``*hh`` by default.

    Before this was added it validated nothing: ``IterMessages`` does not
    enforce ``NMEAMessage.is_valid``, so a corrupted sentence decoded to a
    plausible-looking position. The defect hid itself -- ``VDM_TYPE1`` was
    published here on channel A while carrying channel B's ``5C``, and
    passed only because nothing checked.
    """

    def test_the_canonical_vector_carries_a_valid_checksum(self):
        star = VDM_TYPE1.rfind("*")
        assert nmea_checksum(VDM_TYPE1) == VDM_TYPE1[star + 1 :]

    def test_corrupted_checksum_is_rejected(self):
        corrupted = VDM_TYPE1[:-2] + "00"
        assert decode_ais(corrupted) == []
        # ...and the payload really is otherwise decodable, so the rejection
        # is the checksum's doing and not an unrelated parse failure.
        assert len(decode_ais(corrupted, validate_checksum=False)) == 1

    def test_missing_checksum_is_rejected(self):
        assert decode_ais(VDM_TYPE1.split("*")[0]) == []

    def test_valid_sentence_still_decodes(self):
        msgs = decode_ais(VDM_TYPE1)
        assert len(msgs) == 1
        assert msgs[0].mmsi == 366053209

    def test_one_bad_line_does_not_discard_the_good_ones(self):
        text = "\n".join([VDM_TYPE1, VDM_TYPE1[:-2] + "00", VDM_TYPE1])
        assert len(decode_ais(text)) == 2

    def test_position_reports_forwards_the_flag(self):
        corrupted = VDM_TYPE1[:-2] + "00"
        assert len(ais_position_reports(corrupted).mmsi) == 0
        assert len(ais_position_reports(corrupted, validate_checksum=False).mmsi) == 1

    def test_nmea_checksum_ignores_the_leading_marker_and_trailing_hh(self):
        """Both forms of the same sentence must hash identically."""
        without = VDM_TYPE1[: VDM_TYPE1.rfind("*")]
        assert nmea_checksum(VDM_TYPE1) == nmea_checksum(without)

    def test_tag_block_prefix_does_not_break_validation(self):
        """NMEA 4.10 TAG blocks precede the sentence and carry their own ``*hh``.

        The first cut of this validator hashed from position 0, so a
        TAG-blocked line hashed the tag as well and was rejected. That is
        6,774 of the 6,831 sentences in tests/fixtures/ais -- i.e. the
        validator would have discarded almost every real feed while passing
        the hand-written vector in this file.
        """
        tagged = "\\s:2573235,c:1786476163*02\\" + VDM_TYPE1
        assert nmea_checksum(tagged) == nmea_checksum(VDM_TYPE1)
        assert len(decode_ais(tagged)) == 1

        corrupted = tagged[:-2] + "00"
        assert decode_ais(corrupted) == []
