"""``format_tle`` round trips, and its output stays readable by other tools.

An exported function no test reached (gh-49). It is a thin one -- it joins the
stored line strings -- and that shape is exactly why the interesting assertion
is not "does it return a string" but "is what it returns still a TLE".

A TLE is a fixed-column format with a modulo-10 checksum on each line. Anything
that reformats a field, pads differently, or drops the checksum digit produces
something that this library might well re-read while every other tool rejects
it. So the round trip is closed twice: back through ``parse_tle``, and through
the independent ``sgp4`` package, which is the reference implementation the
validation suite already uses elsewhere.

The element set is Vandenberg-tracked ISS (NORAD 25544), epoch 2024-001.5,
taken from the existing test suite so the checksums are known-good.
"""

import pytest

from pytcl.astronomical.tle import format_tle, parse_tle

LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
LINE2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"
NAME = "ISS (ZARYA)"


@pytest.fixture
def named_tle():
    return parse_tle(LINE1, LINE2, name=NAME)


@pytest.fixture
def unnamed_tle():
    return parse_tle(LINE1, LINE2)


class TestRoundTrip:
    """Parse, format, and get back exactly what went in."""

    def test_formatting_a_parsed_tle_reproduces_the_input_byte_for_byte(
        self, named_tle
    ):
        """The strongest statement available: nothing is lost or reformatted.

        Weaker checks -- that the output has three lines, or that the elements
        survive -- would pass for a formatter that silently renormalized a
        field and broke the checksum.
        """
        assert format_tle(named_tle) == f"{NAME}\n{LINE1}\n{LINE2}"

    def test_the_formatted_output_parses_back_to_the_same_elements(self, named_tle):
        """Closing the loop through the parser, not just through string equality."""
        lines = format_tle(named_tle).splitlines()
        reparsed = parse_tle(lines[1], lines[2], name=lines[0])

        assert reparsed.name == named_tle.name
        assert reparsed.line1 == named_tle.line1
        assert reparsed.line2 == named_tle.line2
        assert reparsed.inclination == named_tle.inclination
        assert reparsed.mean_motion == named_tle.mean_motion
        assert reparsed.eccentricity == named_tle.eccentricity

    def test_the_checksums_still_validate_after_a_round_trip(self, named_tle):
        """``parse_tle`` verifies checksums by default, so this is the real test.

        Passing ``verify_checksum=True`` explicitly says so at the call site
        rather than relying on the default staying that way.
        """
        lines = format_tle(named_tle).splitlines()
        parse_tle(lines[1], lines[2], verify_checksum=True)

    def test_repeated_formatting_is_stable(self, named_tle):
        """Format, reparse, format again -- no drift on the second pass."""
        once = format_tle(named_tle)
        lines = once.splitlines()
        twice = format_tle(parse_tle(lines[1], lines[2], name=lines[0]))
        assert once == twice


class TestNameHandling:
    """Line 0 is optional in the format, and both forms must be emitted."""

    def test_the_name_is_included_by_default(self, named_tle):
        assert format_tle(named_tle).splitlines()[0] == NAME

    def test_the_name_can_be_suppressed(self, named_tle):
        output = format_tle(named_tle, include_name=False)
        assert output.splitlines() == [LINE1, LINE2]

    def test_a_tle_with_no_name_emits_two_lines(self, unnamed_tle):
        """``include_name=True`` with an empty name must not emit a blank line.

        A leading empty line is the kind of output that reads fine and then
        breaks whatever consumes it, because line 0 would be blank rather than
        absent.
        """
        assert format_tle(unnamed_tle).splitlines() == [LINE1, LINE2]

    def test_suppressing_the_name_changes_nothing_else(self, named_tle):
        """The two element lines are identical either way."""
        with_name = format_tle(named_tle, include_name=True).splitlines()
        without = format_tle(named_tle, include_name=False).splitlines()
        assert with_name[1:] == without


class TestIndependentReadback:
    """The output has to be a TLE to anyone, not just to this library."""

    def test_the_official_sgp4_package_accepts_the_formatted_lines(self, named_tle):
        """``sgp4`` is the reference implementation of the format.

        Round-tripping through this library's own parser cannot catch a
        convention both sides share and the rest of the world does not.
        """
        sgp4_api = pytest.importorskip(
            "sgp4.api", reason="the official sgp4 package is the format reference"
        )
        lines = format_tle(named_tle).splitlines()
        satellite = sgp4_api.Satrec.twoline2rv(lines[1], lines[2])

        assert satellite.satnum == 25544
        assert satellite.inclo == pytest.approx(named_tle.inclination, rel=1e-9)

    def test_the_formatted_lines_keep_the_fixed_column_width(self, named_tle):
        """Both element lines are exactly 69 characters, by the standard.

        Column positions carry meaning in this format, so a line of the wrong
        length is not merely untidy -- every field after the change is at the
        wrong offset.
        """
        for line in format_tle(named_tle, include_name=False).splitlines():
            assert len(line) == 69, f"line is {len(line)} characters, not 69: {line!r}"
