# Characterization tests

Tests that pin current behavior where correctness has not been established --
capturing what the code does today so that an unintended change is visible,
without claiming the behavior is right.

**This directory is currently empty**, and deliberately so. Values pinned
elsewhere in the suite are not characterization: the hard-coded numbers in
`validation/test_legendre_high_degree.py` come from mpmath at 60 digits and
those in `validation/test_ephemerides.py` from published ephemerides. Both
assert correctness against an outside source, so they belong in `validation/`.

Use this directory when wrapping legacy behavior that needs to be held steady
during a refactor before it can be validated.
