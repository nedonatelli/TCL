# Validation tests

Results checked against something independent of this codebase: a reference
implementation, or published data.

Oracles in use: `scipy`, `pyproj`, `geographiclib`, `astropy`/`pyerfa`, the
official `sgp4` package, `satkit` (independent Rust SGP4 + IAU frames; needs
`uv sync --group validation`), `mpmath` at 50-60 digits, `scikit-learn`,
`PyWavelets`, brute-force enumeration, and constants from CODATA 2018, WGS84,
IERS and IAU. Real-world recordings also serve as references: a vendored ADS-B
air-traffic capture (aircraft broadcast their own ground speed, which the
filter never sees).

This is where the audit suites live. They exist because structural tests passed
while WMM magnetism was roughly 180 degrees wrong -- line coverage is not the
bar, independent agreement is. See the validation classes in CONTRIBUTING.md.
