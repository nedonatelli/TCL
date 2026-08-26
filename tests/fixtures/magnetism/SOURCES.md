# Geomagnetic reference coefficients

Official model coefficients, vendored verbatim so the validation suite can check
this library's embedded tables against the published sources without a network
call or an optional dependency. Retrieved 2026-07-31.

These are the reference side of a REFERENCE-class test (see
`tests/validation/README.md`). Nothing in the library reads them; only
`tests/validation/test_magnetic_coefficients.py` does.

| File | Model | Source |
|------|-------|--------|
| `igrf13coeffs.txt` | IGRF-13, all epochs 1900.0-2020.0 plus 2020-25 secular variation | <https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf13coeffs.txt> |
| `igrf14coeffs.txt` | IGRF-14, all epochs 1900.0-2025.0 plus 2025-30 secular variation. Retrieved 2026-08-26; the embedded `_IGRF14_COF` constant in `pytcl/magnetism/igrf.py` must equal this file byte for byte (enforced by the validation suite) | <https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt> |
| `WMM_2020.COF` | WMM-2020, epoch 2020.0, released 2019-12-10 | NOAA NCEI World Magnetic Model 2020, as redistributed in `pygeomag` 1.1.0 |
| `WMM_2025.COF` | WMM-2025, epoch 2025.0, released 2024-11-13 | NOAA NCEI World Magnetic Model 2025, as redistributed in `pygeomag` 1.1.0 |

```
e65453b7d2ed34ae30f6f7361aaec403e103c2abf59ee46393bd4c4f880c4fa8  WMM_2020.COF
dfa8597825af4e0b87ff4198a5b4fb661b3c49f4cd090cd0164e0259b075582f  WMM_2025.COF
460b8d8beb9b4df84febe4f0b639f0dd54dccfe8ff0970616287b015fa721425  igrf13coeffs.txt
8f8d88403028fc4ee92c4f38d97b46e0a87e2cfc496045b43c9e26c1d6b0903c  igrf14coeffs.txt
```

## Provenance of the WMM files

`igrf13coeffs.txt` is byte-identical to the IAGA download. The two WMM files came
from `pygeomag` rather than direct download: NOAA's `pub/data/geomag/wmm/`
archive paths for the coefficient bundles now return 404, and `pygeomag`
redistributes the official files unmodified. Their leading degree-1 terms match
the published Technical Reports, which is the check that matters:

- WMM-2020 at epoch 2020.0: `g(1,0) = -29404.5`, `g(1,1) = -1450.7`,
  `h(1,1) = 4652.9` nT, with `g_dot(1,0) = 6.7` nT/yr.
- WMM-2025 at epoch 2025.0: `g(1,0) = -29351.8`, `g(1,1) = -1410.8`,
  `h(1,1) = 4545.4` nT, with `g_dot(1,0) = 12.0` nT/yr.

Each file's own header line carries its model name and release date, so the
provenance travels with the data.

## Updating

Replace the file, update the hash above, and run the validation suite. A model
revision that changes coefficients **should** fail the tests -- that is the point
of pinning them. Treat a failure as a question about which model this library
intends to ship, not as a test to relax.
