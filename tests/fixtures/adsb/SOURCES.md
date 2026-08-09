# Live air traffic, recorded once

A snapshot of real aircraft state vectors, vendored so the tracking stack can be
checked against measurements it did not generate, offline and without a network
call.

These are the reference side of a REFERENCE-class test (see
`tests/validation/README.md`). Nothing in the library reads them; only
`tests/validation/test_adsb_tracking.py` does.

| File | Contents | Source |
|------|----------|--------|
| `adsb_boston.json.gz` | 120 aircraft, 3600 position reports over 5 minutes, within 250 nm of 42.36 N, 71.01 W. Captured 2026-08-05T01:03:40Z. | ADSB.lol, <https://api.adsb.lol/v2/> |

## Why this data and not synthetic

Each record carries the aircraft's position **and the ground speed it broadcast
itself**. The filter estimates velocity from position alone, so the broadcast
speed is an independent measurement of a quantity the filter was never given —
which is what makes this a reference test rather than a self-consistency check.

Real traffic also supplies something no synthetic scenario does: aircraft that
mostly fly straight and occasionally do not. The innovation distribution that
produces — a very low median with a heavy tail — is the signature of manoeuvres
against a constant-velocity model, and it cannot be manufactured without
deciding in advance what the answer should be.

## Fields

`t` is seconds from the start of capture, derived from each report's `seen_pos`
age at poll time, so it is accurate to a fraction of a second rather than
exactly. `alt_ft` is barometric altitude in feet, `gs_kt` ground speed in knots
— both as broadcast. Positions are degrees, rounded to 6 decimal places
(~0.1 m, well inside ADS-B's own accuracy).

Trimmed to the 120 aircraft with the most reports, to keep the file small.

## Licence

**CC0 1.0 (public domain dedication).** ADSB.lol's privacy and licence page
states:

> By sending data to feed.adsb.lol / in.adsb.lol, you agree, to the extent
> possible under law, to waive all copyright and related or neighboring rights
> to the data you are sharing, under the CC0 license.

<https://www.adsb.lol/privacy-license/>

Contributors dedicate their feeds to the public domain, so the aggregated data
carries no redistribution conditions. Attribution is not required; it is given
above because knowing where a fixture came from is worth more than the licence
compels.

Separately, the project's API *code* is BSD 3-Clause (Copyright (c) 2023 Katia
Esposito). That is not what is vendored here.

The underlying ADS-B transmissions are unencrypted broadcasts any receiver can
pick up; what ADSB.lol adds is aggregation from volunteer feeders. If this file
is ever removed, `test_adsb_tracking.py` reads it through a helper that skips
cleanly when absent, so nothing breaks.
