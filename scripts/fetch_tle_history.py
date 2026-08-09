#!/usr/bin/env python3
"""One-time capture of TLE history from Space-Track for the validation fixture.

Requires SPACETRACK_USER and SPACETRACK_PASSWORD in the environment. Never run
by tests or CI; kept for provenance and reproducibility. See
tests/fixtures/tle/SOURCES.md.
"""

import gzip
import http.cookiejar
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

BASE = "https://www.space-track.org"
DAYS = 30

# One satellite per SGP4 regime, verified against CelesTrak's public GP API
# immediately before capture (see tests/fixtures/tle/SOURCES.md for the
# verification numbers and the STARLINK-1007 substitution rationale). The
# decaying entry (ELECTRON R/B) was picked from CelesTrak's last-30-days
# group for having a currently-tracked perigee well under 250 km.
SATELLITES = {
    25544: ("ISS (ZARYA)", "leo-high-drag"),
    45098: ("STARLINK-1184", "leo"),
    28474: ("GPS BIIR-13", "meo-deep-space"),
    41866: ("GOES-16", "geo-deep-space"),
    25485: ("MOLNIYA 1-91", "heo-high-eccentricity"),
    69702: ("ELECTRON R/B", "decaying"),
}

OUT = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "tle"


def main() -> int:
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not user or not password:
        print("SPACETRACK_USER and SPACETRACK_PASSWORD must be set")
        return 1

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    login = urllib.parse.urlencode({"identity": user, "password": password})
    with opener.open(f"{BASE}/ajaxauth/login", login.encode()) as resp:
        if resp.status != 200:
            print(f"login failed: HTTP {resp.status}")
            return 1

    ids = ",".join(str(i) for i in SATELLITES)
    query = (
        f"{BASE}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{ids}/EPOCH/%3Enow-{DAYS}/"
        "orderby/EPOCH%20asc/format/json"
    )
    with opener.open(query) as resp:
        records = json.load(resp)

    fixture: dict = {}
    for norad, (name, regime) in SATELLITES.items():
        rows = [r for r in records if int(r["NORAD_CAT_ID"]) == norad]
        seen: set = set()
        tles = []
        for r in rows:
            if r["EPOCH"] in seen:
                continue
            seen.add(r["EPOCH"])
            tles.append(
                {"epoch": r["EPOCH"], "line1": r["TLE_LINE1"], "line2": r["TLE_LINE2"]}
            )
        if len(tles) < 10:
            print(f"only {len(tles)} TLEs for {norad} ({name}) -- investigate")
            return 1
        fixture[str(norad)] = {"name": name, "regime": regime, "tles": tles}
        print(f"{norad} {name}: {len(tles)} TLEs")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tle_history.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(fixture, handle)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
