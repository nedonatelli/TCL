"""Oracle tests for pytcl.gravity.models.gravity_j2.

Task 2.3 (v2.11.1 correctness patch): ``gravity_j2`` used ``r = a + h``
and treated the geodetic latitude input as if it were geocentric. At the
pole the error is 6572 mGal (0.67%); at 45 deg it is 3277 mGal. The
equator error is only 4.4 mGal, which is why an equator-only test would
miss the defect -- ``normal_gravity`` (an independent Somigliana-formula
implementation) is used as the oracle here, parametrized over latitude
including the pole. See
``.superpowers/sdd/2026-09-14-v2.11.1-tier1-patch/task-2.3-brief.md``.
"""

import numpy as np
import pytest

from pytcl.gravity.models import gravity_j2, normal_gravity

# gravity_j2 is a J2-only truncation of the ellipsoidal field; even with
# the true radius and geocentric latitude it disagrees with the exact
# Somigliana closed form by a genuine, latitude-dependent few-mGal model
# residual (measured up to ~12 mGal at the pole). 30 mGal comfortably
# clears that residual while still catching the task 2.3 defect, whose
# smallest measured error (4.4 mGal at the equator) is the one case this
# tolerance cannot discriminate -- the 45/60/90 degree cases (3277-6572
# mGal pre-fix) do the discriminating.
_TOLERANCE_MGAL = 30
_TOLERANCE = _TOLERANCE_MGAL * 1e-5


@pytest.mark.parametrize("lat_deg", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_gravity_j2_tracks_normal_gravity(lat_deg):
    """J2 and Somigliana agree to within the J2-truncation residual."""
    lat = np.radians(lat_deg)
    got = gravity_j2(lat, 0.0).magnitude
    expected = normal_gravity(lat, 0.0)
    assert abs(got - expected) < _TOLERANCE, f"{abs(got - expected) * 1e5:.0f} mGal"
