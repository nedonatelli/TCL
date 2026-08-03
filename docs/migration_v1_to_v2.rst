Migrating from v1.x to v2.0.0
=============================

v2.0.0 breaks compatibility in a small number of places. There is no
deprecation cycle: the removed names are gone and the changed signatures raise
rather than warn. That is deliberate — a shim that quietly accepts an old call
and does something slightly different is the failure mode this release exists
to remove — but it means an upgrade needs a read-through rather than a bump.

Every change below was made because the old behaviour could give a confidently
wrong answer, and each entry says what that answer was.

.. contents:: On this page
   :local:
   :depth: 1


Spatial containers
------------------

``query(k)`` rejects ``k`` larger than the index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All five indexes — KD-tree, ball tree, R-tree, VP-tree, cover tree — used to
pad a too-large ``k`` with index ``0`` and an infinite distance. Zero is a
*valid* index, so code that read ``result.indices`` without also reading
``result.distances`` silently treated point 0 as a neighbour, once per
overshoot.

.. code-block:: python

   # v1.x: returns indices [0, 1, 2, 0, 0] for a 3-point index
   result = tree.query(points, k=5)

   # v2.0.0: raises ValueError
   result = tree.query(points, k=min(5, tree.n_samples))

``k == n_samples`` remains valid. If you want "up to k", clamp it as above;
the error message says so.

``BoundingBox.volume`` is zero for a degenerate box
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A flat box ``[0,0]-[2,0]`` used to report ``2.0``, because the property
multiplied only the nonzero extents. It now reports ``0.0``, which is the
volume it encloses.

If you relied on the old value you wanted the R-tree insertion measure, not a
volume; that behaviour is retained privately for the tree's own use.


Navigation
----------

INS/GNSS position covariance is in ``[rad, rad, m]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The error state has always been ``[dlat, dlon, dheight]`` in ``[rad, rad, m]``,
but ``initialize_ins_gnss`` placed a metres-valued ``position_std`` directly on
all three diagonal entries.

With the old defaults the two errors cancelled — measurement noise was also
wrongly in metres — so the filter behaved sensibly. The damage appeared the
moment a caller supplied a **correctly scaled** ``position_cov``: the filter's
own covariance was then larger by roughly the square of an Earth radius, and it
absorbed essentially 100% of every measurement regardless of quality.

.. code-block:: python

   from pytcl.navigation.ins_gnss import position_std_to_error_state_units

   # v2.0.0: convert metres to the units the states use
   std = position_std_to_error_state_units(2.0, lat, height)   # 2 m accuracy
   gnss = gnss._replace(position_cov=np.diag([std[0]**2, std[1]**2, 2.0**2]))

If you pass ``position_cov``, convert it. If you rely on the defaults, no code
change is needed, but the filter is now weighted differently — retune if you
had compensated for the old behaviour.

``compute_dop`` needs the user position for HDOP and VDOP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HDOP and VDOP are only horizontal and vertical relative to a local frame. Given
an ECEF geometry matrix they were computed against ECEF x/y/z, which are
horizontal and vertical only at the poles. At 45° latitude the reported values
were close to *each other's* truth.

.. code-block:: python

   # v2.0.0: pass the user position so the matrix is rotated into ENU
   gdop, pdop, hdop, vdop = compute_dop(H_ecef, user_lla=[lat, lon, alt])

Omit ``user_lla`` only when ``H`` is already in a local frame. GDOP and PDOP
are rotation-invariant and were always correct.


Signal processing and statistics
--------------------------------

``detection_probability`` no longer takes ``swerling_case``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All five branches evaluated the same expression, so the argument selected
nothing — a caller asking for a non-fluctuating target got the Swerling 1
answer, which at SNR 10 and Pfa 1e-6 is 0.62 against a true 0.90.

.. code-block:: python

   # v1.x: argument accepted and ignored
   pd = detection_probability(snr, pfa, n_ref, swerling_case=0)

   # v2.0.0: for a real choice of target model
   from pytcl.mathematical_functions.special_functions import (
       swerling_detection_probability,
   )
   pd = swerling_detection_probability(snr, pfa, n_pulses=1, swerling_case=0)

Note the two answer slightly different questions: ``detection_probability``
accounts for threshold estimation from ``n_ref`` reference cells,
``swerling_detection_probability`` for integration over ``n_pulses``.

``snr_loss`` requires ``pfa`` and covers CA only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CFAR loss depends on the operating point, so the old ``1 + c/n_ref``
heuristics — which took no ``pfa`` at all — could not express it. They
understated the loss roughly fourfold.

.. code-block:: python

   # v1.x
   loss = snr_loss(32, method="ca")

   # v2.0.0
   loss = snr_loss(32, pfa=1e-6, pd=0.5, method="ca")

``'go'``, ``'so'`` and ``'os'`` now raise ``NotImplementedError``. The loss is
defined through the detection probability and no closed form is available here
for those three; the previous numbers were not derived from anything.

``nuttall_q`` is now ``rician_cdf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function computes ``1 - Q_1(a, b)``, the Rician CDF, and always did so
correctly. The Nuttall Q function is a different integral. ``nuttall_q``
remains as a deprecated alias that warns; it will be removed in a later
release.

Values that changed without a signature change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These take the same arguments and return different numbers. If you have
baselines recorded against v1.x, they will move:

``matched_filter(...).snr_gain``
   Now ``sum(t^2) / max(t^2)`` rather than ``len(template)``. Identical for a
   constant-modulus template; a 64-point Hann window has 24 effective samples,
   so the old figure was 4.3 dB optimistic.

``optimal_filter``
   Now a linear correlation. The old circular one wrapped a target at the start
   of a record into a phantom at the end reaching 94% of the true peak, across
   samples whose correct value is exactly zero.

``mle_gaussian`` (multivariate)
   ``fisher_info`` and ``covariance`` are now the exact expressions; they used
   to be ``eye(n) * n`` and ``eye(n) / n``, independent of the data.

``q_discrete_white_noise`` with ``dim > 4``
   Now the same discrete gain-vector model used for dims 2–4. It previously
   fell through to a *continuous* white-noise discretization, off by roughly a
   factor of four in the leading term at dim 5.

``tria_sqrt``
   Returns the documented ``(n, n)`` factor. It previously returned ``(n, k)``
   when the product was rank deficient. ``S @ S.T`` is unchanged.

``viewshed``
   Marks the nearest cell rather than the one to the south-west. Results shift
   by up to half a cell in each axis.

MHT ``track.score``
   Now the standard log-likelihood-ratio increment. It carried a factor of 0.5
   that the missed-detection branch did not, so hits and misses accumulated on
   different scales. Nothing reads this field — confirm and delete go by
   M-of-N — so tracking behaviour is unchanged.


GPU filters
-----------

Filter callbacks take the whole batch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The three GPU filters took three different callback contracts.
``batch_ekf_predict`` converted the states to NumPy and called ``f`` once per
track; ``batch_ukf_predict`` looped per sigma point of per track;
``CuPyParticleFilter.predict`` passed the whole population through as a device
array. A callback written for one could not be handed to another, and nothing
documented the difference.

They now share one rule: **a callback receives** ``(N, dim)`` **on the active
backend and returns** ``(N, out_dim)``. Jacobian callbacks return
``(N, out_dim, dim)``. ``N`` is whatever batch the filter holds -- tracks for
the EKF, ``n_tracks * (2 * state_dim + 1)`` sigma points for the UKF, particles
for the particle filter -- so one callable serves all three.

.. code-block:: python

   # v1.x: called once per track with a 1-D NumPy state
   def f(x):
       return np.array([x[0] + x[1], x[1] * 0.99])

   def F_jacobian(x):
       return np.array([[1.0, 1.0], [0.0, 0.99]])

   # v2.0.0: called once with the whole batch, on the active backend
   from pytcl.gpu.utils import get_array_module

   def f(x):
       xp = get_array_module(x)
       return xp.stack([x[:, 0] + x[:, 1], x[:, 1] * 0.99], axis=1)

   def F_jacobian(x):
       xp = get_array_module(x)
       J = xp.array([[1.0, 1.0], [0.0, 0.99]])
       return xp.broadcast_to(J, (x.shape[0], 2, 2))

Write callbacks against :func:`pytcl.gpu.utils.get_array_module` rather than
NumPy directly. Mixing a host NumPy array into the expression raises
``TypeError`` on CuPy, which refuses implicit conversion of a device array;
MLX permits it, so a callback tested only on Apple Silicon can still fail on
CUDA.

``CuPyParticleFilter`` already followed this contract and is unchanged.

Two consequences beyond consistency. The callback is invoked once rather than
once per item: a 200-track UKF prediction went from 1000 invocations to 1, and
a numerical Jacobian over the same batch from 1000 to 5. And the state no
longer round-trips to the host in the middle of the filter.

The numerical-Jacobian step now follows the backend's precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``F_jacobian`` or ``H_jacobian`` is ``None``, the central-difference step
defaults to ``1e-7`` on a float64 backend and ``1e-3`` on float32. The old code
used ``1e-7`` unconditionally, which is below what float32 can resolve -- on
MLX it returned rounding noise rather than a derivative. Pass ``eps``
explicitly if you need the old value.


Storage and I/O
---------------

``SQLStorage`` no longer takes ``db_type``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any value other than ``'sqlite'`` made ``open()`` do nothing at all, after
which every method raised ``RuntimeError``. It advertised backends that did not
exist.

.. code-block:: python

   store = SQLStorage()          # v2.0.0: no arguments

``open(mode="r")`` does not create a database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Opening a nonexistent path for reading used to create an empty file, after
which reads failed with ``sqlite3.OperationalError`` about a missing table
rather than the documented ``KeyError``. It now raises ``FileNotFoundError``
and leaves no file behind. Use ``mode='w'`` or ``'a'`` to create.

``store_array`` replaces on both backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SQLStorage`` replaced an existing name; ``HDF5Storage`` let h5py raise
``ValueError``. Both now replace, which is the contract ``StorageBackend``
states. Metadata is replaced wholesale along with the array.

``get_track_history`` residuals are row-aligned
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Residuals were read from the first row only, so a window beginning with a
prediction reported ``residuals=None`` even when later rows had them — and a
mixed window returned an array *shorter* than ``timestamps``, pairing every
residual with the wrong time.

.. code-block:: python

   history = db.get_track_history(track_id)
   residuals = history["residuals"]          # (N, meas_dim), aligned, or None
   has_residual = ~np.isnan(residuals).any(axis=1)

Rows without a residual now hold ``NaN``. ``None`` means no row in the range
has one.

Unknown track ids raise
^^^^^^^^^^^^^^^^^^^^^^^

``update_track_state`` used to insert the state row and update zero tracks,
leaving history belonging to no track — retrievable by id, so a typo produced
something that looked like a track in every respect but the one that counts.
It and ``merge_tracks`` now raise ``KeyError``.


Removed modules
---------------

``pytcl.logging_config``
   Offered hierarchical loggers, a ``@timed`` decorator and a
   ``TimingContext``. Nothing in the library ever used them; the thirteen
   modules that log call ``logging.getLogger`` from the standard library
   directly. Replace ``get_logger(__name__)`` with
   ``logging.getLogger("pytcl.<subpackage>")`` — the hierarchy it configured is
   what the standard library gives you anyway.

``pytcl.assignment_algorithms.network_simplex``
   Superseded by the Dijkstra-with-potentials implementation in v1.8.0, and its
   one function was separately incorrect. Min-cost flow remains available
   through ``min_cost_flow_successive_shortest_paths``,
   ``min_cost_flow_simplex`` and ``min_cost_assignment_via_flow``; the
   surviving solver is validated against a linear-programming oracle.

Neither module came from the NRL Tracker Component Library — both were
additions made by this port and never wired into anything.


Checking your upgrade
---------------------

The library's own public-API gate is a reasonable model for what to check on
your side: every exported function reached by at least one test, and no
standing exemptions. If you have a test suite, run it against 2.0.0 before
reading this page — the breaks above are designed to be loud, so most of them
will surface as an exception rather than as a changed number.

The ones that will *not* raise are in `Values that changed without a signature
change`_. Those need a baseline comparison.
