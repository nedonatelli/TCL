Typed Configs and Sessions
============================

Two related pieces of the tracker/filter API live here:
:mod:`pytcl.trackers.configs` and :mod:`pytcl.dynamic_estimation.configs`
(typed ``msgspec.Struct`` configuration objects), and :mod:`pytcl.io.session`
(full state snapshot and resume, built on top of those same config types).
Together they answer "what parameters built this object" and "how do I stop
a running tracker/filter and pick it back up later, exactly where it left
off".

Six classes have session support: `SingleTargetTracker`, `MultiTargetTracker`,
`MHTTracker`, `IMMEstimator`, `GaussianSumFilter`, and `RBPFFilter`. No other
class in the library can be saved this way -- see `What Session Support Does
Not Cover`_ at the bottom of this page.

The Session Envelope
------------------------

`save_session` wraps whatever it snapshots in a `SessionEnvelope`: a
`schema_version` (the wire format described below), the `pytcl.__version__`
that produced it, and the tagged snapshot itself. `load_session` decodes
that envelope strictly -- malformed or truncated bytes, or bytes carrying a
`schema_version` newer than this install supports, raise
:class:`~pytcl.core.exceptions.FormatError` rather than returning a
partially-reconstructed object. `SESSION_SCHEMA_VERSION` (currently ``1``)
is exported from :mod:`pytcl.io` for anyone who wants to check it before
decoding.

Typed Configs
-----------------

Five ``msgspec.Struct`` types carry a class's constructor arguments as a
single, serializable object: `SingleTargetConfig`, `MultiTargetConfig`
(:mod:`pytcl.trackers.configs`), and `IMMConfig`, `GaussianSumConfig`,
`RBPFConfig` (:mod:`pytcl.dynamic_estimation.configs`). Each is accepted by
its matching constructor via a keyword-only ``config=`` argument, mutually
exclusive with the individual keyword arguments it replaces -- passing both
raises :class:`~pytcl.core.exceptions.ConfigurationError`.

`SingleTargetConfig`, `MultiTargetConfig` and `IMMConfig` also carry a
``from_arrays`` classmethod that accepts plain arrays/lists (rather than
nested Python lists) and normalizes them, matching how the tracker itself
accepts ``F``/``H``/``Q``/``R``/``transition_matrix``.

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Struct
     - Constructor
     - Carries matrices?
   * - ``SingleTargetConfig``
     - ``SingleTargetTracker``
     - ``F``/``Q`` are ``None`` when built with callable dynamics
   * - ``MultiTargetConfig``
     - ``MultiTargetTracker``
     - Same convention as ``SingleTargetConfig``
   * - ``IMMConfig``
     - ``IMMEstimator``
     - Only ``transition_matrix``; per-mode ``F``/``Q``/``H``/``R`` are set
       separately via ``set_mode_model``/``set_measurement_model``
   * - ``GaussianSumConfig``
     - ``GaussianSumFilter``
     - None -- component-count/merge/prune parameters only
   * - ``RBPFConfig``
     - ``RBPFFilter``
     - None -- particle-count/resample/merge parameters only

.. code-block:: python

   import numpy as np

   from pytcl.core.exceptions import ConfigurationError
   from pytcl.dynamic_estimation.configs import GaussianSumConfig, IMMConfig, RBPFConfig
   from pytcl.trackers import SingleTargetTracker
   from pytcl.trackers.configs import MultiTargetConfig, SingleTargetConfig

   F4 = np.eye(4)
   H24 = np.eye(2, 4)
   Q4 = 0.01 * np.eye(4)
   R2 = 0.1 * np.eye(2)

   single_cfg = SingleTargetConfig.from_arrays(
       state_dim=4, meas_dim=2, H=H24, R=R2, F=F4, Q=Q4, gate_threshold=9.21,
   )
   print(single_cfg.state_dim, single_cfg.meas_dim)  # 4 2

   multi_cfg = MultiTargetConfig.from_arrays(
       state_dim=4, meas_dim=2, H=H24, R=R2, F=F4, Q=Q4, confirm_hits=2,
   )
   print(multi_cfg.confirm_hits, multi_cfg.gate_probability)  # 2 0.99

   imm_cfg = IMMConfig.from_arrays(
       n_modes=2, state_dim=4, transition_matrix=[[0.95, 0.05], [0.05, 0.95]],
   )
   gsf_cfg = GaussianSumConfig(max_components=8, prune_threshold=1e-4)
   rbpf_cfg = RBPFConfig(max_particles=200)
   print(imm_cfg.n_modes, gsf_cfg.max_components, rbpf_cfg.max_particles)  # 2 8 200

   # config= and individual arguments are mutually exclusive.
   try:
       SingleTargetTracker(state_dim=4, meas_dim=2, config=single_cfg)
   except ConfigurationError as exc:
       print("rejected:", "not both" in str(exc))  # True

   tracker = SingleTargetTracker(config=single_cfg)
   print(tracker.state_dim, tracker.meas_dim, tracker.gate_threshold)  # 4 2 9.21

`MHTConfig` (:mod:`pytcl.trackers.mht`) is a sixth, related Struct, but it
does not follow the pattern above: it carries only MHT's own algorithm
parameters (``n_scan``, ``max_hypotheses``, ``detection_prob``, and so on),
never ``F``/``H``/``Q``/``R``, so `MHTTracker` takes it as a plain optional
``config`` argument with no individual-argument equivalents to conflict
with. As of this release `MHTConfig` is a frozen ``msgspec.Struct`` rather
than a ``NamedTuple``: attribute access and keyword construction are
unchanged, but indexing, unpacking and ``_replace`` no longer work, and
assigning an attribute raises ``AttributeError`` rather than silently
succeeding.

.. code-block:: python

   from pytcl.trackers import MHTConfig

   mht_cfg = MHTConfig(n_scan=2, max_hypotheses=50)
   print(mht_cfg.n_scan, mht_cfg.max_hypotheses)  # 2 50

   try:
       mht_cfg.n_scan = 5
   except AttributeError:
       print("frozen: True")

Saving and Restoring a Tracker
-----------------------------------

`save_session` snapshots one of the six supported objects to ``bytes``;
`load_session` reconstructs it. `save_session_file`/`load_session_file` are
the same pair against a path instead of ``bytes``. A resumed object is not
just state-equal to the original -- calling ``predict``/``update`` (or
``process``) on it after resume produces results bit-identical to calling
the same sequence on the original, uninterrupted object, for
`SingleTargetTracker`, `MultiTargetTracker`, `MHTTracker` and
`IMMEstimator`. `GaussianSumFilter` and `RBPFFilter` carry the same
guarantee only when constructed with an instance ``rng=``; built with the
default global RNG, they still resume, but their random draws diverge from
an uninterrupted run's -- see `RNG Reproducibility`_ below.

.. code-block:: python

   import numpy as np

   from pytcl.io import load_session, save_session
   from pytcl.trackers import SingleTargetTracker

   F4 = np.eye(4)
   H24 = np.eye(2, 4)
   Q4 = 0.01 * np.eye(4)
   R2 = 0.1 * np.eye(2)

   tracker = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
   tracker.initialize(np.array([0.0, 1.0, 0.0, 0.5]), np.eye(4))
   tracker.predict(1.0)

   data = save_session(tracker)          # bytes, msgpack by default
   resumed = load_session(data)
   print(type(resumed).__name__, resumed.is_initialized)  # SingleTargetTracker True

   z = np.array([1.1, 0.6])
   tracker.predict(1.0)
   tracker.update(z)
   resumed.predict(1.0)
   resumed.update(z)
   print(np.array_equal(tracker.state.state, resumed.state.state))            # True
   print(np.array_equal(tracker.state.covariance, resumed.state.covariance))  # True

`MultiTargetTracker` sessions round-trip the full track table -- every
track's id, state, covariance, status, hit/miss counters, and the tracker's
own ``next_id`` counter (so a resumed tracker never reissues a track id that
already exists):

.. code-block:: python

   from pytcl.trackers import MultiTargetTracker

   mt_tracker = MultiTargetTracker(4, 2, F4, H24, Q4, R2, confirm_hits=1)
   mt_tracker.process([np.array([0.0, 0.0]), np.array([10.0, 10.0])], dt=1.0)

   mt_resumed = load_session(save_session(mt_tracker))
   print(len(mt_tracker.tracks), len(mt_resumed.tracks))  # 2 2

   z2 = [np.array([0.1, 0.2]), np.array([10.1, 10.2])]
   original_ids = [t.id for t in mt_tracker.process(z2, dt=1.0)]
   resumed_ids = [t.id for t in mt_resumed.process(z2, dt=1.0)]
   print(original_ids == resumed_ids)  # True

`MHTTracker` sessions carry the same construction recipe (``state_dim``,
``meas_dim``, ``H``, ``R``, plus the ``config``) as `SingleTargetSnapshot`/
`MultiTargetSnapshot`, along with the hypothesis tree's tracks, hypotheses
and id counters -- restoring it uses the same `load_session` call, and the
same rehydrate pattern described next.

Rehydrating Callable Dynamics
----------------------------------

`SingleTargetTracker`, `MultiTargetTracker` and `MHTTracker` all accept
``F``/``Q`` as either fixed matrices or callables ``F(dt) -> ndarray`` (a
time-varying model). A snapshot can serialize a fixed matrix, but not a
Python callable, so a tracker built with callable dynamics saves a config
where ``F``/``Q`` are ``None`` -- and `load_session` needs the callable back
from the caller to finish reconstructing it. This is the one place
`load_session` takes keyword arguments: pass ``F=``/``Q=`` for exactly the
matrices the snapshot's config lacks.

The rule is checked one matrix at a time and in both directions: omitting
``F=``/``Q=`` when the snapshot needs it raises
:class:`~pytcl.core.exceptions.ConfigurationError` rather than restoring a
tracker that cannot predict, and passing ``F=``/``Q=`` when the snapshot
*already* carries that matrix also raises -- silently overriding saved
dynamics would be worse than refusing.

.. code-block:: python

   from pytcl.core.exceptions import ConfigurationError

   callable_tracker = SingleTargetTracker(4, 2, lambda dt: F4, H24, Q4, R2)
   callable_tracker.initialize(np.zeros(4), np.eye(4))
   callable_data = save_session(callable_tracker)

   try:
       load_session(callable_data)  # no F= given
   except ConfigurationError as exc:
       print("needs F=:", "callable" in str(exc))  # True

   restored = load_session(callable_data, F=lambda dt: F4)
   print(restored.is_initialized)  # True

Self-Contained Snapshots: IMM, Gaussian Sum, RBPF
--------------------------------------------------------

`IMMEstimator`, `GaussianSumFilter` and `RBPFFilter` have no
callable-dynamics escape hatch at all: their models arrive per call
(``predict(f, F, Q)`` and friends), not at construction, so every snapshot
of these three is fully self-contained. `load_session` rejects *any*
keyword argument for these snapshot types, including ``F=``/``Q=`` -- there
is nothing for them to rehydrate.

.. code-block:: python

   from pytcl.dynamic_estimation import IMMEstimator

   imm = IMMEstimator(2, 2, [[0.9, 0.1], [0.1, 0.9]])
   imm.set_mode_model(0, np.eye(2), 0.01 * np.eye(2))
   imm.set_mode_model(1, np.eye(2), 0.01 * np.eye(2))
   imm.set_measurement_model(np.eye(2), 0.1 * np.eye(2))
   imm.initialize(np.zeros(2), np.eye(2))

   imm_data = save_session(imm)
   try:
       load_session(imm_data, F=np.eye(2))
   except ConfigurationError as exc:
       print("self-contained:", "no rehydration" in str(exc))  # True

RNG Reproducibility
------------------------

`RBPFFilter` and `GaussianSumFilter` both accept an optional
``rng: numpy.random.Generator`` at construction. When an instance ``rng``
is given, its `PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_
bit-generator state is captured in the session snapshot, so a resumed
filter continues drawing from *exactly* the same random stream as an
uninterrupted one -- every particle/component draw after resume matches
bit-for-bit. Session support for instance RNGs is PCG64-only, matching
``numpy.random.Generator``'s default: constructing a filter with a
different bit-generator (MT19937, Philox, SFC64, ...) and then calling
`save_session` raises :class:`~pytcl.core.exceptions.ConfigurationError`
at save time, naming the offending bit-generator class, rather than
producing a session that cannot be restored. Restoring a *saved* PCG64
state onto a mismatched generator is a separate, restore-time failure
that surfaces via numpy's own state-assignment validation.

When ``rng`` is omitted (the default), the filter falls back to the legacy
global ``numpy.random`` state. That state is **not** captured by a
session -- a resumed filter falls back to that same global state, which has
moved on by however much other code in the process consumed it since the
session was saved. Stated plainly: global-RNG filters resume, but their
random draws diverge from what an uninterrupted run would have produced.
Pass an instance ``rng`` whenever bit-reproducible resume matters.

.. code-block:: python

   from pytcl.dynamic_estimation import RBPFFilter

   def build_rbpf():
       f = RBPFFilter(max_particles=8, rng=np.random.Generator(np.random.PCG64(42)))
       f.initialize(np.zeros(2), np.zeros(2), np.eye(2), num_particles=8)
       return f

   uninterrupted = build_rbpf()
   resumed_rbpf = load_session(save_session(build_rbpf()))

   g_mat, f_mat = np.eye(2), np.eye(2)
   Qy = Qx = 0.01 * np.eye(2)
   for filt in (uninterrupted, resumed_rbpf):
       filt.predict(lambda y: g_mat @ y, Qy, lambda x, y: f_mat @ x, f_mat, Qx)

   print(all(
       np.array_equal(pa.y, pb.y) and np.array_equal(pa.x, pb.x)
       for pa, pb in zip(uninterrupted.get_particles(), resumed_rbpf.get_particles())
   ))  # True

`GaussianSumFilter` follows the identical contract (construct with
``rng=``, resume bit-reproducibly; omit it and fall back to the
non-reproducible global state).

Format Notes: msgpack vs JSON
-----------------------------------

`save_session`/`load_session` share the same two wire formats as
:mod:`pytcl.io.serialize` (see :doc:`results_io`), selected by ``fmt``:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Property
     - ``fmt="msgpack"`` (default)
     - ``fmt="json"``
   * - Encoding
     - Compact binary
     - Human-readable text
   * - NaN / Inf anywhere in the snapshot
     - Preserved exactly (bit-identical ``float64``)
     - Not representable -- ``save_session`` raises ``ValueError`` *before*
       writing anything
   * - Finite-value round-trip
     - Bit-exact
     - Bit-exact

The non-finite check walks the *entire* snapshot -- nested configs, per-track
lists, RNG state -- not just the top-level fields, so a stray ``NaN`` deep in
an ``MultiTargetTracker``'s track table is caught the same way a ``NaN`` in a
`SingleTargetTracker`'s state vector is.

.. code-block:: python

   nan_tracker = SingleTargetTracker(4, 2, F4, H24, Q4, R2)
   nan_tracker.initialize(np.array([1.0, np.nan, 0.0, 0.0]), np.eye(4))

   back = load_session(save_session(nan_tracker, fmt="msgpack"), fmt="msgpack")
   print(back.state.state.tobytes() == nan_tracker.state.state.tobytes())  # True

   try:
       save_session(nan_tracker, fmt="json")
   except ValueError as exc:
       print("json rejects non-finite:", "non-finite" in str(exc))  # True

What Session Support Does Not Cover
-----------------------------------------

Two things in the library look like candidates for `save_session` but are
not supported:

The CuPy-backed batch classes in :mod:`pytcl.gpu` (batch Kalman/EKF/UKF and
particle-filter state living on a GPU device array) have no snapshotter
registered. Their state is a device array, not the host ``ndarray`` values
every snapshot type here assumes, and batching changes what "one saved object"
even means (one snapshot per batch element, or one for the whole batch).
Neither question is answered by this module -- move the relevant tracks to
a CPU-side tracker first if you need to persist and resume them.

`ConstrainedEKF` (:mod:`pytcl.dynamic_estimation.kalman.constrained`) is not
one of the six supported classes either. Its constraints are
``ConstraintFunction`` callables added via ``add_constraint`` after
construction, the same category of unpicklable Python object that the
rehydrate pattern above works around for ``F``/``Q`` -- but `load_session`
has no rehydration hook for constraint callables, so a `ConstrainedEKF`
snapshot could not fully reconstruct the filter's behavior even if one were
added.
