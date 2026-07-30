Data Structures & Containers
============================

This guide covers the containers the library actually provides: ``Track``,
``TrackList``, ``MeasurementSet``, ``ClusterSet``, and the spatial indices.

.. note::

   Earlier revisions of this page documented a ``TrackSet`` class imported from
   ``tcl.tracking_containers``, with attributes such as ``track.uid``,
   ``track.position`` and ``track.gate_size``. No such class, module or
   attributes exist. The container that fills that role is ``TrackList`` in
   ``pytcl.containers``, described below. Every example on this page is
   executed by ``tests/test_docs_data_structures.py``.

Track
-----

``Track`` is an immutable snapshot of one target, produced by
``MultiTargetTracker``. It is a ``NamedTuple``, so it unpacks and compares like
a tuple and cannot be mutated in place.

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Field
     - Type
     - Meaning
   * - ``id``
     - ``int``
     - Identifier, stable for the life of the track
   * - ``state``
     - ``ndarray``
     - State vector, layout set by the model you gave the tracker
   * - ``covariance``
     - ``ndarray``
     - State covariance, ``(state_dim, state_dim)``
   * - ``status``
     - ``TrackStatus``
     - ``TENTATIVE``, ``CONFIRMED`` or ``DELETED``
   * - ``hits``
     - ``int``
     - Number of associated detections
   * - ``misses``
     - ``int``
     - Consecutive scans without an association
   * - ``time``
     - ``float``
     - Time of the most recent update

There is no ``position`` or ``velocity`` attribute: which elements of ``state``
are position depends on the transition matrix you supplied. With
``f_constant_velocity`` the layout is ``[x, vx, y, vy]``, so position is
``state[0]`` and ``state[2]`` — which is why ``TrackList.positions()`` takes the
indices as an argument.

.. code-block:: python

   import numpy as np
   from pytcl.trackers import MultiTargetTracker, TrackStatus

   F = np.array([[1.0, 1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0],
                 [0.0, 0.0, 0.0, 1.0]])
   H = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]])

   tracker = MultiTargetTracker(
       state_dim=4, meas_dim=2, F=F, H=H,
       Q=np.eye(4) * 0.01, R=np.eye(2) * 4.0, confirm_hits=2,
   )

   for k in range(6):
       tracker.process(
           [np.array([k * 2.0, k * 1.0]), np.array([50.0 + k, 60.0 - k * 0.5])],
           dt=1.0,
       )

   track = tracker.tracks[0]
   print(track.id, track.status is TrackStatus.CONFIRMED, track.hits)

TrackList
---------

``TrackList`` is an immutable collection of ``Track`` objects. Every operation
returns a new list rather than modifying in place, so a filtered view can be
passed around without any risk of aliasing.

.. code-block:: python

   from pytcl.containers import TrackList
   from pytcl.trackers import TrackStatus

   tracks = TrackList.from_tracker(tracker)

   tracks.track_ids                 # [0, 1]
   tracks.stats()                   # TrackListStats(n_tracks=2, n_confirmed=2, ...)

   tracks.confirmed                 # property, not a call
   tracks.tentative

   tracks.states()                  # (n_tracks, state_dim)
   tracks.covariances()             # (n_tracks, state_dim, state_dim)
   tracks.positions(indices=(0, 2)) # (n_tracks, 2), defaults to [x, y] of [x, vx, y, vy]

   tracks.get_by_id(0)
   tracks.filter_by_status(TrackStatus.CONFIRMED)
   tracks.filter_by_region([0.0, 0.0], radius=20.0)
   tracks.filter_by_time(min_time=2.0)
   tracks.filter_by_predicate(lambda t: t.hits >= 5)

``confirmed`` and ``tentative`` are properties; ``track_ids`` is too. The
filters and accessors are methods. ``add``, ``remove`` and ``merge`` also
return new lists.

MeasurementSet
--------------

``MeasurementSet`` holds detections with their times, covariances and sensor
of origin. ``Measurement`` is a ``NamedTuple`` with fields
``(value, time, covariance, sensor_id, id)``.

.. code-block:: python

   import numpy as np
   from pytcl.containers import MeasurementSet

   detections = MeasurementSet.from_arrays(
       values=np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
       times=np.array([0.0, 1.0, 2.0, 3.0]),
       covariances=np.stack([np.eye(2)] * 4),
   )

   detections.times          # property: array([0., 1., 2., 3.])
   detections.time_range     # property: (0.0, 3.0)
   detections.sensors        # property

   detections.values()                        # (n, meas_dim)
   detections.at_time(2.0)
   detections.in_time_window(1.0, 2.0)
   detections.in_region([1.0, 2.0], radius=3.0)
   detections.nearest_to([2.0, 4.0], k=2)     # MeasurementQuery

A ``MeasurementSet`` iterates over its ``Measurement`` objects and supports
``len()``. Because each measurement carries its own covariance, a set converted
from polar detections feeds straight into the tracker's per-detection
covariance argument:

.. code-block:: python

   scan = detections.at_time(2.0)
   tracker.process(
       [m.value for m in scan],
       dt=1.0,
       measurement_covariances=[m.covariance for m in scan],
   )

See :doc:`architecture` for why a single ``R`` is not sufficient for converted
measurements.

ClusterSet
----------

``ClusterSet`` groups tracks that belong together — formations, or targets too
close to resolve individually.

.. code-block:: python

   from pytcl.containers import ClusterSet

   # method is a name -- 'dbscan' or 'kmeans' -- not a function
   clusters = ClusterSet.from_tracks(tracks, method="dbscan", eps=30.0)
   clusters.cluster_ids
   clusters.all_stats()

Spatial Indices
---------------

Four index structures, all with the same query surface, for finding
neighbours without a linear scan:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Class
     - Constructor
     - Suited to
   * - ``KDTree``
     - ``KDTree(data, leaf_size=10)``
     - Low-dimensional Euclidean data
   * - ``BallTree``
     - ``BallTree(data)``
     - Higher dimensions, still a metric space
   * - ``RTree``
     - ``RTree(max_entries=10)``
     - Extended objects and bounding boxes
   * - ``VPTree``
     - ``VPTree(data, metric=None)``
     - Arbitrary metrics
   * - ``CoverTree``
     - ``CoverTree(data, metric=None, base=2.0)``
     - Arbitrary metrics, bounded intrinsic dimension

.. code-block:: python

   import numpy as np
   from pytcl.containers import KDTree

   points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [5.0, 5.0]])
   index = KDTree(points)

   index.query(np.array([1.1, 1.1]), k=2)      # k nearest
   index.query_radius(np.array([0.0, 0.0]), r=2.0)

``KDTree``, ``VPTree`` and ``CoverTree`` take their data at construction.
``RTree`` is built incrementally with ``insert_point`` / ``insert_points``, or
in one go with ``RTree.from_points``.

Persistence
-----------

Tracks are written and read with ``pytcl.io``. The round trip is exact —
states, covariances, timestamps and metadata all come back unchanged.

.. code-block:: python

   import numpy as np
   from pytcl.io import TrackHDF5Storage

   storage = TrackHDF5Storage("tracks.h5")
   storage.open("w")
   storage.store_track(
       "track_0",
       states=np.zeros((10, 4)),
       covariances=np.stack([np.eye(4)] * 10),
       timestamps=np.arange(10.0),
       metadata={"status": "confirmed"},
   )
   storage.close()

   storage = TrackHDF5Storage("tracks.h5")
   storage.open("r")
   storage.list_tracks()                    # ['track_0']
   record = storage.retrieve_track("track_0")
   record["states"], record["covariances"], record["metadata"]
   storage.close()

``get_track_trajectory`` returns just the state history, optionally windowed by
time.

See Also
--------

- :doc:`architecture` - how the containers fit into the tracking pipeline
- :doc:`api/containers` - full container API reference
- :doc:`api/trackers` - ``Track``, ``TrackStatus`` and the trackers
