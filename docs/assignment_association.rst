Assignment & Data Association
=============================

*Comprehensive guide to solving the data association problem: matching measurements to tracks in multi-target tracking scenarios.*

Data association determines which measurements belong to which tracks—a critical problem in multi-target tracking. Poor association decisions cascade through the entire system. This guide shows how to solve it with the functions in :mod:`pytcl.assignment_algorithms`.

**Table of Contents:**

- Problem Overview
- Association Algorithms
- Cost Matrix Construction
- Gating & Validation
- Performance Metrics
- Practical Implementation
- Common Issues & Solutions
- Best Practices

Problem Overview
----------------

**The Data Association Problem:**

Given:

- :math:`N_t` existing tracks with predicted positions
- :math:`N_m` new measurements from sensors

Find:

- Associations: which measurements go to which tracks
- Track initiations: which measurements are new tracks
- Track deletions: which tracks should be terminated

**Challenge: Combinatorial Explosion**

Naive approach checks all possible associations: :math:`(N_t + 1)^{N_m}` possibilities.

Example: 100 tracks, 50 measurements → :math:`101^{50} \approx 10^{100}` combinations!

**Why Association Matters:**

- Wrong associations prevent filters from converging
- Cause track divergence and coasted tracks
- False initiations create ghost tracks
- Correct association enables robust tracking


Association Algorithms
----------------------

All algorithms below operate on a cost matrix of shape ``(n_tracks, n_measurements)``
whose entries are squared Mahalanobis distances (see *Cost Matrix Construction*).
Each returns an ``AssociationResult`` with ``track_to_measurement`` (per track, the
assigned measurement index or -1), ``measurement_to_track`` (the reverse mapping),
``costs``, and ``total_cost``.

**1. Nearest Neighbor (NN)**

Simplest approach: greedily match each track to its closest measurement.

Advantages:

- Fast, low complexity
- Works well when tracks are widely separated

Disadvantages:

- Greedy (not globally optimal)
- Fails with close tracks
- No conflict resolution

.. code-block:: python

    import numpy as np
    from pytcl.assignment_algorithms import (
        compute_association_cost,
        nearest_neighbor,
    )

    # Two tracks with state [x, vx]; three position-only measurements
    track_predictions = np.array([[0.0, 1.0],
                                  [5.0, -1.0]])
    track_covariances = np.array([np.eye(2), np.eye(2)])
    measurements = np.array([[0.1], [4.9], [10.0]])
    H = np.array([[1.0, 0.0]])  # measure position only

    # Squared Mahalanobis distance for every track/measurement pair
    cost = compute_association_cost(track_predictions, track_covariances,
                                    measurements, H)
    print(np.round(cost, 3))
    # [[1.000e-02 2.401e+01 1.000e+02]
    #  [2.401e+01 1.000e-02 2.500e+01]]

    result = nearest_neighbor(cost, gate_threshold=9.21)
    print(result.track_to_measurement)  # [0 1]
    print(result.measurement_to_track)  # [ 0  1 -1]  (meas 2 unassigned)


**2. Global Nearest Neighbor (GNN)**

Globally optimal assignment via the Hungarian algorithm
(:func:`~pytcl.assignment_algorithms.hungarian` under the hood).

Advantages:

- Globally optimal solution
- O(N^3) with Hungarian algorithm
- Handles ambiguous cases

Disadvantages:

- Slower than NN (still real-time for ~100 objects)
- Requires complete cost matrix
- Doesn't use historical ambiguity information

Greedy NN can lock in a bad early choice that GNN avoids:

.. code-block:: python

    from pytcl.assignment_algorithms import gnn_association

    ambiguous_cost = np.array([[1.0, 3.0],
                               [2.0, 100.0]])

    nn = nearest_neighbor(ambiguous_cost)
    print(nn.track_to_measurement, nn.total_cost)   # [0 1] 101.0

    gnn = gnn_association(ambiguous_cost)
    print(gnn.track_to_measurement, gnn.total_cost)  # [1 0] 5.0

``gated_gnn_association`` combines cost computation, chi-squared gating, and
GNN assignment in one call:

.. code-block:: python

    from pytcl.assignment_algorithms import gated_gnn_association

    result = gated_gnn_association(track_predictions, track_covariances,
                                   measurements, H, gate_probability=0.99)
    print(result.track_to_measurement)  # [0 1]
    print(result.measurement_to_track)  # [ 0  1 -1]


**3. Joint Probabilistic Data Association (JPDA)**

Treats ambiguous associations probabilistically using Bayesian updates.

Advantages:

- Handles ambiguous associations gracefully
- Produces weighted updates (less filter divergence)
- Better than deterministic assignment for clutter

Disadvantages:

- More complex implementation
- Slower than GNN
- Can lose track of targets in dense scenarios

:func:`~pytcl.assignment_algorithms.jpda` computes the association probability
matrix; :func:`~pytcl.assignment_algorithms.jpda_update` also performs the
probability-weighted state update. Each row of ``association_probs`` covers one
track and sums to 1 across all measurements plus a final missed-detection column.

.. code-block:: python

    from pytcl.assignment_algorithms import jpda, jpda_update

    states = [np.array([0.0, 1.0]), np.array([5.0, -1.0])]
    covariances = [0.5 * np.eye(2), 0.5 * np.eye(2)]
    meas = np.array([[-0.6], [0.8], [4.9]])  # two candidates near track 0
    R = np.array([[0.25]])

    result = jpda(states, covariances, meas, H, R,
                  detection_prob=0.9, clutter_density=1e-3)
    print(np.round(result.association_probs, 3))
    # [[0.546 0.453 0.    0.   ]   <- track 0 split between meas 0 and 1
    #  [0.    0.    1.    0.   ]]  <- track 1 firmly matched to meas 2
    print(result.association_probs.sum(axis=1))  # [1. 1.]

    update = jpda_update(states, covariances, meas, H, R,
                         detection_prob=0.9, clutter_density=1e-3)
    print([np.round(x, 3) for x in update.states])
    # [array([0.023, 1.   ]), array([ 4.933, -1.   ])]


**4. Multiple Hypothesis Tracking (MHT)**

MHT maintains multiple competing association hypotheses over time and prunes
low-probability ones. A full MHT implementation involves hypothesis trees,
track scoring, and N-scan pruning, which is beyond the scope of this guide.
Its core computational engine, however, is enumerating the k best assignments
of the current cost matrix—exactly what
:func:`~pytcl.assignment_algorithms.murty` and
:func:`~pytcl.assignment_algorithms.kbest_assign2d` provide
(:func:`~pytcl.assignment_algorithms.ranked_assignments` enumerates until a
cost threshold instead of a fixed k):

.. code-block:: python

    from pytcl.assignment_algorithms import murty

    cost_mht = np.array([[10.0, 5.0, 13.0],
                         [3.0, 15.0, 8.0],
                         [12.0, 7.0, 9.0]])

    kbest = murty(cost_mht, k=3)
    print(kbest.costs)  # [17. 23. 25.]
    for a in kbest.assignments:
        print(a.row_indices, "->", a.col_indices, a.cost)
    # [0 1 2] -> [1 0 2] 17.0
    # [0 1 2] -> [2 0 1] 23.0
    # [0 1 2] -> [1 2 0] 25.0

Each hypothesis branch spawns from one of these ranked assignments; hypothesis
probabilities follow from the assignment costs.

Advantages:

- Handles very ambiguous scenarios
- Can recover from wrong associations
- Best performance in complex clutter

Disadvantages:

- Exponential complexity (must prune aggressively)
- High implementation complexity
- Computationally expensive


Cost Matrix Construction
------------------------

**Mahalanobis Distance-Based Costs:**

The standard association cost is the squared Mahalanobis distance of the
innovation :math:`d^2 = (z - \hat{z})^T S^{-1} (z - \hat{z})`:

.. code-block:: python

    from pytcl.assignment_algorithms import mahalanobis_distance

    innovation = np.array([1.0, 0.5])
    S = np.array([[2.0, 0.0], [0.0, 1.0]])
    print(mahalanobis_distance(innovation, S))  # 0.75 (squared distance)

:func:`~pytcl.assignment_algorithms.compute_association_cost` builds the full
``(n_tracks, n_measurements)`` matrix from predicted states, covariances, and an
optional measurement matrix (see the NN example above). For probabilistic
methods, :func:`~pytcl.assignment_algorithms.compute_likelihood_matrix` returns
Gaussian likelihoods plus a boolean gating mask:

.. code-block:: python

    from pytcl.assignment_algorithms import compute_likelihood_matrix

    L, gated = compute_likelihood_matrix(states, covariances, meas, H, R,
                                         gate_threshold=9.21)
    print(np.round(L, 4))
    # [[0.3624 0.3007 0.    ]
    #  [0.     0.     0.4576]]
    print(gated)
    # [[ True  True False]
    #  [False False  True]]


Gating & Validation
-------------------

**Gating Regions:**

Gating excludes unlikely pairs before assignment, reducing computational load
and false associations. For Gaussian innovations the squared Mahalanobis
distance is chi-squared distributed with the measurement dimension as degrees
of freedom, so the gate threshold should be a chi-squared quantile—**it depends
on the measurement dimension**, not on a universal "3-sigma" rule. Use
:func:`~pytcl.assignment_algorithms.chi2_gate_threshold`:

.. code-block:: python

    from scipy.stats import chi2
    from pytcl.assignment_algorithms import chi2_gate_threshold

    for dof in (1, 2, 3):
        print(dof, round(chi2_gate_threshold(0.99, dof), 2))
    # 1 6.63
    # 2 9.21
    # 3 11.34

    # A fixed threshold of 9.0 is NOT "3-sigma ~ 99.7%": acceptance
    # probability depends on the measurement dimension.
    print(round(chi2.cdf(9.0, df=2), 4))  # 0.9889
    print(round(chi2.cdf(9.0, df=3), 4))  # 0.9707

**Gate tests:**

.. code-block:: python

    from pytcl.assignment_algorithms import (
        compute_gate_volume,
        ellipsoidal_gate,
        rectangular_gate,
    )

    gate = chi2_gate_threshold(0.99, 2)
    innovation = np.array([1.0, 0.5])
    S = np.array([[2.0, 0.0], [0.0, 1.0]])

    print(ellipsoidal_gate(innovation, S, gate))        # True
    print(rectangular_gate(innovation, S, num_sigmas=3.0))  # True

    # Gate volume (needed for clutter density in JPDA)
    print(round(compute_gate_volume(S, gate), 2))  # 40.92

**Gating many measurements against one track:**

.. code-block:: python

    from pytcl.assignment_algorithms import gate_measurements

    z_pred = np.array([0.0, 0.0])
    S = np.eye(2)
    candidates = np.array([[0.5, 0.5], [5.0, 5.0], [1.0, -1.0]])

    valid_idx, distances = gate_measurements(z_pred, S, candidates, gate)
    print(valid_idx)   # [0 2]
    print(distances)   # [0.5 2. ]


Performance Metrics
-------------------

**Association Quality Metrics:**

:mod:`pytcl.performance_evaluation` provides track-level association metrics.
Both take per-observation label arrays: the ground-truth target label and the
estimated track label of each observation.

.. code-block:: python

    from pytcl.performance_evaluation import track_fragmentation, track_purity

    # 8 observations from 2 true targets, assigned to 3 estimated tracks
    true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    est_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])

    # Purity: fraction of observations consistent with one target per track
    print(track_purity(true_labels, est_labels))  # 0.875

    # Fragmentation: how often a true target switches estimated track
    print(track_fragmentation(true_labels, est_labels))  # 2

Total assignment cost is available directly from any ``AssociationResult`` via
``result.total_cost``, and per-pair costs via ``result.costs``.


Practical Implementation
------------------------

**Complete Association Pipeline:**

One predict-associate-update cycle for a two-track scenario with clutter,
using the Kalman filter functions from :mod:`pytcl.dynamic_estimation.kalman`:

.. code-block:: python

    from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
    from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

    dt = 1.0
    F = f_constant_velocity(dt, num_dims=1)          # state [x, vx]
    Q = q_constant_velocity(dt, sigma_a=0.2, num_dims=1)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.25]])

    tracks_x = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
    tracks_P = [np.eye(2), np.eye(2)]
    measurements = np.array([[1.2], [8.7], [30.0]])  # last one is clutter

    # 1. Predict every track
    preds = [kf_predict(x, P, F, Q) for x, P in zip(tracks_x, tracks_P)]

    # 2. Gate + associate
    result = gated_gnn_association(
        np.array([p.x for p in preds]),
        np.array([p.P for p in preds]),
        measurements, H, gate_probability=0.99,
    )
    print(result.track_to_measurement)  # [0 1]
    print(result.measurement_to_track)  # [ 0  1 -1]

    # 3. Update assigned tracks; coast the rest
    for i, m_idx in enumerate(result.track_to_measurement):
        if m_idx >= 0:
            upd = kf_update(preds[i].x, preds[i].P,
                            measurements[m_idx], H, R)
            tracks_x[i], tracks_P[i] = upd.x, upd.P
        else:
            tracks_x[i], tracks_P[i] = preds[i].x, preds[i].P

    # 4. Unassociated measurements seed new (tentative) tracks
    new_track_seeds = [j for j, t in enumerate(result.measurement_to_track)
                       if t < 0]
    print([np.round(x, 3) for x in tracks_x])
    # [array([1.178, 1.09 ]), array([ 8.733, -1.135])]
    print(new_track_seeds)  # [2]


Common Issues & Solutions
-------------------------

**Problem: Closely-spaced tracks cause mis-associations**

Solution: tighten the gate probability and penalize non-assignment less, so
ambiguous measurements are dropped rather than forced onto the wrong track:

.. code-block:: python

    def association_params(track_spacing_min):
        """Pick gating parameters based on track density."""
        if track_spacing_min < 100.0:  # meters: crowded scene
            gate_probability = 0.95   # tighter gate
            cost_of_non_assignment = chi2_gate_threshold(0.95, 2)
        else:
            gate_probability = 0.99
            cost_of_non_assignment = chi2_gate_threshold(0.99, 2)
        return gate_probability, cost_of_non_assignment

    print(tuple(round(v, 2) for v in association_params(50.0)))   # (0.95, 5.99)
    print(tuple(round(v, 2) for v in association_params(500.0)))  # (0.99, 9.21)

For genuinely ambiguous regions, switch from GNN to
:func:`~pytcl.assignment_algorithms.jpda_update`, which spreads the update
across all gated measurements instead of committing to one.

**Problem: Poor filter performance after wrong association**

Solution: monitor the innovation sequence of each track. A healthy track has
NIS values that follow a chi-squared distribution with the measurement
dimension as degrees of freedom; a run of large values indicates a wrong
association or model mismatch. See ``nis_sequence`` and ``consistency_test``
in :mod:`pytcl.performance_evaluation`, covered in :doc:`kalman_filter_tuning`.

**Problem: Ghost tracks from false alarms**

Solution: require M-of-N confirmation before treating a tentative track as
real (e.g. 3 associated measurements in 5 scans), and delete confirmed tracks
after several consecutive coasts. Raising ``cost_of_non_assignment`` also
makes it harder for isolated clutter to steal measurements from real tracks.


Best Practices
--------------

1. **Use GNN over NN for accuracy**

   - GNN is globally optimal and only modestly slower
   - Worth it for any system with moderate object count

2. **Tune gate threshold carefully**

   - Derive it from a chi-squared quantile with
     ``chi2_gate_threshold(probability, num_dimensions)`` — e.g. 9.21 for
     99% at 2 measurement dimensions, 11.34 at 3
   - Tighten (lower probability) if too many false associations
   - Loosen (higher probability) if valid measurements are rejected

3. **Include a cost of non-assignment**

   - Prevents forced associations and false initiations from isolated clutter
   - A chi-squared quantile near the gate threshold is a good starting point

4. **Monitor association quality**

   - Check for measurement-to-track oscillation
   - Track innovation sequences (should be white noise)
   - Count gate violations (should match ``1 - gate_probability``)

5. **Adaptive gating**

   - Vary gate probability based on track density
   - Reduce gate in crowded regions
   - Expand gate in sparse regions

6. **JPDA for ambiguous regions**

   - Use when standard assignment fails
   - Probabilistic updates reduce divergence
   - Good for sports tracking, air traffic

7. **MHT for very complex scenarios**

   - Multiple simultaneous ambiguities
   - High clutter density
   - Safety-critical applications
   - Budget for 100-1000x more computation; build on ``murty`` /
     ``kbest_assign2d`` for hypothesis generation


Troubleshooting
---------------

**Problem: Association stays unstable**

Diagnosis: check whether innovations are white noise. Autocorrelation at
nonzero lags points at wrong associations (try tighter gating), a filter time
constant that is too long, or systematic measurement bias:

.. code-block:: python

    rng = np.random.default_rng(0)
    innovations = rng.normal(0.0, 1.0, size=200)  # replace with real data

    centered = innovations - np.mean(innovations)
    lag1 = (np.dot(centered[:-1], centered[1:])
            / np.dot(centered, centered))
    print(round(lag1, 3))  # 0.039  (|lag1| > 0.3 indicates trouble)


See Also
~~~~~~~~

- :doc:`recipes` - Multi-target tracking with data association
- :doc:`data_structures` - TrackSet management for associations
- :doc:`troubleshooting` - Association debugging
- API: :mod:`pytcl.assignment_algorithms` (assignment, gating, JPDA, k-best)
  and :mod:`pytcl.performance_evaluation` (association metrics)
