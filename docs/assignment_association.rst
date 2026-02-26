Assignment & Data Association
=============================

*Comprehensive guide to solving the data association problem: matching measurements to tracks in multi-target tracking scenarios.*

Data association determines which measurements belong to which tracks—a critical problem in multi-target tracking. Poor association decisions cascade through the entire system.

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
- $N_t$ existing tracks with predicted positions
- $N_m$ new measurements from sensors

Find:
- Associations: which measurements go to which tracks
- Track initiations: which measurements are new tracks
- Track deletions: which tracks should be terminated

**Challenge: Combinatorial Explosion**

Naive approach checks all possible associations: $(N_t + 1)^{N_m}$ possibilities.

Example: 100 tracks, 50 measurements → $101^{50}$ ≈ $10^{100}$ combinations!

**Why Association Matters:**

- Wrong associations prevent filters from converging
- Cause track divergence and coasted tracks
- False initiations create ghost tracks
- Correct association enables robust tracking


Association Algorithms
----------------------

**1. Nearest Neighbor (NN)**

Simplest approach: match each measurement to closest track.

Advantages:
- O(N_m × N_t) complexity
- Fast for real-time systems
- Works well with dense tracks

Disadvantages:
- Greedy (not globally optimal)
- Fails with close tracks
- No conflict resolution

.. code-block:: python

    import numpy as np
    from scipy.spatial.distance import cdist
    
    class NearestNeighbor:
        """Simple NN data association."""
        
        def associate(self, track_positions, measurements, gate_threshold=50.0):
            """
            Args:
                track_positions: N_t × 3 array
                measurements: N_m × 3 array
                gate_threshold: max distance for valid association (m)
            
            Returns:
                associations: list of (measurement_idx, track_idx) tuples
                unassociated_meas: list of measurement indices
                unassociated_tracks: list of track indices
            """
            N_t, N_m = len(track_positions), len(measurements)
            
            # Compute distance matrix: N_m × N_t
            distances = cdist(measurements, track_positions, metric='euclidean')
            
            associations = []
            used_tracks = set()
            
            # For each measurement, find nearest track
            for m_idx in range(N_m):
                valid_tracks = [t_idx for t_idx in range(N_t) 
                               if distances[m_idx, t_idx] < gate_threshold 
                               and t_idx not in used_tracks]
                
                if valid_tracks:
                    # Choose closest
                    closest_t = min(valid_tracks, 
                                   key=lambda t: distances[m_idx, t])
                    associations.append((m_idx, closest_t))
                    used_tracks.add(closest_t)
            
            unassociated_meas = [m for m in range(N_m) 
                                if m not in [a[0] for a in associations]]
            unassociated_tracks = [t for t in range(N_t) 
                                  if t not in used_tracks]
            
            return associations, unassociated_meas, unassociated_tracks


**2. Global Nearest Neighbor (GNN)**

Optimal assignment using cost matrix and Hungarian algorithm.

Advantages:
- Globally optimal solution
- O(N³) with Hungarian algorithm
- Handles ambiguous cases

Disadvantages:
- Slower than NN (still real-time for ~100 objects)
- Requires complete cost matrix
- Doesn't use historical ambiguity information

.. code-block:: python

    from scipy.optimize import linear_sum_assignment
    
    class GlobalNearestNeighbor:
        """Optimal assignment via Hungarian algorithm."""
        
        def associate(self, track_positions, measurements, 
                     covariances, gate_threshold=50.0):
            """
            Args:
                track_positions: N_t × 3 array
                measurements: N_m × 3 array
                covariances: N_t × 3 × 3 array (position uncertainty)
                gate_threshold: max Mahalanobis distance
            
            Returns:
                associations, unassociated_meas, unassociated_tracks
            """
            N_t, N_m = len(track_positions), len(measurements)
            
            # Build cost matrix: N_m × (N_t + 1)
            # Column N_t is cost of new track (fixed penalty)
            cost_matrix = np.full((N_m, N_t + 1), 1e10)
            
            # Fill association costs
            for m_idx, meas in enumerate(measurements):
                for t_idx, track_pos in enumerate(track_positions):
                    # Mahalanobis distance
                    diff = meas - track_pos
                    try:
                        cov_inv = np.linalg.inv(covariances[t_idx])
                        mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                    except np.linalg.LinAlgError:
                        mahal_dist = float('inf')
                    
                    # Gate check
                    if mahal_dist < gate_threshold:
                        cost_matrix[m_idx, t_idx] = mahal_dist
                
                # Cost of new track (no association)
                cost_matrix[m_idx, N_t] = 5.0  # new track penalty
            
            # Solve assignment problem
            meas_indices, assignment = linear_sum_assignment(cost_matrix)
            
            # Extract associations
            associations = []
            unassociated_meas = []
            
            for m_idx in meas_indices:
                t_idx = assignment[m_idx]
                if t_idx < N_t and cost_matrix[m_idx, t_idx] < 1e9:
                    associations.append((m_idx, t_idx))
                else:
                    unassociated_meas.append(m_idx)
            
            unassociated_tracks = [t for t in range(N_t)
                                  if t not in assignment[:N_t]]
            
            return associations, unassociated_meas, unassociated_tracks


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

.. code-block:: python

    class JPDA:
        """Joint Probabilistic Data Association."""
        
        def compute_association_probabilities(self, 
                                             likelihoods,  # N_m × N_t matrix
                                             P_d=0.9,      # detection probability
                                             P_f=1e-4):    # false alarm density
            """
            Compute probability each measurement came from each track.
            
            Args:
                likelihoods: N_m × N_t matrix of likelihood values
                P_d: detection probability (typically 0.8-0.95)
                P_f: false alarm density (returns/volume)
            """
            N_m, N_t = likelihoods.shape
            
            # Initialize association probabilities
            betas = np.zeros((N_m, N_t + 1))  # Column N_t is false alarm
            
            # Normalization constant
            c = P_f
            
            # For each gate hypothesis (simplified - does not enumerate all)
            for m_idx in range(N_m):
                for t_idx in range(N_t):
                    # Probability measurement m came from track t
                    beta_mt = P_d * likelihoods[m_idx, t_idx]
                    betas[m_idx, t_idx] = beta_mt
                    c += beta_mt
                
                # Probability measurement is false alarm
                betas[m_idx, N_t] = P_f
            
            # Normalize probabilities
            betas = betas / c
            
            # Association probability for each track
            # P(track t has measurement) = sum_m P(m associated with t)
            track_probs = np.sum(betas[:, :N_t], axis=0)
            
            return betas, track_probs


**4. Multiple Hypothesis Tracking (MHT)**

Maintains multiple competing track hypotheses, pruning low-probability ones.

Advantages:
- Handles very ambiguous scenarios
- Can recover from wrong associations
- Best performance in complex clutter

Disadvantages:
- Exponential complexity (must prune aggressively)
- High implementation complexity
- Computationally expensive

Example concept (full MHT is much more complex):

.. code-block:: python

    class SimpleMultipleHypothesis:
        """Simplified MHT structure."""
        
        def __init__(self, max_hypotheses=10):
            self.hypotheses = []  # List of hypothesis trees
            self.max_hypotheses = max_hypotheses
        
        def create_hypothesis(self, track_ids, associations, probability):
            """Create new hypothesis for current association."""
            hypothesis = {
                'track_ids': track_ids,
                'associations': associations,
                'probability': probability,
                'likelihood': 1.0
            }
            return hypothesis
        
        def prune_hypotheses(self):
            """Remove low-probability hypotheses."""
            # Keep only top N by probability
            self.hypotheses = sorted(self.hypotheses,
                                    key=lambda h: h['probability'],
                                    reverse=True)[:self.max_hypotheses]


Cost Matrix Construction
------------------------

**Mahalanobis Distance-Based Costs:**

.. code-block:: python

    def build_cost_matrix(tracks, measurements, max_gate=10.0, new_target_cost=15.0):
        """
        Build cost matrix for assignment problem.
        
        Args:
            tracks: List of Track objects with position and covariance
            measurements: List of measurement vectors
            max_gate: maximum Mahalanobis distance for gating
            new_target_cost: cost of initiating new track
        
        Returns:
            cost_matrix: N_m × (N_t + 1) cost matrix
        """
        N_m = len(measurements)
        N_t = len(tracks)
        
        cost_matrix = np.full((N_m, N_t + 1), 1e10)
        
        for m_idx, meas in enumerate(measurements):
            for t_idx, track in enumerate(tracks):
                # Compute Mahalanobis distance
                predicted_pos = track.position
                residual = meas - predicted_pos
                
                # Get covariance (innovation covariance in practice)
                if hasattr(track, 'innovation_cov'):
                    cov = track.innovation_cov
                else:
                    cov = np.eye(3)  # fallback
                
                try:
                    mahal_dist = np.sqrt(residual @ np.linalg.inv(cov) @ residual)
                except np.linalg.LinAlgError:
                    mahal_dist = float('inf')
                
                # Gate check
                if mahal_dist <= max_gate:
                    cost_matrix[m_idx, t_idx] = mahal_dist
            
            # New target column
            cost_matrix[m_idx, N_t] = new_target_cost
        
        return cost_matrix


Gating & Validation
-------------------

**Gating Regions:**

Gating reduces computational load by excluding unlikely associations before assignment.

.. code-block:: python

    class Gate:
        """Validation gate for measurement-track association."""
        
        @staticmethod
        def mahalanobis_gate(residual, innovation_cov, threshold=9.0):
            """
            Check if residual is within Mahalanobis distance gate.
            
            Args:
                residual: measurement - prediction
                innovation_cov: covariance of (measurement - prediction)
                threshold: gate size (typically 9.0 for 3-sigma ~ 99.7%)
            
            Returns:
                is_valid: boolean
                mahal_dist: Mahalanobis distance
            """
            try:
                mahal_dist_sq = residual @ np.linalg.inv(innovation_cov) @ residual
                is_valid = mahal_dist_sq <= threshold
                mahal_dist = np.sqrt(mahal_dist_sq)
            except np.linalg.LinAlgError:
                is_valid = False
                mahal_dist = float('inf')
            
            return is_valid, mahal_dist
        
        @staticmethod
        def elliptical_gate(measurement, predicted_pos, covariance, threshold=9.0):
            """Shorthand for Mahalanobis gating."""
            residual = measurement - predicted_pos
            return Gate.mahalanobis_gate(residual, covariance, threshold)
        
        @staticmethod
        def euclidean_gate(measurement, predicted_pos, max_distance=50.0):
            """Simple Euclidean distance gate."""
            distance = np.linalg.norm(measurement - predicted_pos)
            return distance <= max_distance, distance


**Gating Statistics:**

.. code-block:: python

    def gate_analysis(residuals, innovation_covs, gate_threshold=9.0):
        """Analyze gating performance."""
        mahal_dists = []
        
        for res, cov in zip(residuals, innovation_covs):
            try:
                md = np.sqrt(res @ np.linalg.inv(cov) @ res)
                mahal_dists.append(md)
            except:
                continue
        
        mahal_dists = np.array(mahal_dists)
        
        # Percentage of residuals within gate (should be ~99.7%)
        within_gate = np.sum(mahal_dists <= gate_threshold) / len(mahal_dists)
        
        print(f"Gate stats (threshold={gate_threshold}):")
        print(f"  Within gate: {within_gate*100:.1f}%")
        print(f"  Mean Mahal dist: {np.mean(mahal_dists):.2f}")
        print(f"  Std: {np.std(mahal_dists):.2f}")


Performance Metrics
-------------------

**Association Quality Metrics:**

.. code-block:: python

    class AssociationMetrics:
        """Evaluate data association quality."""
        
        @staticmethod
        def compute_assignment_cost(associations, cost_matrix):
            """Total cost of assignments."""
            total_cost = 0
            for m_idx, t_idx in associations:
                total_cost += cost_matrix[m_idx, t_idx]
            return total_cost
        
        @staticmethod
        def track_purity(associations, ground_truth, measurement_to_track_gt):
            """
            Fraction of associations matching ground truth.
            
            Args:
                associations: list of (m_idx, t_idx) pairs
                ground_truth: ground truth mapping
            
            Returns:
                purity: fraction of correct associations
            """
            correct = 0
            for m_idx, t_idx in associations:
                if (m_idx, t_idx) == ground_truth.get(m_idx, (None, None)):
                    correct += 1
            
            return correct / len(associations) if associations else 0
        
        @staticmethod
        def false_association_rate(associations, cost_matrix, gate_threshold=50.0):
            """Fraction of associations exceeding gate."""
            out_of_gate = 0
            for m_idx, t_idx in associations:
                if cost_matrix[m_idx, t_idx] > gate_threshold:
                    out_of_gate += 1
            
            return out_of_gate / len(associations) if associations else 0
        
        @staticmethod
        def track_fragmentation(tracks, ground_truth_labels):
            """
            How fragmented are true tracks (count hypothesis changes).
            """
            track_changes = {}
            for i, track in enumerate(tracks):
                label = ground_truth_labels[i]
                track_changes[label] = track_changes.get(label, 0) + 1
            
            # Fragmentation = (num_track_ids - num_true_targets) / num_true_targets
            num_true = len(set(ground_truth_labels))
            fragmentation = (len(track_changes) - num_true) / num_true
            return max(0, fragmentation)


Practical Implementation
------------------------

**Complete Association Pipeline:**

.. code-block:: python

    class DataAssociationPipeline:
        """End-to-end data association."""
        
        def __init__(self, algorithm='gnn', gate_threshold=9.0):
            self.algorithm = algorithm
            self.gate_threshold = gate_threshold
            self.gate = Gate()
        
        def associate_measurements(self, tracks, measurements):
            """
            Main entry point for data association.
            
            Returns:
                associations: list of (meas_idx, track_idx)
                unassociated_meas: measurements without tracks
                unassociated_tracks: tracks without measurements
            """
            if len(measurements) == 0:
                return [], [], list(range(len(tracks)))
            
            if len(tracks) == 0:
                return [], list(range(len(measurements))), []
            
            # Gate first: exclude measurements far from any track
            gated_measurements = []
            gated_meas_indices = []
            
            for m_idx, meas in enumerate(measurements):
                has_gate_pass = False
                for track in tracks:
                    is_valid, _ = self.gate.elliptical_gate(
                        meas, track.position, 
                        track.innovation_cov if hasattr(track, 'innovation_cov') else np.eye(3),
                        self.gate_threshold
                    )
                    if is_valid:
                        has_gate_pass = True
                        break
                
                if has_gate_pass:
                    gated_measurements.append(meas)
                    gated_meas_indices.append(m_idx)
            
            # No gated measurements - all are false alarms
            if len(gated_measurements) == 0:
                return [], list(range(len(measurements))), list(range(len(tracks)))
            
            # Build cost matrix
            cost_matrix = build_cost_matrix(tracks, gated_measurements, 
                                          max_gate=self.gate_threshold)
            
            # Associate using selected algorithm
            if self.algorithm == 'gnn':
                tracker = GlobalNearestNeighbor()
                associations, unmeas, untracks = tracker.associate(
                    np.array([t.position for t in tracks]),
                    np.array(gated_measurements),
                    np.array([getattr(t, 'covariance', np.eye(3)) for t in tracks]),
                    self.gate_threshold
                )
            else:  # default: 'nn'
                tracker = NearestNeighbor()
                associations, unmeas, untracks = tracker.associate(
                    np.array([t.position for t in tracks]),
                    np.array(gated_measurements),
                    self.gate_threshold
                )
            
            # Map back to original measurement indices
            associations = [(gated_meas_indices[m_idx], t_idx) 
                           for m_idx, t_idx in associations]
            unmeas = [gated_meas_indices[m_idx] for m_idx in unmeas]
            
            # Include non-gated measurements as unassociated
            non_gated = set(range(len(measurements))) - set(gated_meas_indices)
            unmeas = list(set(unmeas) | non_gated)
            
            return associations, sorted(unmeas), untracks


Common Issues & Solutions
-------------------------

**Problem: Closely-spaced tracks cause mis-associations**

Solution: Use tighter gating and higher cost for ambiguous assignments:

.. code-block:: python

    def adaptive_gating(track_spacing_min, gate_multiplier=1.0):
        """Adjust gate based on track density."""
        if track_spacing_min < 100:  # meters
            gate_threshold = 20.0  # tighter
            new_target_cost = 25.0  # higher
        else:
            gate_threshold = 9.0
            new_target_cost = 15.0
        
        return gate_threshold, new_target_cost


**Problem: Poor filter performance after wrong association**

Solution: Implement feedback tracking association confidence:

.. code-block:: python

    class ConfidenceTracker:
        """Track association confidence for each track."""
        
        def __init__(self, window_size=10):
            self.window_size = window_size
            self.track_innovations = {}  # Track innovations for CI computation
        
        def update_confidence(self, track_uid, innovation, threshold=50.0):
            """Update association confidence based on innovation."""
            if track_uid not in self.track_innovations:
                self.track_innovations[track_uid] = []
            
            self.track_innovations[track_uid].append(innovation)
            
            # Keep only recent innovations
            if len(self.track_innovations[track_uid]) > self.window_size:
                self.track_innovations[track_uid].pop(0)
            
            # Confidence degrades with large innovations
            recent_innovations = self.track_innovations[track_uid]
            large_innovation_ratio = sum(i > threshold for i in recent_innovations) / len(recent_innovations)
            
            confidence = max(0, 1 - large_innovation_ratio)
            return confidence


**Problem: Ghost tracks from false alarms**

Solution: Increase new track threshold and require confirmation:

.. code-block:: python

    class ConfirmedTrackManager:
        """Manage track confirmation logic."""
        
        TENTATIVE_THRESHOLD = 3  # updates before confirmed
        MAX_COASTING = 5  # updates without measurement
        
        def update_track_status(self, track):
            """Update track state based on measurement history."""
            if track.track_type == 'tentative':
                if track.age >= self.TENTATIVE_THRESHOLD:
                    track.track_type = 'confirmed'
            
            elif track.track_type == 'confirmed':
                # Check coasting count
                if hasattr(track, 'coasting_count'):
                    if track.coasting_count > self.MAX_COASTING:
                        track.track_type = 'deleted'
            
            elif track.track_type == 'coasted':
                if track.age > 20:  # Old coasted track
                    track.track_type = 'deleted'


Best Practices
--------------

1. **Use GNN over NN for accuracy**
   - GNN is globally optimal and only ~10× slower
   - Worth it for any system with moderate object count

2. **Tune gate threshold carefully**
   - Start with 9.0 (3-sigma for Gaussian)
   - Tighten if too many false associations
   - Loosen if missing valid measurements

3. **Include new-target penalty**
   - Prevents false initiations from isolated clutter
   - Typical range: 10-20× median Mahalanobis distance

4. **Monitor association quality**
   - Check for measurement-to-track oscillation
   - Track innovation sequences (should be white noise)
   - Count gate violations (should be < 1%)

5. **Adaptive gating**
   - Vary gate based on track density
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
   - Budget for 100-1000× more computation


Troubleshooting
---------------

**Problem: Association stays unstable**

Diagnosis: Check innovation sequences:

.. code-block:: python

    def diagnose_instability(filter_innovations):
        """Check if innovations are white noise."""
        # Autocorrelation should be near-zero at lag > 1
        from scipy import signal
        
        autocorr = signal.correlate(filter_innovations, filter_innovations, mode='same')
        autocorr = autocorr / np.max(autocorr)
        
        # Check lag-1 correlation
        lag1_corr = autocorr[len(autocorr)//2 + 1]
        
        if abs(lag1_corr) > 0.3:
            print("WARNING: Innovations show autocorrelation")
            print("Possible causes:")
            print("  - Filter timeconstant too long")
            print("  - Wrong association (try tighter gating)")
            print("  - Systematic bias in measurements")


See Also
--------

- :doc:`recipes` - Multi-target tracking with data association
- :doc:`data_structures` - TrackSet management for associations
- :doc:`troubleshooting` - Association debugging
- API: `tcl.assignment` module for Hungarian algorithm
