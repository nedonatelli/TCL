Data Structures & Containers
============================

*Comprehensive guide to the primary data structures for storing, managing, and accessing tracked objects, measurements, and states.*

This guide covers the Track, TrackSet, and related container classes that form the foundation for track management in the library.

**Table of Contents:**

- Track Class Overview
- TrackSet Container
- Memory Management & Performance
- Serialization & I/O
- Common Patterns & Workflows
- Best Practices
- Troubleshooting

Track Class
-----------

The **Track** object represents a single target with estimated state, uncertainty, and metadata.

**Track Attributes & Properties:**

.. code-block:: python

    from tcl.tracking_containers import Track, TrackSet
    import numpy as np
    
    class TrackExample:
        """Understand Track structure and attributes."""
        
        def create_basic_track(self):
            """Create a Track with essential information."""
            track = Track(
                uid=1,  # Unique identifier
                state_type='position_velocity',  # State representation
                position=np.array([100.0, 200.0, 50.0]),  # 3D position
                velocity=np.array([5.0, -2.0, 0.1]),  # 3D velocity
                timestamp=0.0
            )
            return track
        
        def track_attributes(self, track):
            """Access common Track properties."""
            
            # Identity
            print(f"Track ID: {track.uid}")
            print(f"Track age: {track.age}")  # Number of updates
            print(f"Track timestamp: {track.timestamp}")
            
            # State
            print(f"Position: {track.position}")
            print(f"Velocity: {track.velocity}")
            
            # Uncertainty
            if hasattr(track, 'covariance'):
                print(f"Covariance shape: {track.covariance.shape}")
                position_uncertainty = np.sqrt(np.diag(track.covariance)[:3])
                print(f"Position std: {position_uncertainty}")
            
            # Metadata
            print(f"Gate size: {track.gate_size}")  # For gating in data association
            print(f"Track type: {track.track_type}")  # e.g., 'confirmed', 'tentative'

**Track Point Representation:**

A Track can be visualized as a sequence of timestamped points:

.. code-block:: python

    def track_point_history(track):
        """Extract track history as points."""
        if hasattr(track, 'history'):
            # Historical positions
            positions = np.array([pt.position for pt in track.history])
            print(f"Track traveled {np.linalg.norm(positions[-1] - positions[0])} m")
        
        if hasattr(track, 'states'):
            # Full state history (position, velocity, etc.)
            for i, state in enumerate(track.states):
                print(f"State {i}: position={state[:3]}, velocity={state[3:6]}")


TrackSet Container
------------------

**TrackSet** is the primary container managing multiple tracks with efficient lookup and modification.

**Basic TrackSet Operations:**

.. code-block:: python

    class TrackSetOperations:
        """Common TrackSet patterns."""
        
        def create_trackset(self):
            """Initialize an empty TrackSet."""
            ts = TrackSet()
            return ts
        
        def add_track(self, ts, track):
            """Add a track to the set."""
            ts.add(track)  # or ts[track.uid] = track
        
        def retrieve_track(self, ts, uid):
            """Get track by ID."""
            track = ts[uid]  # Direct access
            return track
        
        def iterate_tracks(self, ts):
            """Iterate over all tracks."""
            for uid, track in ts.items():
                print(f"Track {uid}: position={track.position}")
        
        def modify_track(self, ts, uid, new_position):
            """Update track state."""
            ts[uid].position = new_position
        
        def remove_track(self, ts, uid):
            """Delete a track."""
            del ts[uid]  # or ts.remove(uid)
        
        def get_all_positions(self, ts):
            """Extract all current positions."""
            positions = {uid: track.position for uid, track in ts.items()}
            return positions
        
        def trackset_size(self, ts):
            """Get number of tracked objects."""
            return len(ts)


**Filtering & Selection:**

.. code-block:: python

    def filter_tracks_by_criteria(ts):
        """Common filtering patterns."""
        
        # Active tracks only
        active_tracks = {uid: t for uid, t in ts.items() 
                        if t.track_type in ['confirmed', 'coasted']}
        
        # Recent tracks (within time window)
        current_time = ts.current_time if hasattr(ts, 'current_time') else 0
        recent_tracks = {uid: t for uid, t in ts.items()
                        if current_time - t.timestamp < 5.0}
        
        # High-confidence tracks
        confirmed_tracks = {uid: t for uid, t in ts.items()
                           if t.track_type == 'confirmed'}
        
        # Tracks in region of interest
        roi_min, roi_max = np.array([-100, -100, 0]), np.array([100, 100, 100])
        roi_tracks = {uid: t for uid, t in ts.items()
                     if np.all(t.position >= roi_min) and 
                        np.all(t.position <= roi_max)}
        
        return active_tracks, recent_tracks, confirmed_tracks, roi_tracks


**Spatial Queries:**

.. code-block:: python

    def spatial_operations(ts):
        """Query tracks by spatial properties."""
        
        # Find closest track to reference point
        ref_point = np.array([0, 0, 0])
        distances = {uid: np.linalg.norm(t.position - ref_point)
                    for uid, t in ts.items()}
        closest_uid = min(distances, key=distances.get)
        closest_track = ts[closest_uid]
        
        # Find all tracks within radius
        search_radius = 50.0
        nearby = {uid: t for uid, t in ts.items()
                 if np.linalg.norm(t.position - ref_point) < search_radius}
        
        # Compute centroid (center of mass)
        if len(ts) > 0:
            centroid = np.mean([t.position for t in ts.values()], axis=0)
            print(f"Track cluster centroid: {centroid}")
        
        # Track separation (minimum distance between any two tracks)
        trackuids = list(ts.keys())
        min_separation = float('inf')
        for i, uid1 in enumerate(trackuids):
            for uid2 in trackuids[i+1:]:
                sep = np.linalg.norm(ts[uid1].position - ts[uid2].position)
                min_separation = min(min_separation, sep)
        
        return closest_track, nearby, min_separation


Memory Management & Performance
-------------------------------

**Memory Consumption Analysis:**

A typical Track object includes:
- State vector: 6-9 floats (48-72 bytes, usually 8 bytes each)
- Covariance matrix: 6×6 to 9×9 (288-648 bytes)
- Metadata/history: 100-500 bytes depending on history length
- **Total per track: ~500-1200 bytes (without history)**

With history (100 points):
- Add ~500-2000 bytes per point
- **Total: ~50-200 KB per track**

**Memory Optimization Patterns:**

.. code-block:: python

    class MemoryOptimization:
        """Efficient TrackSet management."""
        
        def limit_trackset_size(self, ts, max_tracks=1000):
            """Remove oldest/lowest-confidence tracks to stay under limit."""
            while len(ts) > max_tracks:
                # Remove track with lowest confidence (example metric)
                if hasattr(ts[0], 'confidence'):
                    worst_uid = min(ts.keys(), 
                                   key=lambda uid: ts[uid].confidence)
                else:
                    worst_uid = min(ts.keys(), 
                                   key=lambda uid: ts[uid].age)
                del ts[worst_uid]
        
        def limit_track_history(self, track, max_points=50):
            """Keep only recent history points."""
            if hasattr(track, 'history') and len(track.history) > max_points:
                track.history = track.history[-max_points:]
        
        def memory_efficient_update(self, ts, max_memory_mb=500):
            """Monitor and limit total memory usage."""
            import sys
            
            total_bytes = 0
            for track in ts.values():
                # Estimate bytes
                total_bytes += sys.getsizeof(track)
                if hasattr(track, 'covariance'):
                    total_bytes += track.covariance.nbytes
                if hasattr(track, 'history'):
                    total_bytes += sum(sys.getsizeof(p) for p in track.history)
            
            total_mb = total_bytes / (1024 * 1024)
            print(f"TrackSet memory: {total_mb:.1f} MB")
            
            if total_mb > max_memory_mb:
                # Trim oldest tracks
                sorted_uids = sorted(ts.keys(), 
                                    key=lambda uid: ts[uid].timestamp)
                for uid in sorted_uids[:len(ts)//4]:  # Remove 25%
                    del ts[uid]
                    if total_mb <= max_memory_mb:
                        break


**Efficient Access Patterns:**

.. code-block:: python

    def performance_considerations(ts):
        """Best practices for speed."""
        
        # ✓ GOOD: Single pass iteration
        position_sum = np.zeros(3)
        count = 0
        for track in ts.values():
            position_sum += track.position
            count += 1
        mean_position = position_sum / count
        
        # ✗ AVOID: Multiple passes
        # mean_position = np.mean([ts[uid].position for uid in ts.keys()])
        
        # ✓ GOOD: Batch operations
        positions = np.array([t.position for t in ts.values()])
        velocities = np.array([t.velocity for t in ts.values()])
        
        # ✓ GOOD: Cache intermediate results
        uid_list = list(ts.keys())
        for uid in uid_list:
            if uid in ts:  # Check still exists (may be deleted)
                track = ts[uid]
                # Process track


Serialization & I/O
-------------------

**Saving and Loading Tracks:**

.. code-block:: python

    import pickle
    import json
    from pathlib import Path
    
    class TrackSerialization:
        """Persist tracks to disk."""
        
        def save_trackset_pickle(self, ts, filepath):
            """Save TrackSet as binary pickle (fast, preserves numpy arrays)."""
            with open(filepath, 'wb') as f:
                pickle.dump(ts, f)
        
        def load_trackset_pickle(self, filepath):
            """Load TrackSet from pickle."""
            with open(filepath, 'rb') as f:
                ts = pickle.load(f)
            return ts
        
        def export_tracks_to_csv(self, ts, filepath):
            """Export track data as CSV (human-readable)."""
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['UID', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 
                               'Timestamp', 'Age', 'Type'])
                
                for uid, track in ts.items():
                    pos = track.position
                    vel = track.velocity if hasattr(track, 'velocity') else [0]*3
                    writer.writerow([
                        uid,
                        f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                        f"{vel[0]:.2f}", f"{vel[1]:.2f}", f"{vel[2]:.2f}",
                        f"{track.timestamp:.3f}",
                        track.age,
                        track.track_type if hasattr(track, 'track_type') else 'unknown'
                    ])
        
        def export_tracks_to_json(self, ts, filepath):
            """Export tracks as JSON (portable format)."""
            data = []
            for uid, track in ts.items():
                track_data = {
                    'uid': int(uid),
                    'position': track.position.tolist() if hasattr(track.position, 'tolist') else list(track.position),
                    'velocity': track.velocity.tolist() if hasattr(track, 'velocity') and hasattr(track.velocity, 'tolist') else [],
                    'timestamp': float(track.timestamp),
                    'age': int(track.age) if hasattr(track, 'age') else 0,
                    'type': str(track.track_type) if hasattr(track, 'track_type') else 'unknown'
                }
                data.append(track_data)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        def import_tracks_from_json(self, filepath, trackset_class):
            """Load tracks from JSON."""
            ts = trackset_class()
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for track_data in data:
                track = trackset_class.Track(  # Adapt to actual class
                    uid=track_data['uid'],
                    position=np.array(track_data['position']),
                    velocity=np.array(track_data.get('velocity', [0]*3)),
                    timestamp=track_data['timestamp']
                )
                ts.add(track)
            
            return ts


Common Patterns & Workflows
----------------------------

**Pattern 1: Track Lifecycle Management**

.. code-block:: python

    class TrackLifecycle:
        """Manage track creation, confirmation, and deletion."""
        
        def __init__(self):
            self.ts = TrackSet()
            self.tentative_age_threshold = 3  # Confirm after 3 updates
            self.max_coasting_steps = 10  # Delete if not updated for 10 steps
        
        def add_tentative_track(self, position, velocity=None):
            """Create new unconfirmed track."""
            uid = len(self.ts) + 1  # Simple ID scheme
            track = Track(
                uid=uid,
                position=position,
                velocity=velocity or np.zeros(3),
                track_type='tentative'
            )
            self.ts.add(track)
            return uid
        
        def update_track(self, uid, measurement, dt):
            """Incorporate measurement into track."""
            track = self.ts[uid]
            
            # Simple update: average with measurement
            track.position = (track.position + measurement) / 2
            track.age += 1
            track.timestamp += dt
            
            # Promote tentative to confirmed
            if (track.track_type == 'tentative' and 
                track.age >= self.tentative_age_threshold):
                track.track_type = 'confirmed'
        
        def coast_track(self, uid, dt):
            """Propagate track without measurement (prediction only)."""
            track = self.ts[uid]
            track.position += track.velocity * dt
            track.coasting_steps = getattr(track, 'coasting_steps', 0) + 1
            track.track_type = 'coasted'
        
        def prune_tracks(self):
            """Remove old or unconfirmed tracks."""
            to_delete = []
            
            for uid, track in self.ts.items():
                # Delete tentative tracks that are too old
                if (track.track_type == 'tentative' and 
                    track.age > 20):
                    to_delete.append(uid)
                
                # Delete coasted tracks
                if (hasattr(track, 'coasting_steps') and 
                    track.coasting_steps > self.max_coasting_steps):
                    to_delete.append(uid)
            
            for uid in to_delete:
                del self.ts[uid]
            
            return to_delete


**Pattern 2: Multi-Target Tracking State Extraction**

.. code-block:: python

    def extract_tracking_state(ts, include_velocity=True, include_covariance=False):
        """Convert TrackSet to matrix form for algorithms."""
        
        if len(ts) == 0:
            return np.empty((0, 3)), np.empty((0, 3)) if include_velocity else np.empty((0, 3))
        
        # Position matrix: N × 3
        positions = np.vstack([track.position for track in ts.values()])
        
        if include_velocity:
            # Velocity matrix: N × 3
            velocities = np.vstack([track.velocity if hasattr(track, 'velocity') 
                                   else np.zeros(3) 
                                   for track in ts.values()])
            
            if include_covariance:
                # Covariance diagonal: N × 6
                covs = np.vstack([np.diag(track.covariance) if hasattr(track, 'covariance')
                                 else np.diag([1]*6)
                                 for track in ts.values()])
                return positions, velocities, covs
            
            return positions, velocities
        
        return positions


**Pattern 3: Track Association Updates**

.. code-block:: python

    def update_with_associations(ts, measurements, associations):
        """
        Update tracks based on measurement-to-track associations.
        
        Args:
            ts: TrackSet
            measurements: List of measurement vectors
            associations: Dict or list of (measurement_idx, track_uid) pairs
        """
        updated_uids = set()
        
        # Update associated tracks
        for meas_idx, track_uid in associations:
            if track_uid in ts:
                track = ts[track_uid]
                measurement = measurements[meas_idx]
                
                # Simple update (replace with Kalman filter in practice)
                alpha = 0.5  # Smoothing factor
                track.position = alpha * measurement + (1 - alpha) * track.position
                track.age += 1
                track.coasting_steps = 0  # Reset coasting counter
                
                updated_uids.add(track_uid)
        
        # Coast unassociated tracks
        for uid in ts.keys():
            if uid not in updated_uids:
                track = ts[uid]
                track.position += track.velocity * 0.1  # dt=0.1
                track.coasting_steps = getattr(track, 'coasting_steps', 0) + 1
        
        return updated_uids


Best Practices
--------------

1. **Use Numeric UIDs Consistently**
   - Assign sequential UIDs for efficient array operations
   - Never reuse UIDs within a session (or track deletion time)

2. **Maintain State Invariants**
   - Ensure position and velocity are always synchronized
   - Keep covariance consistent with state updates

3. **Regular Cleanup**
   - Delete coasted tracks after maximum threshold
   - Remove unconfirmed tracks that never mature

4. **Memory Awareness**
   - Limit track history for long-running systems
   - Monitor total memory consumption periodically

5. **Thread Safety** (if applicable)
   - Use locks when multiple threads access same TrackSet
   - Consider separate TrackSets per thread

6. **Backup & Snapshots**
   - Periodically save TrackSet state
   - Enable debugging and post-processing

7. **Metadata Tracking**
   - Include sensor association information
   - Track confidence/quality metrics
   - Store creation and update timestamps

Troubleshooting
---------------

**Problem: TrackSet grows unbounded**

Solution: Implement track pruning:

.. code-block:: python

    def prune_strategy(ts, max_tracks=500, current_time=0):
        """Aggressive pruning strategy."""
        if len(ts) > max_tracks:
            # Score tracks: higher = keep longer
            scores = {}
            for uid, t in ts.items():
                age_score = t.age  # Older is better
                recency_score = current_time - t.timestamp  # Newer is better
                type_score = {'confirmed': 3, 'coasted': 1, 'tentative': 0}.get(t.track_type, 0)
                
                scores[uid] = age_score + 10*type_score - 0.1*recency_score
            
            # Delete lowest-scoring tracks
            worst_uids = sorted(scores.keys(), key=lambda u: scores[u])[:len(ts) - max_tracks]
            for uid in worst_uids:
                del ts[uid]


**Problem: Slow track lookup**

Solution: Maintain spatial index:

.. code-block:: python

    from scipy.spatial import cKDTree
    
    def build_spatial_index(ts):
        """Create KD-tree for spatial queries."""
        if len(ts) == 0:
            return None
        
        positions = np.array([t.position for t in ts.values()])
        uids = np.array(list(ts.keys()))
        
        tree = cKDTree(positions)
        return tree, uids, positions


**Problem: Track state inconsistency**

Solution: Add validation:

.. code-block:: python

    def validate_track(track):
        """Check track state for inconsistencies."""
        errors = []
        
        if not isinstance(track.position, np.ndarray) or track.position.shape != (3,):
            errors.append("Position must be 3D numpy array")
        
        if np.any(np.isnan(track.position)) or np.any(np.isinf(track.position)):
            errors.append("Position contains NaN or Inf")
        
        if hasattr(track, 'covariance'):
            if not np.all(np.isfinite(track.covariance)):
                errors.append("Covariance contains NaN or Inf")
            if np.any(np.linalg.eigvals(track.covariance) < 0):
                errors.append("Covariance is not positive semi-definite")
        
        return errors


References & Further Reading
----------------------------

- TrackSet API documentation
- Track lifecycle management patterns
- Efficient data structure design for real-time systems

See Also
--------

- :doc:`recipes` - Multi-target tracking workflows using TrackSet
- :doc:`api_navigation` - Finding Track and TrackSet functions
- :doc:`troubleshooting` - Common data structure issues
