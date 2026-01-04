# Phase 1: Network Flow Performance Optimization

## Objective
Replace Bellman-Ford O(VE²) algorithm with network simplex O(VE log V) for 50-100x performance improvement.

## Current State
- **Algorithm:** Successive Shortest Paths using Bellman-Ford
- **Time Complexity:** O(V² E) per shortest path search
- **Bottleneck:** Bellman-Ford distance computation on dense graphs
- **Skipped Tests:** 13 performance tests due to timeout
- **File:** `pytcl/assignment_algorithms/network_flow.py`

## Target Algorithm
**Network Simplex Method (NSM)**
- **Time Complexity:** O(VE log V) or even O(V²E)
- **Space Complexity:** O(V + E)
- **Benefits:**
  - Dramatically faster for assignment problems
  - Maintains numerical stability
  - Natural for sparse networks

## Implementation Plan

### Phase 1A: Research & Benchmarking (Week 1)
- [ ] Profile current implementation to identify bottlenecks
- [ ] Benchmark performance on varying problem sizes
- [ ] Document baseline metrics
- [ ] Study network simplex algorithm variants

### Phase 1B: Implement Network Simplex (Week 2)
- [ ] Create `min_cost_flow_simplex()` function
- [ ] Implement tree data structure management
- [ ] Add pivot operations (enter/leave)
- [ ] Handle degenerate cases
- [ ] Implement cost scaling for numerical stability

### Phase 1C: Testing & Integration (Week 2-3)
- [ ] Unit tests for simplex implementation
- [ ] Correctness validation against current solution
- [ ] Re-enable 13 skipped tests
- [ ] Benchmark improvement (target: 50-100x)
- [ ] Integration testing

### Phase 1D: Optimization & Validation (Week 3)
- [ ] Performance tuning
- [ ] Memory optimization
- [ ] Fallback to Bellman-Ford for edge cases
- [ ] Final benchmarks and documentation

## Key Implementation Details

### Network Simplex Algorithm Steps
1. **Initialize spanning tree** - Find initial basic feasible solution
2. **Compute reduced costs** - Using dual variables
3. **While optimality not achieved:**
   - Find entering edge with negative reduced cost
   - Find leaving edge via minimum ratio test
   - Perform pivot operation
   - Update tree and dual variables

### Critical Components
- **Tree representation:** Adjacency list or parent pointers
- **Cycle detection:** Needed for pivot operations
- **Numerical precision:** Handle floating-point errors
- **Degeneracy handling:** Perturbation or specific pivot rules

## Performance Targets
| Aspect | Current | Target | Improvement |
|--------|---------|--------|-------------|
| 2x2 assignment | ~10ms | <0.1ms | 100x |
| 3x3 assignment | ~100ms | <1ms | 100x |
| 10x10 assignment | ~1s | <20ms | 50x |
| Scalability | O(V²E) | O(VE log V) | Polynomial |

## Files to Modify
- `pytcl/assignment_algorithms/network_flow.py` - Main implementation
- `tests/test_network_flow.py` - Test updates (remove skip decorators)
- `CHANGELOG.md` - Document improvements
- `ROADMAP.md` - Mark Phase 1 as complete

## Success Criteria
1. ✅ All 13 previously skipped tests pass
2. ✅ Performance improvement: 50-100x on assignment problems
3. ✅ Numerical accuracy maintained (same results as Bellman-Ford)
4. ✅ Code quality: 100% compliance (black, isort, flake8, mypy)
5. ✅ Test coverage maintained or improved
6. ✅ Benchmark results document improvements

## Development Workflow
```bash
# Create feature branch
git checkout -b feature/phase-1-network-flow-optimization

# Make changes, test frequently
pytest tests/test_network_flow.py -v

# Run benchmarks
pytest benchmarks/test_jpda_bench.py -v

# Quality checks before PR
black pytcl/assignment_algorithms/network_flow.py
isort pytcl/assignment_algorithms/network_flow.py
flake8 pytcl/assignment_algorithms/network_flow.py
mypy pytcl/assignment_algorithms/network_flow.py

# Commit and push
git push origin feature/phase-1-network-flow-optimization
```

## References
- Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). Network Flows: Theory, Algorithms, and Applications.
- Goldberg, A. V. (1997). An Efficient Implementation of a Scaling Minimum-Cost Flow Algorithm.
- Successive Shortest Paths vs. Network Simplex comparison studies

## Notes
- Maintain backward compatibility with current API
- Keep Bellman-Ford as fallback for debugging
- Document algorithm choice in code comments
- Consider scipy.optimize.linear_sum_assignment for reference
