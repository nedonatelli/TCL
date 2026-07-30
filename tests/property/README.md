# Property-based tests

Invariants that must hold across generated inputs, rather than assertions about
particular values: round trips that return the original, transformations that
preserve a norm, orderings that stay sorted.

**This directory is currently empty.** `hypothesis` is already a dev dependency
(`pyproject.toml`, `[dev]`) but nothing imports it. Several areas are natural
candidates -- coordinate round trips, rotation composition, assignment
optimality against brute force on small inputs, filter covariance symmetry and
positive-definiteness.

Note that "hypothesis" appears throughout the tracking code in its domain sense
(multiple hypothesis tracking, `HypothesisTree`). That is unrelated to the
library of the same name.
