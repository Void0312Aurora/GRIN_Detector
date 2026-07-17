from __future__ import annotations

"""
Legacy experimental unwrap candidates.

These methods were explored as coarse priors for the phase-jump / wrap regime and
were removed from the main CLI/evaluation surface after evidence showed they did
not outperform `pseudo_poisson` on the current mixed-wrap benchmark.

This module intentionally exposes no active implementation. The auditable legacy
candidates remain in `reconstruction/pseudo_poisson.py` until a future method is
migrated onto the `UnwrapProblem` / `UnwrapSolution` interface.
"""
