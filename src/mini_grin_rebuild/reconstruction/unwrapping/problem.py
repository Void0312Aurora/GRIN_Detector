from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UnwrapProblem:
    """
    Canonical input bundle for future wrap-aware coarse reconstruction algorithms.

    The intent is to separate:
    - local observation-side quantities (wrapped or sign-ambiguous test edges)
    - known standard-phase reference quantities
    - target quantity we ultimately care about (defect phase / height)

    Future algorithms should consume this structure instead of reaching into the
    pseudo-poisson baseline implementation directly.
    """

    standard_phase: torch.Tensor
    standard_grad_x: torch.Tensor
    standard_grad_y: torch.Tensor
    standard_inc_x: torch.Tensor
    standard_inc_y: torch.Tensor
    test_grad_x_candidate: torch.Tensor
    test_grad_y_candidate: torch.Tensor
    test_inc_x_wrapped: torch.Tensor
    test_inc_y_wrapped: torch.Tensor
    dx: float
    phase_scale: float


@dataclass(frozen=True)
class UnwrapSolution:
    """
    Canonical output bundle for wrap-aware reconstruction.

    `defect_phase` is the main field consumed by the height reconstruction path.
    Optional edge weights / diagnostics are included so future algorithms can be
    compared and visualized without changing the external evaluation interface.
    """

    defect_phase: torch.Tensor
    defect_inc_x: torch.Tensor | None = None
    defect_inc_y: torch.Tensor | None = None
    edge_weight_x: torch.Tensor | None = None
    edge_weight_y: torch.Tensor | None = None
