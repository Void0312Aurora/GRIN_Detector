from __future__ import annotations

from mini_grin_rebuild.reconstruction.pseudo_poisson import (
    reconstruct_defect_first_order_poisson,
    reconstruct_defect_first_order_sign_quadratic_poisson,
    reconstruct_defect_oracle_poisson,
    reconstruct_defect_pseudo_poisson,
)


def reconstruct_defect_coarse_prior(
    *,
    method: str,
    **kwargs,
):
    if method == "pseudo_poisson":
        return reconstruct_defect_pseudo_poisson(**kwargs)
    if method == "first_order_poisson":
        return reconstruct_defect_first_order_poisson(**kwargs)
    if method == "first_order_sign_quadratic_poisson":
        return reconstruct_defect_first_order_sign_quadratic_poisson(**kwargs)
    raise ValueError(f"Unknown coarse prior method: {method!r}")


__all__ = [
    "reconstruct_defect_coarse_prior",
    "reconstruct_defect_first_order_poisson",
    "reconstruct_defect_first_order_sign_quadratic_poisson",
    "reconstruct_defect_oracle_poisson",
    "reconstruct_defect_pseudo_poisson",
]
