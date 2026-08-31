from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_grin_rebuild.models.structured_observation import (  # noqa: E402
    decode_linear_factor,
    deterministic_median,
    extract_polar_harmonics,
    fit_edge_harmonics,
    fit_linear_factor,
    fit_texture_spectrum,
    project_linear_factor,
    reconstruct_edge_harmonics,
    sample_correlated_texture,
    sample_factor_scores,
)


def test_deterministic_median_matches_even_and_odd_definitions() -> None:
    values = torch.tensor([[1.0, 9.0], [3.0, 5.0], [7.0, 2.0], [11.0, 4.0]])
    torch.testing.assert_close(deterministic_median(values, dim=0), torch.tensor([5.0, 4.5]))
    torch.testing.assert_close(
        deterministic_median(values[:3], dim=0),
        torch.tensor([3.0, 5.0]),
    )


def _rho_theta(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.sqrt(xx.square() + yy.square()), torch.atan2(yy, xx)


def test_polar_harmonics_reconstruct_known_fourfold_field() -> None:
    rho, theta = _rho_theta(64)
    radial = torch.exp(-((rho - 0.82) / 0.12).square())
    image = 0.15 + 0.6 * radial + 0.2 * radial * torch.cos(4.0 * theta)
    coefficients, reconstruction = extract_polar_harmonics(
        image[None],
        rho,
        harmonics=(1, 2, 4),
        radial_bins=96,
        rho_max=float(torch.max(rho)),
        smoothing_sigma_bins=0.0,
    )
    assert coefficients.shape == (1, 7, 96)
    rmse = torch.sqrt(torch.mean((reconstruction[0] - image) ** 2))
    assert float(rmse) < 0.025


def test_linear_factor_projection_and_sampling_are_reproducible() -> None:
    values = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    state = fit_linear_factor(values, rank=1)
    scores, reconstruction = project_linear_factor(values, state)
    torch.testing.assert_close(reconstruction, values, atol=1e-5, rtol=1e-5)
    first = sample_factor_scores(scores, 5, generator=torch.Generator().manual_seed(17))
    second = sample_factor_scores(scores, 5, generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    decoded = decode_linear_factor(first, state, (2, 2))
    assert decoded.shape == (5, 2, 2)


def test_correlated_texture_sampling_is_seeded_and_finite() -> None:
    rho, _ = _rho_theta(32)
    generator = torch.Generator().manual_seed(5)
    residuals = torch.randn((6, 32, 32), generator=generator)
    state = fit_texture_spectrum(
        residuals,
        rho,
        highpass_kernel=7,
        spectrum_smoothing=3,
    )
    first = sample_correlated_texture(
        state,
        4,
        generator=torch.Generator().manual_seed(23),
    )
    second = sample_correlated_texture(
        state,
        4,
        generator=torch.Generator().manual_seed(23),
    )
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    assert first.shape == (4, 32, 32)
    assert torch.all(torch.isfinite(first))
    assert float(torch.std(first)) > 0.0


def test_explicit_edge_harmonic_recovers_fourfold_correction() -> None:
    rho, theta = _rho_theta(96)
    envelope = torch.exp(-0.5 * ((rho - 1.0) / 0.075).square())
    target = envelope * (0.08 + 0.12 * torch.cos(4.0 * theta))
    state, correction = fit_edge_harmonics(target[None], rho)
    reconstructed = reconstruct_edge_harmonics(state, state.coefficients)
    torch.testing.assert_close(reconstructed, correction, atol=1e-5, rtol=1e-5)
    valid = (rho >= 0.80) & (rho <= 1.20)
    rmse = torch.sqrt(torch.mean((reconstructed[0][valid] - target[valid]) ** 2))
    assert float(rmse) < 0.01
