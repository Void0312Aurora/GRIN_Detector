from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearFactorState:
    mean: torch.Tensor
    components: torch.Tensor
    scores: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])


@dataclass(frozen=True)
class TextureSpectrumState:
    spectra: torch.Tensor
    target_std: torch.Tensor
    zone_masks: torch.Tensor


@dataclass(frozen=True)
class EdgeHarmonicState:
    basis: torch.Tensor
    envelope: torch.Tensor
    coefficients: torch.Tensor
    center: float
    width: float


def deterministic_median(values: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    """Median using deterministic sorting, including the even-sample midpoint."""

    if values.shape[dim] < 1:
        raise ValueError("median dimension must contain at least one value")
    ordered = torch.sort(values, dim=dim).values
    count = ordered.shape[dim]
    lower_value = ordered.select(dim, (count - 1) // 2)
    upper_value = ordered.select(dim, count // 2)
    return 0.5 * (lower_value + upper_value)


def angular_mode_names(harmonics: Iterable[int]) -> tuple[str, ...]:
    names = ["k0"]
    for harmonic in harmonics:
        k = int(harmonic)
        if k < 1:
            raise ValueError("non-DC harmonics must be positive")
        names.extend((f"cos{k}", f"sin{k}"))
    return tuple(names)


def _angular_basis(
    shape: tuple[int, int],
    *,
    harmonics: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = shape
    yy = torch.arange(height, device=device, dtype=dtype) - 0.5 * (height - 1)
    xx = torch.arange(width, device=device, dtype=dtype) - 0.5 * (width - 1)
    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")
    theta = torch.atan2(y_grid, x_grid)
    modes = [torch.ones_like(theta)]
    for harmonic in harmonics:
        modes.extend((torch.cos(harmonic * theta), torch.sin(harmonic * theta)))
    return torch.stack(modes, dim=0)


def fit_edge_harmonics(
    residuals: torch.Tensor,
    rho: torch.Tensor,
    *,
    harmonics: tuple[int, ...] = (1, 2, 4),
    center: float = 1.0,
    width: float = 0.075,
    band: tuple[float, float] = (0.80, 1.20),
) -> tuple[EdgeHarmonicState, torch.Tensor]:
    """Fit an explicit angular correction localized to the lens edge.

    The radial envelope is fixed from the registered lens coordinate.  The
    coefficients are fitted jointly, so incomplete outer rings do not mix
    the fourfold term into the DC or dipole terms.
    """

    if residuals.ndim != 3 or rho.ndim != 2 or residuals.shape[1:] != rho.shape:
        raise ValueError("residuals must be [N,H,W] and match rho [H,W]")
    if width <= 0.0 or band[0] >= band[1]:
        raise ValueError("invalid edge width or band")
    harmonics = tuple(int(value) for value in harmonics)
    angular_mode_names(harmonics)
    basis = _angular_basis(
        tuple(rho.shape),
        harmonics=harmonics,
        device=residuals.device,
        dtype=residuals.dtype,
    )
    envelope = torch.exp(-0.5 * ((rho - center) / width).square())
    design = basis * envelope[None]
    valid = (rho >= band[0]) & (rho <= band[1])
    design_valid = design[:, valid].T
    gram = design_valid.T @ design_valid
    scale = torch.clamp(torch.trace(gram) / design_valid.shape[1], min=1.0)
    gram = gram + torch.eye(
        design_valid.shape[1],
        device=residuals.device,
        dtype=residuals.dtype,
    ) * (1e-6 * scale)
    right_hand = residuals[:, valid] @ design_valid
    coefficients = torch.linalg.solve(gram, right_hand.T).T
    correction = torch.sum(coefficients[:, :, None, None] * design[None], dim=1)
    return (
        EdgeHarmonicState(
            basis=basis,
            envelope=envelope,
            coefficients=coefficients,
            center=float(center),
            width=float(width),
        ),
        correction,
    )


def reconstruct_edge_harmonics(state: EdgeHarmonicState, coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.ndim != 2 or coefficients.shape[1] != state.basis.shape[0]:
        raise ValueError("edge coefficients have the wrong shape")
    return torch.sum(
        coefficients[:, :, None, None] * state.basis[None] * state.envelope[None, None],
        dim=1,
    )


def _smooth_radial_profiles(coefficients: torch.Tensor, sigma_bins: float) -> torch.Tensor:
    if sigma_bins <= 0.0:
        return coefficients
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    offsets = torch.arange(
        -radius,
        radius + 1,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    kernel = torch.exp(-0.5 * (offsets / sigma_bins).square())
    kernel = kernel / torch.sum(kernel)
    flattened = coefficients.reshape(-1, 1, coefficients.shape[-1])
    padded = F.pad(flattened, (radius, radius), mode="replicate")
    return F.conv1d(padded, kernel[None, None]).reshape_as(coefficients)


def extract_polar_harmonics(
    images: torch.Tensor,
    rho: torch.Tensor,
    *,
    harmonics: tuple[int, ...] = (1, 2, 4),
    radial_bins: int = 128,
    rho_max: float | None = None,
    smoothing_sigma_bins: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit radial profiles for selected circular harmonics.

    The result is a compact, geometry-aware encoding with shape
    ``[N, 1 + 2*len(harmonics), radial_bins]`` plus its Cartesian
    reconstruction.  Every image is fitted independently; callers must fit
    any population-level factor distribution on training images only.
    """

    if images.ndim != 3:
        raise ValueError(f"images must be [N,H,W], got {tuple(images.shape)}")
    if rho.shape != images.shape[1:]:
        raise ValueError(f"rho shape {tuple(rho.shape)} does not match {tuple(images.shape[1:])}")
    if radial_bins < 8:
        raise ValueError("radial_bins must be at least 8")
    harmonics = tuple(int(value) for value in harmonics)
    angular_mode_names(harmonics)
    if rho_max is None:
        rho_max = float(torch.max(rho).detach().cpu())
    if rho_max <= 0.0:
        raise ValueError("rho_max must be positive")

    basis = _angular_basis(
        tuple(images.shape[1:]),
        harmonics=harmonics,
        device=images.device,
        dtype=images.dtype,
    ).flatten(1)
    rho_position = torch.clamp(rho.flatten() / rho_max, 0.0, 1.0) * (radial_bins - 1)
    radial_index = torch.round(rho_position).to(torch.long)
    # The square crop truncates large-radius rings, so sine/cosine modes are
    # not orthogonal there.  Solve all angular coefficients jointly in every
    # radial bin instead of using independent Fourier inner products.
    mode_count = basis.shape[0]
    gram = torch.zeros(
        (radial_bins, mode_count, mode_count),
        device=images.device,
        dtype=images.dtype,
    )
    for first in range(mode_count):
        for second in range(first, mode_count):
            value = torch.zeros(radial_bins, device=images.device, dtype=images.dtype)
            value.scatter_add_(0, radial_index, basis[first] * basis[second])
            gram[:, first, second] = value
            gram[:, second, first] = value
    right_hand = torch.zeros(
        (images.shape[0], radial_bins, mode_count),
        device=images.device,
        dtype=images.dtype,
    )
    flat_images = images.flatten(1)
    expanded_index = radial_index[None].expand(images.shape[0], -1)
    for mode_index, mode in enumerate(basis):
        right_hand[:, :, mode_index].scatter_add_(
            1,
            expanded_index,
            flat_images * mode[None],
        )
    trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    ridge = torch.clamp(trace / mode_count, min=1.0) * 1e-6
    regularized = gram + ridge[:, None, None] * torch.eye(
        mode_count,
        device=images.device,
        dtype=images.dtype,
    )[None]
    solved = torch.linalg.solve(regularized, right_hand.permute(1, 2, 0))
    coefficient_tensor = solved.permute(2, 1, 0).contiguous()
    coefficient_tensor = _smooth_radial_profiles(coefficient_tensor, smoothing_sigma_bins)
    reconstruction = reconstruct_polar_harmonics(
        coefficient_tensor,
        rho,
        harmonics=harmonics,
        rho_max=rho_max,
    )
    return coefficient_tensor, reconstruction


def reconstruct_polar_harmonics(
    coefficients: torch.Tensor,
    rho: torch.Tensor,
    *,
    harmonics: tuple[int, ...] = (1, 2, 4),
    rho_max: float | None = None,
) -> torch.Tensor:
    if coefficients.ndim != 3:
        raise ValueError("coefficients must be [N,M,B]")
    harmonics = tuple(int(value) for value in harmonics)
    expected_modes = 1 + 2 * len(harmonics)
    if coefficients.shape[1] != expected_modes:
        raise ValueError(
            f"expected {expected_modes} angular modes, got {coefficients.shape[1]}"
        )
    if rho_max is None:
        rho_max = float(torch.max(rho).detach().cpu())
    basis = _angular_basis(
        tuple(rho.shape),
        harmonics=harmonics,
        device=coefficients.device,
        dtype=coefficients.dtype,
    ).flatten(1)
    position = torch.clamp(rho.flatten() / rho_max, 0.0, 1.0) * (coefficients.shape[-1] - 1)
    lower = torch.floor(position).to(torch.long)
    upper = torch.clamp(lower + 1, max=coefficients.shape[-1] - 1)
    fraction = position - lower.to(position.dtype)
    lower_values = coefficients.index_select(2, lower)
    upper_values = coefficients.index_select(2, upper)
    interpolated = lower_values * (1.0 - fraction)[None, None] + upper_values * fraction[None, None]
    reconstruction = torch.sum(interpolated * basis[None], dim=1)
    return reconstruction.reshape(coefficients.shape[0], *rho.shape)


def fit_linear_factor(values: torch.Tensor, rank: int) -> LinearFactorState:
    if values.ndim < 2:
        raise ValueError("values must have a sample dimension and at least one feature dimension")
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[0] < 2:
        raise ValueError("at least two samples are required")
    rank = min(int(rank), flat.shape[0] - 1, flat.shape[1])
    if rank < 1:
        raise ValueError("rank must be positive")
    mean = torch.mean(flat, dim=0, keepdim=True)
    centered = flat - mean
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    components = right[:rank]
    scores = centered @ components.T
    # Make component signs deterministic for stable plots and serialized states.
    for index in range(rank):
        pivot = torch.argmax(torch.abs(components[index]))
        if components[index, pivot] < 0:
            components[index] = -components[index]
            scores[:, index] = -scores[:, index]
    return LinearFactorState(mean=mean, components=components, scores=scores)


def project_linear_factor(values: torch.Tensor, state: LinearFactorState) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = values.shape
    flat = values.reshape(values.shape[0], -1)
    scores = (flat - state.mean) @ state.components.T
    reconstruction = state.mean + scores @ state.components
    return scores, reconstruction.reshape(original_shape)


def sample_factor_scores(
    scores: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1:
        raise ValueError("count must be positive")
    centered = scores - torch.mean(scores, dim=0, keepdim=True)
    covariance = centered.T @ centered / max(scores.shape[0] - 1, 1)
    scale = max(float(torch.trace(covariance).detach().cpu()) / scores.shape[1], 1e-8)
    covariance = covariance + torch.eye(
        scores.shape[1],
        dtype=scores.dtype,
        device=scores.device,
    ) * (1e-6 * scale)
    root = torch.linalg.cholesky(covariance)
    standard = torch.randn(
        (count, scores.shape[1]),
        dtype=scores.dtype,
        device=scores.device,
        generator=generator,
    )
    return torch.mean(scores, dim=0, keepdim=True) + standard @ root.T


def decode_linear_factor(scores: torch.Tensor, state: LinearFactorState, shape: tuple[int, ...]) -> torch.Tensor:
    flat = state.mean + scores @ state.components
    return flat.reshape(scores.shape[0], *shape)


def radial_zone_masks(rho: torch.Tensor, softness: float = 0.035) -> torch.Tensor:
    if rho.ndim != 2:
        raise ValueError("rho must be [H,W]")
    if softness <= 0.0:
        raise ValueError("softness must be positive")
    interior = torch.sigmoid((0.78 - rho) / softness)
    outside = torch.sigmoid((rho - 1.12) / softness)
    seam = torch.clamp(1.0 - interior - outside, min=0.0)
    masks = torch.stack((interior, seam, outside), dim=0)
    return masks / torch.clamp(torch.sum(masks, dim=0, keepdim=True), min=1e-8)


def _smooth_log_spectrum(spectrum: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return spectrum
    if kernel_size % 2 == 0:
        raise ValueError("spectrum smoothing kernel must be odd")
    pad = kernel_size // 2
    logged = torch.log(torch.clamp(spectrum, min=1e-12))[None]
    padded = F.pad(logged, (pad, pad, pad, pad), mode="replicate")
    smoothed = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    return torch.exp(smoothed[0])


def fit_texture_spectrum(
    residuals: torch.Tensor,
    rho: torch.Tensor,
    *,
    highpass_kernel: int = 17,
    spectrum_smoothing: int = 9,
) -> TextureSpectrumState:
    if residuals.ndim != 3:
        raise ValueError("residuals must be [N,H,W]")
    if highpass_kernel < 3 or highpass_kernel % 2 == 0:
        raise ValueError("highpass_kernel must be odd and at least 3")
    pad = highpass_kernel // 2
    padded = F.pad(residuals[:, None], (pad, pad, pad, pad), mode="reflect")
    lowpass = F.avg_pool2d(padded, kernel_size=highpass_kernel, stride=1)[:, 0]
    highpass = residuals - lowpass
    masks = radial_zone_masks(rho).to(device=residuals.device, dtype=residuals.dtype)
    spectra = []
    target_std = []
    for mask in masks:
        weighted = highpass * torch.sqrt(mask)[None]
        spectrum = torch.mean(
            torch.abs(torch.fft.rfft2(weighted, norm="ortho")).square(),
            dim=0,
        )
        spectra.append(_smooth_log_spectrum(spectrum, spectrum_smoothing))
        variance = torch.sum(highpass.square() * mask[None]) / (
            residuals.shape[0] * torch.sum(mask)
        )
        target_std.append(torch.sqrt(torch.clamp(variance, min=1e-12)))
    return TextureSpectrumState(
        spectra=torch.stack(spectra, dim=0),
        target_std=torch.stack(target_std),
        zone_masks=masks,
    )


def sample_correlated_texture(
    state: TextureSpectrumState,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if count < 1:
        raise ValueError("count must be positive")
    height, width = state.zone_masks.shape[1:]
    output = torch.zeros(
        (count, height, width),
        dtype=state.zone_masks.dtype,
        device=state.zone_masks.device,
    )
    for zone, mask in enumerate(state.zone_masks):
        white = torch.randn(
            (count, height, width),
            dtype=output.dtype,
            device=output.device,
            generator=generator,
        )
        frequency = torch.fft.rfft2(white, norm="ortho")
        field = torch.fft.irfft2(
            frequency * torch.sqrt(torch.clamp(state.spectra[zone], min=1e-12))[None],
            s=(height, width),
            norm="ortho",
        )
        weight_sum = torch.sum(mask)
        mean = torch.sum(field * mask[None], dim=(1, 2), keepdim=True) / weight_sum
        centered = field - mean
        std = torch.sqrt(
            torch.sum(centered.square() * mask[None], dim=(1, 2), keepdim=True) / weight_sum
        )
        normalized = centered * (state.target_std[zone] / torch.clamp(std, min=1e-8))
        output = output + normalized * mask[None]
    return output


__all__ = [
    "EdgeHarmonicState",
    "LinearFactorState",
    "TextureSpectrumState",
    "angular_mode_names",
    "decode_linear_factor",
    "deterministic_median",
    "fit_edge_harmonics",
    "extract_polar_harmonics",
    "fit_linear_factor",
    "fit_texture_spectrum",
    "project_linear_factor",
    "reconstruct_edge_harmonics",
    "radial_zone_masks",
    "reconstruct_polar_harmonics",
    "sample_correlated_texture",
    "sample_factor_scores",
]
