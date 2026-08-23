from __future__ import annotations

from dataclasses import dataclass


CONTINUOUS_COARSE_LEVELS = 8
CONTINUOUS_RESIDUAL_LEVELS = 8


@dataclass(frozen=True, slots=True)
class FactorizedValue:
    coarse: int
    residual: int


def quantize_unsigned(
    value: int,
    *,
    maximum: int,
    coarse_levels: int = CONTINUOUS_COARSE_LEVELS,
    residual_levels: int = CONTINUOUS_RESIDUAL_LEVELS,
) -> FactorizedValue:
    """Factorize an unsigned integer into coarse and residual categorical heads."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if not 0 <= value <= maximum:
        raise ValueError(f"value must be in 0..{maximum}")
    if coarse_levels < 2 or residual_levels < 2:
        raise ValueError("coarse_levels and residual_levels must be at least 2")

    normalized = value / maximum
    coarse = min(coarse_levels - 1, int(normalized * coarse_levels))
    lo = coarse / coarse_levels
    hi = (coarse + 1) / coarse_levels
    local = (normalized - lo) / max(1e-12, hi - lo)
    residual = round(local * (residual_levels - 1))
    residual = max(0, min(residual_levels - 1, residual))
    return FactorizedValue(coarse=coarse, residual=residual)


def dequantize_unsigned(
    value: FactorizedValue,
    *,
    maximum: int,
    coarse_levels: int = CONTINUOUS_COARSE_LEVELS,
    residual_levels: int = CONTINUOUS_RESIDUAL_LEVELS,
) -> int:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if not 0 <= value.coarse < coarse_levels:
        raise ValueError("coarse index is outside range")
    if not 0 <= value.residual < residual_levels:
        raise ValueError("residual index is outside range")

    lo = value.coarse / coarse_levels
    hi = (value.coarse + 1) / coarse_levels
    normalized = lo + value.residual / (residual_levels - 1) * (hi - lo)
    return max(0, min(maximum, round(normalized * maximum)))
