"""Internal shared types for cross-boundary data contracts.

These types define the data shapes passed between provider, pipeline,
and analysis components. They are internal (prefixed ``_``) and NOT
re-exported from ``satellitehub.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

BandList = list[str]
"""Ordered list of spectral band identifiers (e.g., ``['B4', 'B8']``)."""

LocationHash = str
"""Hex digest uniquely identifying a location for cache keys."""

TimeRange = tuple[str, str]
"""ISO-8601 date pair ``(start, end)`` bounding a query window."""


@dataclass
class RawData:
    """Raw data returned by a data provider.

    Per AD-5: numpy arrays are the canonical data representation.

    Args:
        data: Multi-band raster array from the provider.
        metadata: Provider-specific metadata (timestamps, product IDs, etc.).

    Example:
        >>> import numpy as np
        >>> raw = RawData(data=np.zeros((3, 100, 100), dtype=np.float32))
        >>> raw.metadata
        {}
    """

    data: npt.NDArray[np.floating[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskedData:
    """Cloud-masked data ready for analysis.

    Args:
        data: Raster array with cloudy pixels set to NaN.
        mask: Boolean mask where ``True`` indicates a valid (cloud-free) pixel.
        cloud_free_ratio: Fraction of pixels that are cloud-free (0.0–1.0).

    Example:
        >>> import numpy as np
        >>> masked = MaskedData(
        ...     data=np.ones((100, 100), dtype=np.float32),
        ...     cloud_free_ratio=0.85,
        ... )
    """

    data: npt.NDArray[np.floating[Any]]
    mask: npt.NDArray[np.bool_] | None = None
    cloud_free_ratio: float = 0.0


@dataclass
class QualityAssessment:
    """Quality assessment produced by the pipeline for every result.

    Args:
        confidence: Overall confidence score (0.0–1.0).
        observation_count: Total satellite passes or data points available.
        cloud_free_count: Number of usable (cloud-free) observations.
        cloud_cover_percentages: Per-observation cloud cover fractions (0.0–1.0).
        warnings: Human-readable quality warnings.

    Example:
        >>> qa = QualityAssessment(
        ...     confidence=0.78, observation_count=6, cloud_free_count=4,
        ...     cloud_cover_percentages=[0.1, 0.9, 0.05, 0.85, 0.2, 0.95],
        ... )
        >>> qa.warnings
        []
    """

    confidence: float = 0.0
    observation_count: int = 0
    cloud_free_count: int = 0
    cloud_cover_percentages: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MethodInfo:
    """Information about an available analysis method.

    Used by `Location.available_methods()` to describe semantic methods
    and their availability status.

    Args:
        name: Method name (e.g., "vegetation_health").
        description: One-line description of what the method does.
        data_sources: List of required data providers.
        available: Whether the method can be used (all providers configured).
        unavailable_reason: Reason why the method is unavailable, if applicable.

    Example:
        >>> info = MethodInfo(
        ...     name="vegetation_health",
        ...     description="Compute NDVI-based vegetation health analysis",
        ...     data_sources=["cdse"],
        ...     available=True,
        ... )
        >>> info.available
        True
    """

    name: str
    description: str
    data_sources: list[str]
    available: bool = True
    unavailable_reason: str = ""

    def __repr__(self) -> str:
        """Return formatted string for readable display."""
        status = "\u2713" if self.available else f"\u2717 ({self.unavailable_reason})"
        sources = ", ".join(self.data_sources) if self.data_sources else "none"
        return f"{self.name}(): {self.description} [sources: {sources}] {status}"
