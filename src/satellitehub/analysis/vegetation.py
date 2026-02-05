"""Vegetation index computation and cloud masking.

Pure computation module — no HTTP, no caching, no provider interaction.
Takes numpy arrays in, returns numpy arrays out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def compute_ndvi(
    red: npt.NDArray[np.floating[Any]],
    nir: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Compute Normalised Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red).  Masked (NaN) pixels propagate
    to NaN in the output.  Where ``nir + red == 0`` the result is NaN
    (avoids division-by-zero).

    Parameters:
        red: Red band array (e.g., Sentinel-2 B04), shape ``(H, W)``.
        nir: Near-infrared band array (e.g., Sentinel-2 B08), same shape.

    Returns:
        NDVI array with the same shape and dtype as the inputs.
        Values are in ``[-1, 1]`` for valid pixels, ``NaN`` otherwise.

    Example:
        >>> import numpy as np
        >>> red = np.array([[0.1, 0.2]], dtype=np.float32)
        >>> nir = np.array([[0.5, 0.4]], dtype=np.float32)
        >>> ndvi = compute_ndvi(red, nir)
        >>> ndvi.shape
        (1, 2)
    """
    red_f = red.astype(np.float64)
    nir_f = nir.astype(np.float64)

    denominator = nir_f + red_f

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi: npt.NDArray[np.floating[Any]] = np.where(
            denominator == 0.0,
            np.nan,
            (nir_f - red_f) / denominator,
        )

    return ndvi.astype(red.dtype)
