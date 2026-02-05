"""Tests for vegetation analysis computations."""

import numpy as np
import numpy.testing as npt
import pytest

from satellitehub._pipeline import _cloud_mask
from satellitehub._types import RawData
from satellitehub.analysis.vegetation import compute_ndvi


class TestComputeNdvi:
    """Tests for compute_ndvi()."""

    @pytest.mark.unit
    def test_ndvi_known_values(self) -> None:
        """Standard NDVI computation with hand-calculable values."""
        # red=0.1, nir=0.5 → NDVI = (0.5-0.1)/(0.5+0.1) = 0.4/0.6 ≈ 0.6667
        red = np.array([[0.1, 0.2]], dtype=np.float32)
        nir = np.array([[0.5, 0.4]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        assert ndvi.shape == (1, 2)
        npt.assert_allclose(ndvi[0, 0], (0.5 - 0.1) / (0.5 + 0.1), atol=1e-5)
        npt.assert_allclose(ndvi[0, 1], (0.4 - 0.2) / (0.4 + 0.2), atol=1e-5)

    @pytest.mark.unit
    def test_ndvi_nan_propagation(self) -> None:
        """NaN inputs (from cloud mask) propagate to NaN output."""
        red = np.array([[0.1, np.nan, 0.2]], dtype=np.float32)
        nir = np.array([[0.5, 0.4, np.nan]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        assert not np.isnan(ndvi[0, 0])  # valid
        assert np.isnan(ndvi[0, 1])  # red NaN
        assert np.isnan(ndvi[0, 2])  # nir NaN

    @pytest.mark.unit
    def test_ndvi_division_by_zero_produces_nan(self) -> None:
        """Where nir + red == 0, result is NaN (not crash or inf)."""
        red = np.array([[0.0, 0.1]], dtype=np.float32)
        nir = np.array([[0.0, 0.3]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        assert np.isnan(ndvi[0, 0])  # 0/0 → NaN
        assert not np.isnan(ndvi[0, 1])  # valid

    @pytest.mark.unit
    def test_ndvi_output_range(self) -> None:
        """All non-NaN NDVI values are in [-1, 1]."""
        rng = np.random.default_rng(42)
        # Random reflectance values in [0, 1]
        red = rng.random((10, 10)).astype(np.float32)
        nir = rng.random((10, 10)).astype(np.float32)

        ndvi = compute_ndvi(red, nir)

        valid = ndvi[~np.isnan(ndvi)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    @pytest.mark.unit
    def test_ndvi_extreme_values(self) -> None:
        """NDVI at theoretical extremes: pure NIR and pure Red."""
        # nir=1, red=0 → NDVI = 1.0 (dense vegetation)
        # nir=0, red=1 → NDVI = -1.0 (water/bare soil)
        red = np.array([[0.0, 1.0]], dtype=np.float32)
        nir = np.array([[1.0, 0.0]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        npt.assert_allclose(ndvi[0, 0], 1.0, atol=1e-6)
        npt.assert_allclose(ndvi[0, 1], -1.0, atol=1e-6)

    @pytest.mark.unit
    def test_ndvi_empty_array(self) -> None:
        """Empty input arrays return empty output."""
        red = np.array([], dtype=np.float32).reshape(0, 0)
        nir = np.array([], dtype=np.float32).reshape(0, 0)

        ndvi = compute_ndvi(red, nir)

        assert ndvi.shape == (0, 0)
        assert ndvi.size == 0

    @pytest.mark.unit
    def test_ndvi_single_pixel(self) -> None:
        """Single pixel computation works correctly."""
        red = np.array([[0.2]], dtype=np.float32)
        nir = np.array([[0.8]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        assert ndvi.shape == (1, 1)
        expected = (0.8 - 0.2) / (0.8 + 0.2)
        npt.assert_allclose(ndvi[0, 0], expected, atol=1e-5)

    @pytest.mark.unit
    def test_ndvi_preserves_input_dtype(self) -> None:
        """Output dtype matches input dtype."""
        red32 = np.array([[0.1]], dtype=np.float32)
        nir32 = np.array([[0.5]], dtype=np.float32)
        assert compute_ndvi(red32, nir32).dtype == np.float32

        red64 = np.array([[0.1]], dtype=np.float64)
        nir64 = np.array([[0.5]], dtype=np.float64)
        assert compute_ndvi(red64, nir64).dtype == np.float64


class TestNdviReproducibility:
    """Reproducibility tests (NFR20)."""

    @pytest.mark.unit
    def test_ndvi_bit_identical_repeated_computation(self) -> None:
        """Same inputs produce bit-identical outputs on repeated calls."""
        rng = np.random.default_rng(123)
        red = rng.random((20, 20)).astype(np.float32)
        nir = rng.random((20, 20)).astype(np.float32)

        result1 = compute_ndvi(red, nir)
        result2 = compute_ndvi(red, nir)

        npt.assert_array_equal(result1, result2)

    @pytest.mark.unit
    def test_cloud_mask_then_ndvi_bit_identical(self) -> None:
        """Full pipeline: mask → NDVI is reproducible."""
        rng = np.random.default_rng(456)
        b04 = rng.random((5, 5)).astype(np.float32)
        b08 = rng.random((5, 5)).astype(np.float32)
        scl = np.array(
            [
                [4, 9, 4, 5, 8],
                [6, 4, 10, 4, 4],
                [4, 3, 4, 4, 7],
                [4, 4, 0, 4, 4],
                [11, 4, 4, 1, 4],
            ],
            dtype=np.float32,
        )
        data = np.stack([b04, b08, scl])
        raw = RawData(data=data, metadata={})

        masked1 = _cloud_mask(raw, scl_band_index=2)
        ndvi1 = compute_ndvi(masked1.data[0], masked1.data[1])

        masked2 = _cloud_mask(raw, scl_band_index=2)
        ndvi2 = compute_ndvi(masked2.data[0], masked2.data[1])

        npt.assert_array_equal(ndvi1, ndvi2)
