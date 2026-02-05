"""Tests for the pipeline acquisition and serialization helpers."""

import struct
from unittest.mock import MagicMock

import numpy as np
import pytest

from satellitehub._pipeline import (
    _COUNT_WEIGHT,
    _RATIO_WEIGHT,
    _SCL_MASK_CLASSES,
    _acquire,
    _assess_quality,
    _build_result,
    _cloud_mask,
    _deserialize_raw_data,
    _serialize_raw_data,
)
from satellitehub._types import QualityAssessment, RawData
from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.location import Location
from satellitehub.providers.base import CatalogEntry
from satellitehub.results import ResultMetadata, VegetationResult

# ── Serialization round-trip tests ─────────────────────────────────


class TestSerializationRoundTrip:
    """Tests for _serialize_raw_data / _deserialize_raw_data."""

    @pytest.mark.unit
    def test_round_trip_preserves_data(self) -> None:
        data = np.random.default_rng(42).random((2, 50, 50)).astype(np.float32)
        raw = RawData(
            data=data,
            metadata={"product_id": "S2A_123", "bands": ["B04", "B08"]},
        )
        serialized = _serialize_raw_data(raw)
        restored = _deserialize_raw_data(serialized)

        np.testing.assert_array_equal(restored.data, raw.data)
        assert restored.metadata == raw.metadata

    @pytest.mark.unit
    def test_round_trip_preserves_shape(self) -> None:
        data = np.zeros((3, 100, 100), dtype=np.float64)
        raw = RawData(data=data, metadata={})
        restored = _deserialize_raw_data(_serialize_raw_data(raw))

        assert restored.data.shape == (3, 100, 100)
        assert restored.data.dtype == np.float64

    @pytest.mark.unit
    def test_round_trip_preserves_metadata(self) -> None:
        raw = RawData(
            data=np.array([1.0, 2.0], dtype=np.float32),
            metadata={"key": "value", "count": 42},
        )
        restored = _deserialize_raw_data(_serialize_raw_data(raw))
        assert restored.metadata["key"] == "value"
        assert restored.metadata["count"] == 42

    @pytest.mark.unit
    def test_round_trip_empty_array(self) -> None:
        raw = RawData(data=np.array([], dtype=np.float32), metadata={})
        restored = _deserialize_raw_data(_serialize_raw_data(raw))
        assert restored.data.size == 0

    @pytest.mark.unit
    def test_round_trip_zero_sized_multidimensional(self) -> None:
        data = np.zeros((0, 10, 10), dtype=np.float32)
        raw = RawData(data=data, metadata={"note": "empty bands"})
        restored = _deserialize_raw_data(_serialize_raw_data(raw))
        assert restored.data.shape == (0, 10, 10)
        assert restored.data.dtype == np.float32
        assert restored.metadata["note"] == "empty bands"

    @pytest.mark.unit
    def test_serialized_format_has_length_prefix(self) -> None:
        raw = RawData(data=np.array([1.0], dtype=np.float32), metadata={})
        serialized = _serialize_raw_data(raw)
        # First 4 bytes are uint32 big-endian length prefix
        assert len(serialized) > 4
        (header_len,) = struct.unpack(">I", serialized[:4])
        assert header_len > 0


# ── _acquire() tests ──────────────────────────────────────────────


def _make_test_location(tmp_path: object) -> Location:
    """Create a Location with a temporary cache directory."""
    config = Config(cache_dir=str(tmp_path))
    return Location(lat=51.25, lon=22.57, config=config)


class TestAcquireCacheMiss:
    """Tests for _acquire() when cache is empty."""

    @pytest.mark.unit
    def test_cache_miss_calls_provider(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = _make_test_location(tmp_path)
        test_data = np.ones((2, 10, 10), dtype=np.float32)
        raw = RawData(
            data=test_data,
            metadata={"product_id": "P1", "bands": ["B04", "B08"]},
        )
        entry = CatalogEntry(
            provider="cdse",
            product_id="P1",
            timestamp="2024-01-01T10:00:00Z",
        )

        mock_provider = MagicMock()
        mock_provider.search.return_value = [entry]
        mock_provider.download.return_value = raw

        # Pre-populate provider cache (slots-safe)
        loc._providers["cdse"] = mock_provider
        monkeypatch.setattr(
            "satellitehub._pipeline.resolve_credentials_path",
            lambda explicit: None,
        )

        result = _acquire(
            location=loc,
            provider_name="cdse",
            product="sentinel2-l2a",
            bands=["B04", "B08"],
            cloud_max=0.3,
            last_days=30,
        )

        assert result.data.shape == (2, 10, 10)
        mock_provider.search.assert_called_once()
        mock_provider.download.assert_called_once()

    @pytest.mark.unit
    def test_empty_search_returns_empty_raw_data(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = _make_test_location(tmp_path)

        mock_provider = MagicMock()
        mock_provider.search.return_value = []

        loc._providers["cdse"] = mock_provider
        monkeypatch.setattr(
            "satellitehub._pipeline.resolve_credentials_path",
            lambda explicit: None,
        )

        result = _acquire(
            location=loc,
            provider_name="cdse",
            product="sentinel2-l2a",
            bands=None,
            cloud_max=0.3,
            last_days=30,
        )

        assert result.data.size == 0
        mock_provider.download.assert_not_called()


class TestAcquireCacheHit:
    """Tests for _acquire() when cache contains data."""

    @pytest.mark.unit
    def test_cache_hit_skips_provider(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = _make_test_location(tmp_path)

        # Pre-populate cache
        test_data = np.ones((2, 10, 10), dtype=np.float32)
        raw = RawData(data=test_data, metadata={"cached": True})

        from satellitehub.cache import CacheManager

        cache = CacheManager(config=loc.config)
        cache_key = cache._build_cache_key(
            provider="cdse",
            product="sentinel2-l2a",
            location_hash=loc.location_hash,
            time_range=("2024-01-01", "2024-01-31"),
            params={"bands": "B04,B08", "cloud_max": "0.3"},
        )
        serialized = _serialize_raw_data(raw)
        cache.store(
            cache_key=cache_key,
            provider="cdse",
            product="sentinel2-l2a",
            location_hash=loc.location_hash,
            data=serialized,
        )

        mock_provider = MagicMock()
        loc._providers["cdse"] = mock_provider
        monkeypatch.setattr(
            "satellitehub._pipeline.resolve_credentials_path",
            lambda explicit: None,
        )

        # Patch datetime to get consistent time_range matching cache key
        from datetime import datetime, timezone

        import satellitehub._pipeline as pipeline_mod

        class FakeDateTime(datetime):
            @classmethod
            def now(cls, tz: object = None) -> "FakeDateTime":  # type: ignore[override]
                return FakeDateTime(2024, 1, 31, tzinfo=timezone.utc)

        monkeypatch.setattr(pipeline_mod, "datetime", FakeDateTime)

        result = _acquire(
            location=loc,
            provider_name="cdse",
            product="sentinel2-l2a",
            bands=["B04", "B08"],
            cloud_max=0.3,
            last_days=30,
        )

        assert result.metadata.get("cached") is True
        mock_provider.search.assert_not_called()
        mock_provider.download.assert_not_called()


class TestAcquireErrorPropagation:
    """Tests that infrastructure errors propagate from _acquire."""

    @pytest.mark.unit
    def test_provider_error_propagates(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = _make_test_location(tmp_path)

        mock_provider = MagicMock()
        mock_provider.search.side_effect = ProviderError(
            what="CDSE unreachable",
            cause="Connection timeout",
            fix="Check network",
        )

        loc._providers["cdse"] = mock_provider
        monkeypatch.setattr(
            "satellitehub._pipeline.resolve_credentials_path",
            lambda explicit: None,
        )

        with pytest.raises(ProviderError, match="CDSE unreachable"):
            _acquire(
                location=loc,
                provider_name="cdse",
                product="sentinel2-l2a",
                bands=None,
                cloud_max=0.3,
                last_days=30,
            )

    @pytest.mark.unit
    def test_configuration_error_propagates(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loc = _make_test_location(tmp_path)

        mock_provider = MagicMock()
        mock_provider.authenticate.side_effect = ConfigurationError(
            what="Invalid credentials",
            cause="Expired token",
            fix="Regenerate at CDSE portal",
        )

        loc._providers["cdse"] = mock_provider
        monkeypatch.setattr(
            "satellitehub._pipeline.resolve_credentials_path",
            lambda explicit: "/fake/creds.json",
        )
        monkeypatch.setattr(
            "satellitehub._pipeline.load_credentials",
            lambda path: {"cdse": {"username": "u", "password": "p"}},
        )

        with pytest.raises(ConfigurationError, match="Invalid credentials"):
            _acquire(
                location=loc,
                provider_name="cdse",
                product="sentinel2-l2a",
                bands=None,
                cloud_max=0.3,
                last_days=30,
            )


# ── _cloud_mask() tests ──────────────────────────────────────────


class TestCloudMaskStandard:
    """Tests for _cloud_mask() with typical Sentinel-2 data."""

    @pytest.mark.unit
    def test_cloud_mask_mixed_pixels(self) -> None:
        """Standard scene with a mix of clear and cloudy pixels."""
        # 3 bands: B04, B08, SCL (index 2)
        # 2x2 pixels: top-left=vegetation(4), top-right=cloud(9),
        #             bot-left=water(6), bot-right=cirrus(10)
        b04 = np.array([[0.1, 0.2], [0.05, 0.3]], dtype=np.float32)
        b08 = np.array([[0.5, 0.4], [0.3, 0.6]], dtype=np.float32)
        scl = np.array([[4, 9], [6, 10]], dtype=np.float32)
        data = np.stack([b04, b08, scl])  # shape (3, 2, 2)
        raw = RawData(data=data, metadata={"bands": ["B04", "B08", "SCL"]})

        result = _cloud_mask(raw, scl_band_index=2)

        # SCL band removed → 2 bands remain
        assert result.data.shape == (2, 2, 2)
        # Valid pixels (vegetation=4, water=6) keep values
        assert not np.isnan(result.data[0, 0, 0])  # B04, vegetation
        assert not np.isnan(result.data[0, 1, 0])  # B04, water
        # Cloudy pixels (cloud=9, cirrus=10) are NaN
        assert np.isnan(result.data[0, 0, 1])  # B04, cloud
        assert np.isnan(result.data[0, 1, 1])  # B04, cirrus
        # cloud_free_ratio: 2 of 4 pixels are valid
        assert result.cloud_free_ratio == pytest.approx(0.5)
        # Mask shape matches spatial dimensions
        assert result.mask is not None
        assert result.mask.shape == (2, 2)

    @pytest.mark.unit
    def test_cloud_mask_all_clear(self) -> None:
        """All pixels are clear (vegetation class 4)."""
        scl = np.full((3, 3), 4, dtype=np.float32)  # all vegetation
        b04 = np.ones((3, 3), dtype=np.float32) * 0.1
        data = np.stack([b04, scl])  # shape (2, 3, 3), SCL at index 1
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=1)

        assert result.data.shape == (1, 3, 3)
        assert not np.any(np.isnan(result.data))
        assert result.cloud_free_ratio == pytest.approx(1.0)

    @pytest.mark.unit
    def test_cloud_mask_all_cloudy(self) -> None:
        """All pixels are cloudy (cloud high probability class 9)."""
        scl = np.full((3, 3), 9, dtype=np.float32)  # all cloud
        b04 = np.ones((3, 3), dtype=np.float32) * 0.1
        data = np.stack([b04, scl])
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=1)

        assert result.data.shape == (1, 3, 3)
        assert np.all(np.isnan(result.data))
        assert result.cloud_free_ratio == pytest.approx(0.0)

    @pytest.mark.unit
    def test_cloud_mask_scl_classes_categorization(self) -> None:
        """Every SCL class is correctly categorized as mask or keep."""
        # Create 1x12 pixels, one per SCL class (0–11)
        scl = np.arange(12, dtype=np.float32).reshape(1, 12)
        b04 = np.ones((1, 12), dtype=np.float32)
        data = np.stack([b04, scl])  # shape (2, 1, 12), SCL at index 1
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=1)

        assert result.data.shape == (1, 1, 12)
        assert result.mask is not None
        # Masked classes: {0, 1, 3, 8, 9, 10}
        for cls_val in (0, 1, 3, 8, 9, 10):
            assert np.isnan(result.data[0, 0, cls_val]), (
                f"SCL class {cls_val} should be masked (NaN)"
            )
            assert not result.mask[0, cls_val], (
                f"SCL class {cls_val} should be False in mask"
            )
        # Kept classes: {2, 4, 5, 6, 7, 11}
        for cls_val in (2, 4, 5, 6, 7, 11):
            assert not np.isnan(result.data[0, 0, cls_val]), (
                f"SCL class {cls_val} should be kept (not NaN)"
            )
            assert result.mask[0, cls_val], (
                f"SCL class {cls_val} should be True in mask"
            )

    @pytest.mark.unit
    def test_cloud_mask_empty_array(self) -> None:
        """Empty input returns empty MaskedData with SCL band removed."""
        data = np.zeros((3, 0, 0), dtype=np.float32)
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=2)

        # SCL band removed → 2 spectral bands remain even for empty arrays
        assert result.data.shape == (2, 0, 0)
        assert result.data.size == 0
        assert result.mask is None
        assert result.cloud_free_ratio == 0.0

    @pytest.mark.unit
    def test_cloud_mask_scl_at_index_zero(self) -> None:
        """SCL band at index 0 is correctly removed."""
        scl = np.array([[4, 9], [6, 10]], dtype=np.float32)
        b04 = np.ones((2, 2), dtype=np.float32) * 0.1
        b08 = np.ones((2, 2), dtype=np.float32) * 0.5
        data = np.stack([scl, b04, b08])  # shape (3, 2, 2), SCL at index 0
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=0)

        # SCL removed → 2 bands remain (B04, B08)
        assert result.data.shape == (2, 2, 2)
        # Vegetation(4) and Water(6) pixels are valid
        assert not np.isnan(result.data[0, 0, 0])
        assert not np.isnan(result.data[0, 1, 0])
        # Cloud(9) and Cirrus(10) pixels are NaN
        assert np.isnan(result.data[0, 0, 1])
        assert np.isnan(result.data[0, 1, 1])

    @pytest.mark.unit
    def test_cloud_mask_preserves_input_dtype(self) -> None:
        """Output dtype matches input dtype for floating-point inputs."""
        scl = np.full((2, 2), 4, dtype=np.float32)
        b04 = np.ones((2, 2), dtype=np.float32)
        data = np.stack([b04, scl])
        raw = RawData(data=data, metadata={})

        result = _cloud_mask(raw, scl_band_index=1)

        assert result.data.dtype == np.float32

    @pytest.mark.unit
    def test_cloud_mask_constants_match_esa_spec(self) -> None:
        """Verify _SCL_MASK_CLASSES matches ESA specification."""
        assert frozenset({0, 1, 3, 8, 9, 10}) == _SCL_MASK_CLASSES


# ── Quality assessment tests ──────────────────────────────────────


class TestAssessQuality:
    """Tests for _assess_quality()."""

    @pytest.mark.unit
    def test_assess_quality_standard_case(self) -> None:
        """4 of 6 cloud-free → confidence ~0.67–0.78, excluded warning."""
        qa = _assess_quality(
            observation_count=6,
            cloud_free_count=4,
            cloud_cover_percentages=[0.1, 0.9, 0.05, 0.85, 0.2, 0.95],
        )

        assert 0.67 <= qa.confidence <= 0.78
        assert qa.observation_count == 6
        assert qa.cloud_free_count == 4
        assert any("2 passes excluded" in w for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_all_clear(self) -> None:
        """6 of 6 cloud-free → confidence ~1.0, no warnings."""
        qa = _assess_quality(
            observation_count=6,
            cloud_free_count=6,
            cloud_cover_percentages=[0.05, 0.1, 0.02, 0.08, 0.15, 0.03],
        )

        assert qa.confidence == pytest.approx(1.0, abs=0.01)
        assert qa.warnings == []

    @pytest.mark.unit
    def test_assess_quality_zero_cloud_free(self) -> None:
        """0 of 6 cloud-free → confidence 0.0, 'Insufficient' warning."""
        qa = _assess_quality(
            observation_count=6,
            cloud_free_count=0,
            cloud_cover_percentages=[0.9, 0.95, 0.85, 0.92, 0.88, 0.91],
        )

        assert qa.confidence == 0.0
        assert any("Insufficient cloud-free observations" in w for w in qa.warnings)
        assert any("0 of 6" in w for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_marginal_one_pass(self) -> None:
        """1 of 6 cloud-free → low confidence 0.1–0.3."""
        qa = _assess_quality(
            observation_count=6,
            cloud_free_count=1,
        )

        assert 0.1 <= qa.confidence <= 0.3
        assert any("Limited cloud-free observations" in w for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_marginal_two_passes(self) -> None:
        """2 of 6 cloud-free → confidence 0.2–0.4."""
        qa = _assess_quality(
            observation_count=6,
            cloud_free_count=2,
        )

        assert 0.15 <= qa.confidence <= 0.40
        assert any("Limited cloud-free observations" in w for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_single_observation_clear(self) -> None:
        """1 of 1 cloud-free → moderate confidence, penalized for low count."""
        qa = _assess_quality(
            observation_count=1,
            cloud_free_count=1,
        )

        # Single observation should be capped by the ≤2 cloud-free penalty
        assert qa.confidence <= 0.30
        assert qa.confidence > 0.0

    @pytest.mark.unit
    def test_assess_quality_zero_observations(self) -> None:
        """0 total observations → confidence 0.0, 'No satellite' warning."""
        qa = _assess_quality(
            observation_count=0,
            cloud_free_count=0,
        )

        assert qa.confidence == 0.0
        assert any("No satellite observations" in w for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_cloud_cover_percentages_included(self) -> None:
        """cloud_cover_percentages passed through to output."""
        pcts = [0.1, 0.2, 0.3]
        qa = _assess_quality(
            observation_count=3,
            cloud_free_count=3,
            cloud_cover_percentages=pcts,
        )

        assert qa.cloud_cover_percentages == [0.1, 0.2, 0.3]

    @pytest.mark.unit
    def test_assess_quality_cloud_cover_percentages_default_none(self) -> None:
        """cloud_cover_percentages defaults to empty list when None."""
        qa = _assess_quality(
            observation_count=3,
            cloud_free_count=3,
            cloud_cover_percentages=None,
        )

        assert qa.cloud_cover_percentages == []

    @pytest.mark.unit
    def test_assess_quality_warnings_are_strings(self) -> None:
        """All warnings are human-readable strings."""
        qa = _assess_quality(
            observation_count=4,
            cloud_free_count=0,
        )

        assert len(qa.warnings) > 0
        assert all(isinstance(w, str) for w in qa.warnings)
        assert all(len(w) > 10 for w in qa.warnings)

    @pytest.mark.unit
    def test_assess_quality_reproducibility(self) -> None:
        """Identical inputs produce bit-identical outputs (NFR20)."""
        pcts = [0.1, 0.9, 0.2, 0.85, 0.15]

        qa1 = _assess_quality(
            observation_count=5,
            cloud_free_count=3,
            cloud_cover_percentages=list(pcts),
        )
        qa2 = _assess_quality(
            observation_count=5,
            cloud_free_count=3,
            cloud_cover_percentages=list(pcts),
        )

        assert qa1.confidence == qa2.confidence
        assert qa1.observation_count == qa2.observation_count
        assert qa1.cloud_free_count == qa2.cloud_free_count
        assert qa1.cloud_cover_percentages == qa2.cloud_cover_percentages
        assert qa1.warnings == qa2.warnings

    @pytest.mark.unit
    def test_assess_quality_cloud_free_exceeds_observation_count(self) -> None:
        """cloud_free_count > observation_count is clamped to observation_count."""
        qa = _assess_quality(
            observation_count=3,
            cloud_free_count=10,
        )

        # cloud_free_count should be clamped to 3
        assert qa.cloud_free_count == 3
        assert 0.0 <= qa.confidence <= 1.0

    @pytest.mark.unit
    def test_assess_quality_negative_observation_count(self) -> None:
        """Negative observation_count is clamped to 0."""
        qa = _assess_quality(
            observation_count=-5,
            cloud_free_count=3,
        )

        assert qa.observation_count == 0
        assert qa.cloud_free_count == 0
        assert qa.confidence == 0.0

    @pytest.mark.unit
    def test_assess_quality_weight_constants_sum_to_one(self) -> None:
        """Confidence weight constants sum to 1.0."""
        total = _RATIO_WEIGHT + _COUNT_WEIGHT
        assert total == pytest.approx(1.0)

    @pytest.mark.unit
    def test_assess_quality_cloud_cover_percentages_defensive_copy(self) -> None:
        """Caller's list is not mutated by _assess_quality."""
        original = [0.1, 0.2, 0.3]
        original_copy = list(original)
        qa = _assess_quality(
            observation_count=3,
            cloud_free_count=3,
            cloud_cover_percentages=original,
        )

        # Caller's list is unchanged
        assert original == original_copy
        # Mutating the output doesn't affect the original
        qa.cloud_cover_percentages.append(0.99)
        assert original == original_copy


# ── _build_result() tests ────────────────────────────────────────


class TestBuildResult:
    """Tests for _build_result() pipeline helper."""

    @pytest.mark.unit
    def test_build_result_standard_case(self) -> None:
        """Standard NDVI array produces correct mean, std, confidence."""
        ndvi = np.array([[0.4, 0.5], [0.3, 0.6]], dtype=np.float32)
        quality = QualityAssessment(
            confidence=0.78,
            observation_count=6,
            cloud_free_count=4,
            warnings=["2 passes excluded (cloud cover >80%)"],
        )
        meta = ResultMetadata(source="cdse", observation_count=6)

        result = _build_result(ndvi, quality, meta)

        assert isinstance(result, VegetationResult)
        assert result.confidence == 0.78
        assert result.mean_ndvi == pytest.approx(0.45, abs=0.01)
        assert result.ndvi_std == pytest.approx(0.1118, abs=0.01)
        assert result.trend is None
        assert result.observation_count == 6
        assert result.cloud_free_count == 4
        assert result.warnings == ["2 passes excluded (cloud cover >80%)"]
        # Metadata passed by reference (not copied) to avoid overhead
        assert result.metadata is meta

    @pytest.mark.unit
    def test_build_result_with_nan_pixels(self) -> None:
        """NaN pixels (cloud-masked) are ignored in mean/std computation."""
        ndvi = np.array([[0.4, np.nan], [0.6, np.nan]], dtype=np.float32)
        quality = QualityAssessment(
            confidence=0.3,
            observation_count=4,
            cloud_free_count=2,
        )
        meta = ResultMetadata(source="cdse")

        result = _build_result(ndvi, quality, meta)

        assert result.mean_ndvi == pytest.approx(0.5, abs=0.01)
        assert result.ndvi_std == pytest.approx(0.1, abs=0.01)

    @pytest.mark.unit
    def test_build_result_all_nan(self) -> None:
        """All-NaN NDVI array produces NaN mean and std."""
        ndvi = np.full((3, 3), np.nan, dtype=np.float32)
        quality = QualityAssessment(
            confidence=0.0,
            observation_count=3,
            cloud_free_count=0,
            warnings=["Insufficient cloud-free observations: 0 of 3 passes usable"],
        )
        meta = ResultMetadata(source="cdse")

        result = _build_result(ndvi, quality, meta)

        assert np.isnan(result.mean_ndvi)
        assert np.isnan(result.ndvi_std)
        assert result.confidence == 0.0

    @pytest.mark.unit
    def test_build_result_empty_array(self) -> None:
        """Empty NDVI array produces NaN mean and std."""
        ndvi = np.array([], dtype=np.float32)
        quality = QualityAssessment(
            confidence=0.0,
            observation_count=0,
            cloud_free_count=0,
            warnings=["No satellite observations available for the requested period"],
        )
        meta = ResultMetadata(source="cdse")

        result = _build_result(ndvi, quality, meta)

        assert np.isnan(result.mean_ndvi)
        assert np.isnan(result.ndvi_std)
        assert result.confidence == 0.0
        assert result.data.size == 0

    @pytest.mark.unit
    def test_build_result_warnings_defensively_copied(self) -> None:
        """Warnings list is defensively copied from QualityAssessment."""
        original_warnings = ["test warning"]
        quality = QualityAssessment(
            confidence=0.5,
            observation_count=3,
            cloud_free_count=2,
            warnings=original_warnings,
        )
        meta = ResultMetadata(source="cdse")
        ndvi = np.array([[0.5]], dtype=np.float32)

        result = _build_result(ndvi, quality, meta)

        # Mutating the result's warnings doesn't affect the original
        result.warnings.append("extra")
        assert original_warnings == ["test warning"]

    @pytest.mark.unit
    def test_build_result_reproducibility(self) -> None:
        """Identical inputs produce identical results (NFR20)."""
        ndvi = np.array([[0.4, 0.5], [0.3, 0.6]], dtype=np.float32)
        quality = QualityAssessment(
            confidence=0.78,
            observation_count=6,
            cloud_free_count=4,
        )
        meta = ResultMetadata(source="cdse")

        r1 = _build_result(ndvi.copy(), quality, meta)
        r2 = _build_result(ndvi.copy(), quality, meta)

        assert r1.mean_ndvi == r2.mean_ndvi
        assert r1.ndvi_std == r2.ndvi_std
        assert r1.confidence == r2.confidence
        assert r1.warnings == r2.warnings
