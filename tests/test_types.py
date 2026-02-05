"""Tests for the SatelliteHub internal types."""

from __future__ import annotations

import numpy as np
import pytest

from satellitehub._types import (
    BandList,
    LocationHash,
    MaskedData,
    QualityAssessment,
    RawData,
    TimeRange,
)


@pytest.mark.unit
class TestTypeAliases:
    """Verify type aliases resolve correctly."""

    def test_band_list_is_list_of_str(self) -> None:
        bands: BandList = ["B4", "B8"]
        assert isinstance(bands, list)
        assert all(isinstance(b, str) for b in bands)

    def test_location_hash_is_str(self) -> None:
        h: LocationHash = "abc123"
        assert isinstance(h, str)

    def test_time_range_is_tuple_of_str(self) -> None:
        tr: TimeRange = ("2024-01-01", "2024-06-30")
        assert isinstance(tr, tuple)
        assert len(tr) == 2
        assert all(isinstance(t, str) for t in tr)


@pytest.mark.unit
class TestRawData:
    """Verify RawData dataclass construction and defaults."""

    def test_construct_with_data_and_metadata(self) -> None:
        arr = np.zeros((3, 100, 100), dtype=np.float32)
        meta = {"product_id": "S2A_123", "timestamp": "2024-06-15"}
        raw = RawData(data=arr, metadata=meta)
        assert raw.data.shape == (3, 100, 100)
        assert raw.metadata["product_id"] == "S2A_123"

    def test_metadata_defaults_to_empty_dict(self) -> None:
        arr = np.ones((1, 10, 10), dtype=np.float64)
        raw = RawData(data=arr)
        assert raw.metadata == {}
        assert isinstance(raw.metadata, dict)

    def test_data_dtype_is_floating(self) -> None:
        arr = np.zeros((2, 5, 5), dtype=np.float32)
        raw = RawData(data=arr)
        assert np.issubdtype(raw.data.dtype, np.floating)


@pytest.mark.unit
class TestMaskedData:
    """Verify MaskedData dataclass construction and defaults."""

    def test_construct_with_all_fields(self) -> None:
        data = np.ones((100, 100), dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.bool_)
        masked = MaskedData(data=data, mask=mask, cloud_free_ratio=0.85)
        assert masked.data.shape == (100, 100)
        assert masked.mask is not None
        assert masked.mask.shape == (100, 100)
        assert masked.cloud_free_ratio == 0.85

    def test_mask_defaults_to_none(self) -> None:
        data = np.zeros((50, 50), dtype=np.float32)
        masked = MaskedData(data=data)
        assert masked.mask is None

    def test_cloud_free_ratio_defaults_to_zero(self) -> None:
        data = np.zeros((10, 10), dtype=np.float32)
        masked = MaskedData(data=data)
        assert masked.cloud_free_ratio == 0.0


@pytest.mark.unit
class TestQualityAssessment:
    """Verify QualityAssessment dataclass construction and defaults."""

    def test_construct_with_all_fields(self) -> None:
        qa = QualityAssessment(
            confidence=0.78,
            observation_count=6,
            cloud_free_count=4,
            cloud_cover_percentages=[0.1, 0.9, 0.05, 0.85, 0.2, 0.95],
            warnings=["Low cloud-free ratio"],
        )
        assert qa.confidence == 0.78
        assert qa.observation_count == 6
        assert qa.cloud_free_count == 4
        assert qa.cloud_cover_percentages == [0.1, 0.9, 0.05, 0.85, 0.2, 0.95]
        assert qa.warnings == ["Low cloud-free ratio"]

    def test_all_defaults(self) -> None:
        qa = QualityAssessment()
        assert qa.confidence == 0.0
        assert qa.observation_count == 0
        assert qa.cloud_free_count == 0
        assert qa.cloud_cover_percentages == []
        assert qa.warnings == []

    def test_warnings_defaults_to_empty_list(self) -> None:
        qa = QualityAssessment(confidence=0.9, observation_count=10, cloud_free_count=8)
        assert qa.warnings == []
        assert isinstance(qa.warnings, list)

    def test_cloud_cover_percentages_defaults_to_empty_list(self) -> None:
        qa = QualityAssessment(confidence=0.5, observation_count=3, cloud_free_count=2)
        assert qa.cloud_cover_percentages == []
        assert isinstance(qa.cloud_cover_percentages, list)

    def test_default_lists_are_independent(self) -> None:
        qa1 = QualityAssessment()
        qa2 = QualityAssessment()
        qa1.warnings.append("test")
        qa1.cloud_cover_percentages.append(0.5)
        assert qa2.warnings == []
        assert qa2.cloud_cover_percentages == []
