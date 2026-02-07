"""Tests for the Landsat provider (Planetary Computer STAC API)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from satellitehub.config import Config
from satellitehub.exceptions import ProviderError
from satellitehub.providers.base import (
    CatalogEntry,
    ProviderCredentials,
    ProviderStatus,
)
from satellitehub.providers.landsat import (
    _BBOX_BUFFER_DEG,
    _COLLECTION,
    _DEFAULT_BANDS,
    _DEFAULT_TIMEOUT,
    _INITIAL_BACKOFF,
    _LANDSAT_L2_BANDS,
    _MAX_BACKOFF,
    _MAX_RETRIES,
    _READ_TIMEOUT,
    _RETRYABLE_STATUS_CODES,
    _STAC_SEARCH_URL,
    _STAC_URL,
    _STATUS_TIMEOUT,
    _SUCCESS_STATUS_CODES,
    LandsatProvider,
    _compute_bbox,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> LandsatProvider:
    """Create a LandsatProvider with default config."""
    return LandsatProvider(config=Config())


@pytest.fixture
def credentials() -> ProviderCredentials:
    """Create empty credentials (Landsat via PC is public)."""
    return ProviderCredentials()


def _make_location(lat: float = 52.23, lon: float = 21.01) -> Any:
    """Create a mock Location object with lat/lon properties (Warsaw default)."""
    loc = MagicMock()
    loc.lat = lat
    loc.lon = lon
    return loc


def _mock_stac_search_response(num_features: int = 2) -> MagicMock:
    """Return a mock Response for STAC search with Landsat 8/9 results."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    features = []
    for i in range(num_features):
        features.append({
            "id": f"LC09_L2SP_182025_20250615_{i:02d}",
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [20.5, 51.5], [21.5, 51.5], [21.5, 52.5],
                        [20.5, 52.5], [20.5, 51.5],
                    ]
                ],
            },
            "properties": {
                "datetime": f"2025-06-15T10:30:0{i}Z",
                "platform": "landsat-9" if i == 0 else "landsat-8",
                "eo:cloud_cover": 10.5 + i * 5,
                "sun_elevation": 45.2,
                "sun_azimuth": 135.0,
                "landsat:wrs_path": "182",
                "landsat:wrs_row": "025",
                "landsat:correction": "L2SP",
            },
            "assets": {
                "blue": {"href": "https://example.com/B2.tif"},
                "green": {"href": "https://example.com/B3.tif"},
                "red": {"href": "https://example.com/B4.tif"},
                "nir08": {"href": "https://example.com/B5.tif"},
                "qa_pixel": {"href": "https://example.com/QA.tif"},
            },
        })
    resp.json.return_value = {"features": features}
    return resp


def _mock_stac_item_response() -> MagicMock:
    """Return a mock Response for fetching a single STAC item."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "id": "LC09_L2SP_182025_20250615_00",
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[20.5, 51.5], [21.5, 51.5], [21.5, 52.5], [20.5, 52.5], [20.5, 51.5]]
            ],
        },
        "properties": {
            "datetime": "2025-06-15T10:30:00Z",
            "platform": "landsat-9",
            "eo:cloud_cover": 10.5,
        },
        "assets": {
            "blue": {"href": "https://unsigned.example.com/B2.tif"},
            "green": {"href": "https://unsigned.example.com/B3.tif"},
            "red": {"href": "https://unsigned.example.com/B4.tif"},
            "nir08": {"href": "https://unsigned.example.com/B5.tif"},
            "qa_pixel": {"href": "https://unsigned.example.com/QA.tif"},
        },
    }
    return resp


def _mock_sign_response() -> MagicMock:
    """Return a mock Response for Planetary Computer URL signing."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "id": "LC09_L2SP_182025_20250615_00",
        "type": "Feature",
        "assets": {
            "blue": {"href": "https://signed.example.com/B2.tif?token=abc123"},
            "green": {"href": "https://signed.example.com/B3.tif?token=abc123"},
            "red": {"href": "https://signed.example.com/B4.tif?token=abc123"},
            "nir08": {"href": "https://signed.example.com/B5.tif?token=abc123"},
            "qa_pixel": {"href": "https://signed.example.com/QA.tif?token=abc123"},
        },
    }
    return resp


def _mock_empty_search_response() -> MagicMock:
    """Return a mock Response for STAC search with no results."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"features": []}
    return resp


def _mock_collection_response() -> MagicMock:
    """Return a mock Response for collection status check."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "id": _COLLECTION,
        "title": "Landsat Collection 2 Level-2",
    }
    return resp


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestLandsatConstants:
    """Test module-level constants are defined correctly."""

    def test_stac_url_defined(self) -> None:
        """STAC URL constant is defined."""
        assert _STAC_URL == "https://planetarycomputer.microsoft.com/api/stac/v1"

    def test_stac_search_url_defined(self) -> None:
        """STAC search URL constant is defined."""
        assert f"{_STAC_URL}/search" == _STAC_SEARCH_URL

    def test_collection_defined(self) -> None:
        """Collection ID constant is defined."""
        assert _COLLECTION == "landsat-c2-l2"

    def test_timeout_constants_defined(self) -> None:
        """Timeout constants are defined."""
        assert _DEFAULT_TIMEOUT == 30
        assert _STATUS_TIMEOUT == 10
        assert _READ_TIMEOUT == 300

    def test_retry_constants_defined(self) -> None:
        """Retry constants are defined."""
        assert _MAX_RETRIES == 3
        assert _INITIAL_BACKOFF == 1.0
        assert _MAX_BACKOFF == 60.0
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 200 in _SUCCESS_STATUS_CODES

    def test_bbox_buffer_defined(self) -> None:
        """Bounding box buffer constant is defined."""
        assert pytest.approx(0.1) == _BBOX_BUFFER_DEG

    def test_default_bands_defined(self) -> None:
        """Default bands list is defined."""
        assert len(_DEFAULT_BANDS) > 0
        assert "B4" in _DEFAULT_BANDS  # Red
        assert "B5" in _DEFAULT_BANDS  # NIR
        assert "QA_PIXEL" in _DEFAULT_BANDS

    def test_landsat_l2_bands_defined(self) -> None:
        """Landsat L2 bands list is defined."""
        assert len(_LANDSAT_L2_BANDS) > 0
        assert "B2" in _LANDSAT_L2_BANDS
        assert "B5" in _LANDSAT_L2_BANDS


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestComputeBbox:
    """Test _compute_bbox helper function."""

    def test_compute_bbox_returns_correct_format(self) -> None:
        """Bounding box is returned as [min_lon, min_lat, max_lon, max_lat]."""
        bbox = _compute_bbox(52.23, 21.01)

        assert len(bbox) == 4
        assert bbox[0] < bbox[2]  # min_lon < max_lon
        assert bbox[1] < bbox[3]  # min_lat < max_lat

    def test_compute_bbox_centered_on_point(self) -> None:
        """Bounding box is centered on the input point."""
        lat, lon = 52.23, 21.01
        bbox = _compute_bbox(lat, lon)

        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2

        assert center_lon == pytest.approx(lon, abs=0.001)
        assert center_lat == pytest.approx(lat, abs=0.001)

    def test_compute_bbox_respects_buffer(self) -> None:
        """Bounding box uses specified buffer."""
        lat, lon = 52.23, 21.01
        buffer = 0.5

        bbox = _compute_bbox(lat, lon, buffer_deg=buffer)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        assert width == pytest.approx(buffer * 2, abs=0.001)
        assert height == pytest.approx(buffer * 2, abs=0.001)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestLandsatProviderInit:
    """Test LandsatProvider initialization."""

    def test_init_creates_session(self, provider: LandsatProvider) -> None:
        """Init creates requests.Session."""
        assert provider._session is not None
        assert isinstance(provider._session, requests.Session)

    def test_init_not_authenticated(self, provider: LandsatProvider) -> None:
        """Init sets authenticated to False."""
        assert provider._authenticated is False

    def test_name_property(self, provider: LandsatProvider) -> None:
        """Provider name is 'landsat'."""
        assert provider.name == "landsat"
        assert provider._name == "landsat"


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


class TestLandsatAuthenticate:
    """Test LandsatProvider.authenticate() method."""

    def test_authenticate_succeeds_as_noop(
        self, provider: LandsatProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate succeeds as no-op for public API."""
        provider.authenticate(credentials)

        assert provider._authenticated is True

    def test_authenticate_logs_debug_message(
        self,
        provider: LandsatProvider,
        credentials: ProviderCredentials,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Authenticate logs debug message about public API."""
        with caplog.at_level(logging.DEBUG):
            provider.authenticate(credentials)

        assert "Landsat authentication skipped (public API)" in caplog.text

    def test_authenticate_works_with_empty_credentials(
        self, provider: LandsatProvider
    ) -> None:
        """Authenticate works even with empty credentials."""
        provider.authenticate(ProviderCredentials())

        assert provider._authenticated is True


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------


class TestLandsatSearch:
    """Test LandsatProvider.search() method."""

    def test_search_returns_catalog_entries(self, provider: LandsatProvider) -> None:
        """Search returns list of CatalogEntry objects."""
        location = _make_location(lat=52.23, lon=21.01)
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        entries = provider.search(location, time_range)

        assert len(entries) == 2
        assert all(isinstance(e, CatalogEntry) for e in entries)
        assert all(e.provider == "landsat" for e in entries)

    def test_search_filters_to_landsat_8_9(self, provider: LandsatProvider) -> None:
        """Search only returns Landsat 8/9 results."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        # Create response with mixed platforms
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "features": [
                {
                    "id": "landsat-9-scene",
                    "properties": {"platform": "landsat-9", "eo:cloud_cover": 10},
                    "geometry": {},
                    "assets": {},
                },
                {
                    "id": "landsat-7-scene",
                    "properties": {"platform": "landsat-7", "eo:cloud_cover": 10},
                    "geometry": {},
                    "assets": {},
                },
            ]
        }
        provider._session.post = MagicMock(return_value=resp)

        entries = provider.search(location, time_range)

        assert len(entries) == 1
        assert entries[0].metadata["platform"] == "landsat-9"

    def test_search_sends_correct_stac_query(self, provider: LandsatProvider) -> None:
        """Search sends correct STAC API query."""
        location = _make_location(lat=52.23, lon=21.01)
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        provider.search(location, time_range, cloud_cover_max=0.2)

        call_kwargs = provider._session.post.call_args.kwargs
        assert call_kwargs["json"]["collections"] == [_COLLECTION]
        assert "bbox" in call_kwargs["json"]
        assert call_kwargs["json"]["query"]["eo:cloud_cover"]["lt"] == 20.0

    def test_search_returns_empty_when_no_results(
        self, provider: LandsatProvider
    ) -> None:
        """Search returns empty list when no scenes match."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_empty_search_response())

        entries = provider.search(location, time_range)

        assert entries == []

    def test_search_normalizes_cloud_cover_percentage(
        self, provider: LandsatProvider
    ) -> None:
        """Search normalizes cloud cover from 0-1 to 0-100 for STAC query."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        # 0.3 should be converted to 30%
        provider.search(location, time_range, cloud_cover_max=0.3)

        call_kwargs = provider._session.post.call_args.kwargs
        assert call_kwargs["json"]["query"]["eo:cloud_cover"]["lt"] == 30.0

    def test_search_entry_contains_metadata(self, provider: LandsatProvider) -> None:
        """Search entry contains expected metadata."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        entries = provider.search(location, time_range)

        entry = entries[0]
        assert "platform" in entry.metadata
        assert "wrs_path" in entry.metadata
        assert "wrs_row" in entry.metadata
        assert entry.timestamp != ""
        assert 0.0 <= entry.cloud_cover <= 1.0

    def test_search_raises_on_http_error(self, provider: LandsatProvider) -> None:
        """Search raises ProviderError on HTTP error."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 503
        provider._session.post = MagicMock(return_value=error_resp)

        with pytest.raises(ProviderError) as exc_info:
            provider.search(location, time_range)

        assert "Landsat catalog returned error" in str(exc_info.value)

    def test_search_raises_on_network_error(self, provider: LandsatProvider) -> None:
        """Search raises ProviderError on network error."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(
            side_effect=requests.ConnectionError("Network error")
        )

        with pytest.raises(ProviderError) as exc_info:
            provider.search(location, time_range)

        assert "Landsat catalog search failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Download Tests
# ---------------------------------------------------------------------------


class TestLandsatDownload:
    """Test LandsatProvider.download() method."""

    def test_download_missing_product_id_raises_error(
        self, provider: LandsatProvider
    ) -> None:
        """Download raises ProviderError when product_id is empty."""
        entry = CatalogEntry(provider="landsat", product_id="")

        with pytest.raises(ProviderError) as exc_info:
            provider.download(entry)

        assert "product_id is empty" in str(exc_info.value)

    def test_download_fetches_stac_item(self, provider: LandsatProvider) -> None:
        """Download fetches STAC item for the product."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
            metadata={"platform": "landsat-9"},
        )

        # Mock the fetch and sign requests
        with patch.object(provider, "_fetch_stac_item") as mock_fetch:
            mock_fetch.return_value = _mock_stac_item_response().json()
            with patch.object(provider, "_sign_item") as mock_sign:
                mock_sign.return_value = _mock_sign_response().json()
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.zeros((4, 100, 100), dtype=np.float32),
                        ["B2", "B3", "B4", "B5"],
                    )

                    provider.download(entry)

        mock_fetch.assert_called_once_with("LC09_L2SP_182025_20250615_00")

    def test_download_signs_urls(self, provider: LandsatProvider) -> None:
        """Download signs asset URLs via Planetary Computer."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
            metadata={"platform": "landsat-9"},
        )

        item = _mock_stac_item_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item") as mock_sign:
                mock_sign.return_value = _mock_sign_response().json()
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.zeros((4, 100, 100), dtype=np.float32),
                        ["B2", "B3", "B4", "B5"],
                    )

                    provider.download(entry)

        mock_sign.assert_called_once()

    def test_download_returns_rawdata(self, provider: LandsatProvider) -> None:
        """Download returns RawData with correct structure."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
            metadata={"platform": "landsat-9"},
        )

        item = _mock_stac_item_response().json()
        signed = _mock_sign_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item", return_value=signed):
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.zeros((4, 100, 100), dtype=np.float32),
                        ["B2", "B3", "B4", "B5"],
                    )

                    result = provider.download(entry)

        assert result is not None
        assert isinstance(result.data, np.ndarray)
        assert result.data.shape == (4, 100, 100)
        assert result.metadata["product_id"] == "LC09_L2SP_182025_20250615_00"
        assert result.metadata["bands"] == ["B2", "B3", "B4", "B5"]

    def test_download_with_specific_bands(self, provider: LandsatProvider) -> None:
        """Download respects band selection."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
            metadata={"platform": "landsat-9"},
        )

        item = _mock_stac_item_response().json()
        signed = _mock_sign_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item", return_value=signed):
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.zeros((2, 100, 100), dtype=np.float32),
                        ["B4", "B5"],
                    )

                    provider.download(entry, bands=["B4", "B5"])

        # Verify the bands argument was passed
        mock_download.assert_called_once()
        assert mock_download.call_args[0][1] == ["B4", "B5"]

    def test_download_raises_when_item_not_found(
        self, provider: LandsatProvider
    ) -> None:
        """Download raises ProviderError when STAC item not found."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="nonexistent",
            timestamp="2025-06-15T10:30:00Z",
        )

        with patch.object(provider, "_fetch_stac_item", return_value=None):
            with pytest.raises(ProviderError) as exc_info:
                provider.download(entry)

        assert "Landsat item not found" in str(exc_info.value)

    def test_download_raises_when_signing_fails(
        self, provider: LandsatProvider
    ) -> None:
        """Download raises ProviderError when URL signing fails."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
        )

        item = _mock_stac_item_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item", return_value=None):
                with pytest.raises(ProviderError) as exc_info:
                    provider.download(entry)

        assert "URL signing failed" in str(exc_info.value)

    def test_download_raises_when_no_bands_extracted(
        self, provider: LandsatProvider
    ) -> None:
        """Download raises ProviderError when no bands could be extracted."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
        )

        item = _mock_stac_item_response().json()
        signed = _mock_sign_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item", return_value=signed):
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.array([], dtype=np.float32),
                        [],
                    )

                    with pytest.raises(ProviderError) as exc_info:
                        provider.download(entry)

        assert "no usable data" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Check Status Tests
# ---------------------------------------------------------------------------


class TestLandsatCheckStatus:
    """Test LandsatProvider.check_status() method."""

    def test_check_status_available(self, provider: LandsatProvider) -> None:
        """check_status returns available=True when PC is operational."""
        provider._session.get = MagicMock(return_value=_mock_collection_response())

        status = provider.check_status()

        assert isinstance(status, ProviderStatus)
        assert status.available is True
        assert status.message == ""

    def test_check_status_unavailable_on_error_status(
        self, provider: LandsatProvider
    ) -> None:
        """check_status returns available=False on HTTP error."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 503
        provider._session.get = MagicMock(return_value=error_resp)

        status = provider.check_status()

        assert status.available is False
        assert "503" in status.message

    def test_check_status_unavailable_on_network_error(
        self, provider: LandsatProvider
    ) -> None:
        """check_status returns available=False on network error (never raises)."""
        provider._session.get = MagicMock(
            side_effect=requests.ConnectionError("Network error")
        )

        status = provider.check_status()

        assert status.available is False
        assert "unreachable" in status.message.lower()

    def test_check_status_uses_status_timeout(self, provider: LandsatProvider) -> None:
        """check_status uses short timeout."""
        provider._session.get = MagicMock(return_value=_mock_collection_response())

        provider.check_status()

        call_kwargs = provider._session.get.call_args.kwargs
        assert call_kwargs.get("timeout") == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------


class TestLandsatRetryLogic:
    """Test retry and backoff logic."""

    def test_compute_backoff_increases_exponentially(
        self, provider: LandsatProvider
    ) -> None:
        """Backoff increases exponentially with attempt number."""
        backoff_0 = provider._compute_backoff(0)
        backoff_1 = provider._compute_backoff(1)
        backoff_2 = provider._compute_backoff(2)

        # Allow for jitter
        assert backoff_0 < 2.0  # ~1.0 + jitter
        assert backoff_1 < 3.0  # ~2.0 + jitter
        assert backoff_2 < 5.0  # ~4.0 + jitter

    def test_compute_backoff_respects_max(self, provider: LandsatProvider) -> None:
        """Backoff doesn't exceed max backoff."""
        backoff = provider._compute_backoff(100)  # Very high attempt

        assert backoff <= _MAX_BACKOFF * 1.1  # Allow 10% jitter

    def test_retry_request_exhausts_retries(self, provider: LandsatProvider) -> None:
        """_retry_request raises after exhausting retries."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 500
        provider._session.request = MagicMock(return_value=error_resp)

        with patch("time.sleep"), pytest.raises(ProviderError) as exc_info:
            provider._retry_request("get", "http://test.com")

        assert "after retries" in str(exc_info.value).lower()

    def test_retry_request_succeeds_after_transient_failure(
        self, provider: LandsatProvider
    ) -> None:
        """_retry_request succeeds after transient failures."""
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 500

        success_resp = MagicMock(spec=requests.Response)
        success_resp.status_code = 200

        call_count = [0]

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return fail_resp
            return success_resp

        provider._session.request = MagicMock(side_effect=mock_request)

        with patch("time.sleep"):
            result = provider._retry_request("get", "http://test.com")

        assert result.status_code == 200
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Session Usage Tests
# ---------------------------------------------------------------------------


class TestLandsatSessionUsage:
    """Test all HTTP calls use requests.Session for vcrpy compatibility."""

    def test_search_uses_session(self, provider: LandsatProvider) -> None:
        """search() uses self._session for HTTP calls."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        provider.search(location, time_range)

        provider._session.post.assert_called()

    def test_check_status_uses_session(self, provider: LandsatProvider) -> None:
        """check_status() uses self._session for HTTP calls."""
        provider._session.get = MagicMock(return_value=_mock_collection_response())

        provider.check_status()

        provider._session.get.assert_called()


# ---------------------------------------------------------------------------
# Parse STAC Item Tests
# ---------------------------------------------------------------------------


class TestParseStacItem:
    """Test _parse_stac_item static method."""

    def test_parse_landsat_9_item(self) -> None:
        """Parses Landsat 9 item correctly."""
        item = {
            "id": "LC09_L2SP_182025_20250615_00",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {
                "platform": "landsat-9",
                "datetime": "2025-06-15T10:30:00Z",
                "eo:cloud_cover": 15.5,
                "sun_elevation": 45.2,
            },
            "assets": {
                "blue": {"href": "https://example.com/B2.tif"},
                "red": {"href": "https://example.com/B4.tif"},
            },
        }

        entry = LandsatProvider._parse_stac_item(item)

        assert entry is not None
        assert entry.provider == "landsat"
        assert entry.product_id == "LC09_L2SP_182025_20250615_00"
        assert entry.timestamp == "2025-06-15T10:30:00Z"
        assert entry.cloud_cover == pytest.approx(0.155)
        assert entry.metadata["platform"] == "landsat-9"

    def test_parse_landsat_8_item(self) -> None:
        """Parses Landsat 8 item correctly."""
        item = {
            "id": "LC08_L2SP_182025_20250615_00",
            "geometry": {},
            "properties": {
                "platform": "landsat-8",
                "datetime": "2025-06-15T10:30:00Z",
                "eo:cloud_cover": 5.0,
            },
            "assets": {},
        }

        entry = LandsatProvider._parse_stac_item(item)

        assert entry is not None
        assert entry.metadata["platform"] == "landsat-8"

    def test_parse_rejects_landsat_7(self) -> None:
        """Rejects Landsat 7 items."""
        item = {
            "id": "LE07_L2SP_182025_20250615_00",
            "geometry": {},
            "properties": {
                "platform": "landsat-7",
                "datetime": "2025-06-15T10:30:00Z",
                "eo:cloud_cover": 5.0,
            },
            "assets": {},
        }

        entry = LandsatProvider._parse_stac_item(item)

        assert entry is None

    def test_parse_extracts_available_bands(self) -> None:
        """Extracts available bands from assets."""
        item = {
            "id": "LC09_test",
            "geometry": {},
            "properties": {
                "platform": "landsat-9",
                "datetime": "2025-06-15T10:30:00Z",
                "eo:cloud_cover": 5.0,
            },
            "assets": {
                "blue": {"href": "url"},
                "green": {"href": "url"},
                "red": {"href": "url"},
                "nir08": {"href": "url"},
            },
        }

        entry = LandsatProvider._parse_stac_item(item)

        assert entry is not None
        assert "B2" in entry.bands_available
        assert "B3" in entry.bands_available
        assert "B4" in entry.bands_available
        assert "B5" in entry.bands_available


# ---------------------------------------------------------------------------
# Logging Tests
# ---------------------------------------------------------------------------


class TestLandsatLogging:
    """Test logging behavior."""

    def test_search_logs_debug_info(
        self, provider: LandsatProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """search() logs debug information about the query."""
        location = _make_location()
        time_range = ("2025-06-01", "2025-06-30")

        provider._session.post = MagicMock(return_value=_mock_stac_search_response())

        with caplog.at_level(logging.DEBUG):
            provider.search(location, time_range)

        assert "Searching Landsat catalog" in caplog.text

    def test_download_logs_info(
        self, provider: LandsatProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """download() logs info about the download."""
        entry = CatalogEntry(
            provider="landsat",
            product_id="LC09_L2SP_182025_20250615_00",
            timestamp="2025-06-15T10:30:00Z",
        )

        item = _mock_stac_item_response().json()
        signed = _mock_sign_response().json()
        with patch.object(provider, "_fetch_stac_item", return_value=item):
            with patch.object(provider, "_sign_item", return_value=signed):
                with patch.object(provider, "_download_bands") as mock_download:
                    mock_download.return_value = (
                        np.zeros((4, 100, 100), dtype=np.float32),
                        ["B2", "B3", "B4", "B5"],
                    )

                    with caplog.at_level(logging.INFO):
                        provider.download(entry)

        assert "Downloading Landsat product" in caplog.text
        assert "Download complete" in caplog.text
