"""Tests for the IMGW provider (Story 4.2)."""

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
from satellitehub.providers.imgw import (
    _DEFAULT_TIMEOUT,
    _EARTH_RADIUS_KM,
    _IMGW_API_URL,
    _IMGW_STATIONS,
    _IMGW_SYNOP_URL,
    _INITIAL_BACKOFF,
    _MAX_BACKOFF,
    _MAX_RETRIES,
    _MAX_STATION_DISTANCE_KM,
    _RETRYABLE_STATUS_CODES,
    _STATUS_TIMEOUT,
    _SUCCESS_STATUS_CODES,
    _SUPPORTED_STATION_TYPE,
    _haversine_distance,
    IMGWProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> IMGWProvider:
    """Create an IMGWProvider with default config."""
    return IMGWProvider(config=Config())


@pytest.fixture
def credentials() -> ProviderCredentials:
    """Create empty credentials (IMGW is public)."""
    return ProviderCredentials()


def _make_location(lat: float = 52.23, lon: float = 21.01) -> Any:
    """Create a mock Location object with lat/lon properties (Warsaw default)."""
    loc = MagicMock()
    loc.lat = lat
    loc.lon = lon
    return loc


def _mock_synop_success_response() -> MagicMock:
    """Return a mock Response for successful synoptic data."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "id_stacji": "12375",
        "stacja": "Warszawa",
        "data_pomiaru": "2024-01-15",
        "godzina_pomiaru": "12",
        "temperatura": "5.2",
        "predkosc_wiatru": "12",
        "kierunek_wiatru": "240",
        "wilgotnosc_wzgledna": "78",
        "suma_opadu": "0.0",
        "cisnienie": "1013.2",
    }
    return resp


def _mock_synop_list_response() -> MagicMock:
    """Return a mock Response for all stations list."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = [
        {
            "id_stacji": "12375",
            "stacja": "Warszawa",
            "data_pomiaru": "2024-01-15",
            "godzina_pomiaru": "12",
            "temperatura": "5.2",
            "suma_opadu": "0.0",
        }
    ]
    return resp


# ---------------------------------------------------------------------------
# Task 1: Constants Tests
# ---------------------------------------------------------------------------


class TestIMGWConstants:
    """Test module-level constants are defined correctly."""

    def test_imgw_api_url_defined(self) -> None:
        """IMGW API URL constant is defined."""
        assert _IMGW_API_URL == "https://danepubliczne.imgw.pl/api/data"

    def test_imgw_synop_url_defined(self) -> None:
        """IMGW synop URL constant is defined."""
        assert _IMGW_SYNOP_URL == f"{_IMGW_API_URL}/synop"

    def test_supported_station_type_defined(self) -> None:
        """Supported station type is defined (MVP: synoptic only)."""
        assert _SUPPORTED_STATION_TYPE == "synop"

    def test_timeout_constants_defined(self) -> None:
        """Timeout constants are defined per NFR13."""
        assert _DEFAULT_TIMEOUT == 30
        assert _STATUS_TIMEOUT == 10

    def test_retry_constants_defined(self) -> None:
        """Retry constants are defined per NFR18."""
        assert _MAX_RETRIES == 3
        assert _INITIAL_BACKOFF == 1.0
        assert _MAX_BACKOFF == 60.0
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 200 in _SUCCESS_STATUS_CODES

    def test_station_search_constants_defined(self) -> None:
        """Station search constants are defined."""
        assert _MAX_STATION_DISTANCE_KM == 50.0
        assert _EARTH_RADIUS_KM == 6371.0

    def test_imgw_stations_defined(self) -> None:
        """IMGW stations metadata is defined."""
        assert len(_IMGW_STATIONS) > 0
        assert "12375" in _IMGW_STATIONS  # Warszawa
        assert _IMGW_STATIONS["12375"]["name"] == "Warszawa-Okecie"


# ---------------------------------------------------------------------------
# Task 1: Initialization Tests
# ---------------------------------------------------------------------------


class TestIMGWProviderInit:
    """Test IMGWProvider initialization."""

    def test_init_creates_session(self, provider: IMGWProvider) -> None:
        """Init creates requests.Session."""
        assert provider._session is not None
        assert isinstance(provider._session, requests.Session)

    def test_init_not_authenticated(self, provider: IMGWProvider) -> None:
        """Init sets authenticated to False."""
        assert provider._authenticated is False

    def test_name_property(self, provider: IMGWProvider) -> None:
        """Provider name is 'imgw'."""
        assert provider.name == "imgw"
        assert provider._name == "imgw"


# ---------------------------------------------------------------------------
# Task 2: Authentication Tests
# ---------------------------------------------------------------------------


class TestIMGWAuthenticate:
    """Test IMGWProvider.authenticate() method."""

    def test_authenticate_succeeds_as_noop(
        self, provider: IMGWProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate succeeds as no-op for public API."""
        provider.authenticate(credentials)

        assert provider._authenticated is True

    def test_authenticate_logs_debug_message(
        self,
        provider: IMGWProvider,
        credentials: ProviderCredentials,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Authenticate logs debug message about public API."""
        with caplog.at_level(logging.DEBUG):
            provider.authenticate(credentials)

        assert "IMGW authentication skipped (public API)" in caplog.text

    def test_authenticate_works_with_empty_credentials(
        self, provider: IMGWProvider
    ) -> None:
        """Authenticate works even with empty credentials."""
        provider.authenticate(ProviderCredentials())

        assert provider._authenticated is True


# ---------------------------------------------------------------------------
# Task 3: Haversine Distance Tests
# ---------------------------------------------------------------------------


class TestHaversineDistance:
    """Test Haversine distance calculation."""

    def test_haversine_same_point_is_zero(self) -> None:
        """Distance between same point is zero."""
        distance = _haversine_distance(52.23, 21.01, 52.23, 21.01)

        assert distance == pytest.approx(0.0, abs=0.001)

    def test_haversine_warsaw_to_krakow(self) -> None:
        """Distance Warsaw to Krakow is approximately 250km."""
        # Warsaw: 52.23, 21.01
        # Krakow: 50.06, 19.94
        distance = _haversine_distance(52.23, 21.01, 50.06, 19.94)

        # Actual distance is ~250km
        assert 240 < distance < 260

    def test_haversine_warsaw_to_gdansk(self) -> None:
        """Distance Warsaw to Gdansk is approximately 300km."""
        # Warsaw: 52.23, 21.01
        # Gdansk: 54.35, 18.65
        distance = _haversine_distance(52.23, 21.01, 54.35, 18.65)

        # Actual distance is ~300km
        assert 280 < distance < 320

    def test_haversine_short_distance(self) -> None:
        """Short distances calculated correctly."""
        # Points ~10km apart
        distance = _haversine_distance(52.23, 21.01, 52.32, 21.01)

        assert 9 < distance < 11


# ---------------------------------------------------------------------------
# Task 3 & 4: Find Nearest Stations Tests
# ---------------------------------------------------------------------------


class TestFindNearestStations:
    """Test _find_nearest_stations method."""

    def test_find_stations_near_warsaw(self, provider: IMGWProvider) -> None:
        """Find stations near Warsaw returns Warsaw station."""
        # Warsaw center: 52.23, 21.01
        stations = provider._find_nearest_stations(52.23, 21.01)

        assert len(stations) > 0
        # First station should be Warszawa-Okecie
        station_ids = [s[0] for s in stations]
        assert "12375" in station_ids  # Warszawa

    def test_find_stations_sorted_by_distance(self, provider: IMGWProvider) -> None:
        """Stations are sorted by distance (nearest first)."""
        stations = provider._find_nearest_stations(52.23, 21.01, max_distance_km=100)

        if len(stations) > 1:
            distances = [s[2] for s in stations]
            assert distances == sorted(distances)

    def test_find_stations_respects_max_distance(self, provider: IMGWProvider) -> None:
        """Only returns stations within max_distance_km."""
        # Use very small radius
        stations = provider._find_nearest_stations(52.23, 21.01, max_distance_km=1.0)

        # May return empty if no stations within 1km
        for _, _, distance in stations:
            assert distance <= 1.0

    def test_find_stations_empty_when_no_stations_in_range(
        self, provider: IMGWProvider
    ) -> None:
        """Returns empty list when no stations in range."""
        # Location far from Poland (Atlantic Ocean)
        stations = provider._find_nearest_stations(40.0, -30.0, max_distance_km=50)

        assert stations == []


# ---------------------------------------------------------------------------
# Task 4: Search Tests
# ---------------------------------------------------------------------------


class TestIMGWSearch:
    """Test IMGWProvider.search() method."""

    def test_search_returns_catalog_entries(self, provider: IMGWProvider) -> None:
        """Search returns list of CatalogEntry objects."""
        location = _make_location(lat=52.23, lon=21.01)  # Warsaw
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        assert len(entries) > 0
        assert all(isinstance(e, CatalogEntry) for e in entries)
        assert all(e.provider == "imgw" for e in entries)

    def test_search_entry_contains_station_metadata(
        self, provider: IMGWProvider
    ) -> None:
        """Search entry contains station metadata."""
        location = _make_location(lat=52.23, lon=21.01)
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        entry = entries[0]
        assert "station_id" in entry.metadata
        assert "station_name" in entry.metadata
        assert "distance_km" in entry.metadata
        assert "start_date" in entry.metadata
        assert "end_date" in entry.metadata

    def test_search_returns_empty_when_no_stations_in_range(
        self, provider: IMGWProvider
    ) -> None:
        """Search returns empty list when no stations nearby."""
        # Location far from Poland
        location = _make_location(lat=40.0, lon=-30.0)
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        assert entries == []

    def test_search_respects_custom_max_distance(
        self, provider: IMGWProvider
    ) -> None:
        """Search respects custom max_distance_km parameter."""
        location = _make_location(lat=52.23, lon=21.01)
        time_range = ("2024-01-01", "2024-01-31")

        # Use very small radius
        entries = provider.search(
            location, time_range, max_distance_km=1.0
        )

        # May return empty or fewer stations
        for entry in entries:
            distance = float(entry.metadata["distance_km"])
            assert distance <= 1.0

    def test_search_includes_bands_available(self, provider: IMGWProvider) -> None:
        """Search entries include available measurements."""
        location = _make_location(lat=52.23, lon=21.01)
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        assert entries[0].bands_available is not None
        assert "temperature" in entries[0].bands_available
        assert "precipitation" in entries[0].bands_available


# ---------------------------------------------------------------------------
# Task 5: Download Tests
# ---------------------------------------------------------------------------


class TestIMGWDownload:
    """Test IMGWProvider.download() method."""

    def test_download_retrieves_measurements(self, provider: IMGWProvider) -> None:
        """Download retrieves station measurements."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="synop:12375:2024-01-01:2024-01-31",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa-Okecie",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_success_response()
        )

        result = provider.download(entry)

        assert result is not None
        assert isinstance(result.data, np.ndarray)
        assert result.metadata["provider"] == "imgw"
        assert result.metadata["station_id"] == "12375"

    def test_download_missing_station_id_raises_provider_error(
        self, provider: IMGWProvider
    ) -> None:
        """Download raises ProviderError when station_id missing."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="synop:12375:2024-01-01:2024-01-31",
            metadata={},  # No station_id
        )

        with pytest.raises(ProviderError) as exc_info:
            provider.download(entry)

        assert "Missing station_id" in str(exc_info.value)

    def test_download_handles_list_response(self, provider: IMGWProvider) -> None:
        """Download handles API responses that return a list."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="synop:12375:2024-01-01:2024-01-31",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_list_response()
        )

        result = provider.download(entry)

        assert result is not None
        assert result.metadata["provider"] == "imgw"

    def test_download_with_specific_bands(self, provider: IMGWProvider) -> None:
        """Download filters to specific measurements."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="synop:12375:2024-01-01:2024-01-31",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_success_response()
        )

        result = provider.download(entry, bands=["temperature"])

        assert result is not None
        assert "temperature" in result.metadata["measurements"]

    def test_download_retry_on_transient_failure(
        self, provider: IMGWProvider
    ) -> None:
        """Download retries on transient failures."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="synop:12375:2024-01-01:2024-01-31",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        # First call fails, second succeeds
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 500

        call_count = [0]

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return fail_resp
            return _mock_synop_success_response()

        provider._session.request = MagicMock(side_effect=mock_request)

        with patch("time.sleep"):  # Skip actual sleep
            result = provider.download(entry)

        assert result is not None
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Task 6: Check Status Tests
# ---------------------------------------------------------------------------


class TestIMGWCheckStatus:
    """Test IMGWProvider.check_status() method."""

    def test_check_status_available(self, provider: IMGWProvider) -> None:
        """check_status returns available=True when IMGW is operational."""
        provider._session.get = MagicMock(
            return_value=_mock_synop_list_response()
        )

        status = provider.check_status()

        assert isinstance(status, ProviderStatus)
        assert status.available is True
        assert status.message == ""

    def test_check_status_unavailable_on_error_status(
        self, provider: IMGWProvider
    ) -> None:
        """check_status returns available=False on HTTP error."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 503
        provider._session.get = MagicMock(return_value=error_resp)

        status = provider.check_status()

        assert status.available is False
        assert "503" in status.message

    def test_check_status_unavailable_on_network_error(
        self, provider: IMGWProvider
    ) -> None:
        """check_status returns available=False on network error (never raises)."""
        provider._session.get = MagicMock(
            side_effect=requests.ConnectionError("Network error")
        )

        status = provider.check_status()

        assert status.available is False
        assert "unreachable" in status.message.lower()

    def test_check_status_uses_status_timeout(self, provider: IMGWProvider) -> None:
        """check_status uses short timeout."""
        provider._session.get = MagicMock(
            return_value=_mock_synop_list_response()
        )

        provider.check_status()

        call_kwargs = provider._session.get.call_args.kwargs
        assert call_kwargs.get("timeout") == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# Task 7: Error Propagation Tests
# ---------------------------------------------------------------------------


class TestIMGWErrorPropagation:
    """Test exception propagation."""

    def test_provider_error_propagates(self, provider: IMGWProvider) -> None:
        """ProviderError propagates correctly from download."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="test",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        # Non-retryable error
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 404
        provider._session.request = MagicMock(return_value=error_resp)

        with pytest.raises(ProviderError) as exc_info:
            provider.download(entry)

        assert "IMGW request failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 7: Retry Logic Tests
# ---------------------------------------------------------------------------


class TestIMGWRetryLogic:
    """Test retry and backoff logic."""

    def test_compute_backoff_increases_exponentially(
        self, provider: IMGWProvider
    ) -> None:
        """Backoff increases exponentially with attempt number."""
        backoff_0 = provider._compute_backoff(0)
        backoff_1 = provider._compute_backoff(1)
        backoff_2 = provider._compute_backoff(2)

        # Allow for jitter
        assert backoff_0 < 2.0  # ~1.0 + jitter
        assert backoff_1 < 3.0  # ~2.0 + jitter
        assert backoff_2 < 5.0  # ~4.0 + jitter

    def test_compute_backoff_respects_max(self, provider: IMGWProvider) -> None:
        """Backoff doesn't exceed max backoff."""
        backoff = provider._compute_backoff(100)  # Very high attempt

        assert backoff <= _MAX_BACKOFF * 1.1  # Allow 10% jitter

    def test_retry_request_exhausts_retries(self, provider: IMGWProvider) -> None:
        """_retry_request raises after exhausting retries."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 500
        provider._session.request = MagicMock(return_value=error_resp)

        with patch("time.sleep"):  # Skip actual sleep
            with pytest.raises(ProviderError) as exc_info:
                provider._retry_request("get", "http://test.com")

        assert "after retries" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Task 7: Session Tests (vcrpy compatibility)
# ---------------------------------------------------------------------------


class TestIMGWSessionUsage:
    """Test all HTTP calls use requests.Session for vcrpy compatibility."""

    def test_download_uses_session(self, provider: IMGWProvider) -> None:
        """download() uses self._session for HTTP calls."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="test",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_success_response()
        )

        provider.download(entry)

        provider._session.request.assert_called()

    def test_check_status_uses_session(self, provider: IMGWProvider) -> None:
        """check_status() uses self._session for HTTP calls."""
        provider._session.get = MagicMock(
            return_value=_mock_synop_list_response()
        )

        provider.check_status()

        provider._session.get.assert_called()

    def test_retry_request_uses_session(self, provider: IMGWProvider) -> None:
        """_retry_request() uses self._session for HTTP calls."""
        success_resp = MagicMock(spec=requests.Response)
        success_resp.status_code = 200
        provider._session.request = MagicMock(return_value=success_resp)

        provider._retry_request("get", "http://test.com")

        provider._session.request.assert_called()


# ---------------------------------------------------------------------------
# Code Review Fixes: Logging and Edge Case Tests
# ---------------------------------------------------------------------------


class TestIMGWProgressLogging:
    """Test progress logging per code review Issue #4."""

    def test_download_logs_station_info(
        self,
        provider: IMGWProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """download() logs station info at INFO level."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="test",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa-Okecie",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_success_response()
        )

        with caplog.at_level(logging.INFO):
            provider.download(entry)

        assert "Downloading IMGW data from station 12375" in caplog.text
        assert "Warszawa-Okecie" in caplog.text

    def test_download_logs_mvp_limitation(
        self,
        provider: IMGWProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """download() logs MVP limitation at DEBUG level."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="test",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        provider._session.request = MagicMock(
            return_value=_mock_synop_success_response()
        )

        with caplog.at_level(logging.DEBUG):
            provider.download(entry)

        assert "current observation only" in caplog.text


class TestIMGWEmptyMeasurements:
    """Test edge case of empty measurements per code review Issue #7."""

    def test_parse_measurements_empty_response_logs_warning(
        self,
        provider: IMGWProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_parse_measurements logs warning when all fields are empty."""
        empty_data = {
            "id_stacji": "12375",
            "stacja": "Warszawa",
            "data_pomiaru": "2024-01-15",
            "godzina_pomiaru": "12",
            "temperatura": "",  # Empty
            "suma_opadu": "",  # Empty
            "wilgotnosc_wzgledna": "",  # Empty
            "cisnienie": "",  # Empty
        }

        with caplog.at_level(logging.WARNING):
            result = provider._parse_measurements(empty_data, None)

        assert len(result["data"]) == 0
        assert len(result["fields"]) == 0
        assert "No valid measurements extracted" in caplog.text

    def test_download_handles_empty_measurements(
        self, provider: IMGWProvider
    ) -> None:
        """download() handles response with all empty measurements."""
        entry = CatalogEntry(
            provider="imgw",
            product_id="test",
            metadata={
                "station_id": "12375",
                "station_name": "Warszawa",
                "distance_km": "10.5",
                "lat": "52.1656",
                "lon": "20.9667",
            },
        )

        # Response with all empty fields
        empty_resp = MagicMock(spec=requests.Response)
        empty_resp.status_code = 200
        empty_resp.json.return_value = {
            "id_stacji": "12375",
            "stacja": "Warszawa",
            "data_pomiaru": "2024-01-15",
            "godzina_pomiaru": "12",
            "temperatura": None,
            "suma_opadu": None,
        }

        provider._session.request = MagicMock(return_value=empty_resp)

        result = provider.download(entry)

        # Should succeed but with empty data
        assert result is not None
        assert len(result.data) == 0
        assert result.metadata["measurements"] == []

        provider._session.request.assert_called()
