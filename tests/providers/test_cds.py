"""Tests for the CDS provider (Story 4.1)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.providers.base import (
    CatalogEntry,
    ProviderCredentials,
    ProviderStatus,
)
from satellitehub.providers.cds import (
    _CDS_API_URL,
    _CDS_HOWTO_URL,
    _CDS_PORTAL_URL,
    _CDS_RESOURCES_URL,
    _DEFAULT_TIMEOUT,
    _DEFAULT_VARIABLES,
    _ERA5_PRODUCT,
    _INITIAL_BACKOFF,
    _LOG_INTERVAL,
    _MAX_BACKOFF,
    _MAX_QUEUE_TIME,
    _MAX_RETRIES,
    _POLL_INTERVAL,
    _READ_TIMEOUT,
    _RETRYABLE_STATUS_CODES,
    _STATUS_TIMEOUT,
    _SUCCESS_STATUS_CODES,
    CDSProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> CDSProvider:
    """Create a CDSProvider with default config."""
    return CDSProvider(config=Config())


@pytest.fixture
def credentials() -> ProviderCredentials:
    """Create placeholder CDS credentials with API key."""
    return ProviderCredentials(api_key="test-api-key-12345")


def _make_location(lat: float = 51.25, lon: float = 22.57) -> Any:
    """Create a mock Location object with lat/lon properties."""
    loc = MagicMock()
    loc.lat = lat
    loc.lon = lon
    return loc


def _mock_auth_success_response() -> MagicMock:
    """Return a mock Response for successful API key validation."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"resources": ["reanalysis-era5-single-levels"]}
    return resp


def _mock_auth_failure_response() -> MagicMock:
    """Return a mock Response for failed API key validation (401)."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 401
    resp.json.return_value = {"error": "Invalid API key"}
    return resp


def _mock_submit_success_response() -> MagicMock:
    """Return a mock Response for successful queue submission."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 202
    resp.json.return_value = {"request_id": "task-123-abc"}
    return resp


def _mock_queue_completed_response() -> MagicMock:
    """Return a mock Response for completed queue status."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "state": "completed",
        "location": "https://download.cds.climate.copernicus.eu/result-123",
    }
    return resp


def _mock_queue_running_response() -> MagicMock:
    """Return a mock Response for running queue status."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"state": "running"}
    return resp


def _create_minimal_netcdf_bytes() -> bytes:
    """Create minimal valid NetCDF bytes for testing."""
    import io

    import xarray as xr

    # Create minimal dataset
    times = pd.date_range("2024-01-15", periods=2, freq="6h")
    ds = xr.Dataset(
        {
            "t2m": (["time"], np.array([280.0, 281.0])),
            "tp": (["time"], np.array([0.001, 0.002])),
        },
        coords={"time": times},
    )

    buffer = io.BytesIO()
    ds.to_netcdf(buffer, engine="h5netcdf")
    buffer.seek(0)
    return buffer.read()


def _mock_download_success_response() -> MagicMock:
    """Return a mock Response for successful download with valid NetCDF."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.content = _create_minimal_netcdf_bytes()
    return resp


# ---------------------------------------------------------------------------
# Task 1: Constants Tests
# ---------------------------------------------------------------------------


class TestCDSConstants:
    """Test module-level constants are defined correctly."""

    def test_cds_api_url_defined(self) -> None:
        """CDS API URL constant is defined."""
        assert _CDS_API_URL == "https://cds.climate.copernicus.eu/api"

    def test_cds_resources_url_defined(self) -> None:
        """CDS resources URL constant is defined."""
        assert f"{_CDS_API_URL}/resources" == _CDS_RESOURCES_URL

    def test_timeout_constants_defined(self) -> None:
        """Timeout constants are defined per NFR13."""
        assert _DEFAULT_TIMEOUT == 30
        assert _STATUS_TIMEOUT == 10
        assert _READ_TIMEOUT == 300

    def test_retry_constants_defined(self) -> None:
        """Retry constants are defined per NFR18."""
        assert _MAX_RETRIES == 3
        assert _INITIAL_BACKOFF == 1.0
        assert _MAX_BACKOFF == 60.0
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 200 in _SUCCESS_STATUS_CODES
        assert 202 in _SUCCESS_STATUS_CODES

    def test_queue_constants_defined(self) -> None:
        """Queue polling constants are defined per NFR14."""
        assert _POLL_INTERVAL == 5.0
        assert _MAX_QUEUE_TIME == 120.0
        assert _LOG_INTERVAL == 30.0

    def test_era5_product_constants_defined(self) -> None:
        """ERA5 product constants are defined."""
        assert _ERA5_PRODUCT == "reanalysis-era5-single-levels"
        assert "2m_temperature" in _DEFAULT_VARIABLES


# ---------------------------------------------------------------------------
# Task 1: Initialization Tests
# ---------------------------------------------------------------------------


class TestCDSProviderInit:
    """Test CDSProvider initialization."""

    def test_init_creates_session(self, provider: CDSProvider) -> None:
        """Init creates requests.Session."""
        assert provider._session is not None
        assert isinstance(provider._session, requests.Session)

    def test_init_api_key_empty(self, provider: CDSProvider) -> None:
        """Init sets api_key to empty string."""
        assert provider._api_key == ""

    def test_init_not_authenticated(self, provider: CDSProvider) -> None:
        """Init sets authenticated to False."""
        assert provider._authenticated is False

    def test_name_property(self, provider: CDSProvider) -> None:
        """Provider name is 'cds'."""
        assert provider.name == "cds"
        assert provider._name == "cds"


# ---------------------------------------------------------------------------
# Task 2: Authentication Tests
# ---------------------------------------------------------------------------


class TestCDSAuthenticate:
    """Test CDSProvider.authenticate() method."""

    def test_authenticate_valid_key_succeeds(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate succeeds with valid API key."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        provider.authenticate(credentials)

        assert provider._authenticated is True
        assert provider._api_key == "test-api-key-12345"
        assert "PRIVATE-TOKEN" in provider._session.headers

    def test_authenticate_missing_key_raises_configuration_error(
        self, provider: CDSProvider
    ) -> None:
        """Authenticate raises ConfigurationError when api_key is missing."""
        credentials = ProviderCredentials()  # No api_key

        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        assert "CDS API key not provided" in str(exc_info.value)
        assert provider._authenticated is False

    def test_authenticate_invalid_key_raises_configuration_error(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate raises ConfigurationError for invalid API key."""
        provider._session.get = MagicMock(return_value=_mock_auth_failure_response())

        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        assert "CDS authentication failed" in str(exc_info.value)
        assert _CDS_HOWTO_URL in str(exc_info.value)
        assert provider._authenticated is False

    def test_authenticate_network_error_raises_configuration_error(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate raises ConfigurationError on network error."""
        provider._session.get = MagicMock(
            side_effect=requests.ConnectionError("Network unreachable")
        )

        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        assert "Cannot reach CDS API" in str(exc_info.value)
        assert _CDS_PORTAL_URL in str(exc_info.value)
        assert provider._authenticated is False

    def test_authenticate_uses_status_timeout(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Authenticate uses short timeout for validation request."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        provider.authenticate(credentials)

        provider._session.get.assert_called_once()
        call_kwargs = provider._session.get.call_args.kwargs
        assert call_kwargs.get("timeout") == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# Task 3: Search Tests
# ---------------------------------------------------------------------------


class TestCDSSearch:
    """Test CDSProvider.search() method."""

    def test_search_returns_catalog_entries(self, provider: CDSProvider) -> None:
        """Search returns list of CatalogEntry objects."""
        location = _make_location()
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        assert len(entries) == 1
        assert isinstance(entries[0], CatalogEntry)
        assert entries[0].provider == "cds"

    def test_search_entry_contains_product_info(self, provider: CDSProvider) -> None:
        """Search entry contains correct product information."""
        location = _make_location(lat=51.25, lon=22.57)
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        entry = entries[0]
        assert entry.metadata["product"] == _ERA5_PRODUCT
        assert entry.metadata["start_date"] == "2024-01-01"
        assert entry.metadata["end_date"] == "2024-01-31"
        assert entry.metadata["lat"] == "51.25"
        assert entry.metadata["lon"] == "22.57"

    def test_search_entry_contains_default_variables(
        self, provider: CDSProvider
    ) -> None:
        """Search entry includes default variables in bands_available."""
        location = _make_location()
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        assert entries[0].bands_available == _DEFAULT_VARIABLES

    def test_search_custom_variables(self, provider: CDSProvider) -> None:
        """Search accepts custom variables parameter."""
        location = _make_location()
        time_range = ("2024-01-01", "2024-01-31")
        custom_vars = ["10m_u_component_of_wind", "surface_pressure"]

        entries = provider.search(location, time_range, variables=custom_vars)

        assert entries[0].bands_available == custom_vars

    def test_search_always_returns_one_entry_era5_global_coverage(
        self, provider: CDSProvider
    ) -> None:
        """Search always returns one entry because ERA5 has global coverage.

        ERA5 is a reanalysis product with global coverage from 1940 to present.
        Data availability is verified at download time by the CDS API, not
        during search. This is by design per AC#2 clarification.
        """
        location = _make_location()
        time_range = ("2024-01-01", "2024-01-31")

        entries = provider.search(location, time_range)

        # ERA5 always returns one entry - actual availability checked at download
        assert len(entries) == 1
        assert entries[0].provider == "cds"


# ---------------------------------------------------------------------------
# Task 4: Download Tests
# ---------------------------------------------------------------------------


class TestCDSDownload:
    """Test CDSProvider.download() method."""

    def test_download_requires_authentication(self, provider: CDSProvider) -> None:
        """Download raises ProviderError if not authenticated."""
        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        with pytest.raises(ProviderError) as exc_info:
            provider.download(entry)

        assert "CDS download requires authentication" in str(exc_info.value)

    def test_download_queue_submission_and_polling(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Download submits to queue and polls for completion."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        # Setup mock responses for download flow
        provider._session.request = MagicMock(
            return_value=_mock_submit_success_response()
        )
        provider._session.get = MagicMock(
            side_effect=[
                _mock_queue_completed_response(),
                _mock_download_success_response(),
            ]
        )

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            bands_available=["2m_temperature"],
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        # Mock _retry_request to return success responses
        with patch.object(provider, "_retry_request") as mock_retry:
            mock_retry.side_effect = [
                _mock_submit_success_response(),
                _mock_download_success_response(),
            ]

            result = provider.download(entry)

        assert result is not None
        assert isinstance(result.data, np.ndarray)
        assert result.metadata["provider"] == "cds"

    def test_download_queue_timeout_raises_provider_error(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Download raises ProviderError when queue times out."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        # Mock _submit_request to return a task ID
        with (
            patch.object(provider, "_submit_request", return_value="task-123"),
            patch.object(provider, "_poll_queue") as mock_poll,
        ):
            mock_poll.side_effect = ProviderError(
                what="ERA5 request timed out",
                cause="CDS queue exceeded 2-minute budget",
                fix="Try again later or during off-peak hours",
            )

            with pytest.raises(ProviderError) as exc_info:
                provider.download(entry)

                assert "ERA5 request timed out" in str(exc_info.value)
                assert "2-minute budget" in str(exc_info.value)

    def test_download_retry_on_transient_failure(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Download retries on transient failures during queue submission."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        # Create response that fails then succeeds
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 500

        success_resp = _mock_submit_success_response()

        # Mock session.request to fail first, then succeed
        call_count = [0]

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return fail_resp
            return success_resp

        provider._session.request = MagicMock(side_effect=mock_request)

        # Test retry logic
        with patch("time.sleep"):  # Skip actual sleep
            resp = provider._retry_request("post", "http://test.com")

        assert resp.status_code == 202
        assert call_count[0] == 2  # One failure, one success

    def test_download_full_flow_with_retry(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """Full download flow handles retry correctly."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            bands_available=["2m_temperature"],
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-15",
                "end_date": "2024-01-15",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        # Mock the full flow with retry on submit
        fail_resp = MagicMock(spec=requests.Response)
        fail_resp.status_code = 503

        with patch.object(provider, "_retry_request") as mock_retry:
            mock_retry.side_effect = [
                _mock_submit_success_response(),  # Submit succeeds
                _mock_download_success_response(),  # Download succeeds
            ]
            provider._session.get = MagicMock(
                return_value=_mock_queue_completed_response()
            )

            result = provider.download(entry)

        assert result is not None
        assert result.metadata["provider"] == "cds"


# ---------------------------------------------------------------------------
# Task 5: Check Status Tests
# ---------------------------------------------------------------------------


class TestCDSCheckStatus:
    """Test CDSProvider.check_status() method."""

    def test_check_status_available(self, provider: CDSProvider) -> None:
        """check_status returns available=True when CDS is operational."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        status = provider.check_status()

        assert isinstance(status, ProviderStatus)
        assert status.available is True
        assert status.message == ""

    def test_check_status_unavailable_on_error_status(
        self, provider: CDSProvider
    ) -> None:
        """check_status returns available=False on HTTP error."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 503
        provider._session.get = MagicMock(return_value=error_resp)

        status = provider.check_status()

        assert status.available is False
        assert "503" in status.message

    def test_check_status_unavailable_on_network_error(
        self, provider: CDSProvider
    ) -> None:
        """check_status returns available=False on network error (never raises)."""
        provider._session.get = MagicMock(
            side_effect=requests.ConnectionError("Network error")
        )

        status = provider.check_status()

        assert status.available is False
        assert "unreachable" in status.message.lower()

    def test_check_status_uses_status_timeout(self, provider: CDSProvider) -> None:
        """check_status uses short timeout."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        provider.check_status()

        call_kwargs = provider._session.get.call_args.kwargs
        assert call_kwargs.get("timeout") == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# Task 6: Error Propagation Tests
# ---------------------------------------------------------------------------


class TestCDSErrorPropagation:
    """Test exception propagation."""

    def test_provider_error_propagates(self, provider: CDSProvider) -> None:
        """ProviderError propagates correctly."""
        provider._authenticated = True

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        with patch.object(provider, "_submit_request") as mock_submit:
            mock_submit.side_effect = ProviderError(
                what="Test error",
                cause="Test cause",
                fix="Test fix",
            )

            with pytest.raises(ProviderError) as exc_info:
                provider.download(entry)

            assert "Test error" in str(exc_info.value)

    def test_configuration_error_propagates(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """ConfigurationError propagates correctly."""
        provider._session.get = MagicMock(
            side_effect=requests.Timeout("Connection timed out")
        )

        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        assert "Cannot reach CDS API" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 6: Retry Logic Tests
# ---------------------------------------------------------------------------


class TestCDSRetryLogic:
    """Test retry and backoff logic."""

    def test_compute_backoff_increases_exponentially(
        self, provider: CDSProvider
    ) -> None:
        """Backoff increases exponentially with attempt number."""
        backoff_0 = provider._compute_backoff(0)
        backoff_1 = provider._compute_backoff(1)
        backoff_2 = provider._compute_backoff(2)

        # Allow for jitter
        assert backoff_0 < 2.0  # ~1.0 + jitter
        assert backoff_1 < 3.0  # ~2.0 + jitter
        assert backoff_2 < 5.0  # ~4.0 + jitter

    def test_compute_backoff_respects_max(self, provider: CDSProvider) -> None:
        """Backoff doesn't exceed max backoff."""
        backoff = provider._compute_backoff(100)  # Very high attempt

        assert backoff <= _MAX_BACKOFF * 1.1  # Allow 10% jitter

    def test_retry_request_exhausts_retries(self, provider: CDSProvider) -> None:
        """_retry_request raises after exhausting retries."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 500
        provider._session.request = MagicMock(return_value=error_resp)

        with patch("time.sleep"), pytest.raises(ProviderError) as exc_info:
            provider._retry_request("get", "http://test.com")

        assert "after retries" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Task 6: Session Tests (vcrpy compatibility)
# ---------------------------------------------------------------------------


class TestCDSSessionUsage:
    """Test all HTTP calls use requests.Session for vcrpy compatibility."""

    def test_authenticate_uses_session(
        self, provider: CDSProvider, credentials: ProviderCredentials
    ) -> None:
        """authenticate() uses self._session for HTTP calls."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        provider.authenticate(credentials)

        provider._session.get.assert_called()

    def test_check_status_uses_session(self, provider: CDSProvider) -> None:
        """check_status() uses self._session for HTTP calls."""
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())

        provider.check_status()

        provider._session.get.assert_called()

    def test_retry_request_uses_session(self, provider: CDSProvider) -> None:
        """_retry_request() uses self._session for HTTP calls."""
        success_resp = MagicMock(spec=requests.Response)
        success_resp.status_code = 200
        provider._session.request = MagicMock(return_value=success_resp)

        provider._retry_request("get", "http://test.com")

        provider._session.request.assert_called()


# ---------------------------------------------------------------------------
# Issue #2: Progress Logging Tests (AC#3)
# ---------------------------------------------------------------------------


class TestCDSProgressLogging:
    """Test progress logging per AC#3 and AD-7."""

    def test_download_logs_queue_submission(
        self,
        provider: CDSProvider,
        credentials: ProviderCredentials,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """download() logs queue submission at INFO level."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            bands_available=["2m_temperature"],
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-15",
                "end_date": "2024-01-15",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        with patch.object(provider, "_retry_request") as mock_retry:
            mock_retry.side_effect = [
                _mock_submit_success_response(),
                _mock_download_success_response(),
            ]
            provider._session.get = MagicMock(
                return_value=_mock_queue_completed_response()
            )

            with caplog.at_level(logging.INFO):
                provider.download(entry)

        assert "Submitted ERA5 request, waiting for queue..." in caplog.text

    def test_poll_queue_logs_initial_message(
        self,
        provider: CDSProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_poll_queue() logs initial polling message at INFO level."""
        provider._session.get = MagicMock(
            return_value=_mock_queue_completed_response()
        )

        with caplog.at_level(logging.INFO):
            provider._poll_queue("task-123")

        assert "ERA5 request queued, polling for completion..." in caplog.text


# ---------------------------------------------------------------------------
# Issue #5: Date Range Optimization Tests
# ---------------------------------------------------------------------------


class TestCDSBuildRequest:
    """Test _build_request date range handling."""

    def test_build_request_single_day_uses_single_day(
        self, provider: CDSProvider
    ) -> None:
        """Single day request only includes that day."""
        request = provider._build_request(
            product=_ERA5_PRODUCT,
            variables=["2m_temperature"],
            start_date="2024-03-15",
            end_date="2024-03-15",
            lat=51.25,
            lon=22.57,
        )

        assert request["year"] == ["2024"]
        assert request["month"] == ["03"]
        assert request["day"] == ["15"]

    def test_build_request_single_month_uses_correct_days(
        self, provider: CDSProvider
    ) -> None:
        """Request within single month includes only days in range."""
        request = provider._build_request(
            product=_ERA5_PRODUCT,
            variables=["2m_temperature"],
            start_date="2024-03-10",
            end_date="2024-03-20",
            lat=51.25,
            lon=22.57,
        )

        assert request["year"] == ["2024"]
        assert request["month"] == ["03"]
        assert request["day"] == [f"{d:02d}" for d in range(10, 21)]

    def test_build_request_multi_month_uses_all_days(
        self, provider: CDSProvider
    ) -> None:
        """Request spanning months includes all days (CDS handles filtering)."""
        request = provider._build_request(
            product=_ERA5_PRODUCT,
            variables=["2m_temperature"],
            start_date="2024-03-15",
            end_date="2024-04-15",
            lat=51.25,
            lon=22.57,
        )

        assert request["year"] == ["2024"]
        assert request["month"] == ["03", "04"]
        assert request["day"] == [f"{d:02d}" for d in range(1, 32)]

    def test_build_request_multi_year_uses_all_months(
        self, provider: CDSProvider
    ) -> None:
        """Request spanning years includes all months (CDS handles filtering)."""
        request = provider._build_request(
            product=_ERA5_PRODUCT,
            variables=["2m_temperature"],
            start_date="2023-11-01",
            end_date="2024-02-28",
            lat=51.25,
            lon=22.57,
        )

        assert request["year"] == ["2023", "2024"]
        assert request["month"] == [f"{m:02d}" for m in range(1, 13)]


# ---------------------------------------------------------------------------
# Story 4.3: NetCDF Parsing Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_netcdf_bytes() -> bytes:
    """Create sample NetCDF bytes for testing using xarray.

    Generates a minimal NetCDF file with temperature (t2m) and
    precipitation (tp) variables over a small grid and time range.
    """
    import xarray as xr

    # Create time coordinates (4 timestamps over 2 days)
    times = pd.date_range("2024-01-15", periods=4, freq="6h")

    # Create spatial coordinates (small 2x2 grid)
    lats = np.array([51.0, 51.5])
    lons = np.array([22.0, 22.5])

    # Create temperature data (Kelvin - will be converted to Celsius)
    # Values around 280K = ~7C
    t2m_data = np.array(
        [
            [[278.0, 279.0], [280.0, 281.0]],  # time 0
            [[279.0, 280.0], [281.0, 282.0]],  # time 1
            [[280.0, 281.0], [282.0, 283.0]],  # time 2
            [[279.5, 280.5], [281.5, 282.5]],  # time 3
        ]
    )

    # Create precipitation data (meters - small values)
    tp_data = np.array(
        [
            [[0.001, 0.002], [0.001, 0.002]],  # time 0
            [[0.0, 0.0], [0.0, 0.0]],  # time 1
            [[0.003, 0.004], [0.003, 0.004]],  # time 2
            [[0.001, 0.001], [0.001, 0.001]],  # time 3
        ]
    )

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], t2m_data),
            "tp": (["time", "latitude", "longitude"], tp_data),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )

    # Add attributes for realism
    ds["t2m"].attrs = {"units": "K", "long_name": "2 metre temperature"}
    ds["tp"].attrs = {"units": "m", "long_name": "Total precipitation"}

    # Write to bytes
    import io

    buffer = io.BytesIO()
    ds.to_netcdf(buffer, engine="h5netcdf")
    buffer.seek(0)
    return buffer.read()


class TestCDSNetCDFParsing:
    """Tests for NetCDF parsing functionality (Story 4.3)."""

    def test_parse_netcdf_extracts_timestamps(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf extracts timestamps from NetCDF data."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        timestamps = result.metadata.get("timestamps", [])
        assert len(timestamps) == 4
        # Check timestamps are ISO-8601 format
        assert "2024-01-15" in timestamps[0]
        assert timestamps[0].endswith("Z")

    def test_parse_netcdf_extracts_temperature_in_celsius(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf converts temperature from Kelvin to Celsius."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        variables = result.metadata.get("variables", [])
        assert "2m_temperature" in variables

        # Get temperature column index
        temp_idx = variables.index("2m_temperature")

        # Temperature values should be in Celsius (around 5-10C, not 280K)
        temp_values = result.data[:, temp_idx]
        assert np.all(temp_values > -50)  # Not Kelvin
        assert np.all(temp_values < 50)  # Reasonable Celsius range

        # Verify units in metadata
        units = result.metadata.get("units", {})
        assert units.get("2m_temperature") == "celsius"

    def test_parse_netcdf_computes_spatial_mean(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf computes spatial mean over lat/lon dimensions."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        # Result should have shape (time, variables) = (4, 2)
        assert result.data.shape == (4, 2)

        # Values should be the mean of the 2x2 grid at each timestamp
        # First timestamp t2m: mean of [[278, 279], [280, 281]] = 279.5K = 6.35C
        variables = result.metadata.get("variables", [])
        temp_idx = variables.index("2m_temperature")
        first_temp = result.data[0, temp_idx]

        # Expected: 279.5K - 273.15 = 6.35C
        expected_temp = 279.5 - 273.15
        assert abs(first_temp - expected_temp) < 0.1

    def test_parse_netcdf_includes_variables_in_metadata(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf includes variable names in metadata."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        variables = result.metadata.get("variables", [])
        assert "2m_temperature" in variables
        assert "total_precipitation" in variables
        assert len(variables) == 2

    def test_parse_netcdf_includes_units_in_metadata(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf includes unit information in metadata."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        units = result.metadata.get("units", {})
        assert units.get("2m_temperature") == "celsius"
        assert units.get("total_precipitation") == "m"

    def test_parse_netcdf_returns_rawdata_type(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf returns a RawData object."""
        from satellitehub._types import RawData

        result = provider._parse_netcdf(sample_netcdf_bytes)

        assert isinstance(result, RawData)
        assert isinstance(result.data, np.ndarray)
        assert isinstance(result.metadata, dict)

    def test_parse_netcdf_data_is_float32(
        self, provider: CDSProvider, sample_netcdf_bytes: bytes
    ) -> None:
        """_parse_netcdf returns data as float32 array."""
        result = provider._parse_netcdf(sample_netcdf_bytes)

        assert result.data.dtype == np.float32

    def test_extract_era5_data_handles_missing_variables(
        self, provider: CDSProvider
    ) -> None:
        """_extract_era5_data handles datasets with unknown variable names."""
        import xarray as xr

        # Create dataset with non-standard variable name
        times = pd.date_range("2024-01-15", periods=2, freq="6h")
        ds = xr.Dataset(
            {
                "unknown_var": (["time"], np.array([1.0, 2.0])),
            },
            coords={"time": times},
        )

        result = provider._extract_era5_data(ds)

        # Should fall back to using the unknown variable
        assert result.data.shape[0] == 2
        assert "unknown_var" in result.metadata.get("variables", [])


class TestCDSDownloadWithNetCDF:
    """Integration tests for download with NetCDF parsing."""

    def test_download_returns_parsed_data(
        self,
        provider: CDSProvider,
        credentials: ProviderCredentials,
        sample_netcdf_bytes: bytes,
    ) -> None:
        """Full download flow returns parsed NetCDF data."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            bands_available=["2m_temperature", "total_precipitation"],
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-15",
                "end_date": "2024-01-16",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        # Mock responses
        download_resp = MagicMock(spec=requests.Response)
        download_resp.status_code = 200
        download_resp.content = sample_netcdf_bytes

        with patch.object(provider, "_retry_request") as mock_retry:
            mock_retry.side_effect = [
                _mock_submit_success_response(),
                download_resp,
            ]
            provider._session.get = MagicMock(
                return_value=_mock_queue_completed_response()
            )

            result = provider.download(entry)

        # Verify parsed data structure
        assert result is not None
        assert result.metadata["provider"] == "cds"
        assert "timestamps" in result.metadata
        assert "variables" in result.metadata
        assert result.data.shape[1] == 2  # 2 variables

    def test_download_merges_provider_metadata(
        self,
        provider: CDSProvider,
        credentials: ProviderCredentials,
        sample_netcdf_bytes: bytes,
    ) -> None:
        """Download merges parsed metadata with provider context."""
        # Setup authenticated provider
        provider._session.get = MagicMock(return_value=_mock_auth_success_response())
        provider.authenticate(credentials)

        entry = CatalogEntry(
            provider="cds",
            product_id="test",
            bands_available=["2m_temperature"],
            metadata={
                "product": _ERA5_PRODUCT,
                "start_date": "2024-01-15",
                "end_date": "2024-01-16",
                "lat": "51.25",
                "lon": "22.57",
            },
        )

        download_resp = MagicMock(spec=requests.Response)
        download_resp.status_code = 200
        download_resp.content = sample_netcdf_bytes

        with patch.object(provider, "_retry_request") as mock_retry:
            mock_retry.side_effect = [
                _mock_submit_success_response(),
                download_resp,
            ]
            provider._session.get = MagicMock(
                return_value=_mock_queue_completed_response()
            )

            result = provider.download(entry)

        # Check merged metadata
        assert result.metadata["provider"] == "cds"
        assert result.metadata["product"] == _ERA5_PRODUCT
        assert result.metadata["time_range"] == ("2024-01-15", "2024-01-16")
        assert "bounds" in result.metadata
        assert "crs" in result.metadata
        # Parsed metadata from NetCDF
        assert "timestamps" in result.metadata
        assert "variables" in result.metadata
        assert "units" in result.metadata
