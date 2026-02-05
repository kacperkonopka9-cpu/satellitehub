"""Tests for the CDSE provider (Stories 2.4 and 2.5)."""

from __future__ import annotations

import io
import random
import time
import zipfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.providers.base import (
    CatalogEntry,
    ProviderCredentials,
    ProviderStatus,
)
from satellitehub.providers.cdse import (
    _BAND_RESOLUTION,
    _CATALOG_BASE_URL,
    _CATALOG_URL,
    _CDSE_PORTAL_URL,
    _CLIENT_ID,
    _DEFAULT_TIMEOUT,
    _DOWNLOAD_URL,
    _INITIAL_BACKOFF,
    _MAX_RETRIES,
    _READ_TIMEOUT,
    _RETRYABLE_STATUS_CODES,
    _S2_L2A_BANDS,
    _STATUS_TIMEOUT,
    _TOKEN_URL,
    CDSEProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> CDSEProvider:
    """Create a CDSEProvider with default config."""
    return CDSEProvider(config=Config())


@pytest.fixture
def credentials() -> ProviderCredentials:
    """Create placeholder CDSE credentials."""
    return ProviderCredentials(username="placeholder", password="placeholder")


def _make_location(lat: float = 51.25, lon: float = 22.57) -> Any:
    """Create a mock Location object with lat/lon properties."""
    loc = MagicMock()
    loc.lat = lat
    loc.lon = lon
    return loc


def _mock_auth_success_response() -> MagicMock:
    """Return a mock Response for successful OAuth2 auth."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "mock-token-abc123",
        "expires_in": 600,
        "refresh_token": "mock-refresh-token",
        "token_type": "Bearer",
    }
    return resp


def _mock_auth_failure_response() -> MagicMock:
    """Return a mock Response for failed OAuth2 auth (401)."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 401
    resp.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Invalid user credentials",
    }
    return resp


def _mock_auth_malformed_response() -> MagicMock:
    """Return a mock Response with missing access_token."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {"token_type": "Bearer"}
    return resp


_SAMPLE_ODATA_PRODUCT: dict[str, Any] = {
    "Id": "b3e1f2a4-5678-9abc-def0-123456789abc",
    "Name": "S2B_MSIL2A_20240115T100219_N0510_R122_T33UXP_20240115T131254.SAFE",
    "ContentLength": 1073741824,
    "ContentDate": {
        "Start": "2024-01-15T10:02:19.024Z",
        "End": "2024-01-15T10:02:19.024Z",
    },
    "GeoFootprint": {
        "type": "Polygon",
        "coordinates": [
            [
                [22.0, 51.0],
                [23.0, 51.0],
                [23.0, 52.0],
                [22.0, 52.0],
                [22.0, 51.0],
            ]
        ],
    },
    "Online": True,
    "S3Path": "/eodata/Sentinel-2/MSI/L2A/2024/01/15/S2B_MSIL2A.SAFE",
}


def _mock_search_success_response(
    products: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Return a mock Response for successful catalog search."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "@odata.context": "$metadata#Products",
        "value": products if products is not None else [_SAMPLE_ODATA_PRODUCT],
    }
    return resp


def _mock_search_empty_response() -> MagicMock:
    """Return a mock Response for catalog search with no results."""
    return _mock_search_success_response(products=[])


def _mock_status_success_response() -> MagicMock:
    """Return a mock Response for CDSE health check."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    return resp


def _mock_status_failure_response() -> MagicMock:
    """Return a mock Response for CDSE health check failure."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 503
    return resp


# ---------------------------------------------------------------------------
# Task 6: OAuth2 Authentication Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCDSEAuthentication:
    """Tests for CDSEProvider.authenticate()."""

    def test_successful_auth_stores_token(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful auth stores token and creates session with Bearer header."""
        monkeypatch.setattr(
            provider._session, "post", lambda *a, **kw: _mock_auth_success_response()
        )
        provider.authenticate(credentials)

        assert provider._token == "mock-token-abc123"
        assert provider._session is not None
        assert provider._session.headers["Authorization"] == "Bearer mock-token-abc123"

    def test_invalid_credentials_raises_configuration_error(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP 401 raises ConfigurationError with three-part message."""
        monkeypatch.setattr(
            provider._session, "post", lambda *a, **kw: _mock_auth_failure_response()
        )
        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        err = exc_info.value
        assert err.what != ""
        assert err.cause != ""
        assert err.fix != ""
        assert "authentication failed" in err.what.lower()

    def test_network_error_raises_configuration_error(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Network error raises ConfigurationError with network cause."""

        def raise_connection_error(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("Connection refused")

        monkeypatch.setattr(provider._session, "post", raise_connection_error)

        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        err = exc_info.value
        assert "Cannot reach" in err.what
        assert "Connection refused" in err.cause

    def test_malformed_response_raises_configuration_error(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing access_token in response raises ConfigurationError."""
        monkeypatch.setattr(
            provider._session, "post", lambda *a, **kw: _mock_auth_malformed_response()
        )
        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        err = exc_info.value
        assert "unexpected auth response" in err.what.lower()
        assert "access_token" in err.cause.lower()

    def test_error_includes_cdse_portal_link(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Error fix field includes CDSE portal URL."""
        monkeypatch.setattr(
            provider._session, "post", lambda *a, **kw: _mock_auth_failure_response()
        )
        with pytest.raises(ConfigurationError) as exc_info:
            provider.authenticate(credentials)

        assert _CDSE_PORTAL_URL in exc_info.value.fix

    def test_timeout_is_set_on_request(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auth request uses 30-second timeout (NFR13)."""
        captured_kwargs: dict[str, Any] = {}

        def capture_post(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_auth_success_response()

        monkeypatch.setattr(provider._session, "post", capture_post)
        provider.authenticate(credentials)

        assert captured_kwargs["timeout"] == _DEFAULT_TIMEOUT

    def test_auth_posts_correct_data(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auth POST sends correct grant_type, client_id, credentials."""
        captured_kwargs: dict[str, Any] = {}

        def capture_post(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_auth_success_response()

        monkeypatch.setattr(provider._session, "post", capture_post)
        provider.authenticate(credentials)

        data = captured_kwargs["data"]
        assert data["grant_type"] == "password"
        assert data["client_id"] == _CLIENT_ID
        assert data["username"] == "placeholder"
        assert data["password"] == "placeholder"

    def test_auth_posts_to_correct_url(
        self,
        provider: CDSEProvider,
        credentials: ProviderCredentials,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auth POST targets the correct CDSE token URL."""
        captured_args: list[Any] = []

        def capture_post(*args: Any, **kwargs: Any) -> MagicMock:
            captured_args.extend(args)
            return _mock_auth_success_response()

        monkeypatch.setattr(provider._session, "post", capture_post)
        provider.authenticate(credentials)

        assert captured_args[0] == _TOKEN_URL


# ---------------------------------------------------------------------------
# Task 7: OData Catalog Search Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCDSESearch:
    """Tests for CDSEProvider.search()."""

    def test_search_returns_catalog_entries(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search with results returns list of CatalogEntry objects."""
        monkeypatch.setattr(
            provider._session, "get", lambda *a, **kw: _mock_search_success_response()
        )
        location = _make_location()
        entries = provider.search(location, ("2024-01-01", "2024-01-31"))

        assert len(entries) == 1
        assert isinstance(entries[0], CatalogEntry)
        assert entries[0].provider == "cdse"
        assert entries[0].product_id == _SAMPLE_ODATA_PRODUCT["Id"]

    def test_search_empty_result_returns_empty_list(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search with no matching data returns empty list, no exception."""
        monkeypatch.setattr(
            provider._session, "get", lambda *a, **kw: _mock_search_empty_response()
        )
        location = _make_location()
        entries = provider.search(location, ("2024-01-01", "2024-01-31"))

        assert entries == []

    def test_search_builds_correct_filter(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search builds OData filter with collection, type, spatial, date."""
        captured_kwargs: dict[str, Any] = {}

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_search_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        location = _make_location(lat=51.25, lon=22.57)
        provider.search(location, ("2024-01-01", "2024-01-31"))

        filter_str = captured_kwargs["params"]["$filter"]
        assert "Collection/Name eq 'SENTINEL-2'" in filter_str
        assert "S2MSI2A" in filter_str
        assert "22.57" in filter_str
        assert "51.25" in filter_str
        assert "2024-01-01" in filter_str
        assert "2024-01-31" in filter_str

    def test_cloud_cover_param_applied_to_filter(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cloud_cover_max param adds cloud cover filter clause."""
        captured_kwargs: dict[str, Any] = {}

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_search_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        location = _make_location()
        provider.search(location, ("2024-01-01", "2024-01-31"), cloud_cover_max=0.2)

        filter_str = captured_kwargs["params"]["$filter"]
        assert "cloudCover" in filter_str
        assert "20" in filter_str

    def test_catalog_entry_fields_mapped_correctly(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OData response fields are correctly mapped to CatalogEntry."""
        monkeypatch.setattr(
            provider._session, "get", lambda *a, **kw: _mock_search_success_response()
        )
        location = _make_location()
        entries = provider.search(location, ("2024-01-01", "2024-01-31"))
        entry = entries[0]

        assert entry.product_id == _SAMPLE_ODATA_PRODUCT["Id"]
        assert entry.timestamp == _SAMPLE_ODATA_PRODUCT["ContentDate"]["Start"]
        assert entry.geometry == _SAMPLE_ODATA_PRODUCT["GeoFootprint"]
        assert entry.bands_available == list(_S2_L2A_BANDS)
        assert entry.metadata["name"] == _SAMPLE_ODATA_PRODUCT["Name"]
        assert entry.metadata["s3_path"] == _SAMPLE_ODATA_PRODUCT["S3Path"]

    def test_network_error_raises_provider_error(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Network error during search raises ProviderError."""

        def raise_connection_error(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("Connection refused")

        monkeypatch.setattr(provider._session, "get", raise_connection_error)
        location = _make_location()

        with pytest.raises(ProviderError) as exc_info:
            provider.search(location, ("2024-01-01", "2024-01-31"))

        err = exc_info.value
        assert "catalog search failed" in err.what.lower()
        assert err.cause != ""
        assert err.fix != ""

    def test_search_does_not_send_auth_token(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search works without authentication; session has no auth header."""
        captured_args: list[Any] = []

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_args.extend(args)
            return _mock_search_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        location = _make_location()
        provider.search(location, ("2024-01-01", "2024-01-31"))

        assert captured_args[0] == _CATALOG_URL
        # Session has no auth header (authenticate() was not called)
        assert "Authorization" not in provider._session.headers

    def test_search_uses_correct_timeout(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search request uses 30-second timeout (NFR13)."""
        captured_kwargs: dict[str, Any] = {}

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_search_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        location = _make_location()
        provider.search(location, ("2024-01-01", "2024-01-31"))

        assert captured_kwargs["timeout"] == _DEFAULT_TIMEOUT

    def test_http_error_raises_provider_error(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP error status from catalog raises ProviderError."""
        error_resp = MagicMock(spec=requests.Response)
        error_resp.status_code = 500

        monkeypatch.setattr(provider._session, "get", lambda *a, **kw: error_resp)
        location = _make_location()

        with pytest.raises(ProviderError) as exc_info:
            provider.search(location, ("2024-01-01", "2024-01-31"))

        assert "HTTP 500" in exc_info.value.cause

    def test_search_invalid_json_raises_provider_error(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid JSON response from catalog raises ProviderError."""
        bad_resp = MagicMock(spec=requests.Response)
        bad_resp.status_code = 200
        bad_resp.json.side_effect = ValueError("No JSON object could be decoded")

        monkeypatch.setattr(provider._session, "get", lambda *a, **kw: bad_resp)
        location = _make_location()

        with pytest.raises(ProviderError) as exc_info:
            provider.search(location, ("2024-01-01", "2024-01-31"))

        assert "invalid json" in exc_info.value.what.lower()

    def test_cloud_cover_percentage_passes_through(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cloud_cover_max > 1.0 treated as direct percentage (no normalization)."""
        captured_kwargs: dict[str, Any] = {}

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_search_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        location = _make_location()
        provider.search(location, ("2024-01-01", "2024-01-31"), cloud_cover_max=50)

        filter_str = captured_kwargs["params"]["$filter"]
        assert "cloudCover" in filter_str
        assert "50" in filter_str


# ---------------------------------------------------------------------------
# Task 8: Check Status Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCDSECheckStatus:
    """Tests for CDSEProvider.check_status()."""

    def test_returns_available_true_when_up(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returns ProviderStatus(available=True) when CDSE is healthy."""
        monkeypatch.setattr(
            provider._session, "get", lambda *a, **kw: _mock_status_success_response()
        )
        status = provider.check_status()

        assert isinstance(status, ProviderStatus)
        assert status.available is True

    def test_returns_available_false_when_down(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returns ProviderStatus(available=False) with message when down."""
        monkeypatch.setattr(
            provider._session, "get", lambda *a, **kw: _mock_status_failure_response()
        )
        status = provider.check_status()

        assert status.available is False
        assert status.message != ""

    def test_never_raises_on_network_error(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Network error returns ProviderStatus(available=False), never raises."""

        def raise_error(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("Cannot connect")

        monkeypatch.setattr(provider._session, "get", raise_error)
        status = provider.check_status()

        assert isinstance(status, ProviderStatus)
        assert status.available is False
        assert "Cannot connect" in status.message

    def test_status_uses_correct_url(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """check_status hits the correct CDSE base URL."""
        captured_args: list[Any] = []

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_args.extend(args)
            return _mock_status_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        provider.check_status()

        assert captured_args[0] == _CATALOG_BASE_URL

    def test_status_uses_shorter_timeout(
        self,
        provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """check_status uses 10-second timeout (shorter than default)."""
        captured_kwargs: dict[str, Any] = {}

        def capture_get(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_status_success_response()

        monkeypatch.setattr(provider._session, "get", capture_get)
        provider.check_status()

        assert captured_kwargs["timeout"] == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# Task 5: Constants verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCDSEConstants:
    """Tests for CDSE API constants."""

    def test_token_url_is_correct(self) -> None:
        """Token URL matches CDSE documentation."""
        assert "identity.dataspace.copernicus.eu" in _TOKEN_URL
        assert "openid-connect/token" in _TOKEN_URL

    def test_catalog_url_is_correct(self) -> None:
        """Catalog URL matches CDSE documentation."""
        assert "catalogue.dataspace.copernicus.eu" in _CATALOG_URL
        assert "odata/v1/Products" in _CATALOG_URL

    def test_client_id_is_public(self) -> None:
        """Client ID is the public CDSE client."""
        assert _CLIENT_ID == "cdse-public"

    def test_default_timeout_is_30(self) -> None:
        """Default timeout matches NFR13 (30 seconds)."""
        assert _DEFAULT_TIMEOUT == 30

    def test_status_timeout_is_10(self) -> None:
        """Status timeout is shorter than default."""
        assert _STATUS_TIMEOUT == 10

    def test_s2_l2a_bands_complete(self) -> None:
        """Sentinel-2 L2A band list includes all expected bands."""
        assert "B04" in _S2_L2A_BANDS
        assert "B08" in _S2_L2A_BANDS
        assert "SCL" in _S2_L2A_BANDS
        assert len(_S2_L2A_BANDS) == 13

    def test_download_url_is_zipper_endpoint(self) -> None:
        """Download URL points to CDSE zipper endpoint."""
        assert "zipper.dataspace.copernicus.eu" in _DOWNLOAD_URL
        assert "odata/v1/Products" in _DOWNLOAD_URL

    def test_max_retries_is_3(self) -> None:
        """Max retries matches NFR18 (3 attempts)."""
        assert _MAX_RETRIES == 3

    def test_read_timeout_is_300(self) -> None:
        """Read timeout is 5 minutes for large file streaming."""
        assert _READ_TIMEOUT == 300

    def test_retryable_status_codes(self) -> None:
        """Retryable status codes include 429, 500, 502, 503, 504, 408."""
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 500 in _RETRYABLE_STATUS_CODES
        assert 408 in _RETRYABLE_STATUS_CODES

    def test_band_resolution_mapping(self) -> None:
        """Band resolution mapping covers all standard L2A bands."""
        assert _BAND_RESOLUTION["B04"] == "10m"
        assert _BAND_RESOLUTION["SCL"] == "20m"
        assert _BAND_RESOLUTION["B01"] == "60m"


# ---------------------------------------------------------------------------
# Story 2.5: Retry logic tests
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock response with given status code and headers."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def _make_response_sequence(
    *responses: MagicMock,
) -> Any:
    """Return a callable that yields mock responses in order."""
    remaining = list(responses)

    def fn(*args: Any, **kwargs: Any) -> MagicMock:
        return remaining.pop(0)

    return fn


@pytest.fixture
def auth_provider() -> CDSEProvider:
    """Create an authenticated CDSEProvider."""
    p = CDSEProvider(config=Config())
    p._token = "mock-token-abc123"
    p._session.headers.update({"Authorization": "Bearer mock-token-abc123"})
    return p


def _create_test_zip(bands: list[str]) -> io.BytesIO:
    """Create a minimal test ZIP mimicking Sentinel-2 SAFE structure."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for band in bands:
            res = _BAND_RESOLUTION[band]
            path = (
                f"S2A_TEST.SAFE/GRANULE/L2A_T34UFA/"
                f"IMG_DATA/R{res}/"
                f"T34UFA_20240115T235221_{band}_{res}.jp2"
            )
            zf.writestr(path, b"fake-jp2-data")
    buf.seek(0)
    return buf


def _mock_download_response(zip_buffer: io.BytesIO) -> MagicMock:
    """Create a mock streaming response containing ZIP data."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.headers = {}
    data = zip_buffer.getvalue()
    resp.iter_content.return_value = [data]
    return resp


class _MockDataset:
    """Mock rasterio dataset that returns a test numpy array."""

    def read(self, index: int = 1) -> Any:
        return np.ones((100, 100), dtype=np.float32)

    def close(self) -> None:
        pass

    def __enter__(self) -> _MockDataset:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _MockMemoryFile:
    """Mock rasterio.io.MemoryFile for unit testing."""

    def __init__(self, data: bytes = b"") -> None:
        pass

    def open(self) -> _MockDataset:
        return _MockDataset()

    def __enter__(self) -> _MockMemoryFile:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@pytest.mark.unit
class TestCDSERetry:
    """Tests for CDSEProvider._retry_request() retry logic."""

    def test_successful_request_no_retry(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful request returns response immediately, no retry."""
        call_count = 0

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _mock_response(200)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        resp = auth_provider._retry_request("get", "http://test")

        assert resp.status_code == 200
        assert call_count == 1

    def test_transient_failure_then_success(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single transient failure followed by success retries and succeeds."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
        seq = _make_response_sequence(
            _mock_response(500),
            _mock_response(200),
        )
        monkeypatch.setattr(auth_provider._session, "request", seq)
        resp = auth_provider._retry_request("get", "http://test")

        assert resp.status_code == 200

    def test_429_with_retry_after_header(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP 429 with Retry-After header respects the wait time."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        seq = _make_response_sequence(
            _mock_response(429, headers={"Retry-After": "5"}),
            _mock_response(200),
        )
        monkeypatch.setattr(auth_provider._session, "request", seq)
        auth_provider._retry_request("get", "http://test")

        assert sleep_calls[0] == 5.0

    def test_429_with_millisecond_retry_after(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP 429 with Retry-After > 1000 is treated as milliseconds."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        seq = _make_response_sequence(
            _mock_response(429, headers={"Retry-After": "3000"}),
            _mock_response(200),
        )
        monkeypatch.setattr(auth_provider._session, "request", seq)
        auth_provider._retry_request("get", "http://test")

        assert sleep_calls[0] == 3.0

    def test_429_without_retry_after_uses_backoff(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP 429 without Retry-After falls back to exponential backoff."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(random, "uniform", lambda a, b: b)
        seq = _make_response_sequence(
            _mock_response(429),
            _mock_response(200),
        )
        monkeypatch.setattr(auth_provider._session, "request", seq)
        auth_provider._retry_request("get", "http://test")

        assert sleep_calls[0] == _INITIAL_BACKOFF

    def test_all_retries_exhausted_raises_provider_error(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All retries exhausted raises ProviderError with three-part message."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: _mock_response(500),
        )

        with pytest.raises(ProviderError) as exc_info:
            auth_provider._retry_request("get", "http://test")

        err = exc_info.value
        assert "download failed" in err.what.lower()
        assert str(_MAX_RETRIES) in err.cause
        assert err.fix != ""

    def test_connection_error_triggers_retry(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """requests.ConnectionError triggers retry."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
        call_count = 0

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.ConnectionError("Connection refused")
            return _mock_response(200)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        resp = auth_provider._retry_request("get", "http://test")

        assert resp.status_code == 200
        assert call_count == 2

    def test_timeout_triggers_retry(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """requests.Timeout triggers retry."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
        call_count = 0

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.Timeout("Read timed out")
            return _mock_response(200)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        resp = auth_provider._retry_request("get", "http://test")

        assert resp.status_code == 200
        assert call_count == 2

    def test_401_raises_immediately_no_retry(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HTTP 401 raises ProviderError immediately without retrying."""
        call_count = 0

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _mock_response(401)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)

        with pytest.raises(ProviderError) as exc_info:
            auth_provider._retry_request("get", "http://test")

        assert call_count == 1
        assert "authentication expired" in exc_info.value.what.lower()

    def test_500_502_503_504_trigger_retry(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Server error codes 500, 502, 503, 504 all trigger retries."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        for status in (500, 502, 503, 504):
            call_count = 0

            def mock_request(*args: Any, _s: int = status, **kwargs: Any) -> MagicMock:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _mock_response(_s)
                return _mock_response(200)

            monkeypatch.setattr(auth_provider._session, "request", mock_request)
            resp = auth_provider._retry_request("get", "http://test")
            assert resp.status_code == 200
            assert call_count == 2, f"Expected 2 calls for HTTP {status}"

    def test_non_retryable_error_raises_immediately(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-retryable HTTP error (e.g., 403) raises ProviderError immediately."""
        call_count = 0

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _mock_response(403)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)

        with pytest.raises(ProviderError) as exc_info:
            auth_provider._retry_request("get", "http://test")

        assert call_count == 1
        assert "HTTP 403" in exc_info.value.cause

    def test_backoff_increases_exponentially(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backoff values increase exponentially between retries."""
        sleep_calls: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(random, "uniform", lambda a, b: b)  # max backoff
        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: _mock_response(500),
        )

        with pytest.raises(ProviderError):
            auth_provider._retry_request("get", "http://test")

        assert len(sleep_calls) == _MAX_RETRIES
        assert sleep_calls[0] == _INITIAL_BACKOFF * (2**0)  # 1.0
        assert sleep_calls[1] == _INITIAL_BACKOFF * (2**1)  # 2.0
        assert sleep_calls[2] == _INITIAL_BACKOFF * (2**2)  # 4.0

    def test_timeout_tuple_passed_to_session(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Timeout tuple is forwarded to session.request."""
        captured_kwargs: dict[str, Any] = {}

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_response(200)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        auth_provider._retry_request(
            "get", "http://test", timeout=(_DEFAULT_TIMEOUT, _READ_TIMEOUT)
        )

        assert captured_kwargs["timeout"] == (_DEFAULT_TIMEOUT, _READ_TIMEOUT)

    def test_redirect_followed_with_auth(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redirect responses are followed while preserving Authorization."""
        urls_called: list[str] = []

        redirect_resp = _mock_response(302, headers={"Location": "http://final"})
        final_resp = _mock_response(200)

        def mock_request(method: str, url: str, **kwargs: Any) -> MagicMock:
            urls_called.append(url)
            if url == "http://test":
                return redirect_resp
            return final_resp

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        resp = auth_provider._retry_request("get", "http://test")

        assert resp.status_code == 200
        assert urls_called == ["http://test", "http://final"]

    def test_network_errors_exhausted_raises_with_cause(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All retries exhausted due to network errors includes cause string."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)

        def always_fail(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("Connection refused")

        monkeypatch.setattr(auth_provider._session, "request", always_fail)

        with pytest.raises(ProviderError) as exc_info:
            auth_provider._retry_request("get", "http://test")

        assert "Connection refused" in exc_info.value.cause

    def test_redirect_closes_intermediate_response(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Intermediate redirect response is closed before following."""
        redirect_resp = _mock_response(302, headers={"Location": "http://final"})
        final_resp = _mock_response(200)

        def mock_request(method: str, url: str, **kwargs: Any) -> MagicMock:
            if url == "http://test":
                return redirect_resp
            return final_resp

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        auth_provider._retry_request("get", "http://test")

        redirect_resp.close.assert_called_once()

    def test_default_timeout_set_when_not_provided(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_retry_request defaults timeout to (connect, read) tuple."""
        captured_kwargs: dict[str, Any] = {}

        def mock_request(*args: Any, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_response(200)

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        auth_provider._retry_request("get", "http://test")

        assert captured_kwargs["timeout"] == (_DEFAULT_TIMEOUT, _READ_TIMEOUT)


# ---------------------------------------------------------------------------
# Story 2.5: Download tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCDSEDownload:
    """Tests for CDSEProvider.download()."""

    def test_download_without_auth_raises(
        self,
        provider: CDSEProvider,
    ) -> None:
        """download() without authentication raises ProviderError."""
        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        with pytest.raises(ProviderError) as exc_info:
            provider.download(entry)

        assert "requires authentication" in exc_info.value.what.lower()
        assert "authenticate()" in exc_info.value.fix

    def test_download_empty_product_id_raises(
        self,
        auth_provider: CDSEProvider,
    ) -> None:
        """download() with empty product_id raises ProviderError."""
        entry = CatalogEntry(provider="cdse", product_id="")
        with pytest.raises(ProviderError) as exc_info:
            auth_provider.download(entry)

        assert "product_id" in exc_info.value.cause.lower()

    def test_download_url_correctly_constructed(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Download URL is built from _DOWNLOAD_URL and entry.product_id."""
        captured_urls: list[str] = []
        test_zip = _create_test_zip(["B04"])
        mock_resp = _mock_download_response(test_zip)

        def mock_request(method: str, url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            return mock_resp

        monkeypatch.setattr(auth_provider._session, "request", mock_request)
        monkeypatch.setattr(
            "rasterio.io.MemoryFile",
            _MockMemoryFile,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid-123")
        auth_provider.download(entry, bands=["B04"])

        expected_url = f"{_DOWNLOAD_URL}(test-uuid-123)/$value"
        assert captured_urls[0] == expected_url

    def test_successful_download_returns_raw_data(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful download returns RawData with correct metadata."""
        test_zip = _create_test_zip(["B04", "B08"])
        mock_resp = _mock_download_response(test_zip)

        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: mock_resp,
        )
        monkeypatch.setattr(
            "rasterio.io.MemoryFile",
            _MockMemoryFile,
        )

        entry = CatalogEntry(
            provider="cdse",
            product_id="test-uuid-123",
            timestamp="2024-01-15T10:00:00Z",
        )
        result = auth_provider.download(entry, bands=["B04", "B08"])

        assert result.data.shape == (2, 100, 100)
        assert result.metadata["product_id"] == "test-uuid-123"
        assert result.metadata["bands"] == ["B04", "B08"]
        assert result.metadata["timestamp"] == "2024-01-15T10:00:00Z"
        assert isinstance(result.metadata["download_size_bytes"], int)

    def test_download_specific_bands(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """download(bands=['B04']) extracts only specified bands."""
        test_zip = _create_test_zip(["B04", "B08"])
        mock_resp = _mock_download_response(test_zip)

        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: mock_resp,
        )
        monkeypatch.setattr(
            "rasterio.io.MemoryFile",
            _MockMemoryFile,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        result = auth_provider.download(entry, bands=["B04"])

        assert result.data.shape == (1, 100, 100)
        assert result.metadata["bands"] == ["B04"]

    def test_download_none_bands_uses_all_standard(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """download(bands=None) requests all standard L2A bands."""
        test_zip = _create_test_zip(["B04", "B08"])
        mock_resp = _mock_download_response(test_zip)

        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: mock_resp,
        )
        monkeypatch.setattr(
            "rasterio.io.MemoryFile",
            _MockMemoryFile,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        result = auth_provider.download(entry, bands=None)

        # Only B04 and B08 are in the test ZIP; others log warning
        assert result.data.shape[0] == 2
        assert result.metadata["bands"] == ["B04", "B08"]

    def test_corrupt_zip_raises_provider_error(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corrupt ZIP archive raises ProviderError."""
        corrupt_resp = MagicMock(spec=requests.Response)
        corrupt_resp.status_code = 200
        corrupt_resp.headers = {}
        corrupt_resp.iter_content.return_value = [b"this is not a zip"]

        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: corrupt_resp,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        with pytest.raises(ProviderError) as exc_info:
            auth_provider.download(entry, bands=["B04"])

        assert "corrupt" in exc_info.value.what.lower()

    def test_zero_bands_found_raises(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty ZIP with no matching bands raises ProviderError."""
        empty_zip = io.BytesIO()
        with zipfile.ZipFile(empty_zip, "w") as zf:
            zf.writestr("dummy.txt", "no bands here")
        empty_zip.seek(0)

        mock_resp = _mock_download_response(empty_zip)
        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: mock_resp,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        with pytest.raises(ProviderError) as exc_info:
            auth_provider.download(entry, bands=["B04"])

        assert "no usable data" in exc_info.value.what.lower()

    def test_missing_band_logs_warning_and_continues(
        self,
        auth_provider: CDSEProvider,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing requested band logs WARNING but continues with available bands."""
        # ZIP only has B04, not B08
        test_zip = _create_test_zip(["B04"])
        mock_resp = _mock_download_response(test_zip)

        monkeypatch.setattr(
            auth_provider._session,
            "request",
            lambda *a, **kw: mock_resp,
        )
        monkeypatch.setattr(
            "rasterio.io.MemoryFile",
            _MockMemoryFile,
        )

        entry = CatalogEntry(provider="cdse", product_id="test-uuid")
        with caplog.at_level("WARNING"):
            result = auth_provider.download(entry, bands=["B04", "B08"])

        assert result.data.shape == (1, 100, 100)
        assert any("B08" in msg and "not found" in msg for msg in caplog.messages)
