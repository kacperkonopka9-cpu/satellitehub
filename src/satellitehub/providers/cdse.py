"""Sentinel-2 data access via Copernicus Data Space Ecosystem."""

from __future__ import annotations

import io
import logging
import random
import time
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import requests

from satellitehub._types import RawData
from satellitehub.config import Config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.providers.base import (
    CatalogEntry,
    DataProvider,
    ProviderCredentials,
    ProviderStatus,
)

if TYPE_CHECKING:
    from satellitehub._types import TimeRange
    from satellitehub.location import Location

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CDSE API constants (Task 5)
# ---------------------------------------------------------------------------

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_CATALOG_BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/"
_CLIENT_ID = "cdse-public"
_DEFAULT_TIMEOUT = 30
_STATUS_TIMEOUT = 10
_CDSE_PORTAL_URL = "https://dataspace.copernicus.eu/"

# ---------------------------------------------------------------------------
# Download & retry constants (Story 2.5)
# ---------------------------------------------------------------------------

_DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 60.0  # seconds
_READ_TIMEOUT = 300  # 5-minute read timeout for large file streaming
_CHUNK_SIZE = 8192  # streaming download chunk size in bytes
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 408})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307})

# Band-to-resolution mapping for Sentinel-2 L2A SAFE directory
_BAND_RESOLUTION: dict[str, str] = {
    "B01": "60m",
    "B02": "10m",
    "B03": "10m",
    "B04": "10m",
    "B05": "20m",
    "B06": "20m",
    "B07": "20m",
    "B08": "10m",
    "B8A": "20m",
    "B09": "60m",
    "B11": "20m",
    "B12": "20m",
    "SCL": "20m",
}

# Sentinel-2 L2A standard band list
_S2_L2A_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "SCL",
]


class CDSEProvider(DataProvider):
    """Sentinel-2 data provider via Copernicus Data Space Ecosystem.

    Provides access to Sentinel-2 Level-2A imagery through the CDSE
    OData catalog and download APIs. Uses OAuth2 token authentication.

    Args:
        config: Frozen configuration snapshot from the Location.

    Example:
        >>> from satellitehub.config import Config
        >>> provider = CDSEProvider(config=Config())
        >>> provider.name
        'cdse'
    """

    _name: str = "cdse"

    def __init__(self, config: Config) -> None:
        """Initialize CDSE provider with configuration.

        Args:
            config: SDK configuration captured at Location creation.
        """
        super().__init__(config)
        self._session: requests.Session = requests.Session()
        self._token: str = ""

    def authenticate(self, credentials: ProviderCredentials) -> None:
        """Validate CDSE credentials and obtain OAuth2 access token.

        Exchanges username/password for a bearer token via the CDSE
        identity service. The token is stored on ``self._session`` for
        subsequent authenticated requests.

        Args:
            credentials: CDSE account username and password.

        Raises:
            ConfigurationError: If credentials are invalid, the service
                is unreachable, or the response is malformed.

        Example:
            >>> provider = CDSEProvider(config=Config())
            >>> provider.authenticate(
            ...     ProviderCredentials(username="user", password="pass")
            ... )  # doctest: +SKIP
        """
        try:
            resp = self._session.post(
                _TOKEN_URL,
                data={
                    "grant_type": "password",
                    "client_id": _CLIENT_ID,
                    "username": credentials.username,
                    "password": credentials.password,
                },
                timeout=_DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ConfigurationError(
                what="Cannot reach CDSE authentication service",
                cause=str(exc),
                fix=(
                    "Check internet connection and CDSE status at " + _CDSE_PORTAL_URL
                ),
            ) from exc

        if resp.status_code != 200:  # noqa: PLR2004
            raise ConfigurationError(
                what="CDSE authentication failed",
                cause="Invalid username or password",
                fix=(
                    "Verify credentials at "
                    + _CDSE_PORTAL_URL
                    + " and update ~/.satellitehub/credentials.json"
                ),
            )

        try:
            body = resp.json()
            token: str = body["access_token"]
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(
                what="CDSE returned unexpected auth response",
                cause="Missing access_token in response",
                fix="Try again; if persistent, check CDSE status",
            ) from exc

        self._token = token
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        logger.debug("CDSE authentication successful")

    def _build_odata_filter(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> str:
        """Build an OData ``$filter`` string for Sentinel-2 L2A products.

        Args:
            location: Geographic location for spatial intersection.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Optional parameters. Supports ``cloud_cover_max``
                (float, 0.0--1.0 or 0--100) to filter by cloud cover.

        Returns:
            Combined OData filter string.
        """
        # Format dates as ISO-8601 date strings (YYYY-MM-DD)
        from datetime import datetime

        start_date = time_range[0]
        end_date = time_range[1]
        if isinstance(start_date, datetime):
            start_str = start_date.strftime("%Y-%m-%d")
        else:
            start_str = str(start_date)[:10]  # Take only date part
        if isinstance(end_date, datetime):
            end_str = end_date.strftime("%Y-%m-%d")
        else:
            end_str = str(end_date)[:10]

        filters = [
            "Collection/Name eq 'SENTINEL-2'",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType' and "
                "att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
            ),
            (
                f"OData.CSC.Intersects(area=geography"
                f"'SRID=4326;POINT({location.lon} {location.lat})')"
            ),
            (
                f"ContentDate/Start gt {start_str}T00:00:00.000Z"
                f" and ContentDate/Start lt {end_str}T23:59:59.999Z"
            ),
        ]

        if "cloud_cover_max" in params:
            max_cc = float(params["cloud_cover_max"])
            # Normalize 0.0–1.0 to 0–100 if needed
            if max_cc <= 1.0:
                max_cc *= 100
            filters.append(
                "Attributes/OData.CSC.DoubleAttribute/any("
                "att:att/Name eq 'cloudCover' and "
                f"att/OData.CSC.DoubleAttribute/Value le {max_cc:.0f})"
            )

        return " and ".join(filters)

    @staticmethod
    def _parse_catalog_entry(item: dict[str, Any]) -> CatalogEntry:
        """Parse one OData product JSON object into a CatalogEntry.

        Args:
            item: Single product dict from the OData ``.value`` array.

        Returns:
            Populated ``CatalogEntry``.
        """
        content_date = item.get("ContentDate", {})
        return CatalogEntry(
            provider="cdse",
            product_id=str(item.get("Id", "")),
            timestamp=str(content_date.get("Start", "")),
            cloud_cover=0.0,
            geometry=item.get("GeoFootprint", {}),
            bands_available=list(_S2_L2A_BANDS),
            metadata={
                "name": str(item.get("Name", "")),
                "content_length": str(item.get("ContentLength", "")),
                "online": str(item.get("Online", "")),
                "s3_path": str(item.get("S3Path", "")),
            },
        )

    def search(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> list[CatalogEntry]:
        """Search CDSE catalog for Sentinel-2 L2A data via OData.

        Catalog search is public and does **not** require authentication.
        Returns an empty list when no data matches the query —
        never raises on missing data.

        Args:
            location: Geographic location to search around.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Optional. ``cloud_cover_max`` (float) filters by
                maximum cloud cover percentage.

        Returns:
            List of matching catalog entries, empty if none found.

        Raises:
            ProviderError: If the catalog service is unreachable or
                returns an HTTP error.

        Example:
            >>> provider.search(location, ("2024-01-01", "2024-01-31"))
            []
        """
        filter_string = self._build_odata_filter(location, time_range, **params)
        try:
            resp = self._session.get(
                _CATALOG_URL,
                params={"$filter": filter_string, "$top": "100"},
                timeout=_DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                what="CDSE catalog search failed",
                cause=str(exc),
                fix="Check internet connection and try again",
            ) from exc

        if resp.status_code != 200:  # noqa: PLR2004
            raise ProviderError(
                what="CDSE catalog returned error",
                cause=f"HTTP {resp.status_code}",
                fix="Try again; check CDSE status if persistent",
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderError(
                what="CDSE catalog returned invalid JSON",
                cause=str(exc),
                fix="Try again; check CDSE status if persistent",
            ) from exc

        items: list[dict[str, Any]] = body.get("value", [])
        return [self._parse_catalog_entry(item) for item in items]

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> float | None:
        """Parse ``Retry-After`` header from an HTTP response.

        CDSE may return the value in milliseconds (>1000).

        Args:
            resp: HTTP response potentially containing Retry-After header.

        Returns:
            Wait time in seconds, or ``None`` if header is absent/invalid.
        """
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            value = float(raw)
            # CDSE Retry-After may be in milliseconds
            if value > 1000:  # noqa: PLR2004
                return value / 1000.0
            return value
        except ValueError:
            return None

    @staticmethod
    def _compute_backoff(attempt: int) -> float:
        """Compute exponential backoff with full jitter.

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Wait time in seconds (randomized between 0 and computed cap).
        """
        backoff = min(_MAX_BACKOFF, _INITIAL_BACKOFF * (2**attempt))
        return random.uniform(0, backoff)  # noqa: S311

    def _retry_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute HTTP request with retry, backoff, and redirect handling.

        Follows redirects manually to preserve the ``Authorization`` header
        across cross-domain redirects (Python ``requests`` strips it by
        default). Retries on transient failures and rate limits.

        Args:
            method: HTTP method (``"get"``, ``"post"``, etc.).
            url: Target URL.
            **kwargs: Additional keyword arguments for ``requests.Session.request``.

        Returns:
            Successful HTTP response (status 200).

        Raises:
            ProviderError: If all retries are exhausted, the server returns
                a non-retryable error, or the token has expired (HTTP 401).
        """
        kwargs.setdefault("allow_redirects", False)
        kwargs.setdefault("timeout", (_DEFAULT_TIMEOUT, _READ_TIMEOUT))
        last_status: int = 0
        last_exc: requests.RequestException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.request(method, url, **kwargs)

                # Follow redirects manually (preserves Authorization header)
                redirect_count = 0
                while (
                    resp.status_code in _REDIRECT_STATUS_CODES and redirect_count < 5  # noqa: PLR2004
                ):
                    location = resp.headers.get("Location", "")
                    if not location:
                        break
                    resp.close()
                    resp = self._session.request(method, location, **kwargs)
                    redirect_count += 1

                if resp.status_code == 200:  # noqa: PLR2004
                    return resp

                if resp.status_code == 401:  # noqa: PLR2004
                    raise ProviderError(
                        what="CDSE authentication expired",
                        cause="Bearer token rejected",
                        fix="Call authenticate() with fresh credentials",
                    )

                last_status = resp.status_code

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    raise ProviderError(
                        what="CDSE download failed",
                        cause=f"HTTP {resp.status_code}",
                        fix="Try again; check CDSE status if persistent",
                    )

                # Retryable: compute wait time
                wait: float | None = None
                if resp.status_code == 429:  # noqa: PLR2004
                    wait = self._parse_retry_after(resp)
                if wait is None:
                    wait = self._compute_backoff(attempt)

                logger.error(
                    "CDSE request failed (attempt %d/%d), HTTP %d, retrying...",
                    attempt + 1,
                    _MAX_RETRIES,
                    resp.status_code,
                )
                time.sleep(wait)

            except ProviderError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    logger.error(
                        "CDSE request failed (attempt %d/%d), retrying...",
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(self._compute_backoff(attempt))

        # All retries exhausted
        if last_exc is not None:
            raise ProviderError(
                what="CDSE download failed",
                cause=str(last_exc),
                fix="Check internet connection and try again",
            ) from last_exc

        raise ProviderError(
            what="CDSE download failed",
            cause=f"HTTP {last_status} after {_MAX_RETRIES} retries",
            fix="Try again; check CDSE status if persistent",
        )

    def download(
        self,
        entry: CatalogEntry,
        bands: list[str] | None = None,
    ) -> RawData:
        """Download Sentinel-2 bands for a catalog entry.

        Downloads the full product ZIP from the CDSE OData zipper endpoint,
        extracts requested bands from the SAFE directory structure, and
        reads them into numpy arrays via rasterio.

        Args:
            entry: Catalog entry from ``search()`` to download.
            bands: Optional list of specific bands (e.g., ``['B04', 'B08']``).
                Downloads all standard L2A bands if ``None``.

        Returns:
            Raw data with numpy arrays and provider metadata.

        Raises:
            ProviderError: If download fails after retries exhausted,
                authentication is missing, or the archive contains no
                usable bands.

        Example:
            >>> provider.download(entry, bands=["B04", "B08"])  # doctest: +SKIP
        """
        if not entry.product_id:
            raise ProviderError(
                what="Invalid catalog entry",
                cause="product_id is empty",
                fix="Ensure search() returned valid entries",
            )

        if not self._token:
            raise ProviderError(
                what="CDSE download requires authentication",
                cause="No bearer token set",
                fix="Call authenticate() before download()",
            )

        target_bands = bands if bands is not None else list(_S2_L2A_BANDS)
        url = f"{_DOWNLOAD_URL}({entry.product_id})/$value"

        logger.info("Downloading Sentinel-2 product %s...", entry.product_id)

        # Retry entire download if streaming fails (large files can timeout)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._retry_request(
                    "get",
                    url,
                    stream=True,
                    timeout=(_DEFAULT_TIMEOUT, _READ_TIMEOUT),
                )

                # Stream response into memory buffer
                buf = io.BytesIO()
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        buf.write(chunk)
                download_size = buf.tell()
                buf.seek(0)
                break  # Success
            except (OSError, requests.RequestException) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = min(_INITIAL_BACKOFF * (2**attempt), _MAX_BACKOFF)
                    logger.warning(
                        "Download attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        str(exc)[:100],
                        backoff,
                    )
                    time.sleep(backoff)
        else:
            raise ProviderError(
                what="CDSE download failed after retries",
                cause=str(last_exc)[:200] if last_exc else "Unknown error",
                fix="Check network connection; try again later",
            )

        # Extract bands from SAFE ZIP archive
        array, extracted_bands = self._extract_bands(buf, target_bands)

        logger.info(
            "Download complete: %d bands, %.1f MB",
            array.shape[0] if array.ndim > 2 else 1,  # noqa: PLR2004
            download_size / (1024 * 1024),
        )

        return RawData(
            data=array,
            metadata={
                "product_id": entry.product_id,
                "bands": extracted_bands,
                "timestamp": entry.timestamp,
                "download_size_bytes": download_size,
            },
        )

    @staticmethod
    def _extract_bands(
        zip_buffer: io.BytesIO,
        bands: list[str],
    ) -> tuple[npt.NDArray[np.floating[Any]], list[str]]:
        """Extract spectral band arrays from a Sentinel-2 SAFE ZIP archive.

        Parses the SAFE directory structure inside the ZIP to locate
        band files (JPEG 2000), reads them with rasterio, and stacks
        the arrays. Bands are returned at their native resolution.

        Args:
            zip_buffer: In-memory buffer containing the product ZIP.
            bands: List of band identifiers to extract.

        Returns:
            Tuple of (stacked numpy array with shape
            ``(n_bands, height, width)``, list of band names actually
            extracted).

        Raises:
            ProviderError: If the archive is corrupt or contains no
                matching bands.
        """
        from rasterio.io import MemoryFile  # noqa: PLC0415

        try:
            zf = zipfile.ZipFile(zip_buffer)
        except zipfile.BadZipFile as exc:
            raise ProviderError(
                what="CDSE download produced corrupt archive",
                cause=str(exc),
                fix="Try again; product may be temporarily unavailable",
            ) from exc

        band_arrays: list[npt.NDArray[np.floating[Any]]] = []
        extracted_bands: list[str] = []
        target_shape: tuple[int, ...] | None = None

        with zf:
            names = zf.namelist()

            for band in bands:
                resolution = _BAND_RESOLUTION.get(band)
                if resolution is None:
                    logger.warning("Unknown band %s, skipping", band)
                    continue

                # Match SAFE filename pattern: _B04_10m.jp2
                suffix = f"_{band}_{resolution}.jp2"
                matched = [n for n in names if n.endswith(suffix)]

                if not matched:
                    logger.warning("Band %s not found in archive", band)
                    continue

                jp2_data = zf.read(matched[0])
                with MemoryFile(jp2_data) as memfile, memfile.open() as dataset:
                    arr: npt.NDArray[np.floating[Any]] = dataset.read(1).astype(
                        np.float32
                    )

                    # Set target shape from first 10m band for resampling
                    if target_shape is None and resolution == "10m":
                        target_shape = arr.shape

                    band_arrays.append(arr)
                    extracted_bands.append(band)

        if not band_arrays:
            raise ProviderError(
                what="CDSE download produced no usable data",
                cause="Zero bands found in archive",
                fix="Try again; product may be temporarily unavailable",
            )

        # Resample bands to common shape (nearest neighbor for SCL categorical)
        if target_shape is not None:
            resampled: list[npt.NDArray[np.floating[Any]]] = []
            for arr in band_arrays:
                if arr.shape != target_shape:
                    # Upscale using nearest neighbor (repeat pixels)
                    scale_y = target_shape[0] // arr.shape[0]
                    scale_x = target_shape[1] // arr.shape[1]
                    arr = np.repeat(np.repeat(arr, scale_y, axis=0), scale_x, axis=1)
                    # Crop if slightly larger due to rounding
                    arr = arr[: target_shape[0], : target_shape[1]]
                resampled.append(arr)
            band_arrays = resampled

        return np.stack(band_arrays), extracted_bands

    def check_status(self) -> ProviderStatus:
        """Check CDSE OData catalog operational status.

        Never raises — returns ``ProviderStatus`` with ``available=False``
        and a descriptive message on any failure.

        Returns:
            Current operational status of the CDSE API.

        Example:
            >>> provider = CDSEProvider(config=Config())
            >>> provider.check_status().available  # doctest: +SKIP
            True
        """
        try:
            resp = self._session.get(_CATALOG_BASE_URL, timeout=_STATUS_TIMEOUT)
            if resp.status_code == 200:  # noqa: PLR2004
                return ProviderStatus(available=True)
            return ProviderStatus(
                available=False,
                message=f"CDSE returned HTTP {resp.status_code}",
            )
        except requests.RequestException as exc:
            return ProviderStatus(available=False, message=str(exc))
