"""ERA5 climate data access via Climate Data Store API."""

from __future__ import annotations

import logging
import random
import tempfile
import time
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import requests
import xarray as xr

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
# CDS API constants (Task 1)
# ---------------------------------------------------------------------------

_CDS_API_URL = "https://cds.climate.copernicus.eu/api"
_CDS_RESOURCES_URL = f"{_CDS_API_URL}/resources"
_CDS_PORTAL_URL = "https://cds.climate.copernicus.eu/"
_CDS_HOWTO_URL = "https://cds.climate.copernicus.eu/how-to-api"

# ---------------------------------------------------------------------------
# Timeout constants (NFR13: 30-second connection timeout)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # seconds for connection/request timeout
_STATUS_TIMEOUT = 10  # shorter timeout for status checks
_READ_TIMEOUT = 300  # 5-minute read timeout for downloads

# ---------------------------------------------------------------------------
# Retry constants (NFR18: up to 3 retries with backoff)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 60.0  # seconds
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 408})
_SUCCESS_STATUS_CODES = frozenset({200, 202})  # 200=OK, 202=Accepted (queue)

# ---------------------------------------------------------------------------
# Queue polling constants (NFR14: queue-based retrieval, NFR2: 2-min budget)
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 5.0  # seconds between queue status checks
_MAX_QUEUE_TIME = 120.0  # 2-minute budget for queue completion
_LOG_INTERVAL = 30.0  # log progress every 30 seconds

# ---------------------------------------------------------------------------
# ERA5 product constants
# ---------------------------------------------------------------------------

_ERA5_PRODUCT = "reanalysis-era5-single-levels"
_DEFAULT_VARIABLES = ["2m_temperature", "total_precipitation"]


class CDSProvider(DataProvider):
    """ERA5 climate data provider via the CDS API.

    Provides access to ERA5 reanalysis data (temperature, precipitation,
    wind) through the Copernicus Climate Data Store. Uses API key
    authentication and handles queue-based retrieval transparently.

    Args:
        config: Frozen configuration snapshot from the Location.

    Example:
        >>> from satellitehub.config import Config
        >>> provider = CDSProvider(config=Config())
        >>> provider.name
        'cds'
    """

    _name: str = "cds"

    def __init__(self, config: Config) -> None:
        """Initialize CDS provider with configuration.

        Args:
            config: SDK configuration captured at Location creation.
        """
        super().__init__(config)
        self._session: requests.Session = requests.Session()
        self._api_key: str = ""
        self._authenticated: bool = False

    def authenticate(self, credentials: ProviderCredentials) -> None:
        """Validate CDS API credentials.

        Tests the API key by making a lightweight request to the CDS
        resources endpoint. The key is stored for subsequent requests.

        Args:
            credentials: CDS API key credential (uses ``api_key`` field).

        Raises:
            ConfigurationError: If the API key is missing, invalid,
                or the CDS service is unreachable.

        Example:
            >>> provider = CDSProvider(config=Config())
            >>> provider.authenticate(
            ...     ProviderCredentials(api_key="your-api-key")
            ... )  # doctest: +SKIP
        """
        api_key = credentials.api_key
        if not api_key:
            raise ConfigurationError(
                what="CDS API key not provided",
                cause="Missing api_key in credentials",
                fix="Add CDS API key to ~/.satellitehub/credentials.json",
            )

        # Set API key in session headers
        self._session.headers["PRIVATE-TOKEN"] = api_key

        # Test the key with a lightweight request
        try:
            resp = self._session.get(_CDS_RESOURCES_URL, timeout=_STATUS_TIMEOUT)
        except requests.RequestException as exc:
            raise ConfigurationError(
                what="Cannot reach CDS API",
                cause=str(exc),
                fix=f"Check internet connection and CDS status at {_CDS_PORTAL_URL}",
            ) from exc

        if resp.status_code in (401, 403):
            raise ConfigurationError(
                what="CDS authentication failed",
                cause="Invalid or expired API key",
                fix=f"Register at {_CDS_HOWTO_URL} and generate a new API key",
            )

        if resp.status_code != 200:  # noqa: PLR2004
            raise ConfigurationError(
                what="CDS authentication check failed",
                cause=f"HTTP {resp.status_code}",
                fix=f"Check CDS status at {_CDS_PORTAL_URL}",
            )

        self._api_key = api_key
        self._authenticated = True
        logger.debug("CDS authentication successful")

    def search(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> list[CatalogEntry]:
        """Search CDS catalog for ERA5 data.

        ERA5 is a gridded reanalysis product with global coverage
        from 1940 to present (~5 day latency). Data is always available
        for any valid location and historical time range, so this method
        always returns exactly one entry. Actual data availability is
        verified at download time by the CDS API.

        Args:
            location: Geographic location to search around.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Additional CDS-specific search parameters.
                ``variables``: List of ERA5 variable names.
                ``product``: ERA5 product type (default: single-levels).

        Returns:
            List containing one catalog entry for the requested data.
            ERA5 has global coverage so always returns one entry.

        Example:
            >>> provider.search(location, ("2024-01-01", "2024-01-31"))
            [CatalogEntry(provider='cds', ...)]
        """
        variables = params.get("variables", _DEFAULT_VARIABLES)
        product = params.get("product", _ERA5_PRODUCT)

        start_date = time_range[0]
        end_date = time_range[1]

        # ERA5 data is available from 1940 onwards, with ~5 day latency
        # Generate a single catalog entry representing the available data
        entry = CatalogEntry(
            provider="cds",
            product_id=f"{product}:{start_date}:{end_date}",
            timestamp=start_date,
            cloud_cover=0.0,  # Not applicable for reanalysis data
            geometry={
                "type": "Point",
                "coordinates": [location.lon, location.lat],
            },
            bands_available=(
                list(variables) if isinstance(variables, list) else [variables]
            ),
            metadata={
                "product": product,
                "start_date": start_date,
                "end_date": end_date,
                "lat": str(location.lat),
                "lon": str(location.lon),
            },
        )
        return [entry]

    def download(
        self,
        entry: CatalogEntry,
        bands: list[str] | None = None,
    ) -> RawData:
        """Download ERA5 climate variables for a catalog entry.

        Submits a request to the CDS queue, polls for completion,
        and downloads the result. Progress is logged at INFO level.

        Args:
            entry: Catalog entry from ``search()`` to download.
            bands: Optional list of specific variables to download.
                Uses entry's bands_available if ``None``.

        Returns:
            Raw data with numpy arrays and provider metadata.

        Raises:
            ProviderError: If download fails after retries exhausted,
                queue times out, or authentication is missing.

        Example:
            >>> provider.download(entry, bands=["2m_temperature"])  # doctest: +SKIP
        """
        if not self._authenticated:
            raise ProviderError(
                what="CDS download requires authentication",
                cause="Not authenticated",
                fix="Call authenticate() before download()",
            )

        variables = bands if bands is not None else entry.bands_available
        if not variables:
            variables = _DEFAULT_VARIABLES

        # Parse metadata from entry
        product = entry.metadata.get("product", _ERA5_PRODUCT)
        start_date = entry.metadata.get("start_date", "")
        end_date = entry.metadata.get("end_date", "")
        lat = float(entry.metadata.get("lat", "0"))
        lon = float(entry.metadata.get("lon", "0"))

        # Build request body
        request_body = self._build_request(
            product=product,
            variables=variables,
            start_date=start_date,
            end_date=end_date,
            lat=lat,
            lon=lon,
        )

        # Submit request to CDS queue
        logger.info("Submitted ERA5 request, waiting for queue...")
        task_id = self._submit_request(product, request_body)

        # Poll for completion
        download_url = self._poll_queue(task_id)

        # Download result
        logger.debug("ERA5 request completed, downloading result...")
        raw_data = self._download_result(download_url)

        # Merge parsed metadata with provider context
        merged_metadata = {
            "provider": "cds",
            "product": product,
            "time_range": (start_date, end_date),
            "bounds": {
                "north": lat + 0.25,
                "south": lat - 0.25,
                "east": lon + 0.25,
                "west": lon - 0.25,
            },
            "crs": "EPSG:4326",
            **raw_data.metadata,  # timestamps, variables, units from NetCDF
        }

        return RawData(
            data=raw_data.data,
            metadata=merged_metadata,
        )

    def _build_request(
        self,
        product: str,
        variables: list[str],
        start_date: str,
        end_date: str,
        lat: float,
        lon: float,
    ) -> dict[str, Any]:
        """Build CDS API request body.

        Args:
            product: ERA5 product identifier.
            variables: List of variable names to request.
            start_date: Start date (ISO format).
            end_date: End date (ISO format).
            lat: Latitude of the location.
            lon: Longitude of the location.

        Returns:
            Request body dictionary.
        """
        # Parse dates
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Generate date components for the actual requested range
        years = sorted({str(y) for y in range(start_dt.year, end_dt.year + 1)})

        # Generate months within range
        if start_dt.year == end_dt.year:
            months = [f"{m:02d}" for m in range(start_dt.month, end_dt.month + 1)]
        else:
            # Multi-year: need all months (CDS handles filtering)
            months = [f"{m:02d}" for m in range(1, 13)]

        # Generate days within range
        if start_dt.year == end_dt.year and start_dt.month == end_dt.month:
            days = [f"{d:02d}" for d in range(start_dt.day, end_dt.day + 1)]
        else:
            # Multi-month: need all days (CDS handles filtering)
            days = [f"{d:02d}" for d in range(1, 32)]

        times = ["00:00", "06:00", "12:00", "18:00"]

        # Define bounding box (small area around point)
        area = [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25]  # N, W, S, E

        return {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        }

    def _submit_request(self, product: str, request_body: dict[str, Any]) -> str:
        """Submit request to CDS queue.

        Args:
            product: ERA5 product identifier.
            request_body: Request body dictionary.

        Returns:
            Task ID for polling.

        Raises:
            ProviderError: If submission fails.
        """
        url = f"{_CDS_RESOURCES_URL}/{product}"

        resp = self._retry_request("post", url, json=request_body)

        try:
            body: dict[str, Any] = resp.json()
            task_id: str = str(body.get("request_id", ""))
            if not task_id:
                # Some responses return the ID in "jobID" or similar
                task_id = str(body.get("jobID", body.get("id", "")))
        except (ValueError, KeyError) as exc:
            raise ProviderError(
                what="CDS request submission failed",
                cause="Invalid response format",
                fix="Try again; if persistent, check CDS status",
            ) from exc

        if not task_id:
            raise ProviderError(
                what="CDS request submission failed",
                cause="No task ID in response",
                fix="Try again; if persistent, check CDS status",
            )

        return task_id

    def _poll_queue(self, task_id: str) -> str:
        """Poll CDS queue for request completion.

        Args:
            task_id: Task ID from submission.

        Returns:
            Download URL for completed request.

        Raises:
            ProviderError: If queue times out or request fails.
        """
        url = f"{_CDS_API_URL}/tasks/{task_id}"
        start_time = time.time()
        last_log_time = start_time

        # Log initial polling start
        logger.info("ERA5 request queued, polling for completion...")

        while True:
            elapsed = time.time() - start_time

            if elapsed > _MAX_QUEUE_TIME:
                raise ProviderError(
                    what="ERA5 request timed out",
                    cause="CDS queue exceeded 2-minute budget",
                    fix="Try again later or during off-peak hours",
                )

            try:
                resp = self._session.get(url, timeout=_DEFAULT_TIMEOUT)
            except requests.RequestException as exc:
                raise ProviderError(
                    what="CDS queue polling failed",
                    cause=str(exc),
                    fix="Check internet connection and try again",
                ) from exc

            if resp.status_code != 200:  # noqa: PLR2004
                raise ProviderError(
                    what="CDS queue polling failed",
                    cause=f"HTTP {resp.status_code}",
                    fix="Try again; if persistent, check CDS status",
                )

            try:
                body = resp.json()
            except ValueError as exc:
                raise ProviderError(
                    what="CDS queue polling failed",
                    cause="Invalid JSON response",
                    fix="Try again; if persistent, check CDS status",
                ) from exc

            state = body.get("state", "").lower()

            if state == "completed":
                download_url: str = str(body.get("location", ""))
                if not download_url:
                    raise ProviderError(
                        what="CDS request completed without download URL",
                        cause="Missing location in response",
                        fix="Try again; if persistent, report to CDS support",
                    )
                return download_url

            if state == "failed":
                error_msg = body.get("error", {}).get("message", "Unknown error")
                raise ProviderError(
                    what="ERA5 request failed",
                    cause=error_msg,
                    fix="Check request parameters and try again",
                )

            # Log progress periodically
            if time.time() - last_log_time >= _LOG_INTERVAL:
                logger.info("ERA5 request processing (elapsed: %.0fs)...", elapsed)
                last_log_time = time.time()

            time.sleep(_POLL_INTERVAL)

    def _download_result(self, download_url: str) -> RawData:
        """Download and parse NetCDF result file from CDS.

        Args:
            download_url: URL to download result.

        Returns:
            RawData with parsed ERA5 variables, timestamps, and metadata.

        Raises:
            ProviderError: If download or parsing fails.
        """
        resp = self._retry_request(
            "get",
            download_url,
            stream=True,
            timeout=(_DEFAULT_TIMEOUT, _READ_TIMEOUT),
        )

        # Read content into memory
        content = resp.content
        logger.debug("Downloaded %d bytes from CDS", len(content))

        # Parse NetCDF and extract ERA5 data
        return self._parse_netcdf(content)

    def _parse_netcdf(self, content: bytes) -> RawData:
        """Parse NetCDF bytes into RawData.

        Uses h5netcdf engine for in-memory parsing via BytesIO.
        Falls back to temp file for NetCDF formats requiring seekable streams.

        Args:
            content: Raw bytes of NetCDF file.

        Returns:
            RawData with extracted variables and metadata.

        Raises:
            ProviderError: If parsing fails.
        """
        # Try in-memory parsing first (h5netcdf supports BytesIO for HDF5/NetCDF4)
        try:
            with xr.open_dataset(BytesIO(content), engine="h5netcdf") as ds:
                return self._extract_era5_data(ds)
        except Exception as mem_err:
            logger.debug(
                "In-memory NetCDF parsing failed: %s. Trying temp file fallback.",
                mem_err,
            )

        # Fallback: write to temp file for formats needing seekable stream
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            with xr.open_dataset(tmp_path) as ds:
                return self._extract_era5_data(ds)
        except Exception as file_err:
            raise ProviderError(
                what="Failed to parse ERA5 NetCDF data",
                cause=str(file_err),
                fix="Check CDS response format or try again",
            ) from file_err

    def _extract_era5_data(self, ds: xr.Dataset) -> RawData:
        """Extract ERA5 variables from xarray Dataset.

        Computes spatial mean over lat/lon dimensions, converts temperature
        from Kelvin to Celsius, and formats timestamps as ISO-8601.

        Args:
            ds: Opened xarray Dataset.

        Returns:
            RawData with shape (time, variables) and metadata.
        """
        # ERA5 variable name mappings (CDS API names -> standard names)
        var_mappings: dict[str, str] = {
            "t2m": "2m_temperature",
            "tp": "total_precipitation",
            "2t": "2m_temperature",  # Alternative name
        }

        # Find available variables
        available_vars: list[str] = []
        var_names: list[str] = []
        for cds_name, standard_name in var_mappings.items():
            if cds_name in ds.data_vars:
                available_vars.append(cds_name)
                var_names.append(standard_name)

        if not available_vars:
            # Fallback: use first available data variable
            first_var = list(ds.data_vars.keys())[0] if ds.data_vars else None
            if first_var:
                available_vars = [str(first_var)]
                var_names = [str(first_var)]

        # Extract time dimension
        time_dim = "time" if "time" in ds.dims else "valid_time"
        if time_dim not in ds.dims:
            # Single timestamp case
            timestamps = [datetime.now().isoformat() + "Z"]
            n_times = 1
        else:
            time_values = ds[time_dim].values
            timestamps = [
                np.datetime_as_string(t, unit="s").replace("T", "T") + "Z"
                if isinstance(t, np.datetime64)
                else str(t)
                for t in time_values
            ]
            n_times = len(timestamps)

        # Extract and process each variable
        data_arrays: list[npt.NDArray[np.floating[Any]]] = []
        units: dict[str, str] = {}

        for cds_name, standard_name in zip(available_vars, var_names, strict=True):
            var_data = ds[cds_name]

            # Compute spatial mean over lat/lon
            spatial_dims = [d for d in var_data.dims if d in ("latitude", "longitude")]
            var_mean = var_data.mean(dim=spatial_dims) if spatial_dims else var_data

            # Convert to numpy array, ensuring 1D for time series
            values = var_mean.values.flatten()

            # Temperature conversion: Kelvin -> Celsius
            if standard_name == "2m_temperature":
                values = values - 273.15
                units[standard_name] = "celsius"
            elif standard_name == "total_precipitation":
                # Precipitation is in meters; keep as-is for now
                units[standard_name] = "m"
            else:
                units[standard_name] = str(var_data.attrs.get("units", "unknown"))

            data_arrays.append(values.astype(np.float32))

        # Stack into (time, variables) shape
        if data_arrays:
            data = np.column_stack(data_arrays)
        else:
            data = np.empty((n_times, 0), dtype=np.float32)

        logger.debug(
            "Extracted ERA5 data: shape=%s, variables=%s, timestamps=%d",
            data.shape,
            var_names,
            len(timestamps),
        )

        return RawData(
            data=data,
            metadata={
                "timestamps": timestamps,
                "variables": var_names,
                "units": units,
            },
        )

    def _retry_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute HTTP request with retry and exponential backoff.

        Args:
            method: HTTP method (``"get"``, ``"post"``, etc.).
            url: Target URL.
            **kwargs: Additional keyword arguments for ``requests.Session.request``.

        Returns:
            Successful HTTP response.

        Raises:
            ProviderError: If all retries are exhausted.
        """
        kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
        last_status: int = 0
        last_exc: requests.RequestException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.request(method, url, **kwargs)

                if resp.status_code in _SUCCESS_STATUS_CODES:
                    return resp

                last_status = resp.status_code

                if resp.status_code in (401, 403):
                    raise ProviderError(
                        what="CDS authentication failed",
                        cause="API key rejected",
                        fix="Call authenticate() with valid credentials",
                    )

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    raise ProviderError(
                        what="CDS request failed",
                        cause=f"HTTP {resp.status_code}",
                        fix=f"Check CDS status at {_CDS_PORTAL_URL}",
                    )

                # Retryable status code - compute backoff
                backoff = self._compute_backoff(attempt)
                logger.warning(
                    "CDS request failed (HTTP %d, attempt %d/%d), retrying in %.1fs...",
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)

            except ProviderError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = self._compute_backoff(attempt)
                    logger.warning(
                        "CDS request failed (%s, attempt %d/%d), retrying in %.1fs...",
                        type(exc).__name__,
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)

        # All retries exhausted
        if last_exc is not None:
            raise ProviderError(
                what="CDS request failed after retries",
                cause=str(last_exc),
                fix="Check internet connection and try again",
            ) from last_exc

        raise ProviderError(
            what="CDS request failed after retries",
            cause=f"HTTP {last_status} after {_MAX_RETRIES} retries",
            fix=f"Check CDS status at {_CDS_PORTAL_URL}",
        )

    @staticmethod
    def _compute_backoff(attempt: int) -> float:
        """Compute exponential backoff with jitter.

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Wait time in seconds (randomized).
        """
        base_delay: float = min(_INITIAL_BACKOFF * (2**attempt), _MAX_BACKOFF)
        jitter: float = random.uniform(0, base_delay * 0.1)  # noqa: S311
        return float(base_delay + jitter)

    def check_status(self) -> ProviderStatus:
        """Check CDS operational status.

        Never raises — returns ``ProviderStatus`` with ``available=False``
        and a descriptive message on any failure.

        Note:
            This method checks the public CDS resources endpoint and does
            not require prior authentication. If authenticated, the session
            headers are used; otherwise, the check is unauthenticated.

        Returns:
            Current operational status of the CDS API.

        Example:
            >>> provider = CDSProvider(config=Config())
            >>> provider.check_status().available  # doctest: +SKIP
            True
        """
        try:
            # Resources endpoint is public; auth headers used if present
            resp = self._session.get(_CDS_RESOURCES_URL, timeout=_STATUS_TIMEOUT)
            if resp.status_code == 200:  # noqa: PLR2004
                return ProviderStatus(available=True)
            return ProviderStatus(
                available=False,
                message=f"CDS returned HTTP {resp.status_code}",
            )
        except requests.RequestException as exc:
            return ProviderStatus(
                available=False,
                message=f"CDS API unreachable: {exc}",
            )
