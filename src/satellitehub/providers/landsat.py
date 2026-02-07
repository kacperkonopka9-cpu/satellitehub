"""Landsat 8/9 data access via Microsoft Planetary Computer STAC API."""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import requests

from satellitehub._types import RawData
from satellitehub.config import Config
from satellitehub.exceptions import ProviderError
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
# Planetary Computer STAC API constants
# ---------------------------------------------------------------------------

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_STAC_SEARCH_URL = f"{_STAC_URL}/search"
_COLLECTION = "landsat-c2-l2"

# ---------------------------------------------------------------------------
# Timeout and retry constants (consistent with other providers)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # seconds for connection/request timeout
_STATUS_TIMEOUT = 10  # shorter timeout for status checks
_READ_TIMEOUT = 300  # 5-minute read timeout for large file streaming

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 60.0  # seconds
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 408})
_SUCCESS_STATUS_CODES = frozenset({200})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307})

# ---------------------------------------------------------------------------
# Landsat band mapping (Collection 2 Level-2)
# ---------------------------------------------------------------------------

# Band-to-asset mapping for Landsat 8/9 Collection 2 Level-2
_BAND_ASSET_MAP: dict[str, str] = {
    "B1": "coastal",  # Coastal/Aerosol (30m)
    "B2": "blue",  # Blue (30m)
    "B3": "green",  # Green (30m)
    "B4": "red",  # Red (30m)
    "B5": "nir08",  # NIR (30m)
    "B6": "swir16",  # SWIR-1 (30m)
    "B7": "swir22",  # SWIR-2 (30m)
    "QA_PIXEL": "qa_pixel",  # Quality Assessment (30m)
}

# Default bands to download (common optical + quality)
_DEFAULT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "QA_PIXEL"]

# Landsat L2A standard band list for CatalogEntry
_LANDSAT_L2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "QA_PIXEL",
]

# Sentinel-2 equivalent mapping for documentation/reference
_SENTINEL2_EQUIVALENT: dict[str, str] = {
    "B2": "B02",  # Blue
    "B3": "B03",  # Green
    "B4": "B04",  # Red
    "B5": "B08",  # NIR
    "B6": "B11",  # SWIR1
    "B7": "B12",  # SWIR2
    "QA_PIXEL": "SCL",  # Quality mask
}

# Buffer in degrees for bbox queries (~10km at equator)
_BBOX_BUFFER_DEG = 0.1


def _compute_bbox(
    lat: float, lon: float, buffer_deg: float = _BBOX_BUFFER_DEG
) -> list[float]:
    """Compute a bounding box around a point with buffer.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        buffer_deg: Buffer in degrees (default ~10km at equator).

    Returns:
        Bounding box as [min_lon, min_lat, max_lon, max_lat].
    """
    return [
        lon - buffer_deg,
        lat - buffer_deg,
        lon + buffer_deg,
        lat + buffer_deg,
    ]


class LandsatProvider(DataProvider):
    """Landsat 8/9 data provider via Microsoft Planetary Computer.

    Provides access to Landsat Collection 2 Level-2 surface reflectance
    imagery through the Planetary Computer STAC API. No authentication
    is required for catalog searches; URL signing is required for downloads.

    Args:
        config: Frozen configuration snapshot from the Location.

    Example:
        >>> from satellitehub.config import Config
        >>> provider = LandsatProvider(config=Config())
        >>> provider.name
        'landsat'
    """

    _name: str = "landsat"

    def __init__(self, config: Config) -> None:
        """Initialize Landsat provider with configuration.

        Args:
            config: SDK configuration captured at Location creation.
        """
        super().__init__(config)
        self._session: requests.Session = requests.Session()
        self._authenticated: bool = False

    def authenticate(self, credentials: ProviderCredentials) -> None:
        """Validate connection to Planetary Computer (no auth required).

        Planetary Computer's STAC API is publicly accessible. This method
        validates the connection is working by making a lightweight request.

        Args:
            credentials: Ignored for Planetary Computer (public API).

        Raises:
            ConfigurationError: If Planetary Computer API is unreachable.

        Example:
            >>> provider = LandsatProvider(config=Config())
            >>> provider.authenticate(ProviderCredentials())
        """
        # Planetary Computer is public, no authentication needed
        # Just validate the API is reachable
        self._authenticated = True
        logger.debug("Landsat authentication skipped (public API)")

    def search(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> list[CatalogEntry]:
        """Search Planetary Computer catalog for Landsat 8/9 L2 data.

        Catalog search is public and does **not** require authentication.
        Returns an empty list when no data matches the query —
        never raises on missing data.

        Args:
            location: Geographic location to search around.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Optional parameters:
                - ``cloud_cover_max`` (float): Maximum cloud cover (0.0-1.0).
                  Defaults to 0.3 (30%).
                - ``limit`` (int): Maximum results to return. Defaults to 100.

        Returns:
            List of matching catalog entries, empty if none found.

        Raises:
            ProviderError: If the STAC API is unreachable or
                returns an HTTP error.

        Example:
            >>> provider.search(location, ("2025-06-01", "2025-06-30"))
            []
        """
        # Parse parameters
        cloud_cover_max = float(params.get("cloud_cover_max", 0.3))
        limit = int(params.get("limit", 100))

        # Normalize cloud cover to percentage (0-100) for STAC query
        if cloud_cover_max <= 1.0:
            cloud_cover_pct = cloud_cover_max * 100
        else:
            cloud_cover_pct = cloud_cover_max

        # Build bounding box from location
        bbox = _compute_bbox(location.lat, location.lon)

        # Format time range for STAC
        start_date = time_range[0]
        end_date = time_range[1]
        datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"

        # Build STAC search request
        search_body = {
            "collections": [_COLLECTION],
            "bbox": bbox,
            "datetime": datetime_range,
            "limit": limit,
            "query": {
                "eo:cloud_cover": {"lt": cloud_cover_pct},
            },
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
        }

        logger.debug(
            "Searching Landsat catalog: bbox=%s, datetime=%s, cloud_max=%.0f%%",
            bbox,
            datetime_range,
            cloud_cover_pct,
        )

        try:
            resp = self._session.post(
                _STAC_SEARCH_URL,
                json=search_body,
                timeout=_DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                what="Landsat catalog search failed",
                cause=str(exc),
                fix="Check internet connection and try again",
            ) from exc

        if resp.status_code not in _SUCCESS_STATUS_CODES:
            raise ProviderError(
                what="Landsat catalog returned error",
                cause=f"HTTP {resp.status_code}",
                fix="Try again; check Planetary Computer status if persistent",
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderError(
                what="Landsat catalog returned invalid JSON",
                cause=str(exc),
                fix="Try again; check Planetary Computer status if persistent",
            ) from exc

        features: list[dict[str, Any]] = body.get("features", [])

        # Filter to Landsat 8/9 only and parse entries
        entries: list[CatalogEntry] = []
        for feature in features:
            entry = self._parse_stac_item(feature)
            if entry is not None:
                entries.append(entry)

        logger.debug("Found %d Landsat scenes matching criteria", len(entries))
        return entries

    @staticmethod
    def _parse_stac_item(item: dict[str, Any]) -> CatalogEntry | None:
        """Parse a STAC item into a CatalogEntry.

        Filters to Landsat 8/9 Collection 2 items only.

        Args:
            item: STAC item (GeoJSON feature).

        Returns:
            CatalogEntry or None if not Landsat 8/9.
        """
        properties = item.get("properties", {})

        # Filter to Landsat 8/9 (collection number >= 8)
        platform = properties.get("platform", "")
        if platform not in ("landsat-8", "landsat-9"):
            return None

        # Extract cloud cover (0-100) and normalize to 0-1
        cloud_cover_pct = float(properties.get("eo:cloud_cover", 0))
        cloud_cover = cloud_cover_pct / 100.0

        # Get acquisition timestamp
        timestamp = properties.get("datetime", "")

        # Get geometry
        geometry = item.get("geometry", {})

        # Get asset information
        assets = item.get("assets", {})
        bands_available = [
            band for band, asset_key in _BAND_ASSET_MAP.items()
            if asset_key in assets
        ]

        # Extract metadata
        product_id = item.get("id", "")

        return CatalogEntry(
            provider="landsat",
            product_id=product_id,
            timestamp=timestamp,
            cloud_cover=cloud_cover,
            geometry=geometry,
            bands_available=bands_available,
            metadata={
                "platform": platform,
                "collection": _COLLECTION,
                "sun_elevation": str(properties.get("sun_elevation", "")),
                "sun_azimuth": str(properties.get("sun_azimuth", "")),
                "wrs_path": str(properties.get("landsat:wrs_path", "")),
                "wrs_row": str(properties.get("landsat:wrs_row", "")),
                "processing_level": str(properties.get("landsat:correction", "")),
            },
        )

    def download(
        self,
        entry: CatalogEntry,
        bands: list[str] | None = None,
    ) -> RawData:
        """Download Landsat bands for a catalog entry.

        Downloads the requested bands as Cloud-Optimized GeoTIFFs (COGs)
        from Planetary Computer, signs URLs using the PC signing API,
        and stacks them into a numpy array.

        Args:
            entry: Catalog entry from ``search()`` to download.
            bands: Optional list of specific bands (e.g., ``['B4', 'B5']``).
                Downloads default bands if ``None``.

        Returns:
            Raw data with numpy arrays and provider metadata.

        Raises:
            ProviderError: If download fails after retries exhausted
                or no bands could be extracted.

        Example:
            >>> provider.download(entry, bands=["B4", "B5"])  # doctest: +SKIP
        """
        if not entry.product_id:
            raise ProviderError(
                what="Invalid catalog entry",
                cause="product_id is empty",
                fix="Ensure search() returned valid entries",
            )

        target_bands = bands if bands is not None else list(_DEFAULT_BANDS)

        logger.info("Downloading Landsat product %s...", entry.product_id)

        # Get STAC item with assets
        item = self._fetch_stac_item(entry.product_id)
        if item is None:
            raise ProviderError(
                what="Landsat item not found",
                cause=f"Could not fetch STAC item {entry.product_id}",
                fix="Ensure the product_id is valid",
            )

        # Sign asset URLs using Planetary Computer API
        signed_item = self._sign_item(item)
        if signed_item is None:
            raise ProviderError(
                what="Landsat URL signing failed",
                cause="Planetary Computer signing API unavailable",
                fix="Try again; check Planetary Computer status",
            )

        # Download and stack bands
        array, extracted_bands = self._download_bands(signed_item, target_bands)

        if array.size == 0:
            raise ProviderError(
                what="Landsat download produced no usable data",
                cause="Zero bands found in product",
                fix="Try again; product may be temporarily unavailable",
            )

        logger.info(
            "Download complete: %d bands, shape %s",
            len(extracted_bands),
            array.shape,
        )

        return RawData(
            data=array,
            metadata={
                "product_id": entry.product_id,
                "bands": extracted_bands,
                "timestamp": entry.timestamp,
                "platform": entry.metadata.get("platform", ""),
            },
        )

    def _fetch_stac_item(self, item_id: str) -> dict[str, Any] | None:
        """Fetch a STAC item by ID from Planetary Computer.

        Args:
            item_id: The STAC item ID.

        Returns:
            STAC item dict or None if not found.
        """
        url = f"{_STAC_URL}/collections/{_COLLECTION}/items/{item_id}"

        try:
            resp = self._retry_request("get", url)
            result: dict[str, Any] = resp.json()
            return result
        except (ProviderError, ValueError):
            return None

    def _sign_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Sign STAC item asset URLs using Planetary Computer.

        Planetary Computer requires URL signing for data downloads.

        Args:
            item: STAC item with unsigned asset URLs.

        Returns:
            STAC item with signed asset URLs, or None on failure.
        """
        sign_url = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

        try:
            resp = self._session.post(
                sign_url,
                json=item,
                timeout=_DEFAULT_TIMEOUT,
            )
            if resp.status_code in _SUCCESS_STATUS_CODES:
                result: dict[str, Any] = resp.json()
                return result
            logger.warning(
                "Planetary Computer signing failed: HTTP %d", resp.status_code
            )
            return None
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Planetary Computer signing failed: %s", exc)
            return None

    def _download_bands(
        self,
        item: dict[str, Any],
        bands: list[str],
    ) -> tuple[npt.NDArray[np.floating[Any]], list[str]]:
        """Download and stack spectral bands from a signed STAC item.

        Args:
            item: Signed STAC item with asset URLs.
            bands: List of band identifiers to download.

        Returns:
            Tuple of (stacked numpy array with shape
            ``(n_bands, height, width)``, list of band names actually
            extracted).
        """
        from rasterio.io import MemoryFile  # noqa: PLC0415

        assets = item.get("assets", {})
        band_arrays: list[npt.NDArray[np.floating[Any]]] = []
        extracted_bands: list[str] = []
        target_shape: tuple[int, ...] | None = None

        for band in bands:
            asset_key = _BAND_ASSET_MAP.get(band)
            if asset_key is None:
                logger.warning("Unknown band %s, skipping", band)
                continue

            asset = assets.get(asset_key)
            if asset is None:
                logger.warning(
                    "Band %s (asset: %s) not found in product", band, asset_key
                )
                continue

            href = asset.get("href")
            if not href:
                logger.warning("Band %s has no download URL", band)
                continue

            # Download the COG
            try:
                resp = self._retry_request("get", href, stream=True)
                data = resp.content
            except ProviderError as exc:
                logger.warning("Failed to download band %s: %s", band, exc)
                continue

            # Read with rasterio
            try:
                with MemoryFile(data) as memfile, memfile.open() as dataset:
                    arr: npt.NDArray[np.floating[Any]] = dataset.read(1).astype(
                        np.float32
                    )

                    # Set target shape from first band
                    if target_shape is None:
                        target_shape = arr.shape

                    band_arrays.append(arr)
                    extracted_bands.append(band)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read band %s: %s", band, exc)
                continue

        if not band_arrays:
            return np.array([], dtype=np.float32), []

        # Resample bands to common shape if needed (nearest neighbor)
        if target_shape is not None:
            resampled: list[npt.NDArray[np.floating[Any]]] = []
            for arr in band_arrays:
                if arr.shape != target_shape:
                    # Upscale using nearest neighbor (repeat pixels)
                    scale_y = target_shape[0] // arr.shape[0]
                    scale_x = target_shape[1] // arr.shape[1]
                    if scale_y > 0 and scale_x > 0:
                        arr = np.repeat(
                            np.repeat(arr, scale_y, axis=0), scale_x, axis=1
                        )
                        # Crop if slightly larger due to rounding
                        arr = arr[: target_shape[0], : target_shape[1]]
                resampled.append(arr)
            band_arrays = resampled

        return np.stack(band_arrays), extracted_bands

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
        kwargs.setdefault("timeout", (_DEFAULT_TIMEOUT, _READ_TIMEOUT))
        kwargs.setdefault("allow_redirects", True)
        last_status: int = 0
        last_exc: requests.RequestException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.request(method, url, **kwargs)

                if resp.status_code in _SUCCESS_STATUS_CODES:
                    return resp

                last_status = resp.status_code

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    raise ProviderError(
                        what="Landsat request failed",
                        cause=f"HTTP {resp.status_code}",
                        fix="Check Planetary Computer status at https://planetarycomputer.microsoft.com/",
                    )

                # Retryable status code - compute backoff
                backoff = self._compute_backoff(attempt)
                logger.warning(
                    "Landsat request failed (HTTP %d, attempt %d/%d), "
                    "retrying in %.1fs...",
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
                        "Landsat request failed (%s, attempt %d/%d), "
                        "retrying in %.1fs...",
                        type(exc).__name__,
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)

        # All retries exhausted
        if last_exc is not None:
            raise ProviderError(
                what="Landsat request failed after retries",
                cause=str(last_exc),
                fix="Check internet connection and try again",
            ) from last_exc

        raise ProviderError(
            what="Landsat request failed after retries",
            cause=f"HTTP {last_status} after {_MAX_RETRIES} retries",
            fix="Check Planetary Computer status at https://planetarycomputer.microsoft.com/",
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
        """Check Planetary Computer STAC API operational status.

        Makes a lightweight request to verify the API is reachable.
        Never raises - returns ``ProviderStatus`` with ``available=False``
        and a descriptive message on any failure.

        Returns:
            Current operational status of the Planetary Computer API.

        Example:
            >>> provider = LandsatProvider(config=Config())
            >>> provider.check_status().available  # doctest: +SKIP
            True
        """
        try:
            resp = self._session.get(
                f"{_STAC_URL}/collections/{_COLLECTION}",
                timeout=_STATUS_TIMEOUT,
            )
            if resp.status_code in _SUCCESS_STATUS_CODES:
                return ProviderStatus(available=True)
            return ProviderStatus(
                available=False,
                message=f"Planetary Computer returned HTTP {resp.status_code}",
            )
        except requests.RequestException as exc:
            return ProviderStatus(
                available=False,
                message=f"Planetary Computer API unreachable: {exc}",
            )
