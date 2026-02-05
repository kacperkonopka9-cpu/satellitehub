"""Location model for SatelliteHub spatial queries.

Implements the Location entry point for all data queries and analysis,
with WGS84 coordinate validation, UTM zone determination (NFR11),
and configuration snapshot capture (AD-1).
"""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from satellitehub._types import LocationHash, TimeRange
from satellitehub.config import Config, get_default_config
from satellitehub.exceptions import ConfigurationError, ProviderError
from satellitehub.results import (
    _STABLE_THRESHOLD,
    BaseResult,
    ChangeResult,
    ResultMetadata,
    VegetationResult,
    WeatherResult,
)

if TYPE_CHECKING:
    from satellitehub.providers.base import DataProvider

logger = logging.getLogger(__name__)

_MIN_LAT = -90.0
_MAX_LAT = 90.0
_MIN_LON = -180.0
_MAX_LON = 180.0
_DEFAULT_CLOUD_MAX: float = 0.8  # Allow acquisition, filter via SCL masking


def _compute_utm_zone(lat: float, lon: float) -> int:
    """Compute the UTM zone number from WGS84 coordinates.

    Handles the standard 6-degree zone calculation plus the Norway
    and Svalbard special-case overrides.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.

    Returns:
        UTM zone number (1--60).
    """
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)

    # Norway exception: zone 32V is widened to 9 degrees
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32

    # Svalbard exceptions: zones 32X, 34X, 36X eliminated
    elif 72.0 <= lat <= 84.0:
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37

    return zone


def _compute_utm_epsg(lat: float, lon: float) -> int:
    """Compute the EPSG code for the UTM zone covering *lat*/*lon*.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.

    Returns:
        EPSG code (326XX for northern hemisphere, 327XX for southern).
    """
    zone = _compute_utm_zone(lat, lon)
    hemisphere_offset = 32600 if lat >= 0 else 32700
    return hemisphere_offset + zone


def location(
    lat: float,
    lon: float,
    config: Config | None = None,
) -> Location:
    """Create a location reference from WGS84 coordinates.

    Validates coordinates and captures the current configuration
    snapshot for use by all analysis and data-tier methods (AD-1).

    Args:
        lat: Latitude in WGS84 (valid range: -90 to 90).
        lon: Longitude in WGS84 (valid range: -180 to 180).
        config: Optional Config object. If provided, overrides
            the module-level defaults.

    Returns:
        A ``Location`` object ready for data queries and analysis.

    Raises:
        ConfigurationError: If coordinates are outside WGS84 bounds.

    Example:
        >>> import satellitehub as sh
        >>> field = sh.location(lat=51.25, lon=22.57)
        >>> field.utm_zone
        34
    """
    if not (_MIN_LAT <= lat <= _MAX_LAT):
        raise ConfigurationError(
            what=f"Invalid latitude: {lat}",
            cause=f"Latitude must be between {_MIN_LAT} and {_MAX_LAT}",
            fix="Provide a valid WGS84 latitude value",
        )
    if not (_MIN_LON <= lon <= _MAX_LON):
        raise ConfigurationError(
            what=f"Invalid longitude: {lon}",
            cause=f"Longitude must be between {_MIN_LON} and {_MAX_LON}",
            fix="Provide a valid WGS84 longitude value",
        )

    captured_config = config if config is not None else get_default_config()
    return Location(lat=lat, lon=lon, config=captured_config)


class DataTier:
    """Raw data access methods for a geographic location.

    Provides methods to retrieve raw satellite data without semantic
    analysis. Accessed via ``Location.data`` property.

    Args:
        loc: Parent Location instance.

    Example:
        >>> field = location(lat=51.25, lon=22.57)
        >>> result = field.data.sentinel2(bands=["B04", "B08"])
        >>> result.confidence  # doctest: +SKIP
        0.85
    """

    def __init__(self, loc: Location) -> None:
        """Initialize with parent location reference."""
        self._location = loc

    def sentinel2(
        self,
        bands: list[str] | None = None,
        cloud_max: float = 0.3,
        last_days: int = 60,
    ) -> BaseResult:
        """Retrieve raw Sentinel-2 L2A bands for this location.

        Acquires data through the pipeline (cache → provider → cache store)
        and wraps the result into a ``BaseResult``.

        Data availability issues return a zero-confidence result with
        warnings — this method never raises on missing data.

        Args:
            bands: Specific band identifiers to download (e.g.,
                ``["B04", "B08"]``). Downloads all standard bands if ``None``.
            cloud_max: Maximum acceptable cloud cover fraction (0.0--1.0).
                Defaults to 0.3 (30%).
            last_days: Number of days to look back from today. Defaults to 60.

        Returns:
            ``BaseResult`` with raw band data, confidence, and metadata.

        Example:
            >>> field = location(lat=51.25, lon=22.57)
            >>> result = field.data.sentinel2(bands=["B04", "B08"])
            >>> result.confidence  # doctest: +SKIP
            0.85
        """
        from satellitehub._pipeline import _acquire

        try:
            raw = _acquire(
                location=self._location,
                provider_name="cdse",
                product="sentinel2-l2a",
                bands=bands,
                cloud_max=cloud_max,
                last_days=last_days,
            )
        except (ProviderError, ConfigurationError) as exc:
            logger.warning("Sentinel-2 data acquisition failed: %s", exc)
            return BaseResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=[str(exc)],
            )

        # Empty result from pipeline (no catalog entries)
        if raw.data.size == 0:
            return BaseResult(
                data=raw.data,
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=[
                    "No Sentinel-2 data found for the requested location and time range"
                ],
            )

        # Build metadata from provider's download response.
        # observation_count=1 is correct: _acquire downloads only entries[0].
        meta = raw.metadata
        result_meta = ResultMetadata(
            source="cdse",
            timestamps=[meta.get("timestamp", "")],
            observation_count=1,
            cloud_cover_pct=meta.get("cloud_cover_pct", 0.0),
            bands=meta.get("bands", []),
        )

        return BaseResult(
            data=raw.data,
            confidence=1.0,
            metadata=result_meta,
        )

    def era5(
        self,
        variables: list[str] | None = None,
        last_days: int = 90,
    ) -> BaseResult:
        """Retrieve raw ERA5 climate data for this location.

        Acquires ERA5 reanalysis data through the CDS provider for the
        specified climate variables and time period.

        Data availability issues return a zero-confidence result with
        warnings — this method never raises on missing data.

        Args:
            variables: Climate variables to download (e.g.,
                ``["2m_temperature", "total_precipitation"]``).
                Downloads default variables if ``None``.
            last_days: Number of days to look back from today. Defaults to 90.

        Returns:
            ``BaseResult`` with raw ERA5 data, confidence, and metadata.

        Example:
            >>> field = location(lat=51.25, lon=22.57)
            >>> result = field.data.era5(variables=["2m_temperature"])
            >>> result.confidence  # doctest: +SKIP
            0.85
        """
        from satellitehub._pipeline import _acquire_weather

        try:
            raw = _acquire_weather(
                location=self._location,
                provider_name="cds",
                variables=variables,
                last_days=last_days,
            )
        except (ProviderError, ConfigurationError) as exc:
            logger.warning("ERA5 data acquisition failed: %s", exc)
            return BaseResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cds"),
                warnings=[str(exc)],
            )

        # Empty result from pipeline (no catalog entries)
        if raw.data.size == 0:
            return BaseResult(
                data=raw.data,
                confidence=0.0,
                metadata=ResultMetadata(source="cds"),
                warnings=[
                    "No ERA5 data found for the requested location and time range"
                ],
            )

        # Build metadata from provider's download response
        meta = raw.metadata
        result_meta = ResultMetadata(
            source="cds",
            timestamps=meta.get("timestamps", []),
            observation_count=len(meta.get("timestamps", [])),
            bands=meta.get("variables", []),
        )

        return BaseResult(
            data=raw.data,
            confidence=1.0,
            metadata=result_meta,
        )

    def imgw(
        self,
        station_type: str = "synop",
        last_days: int = 30,
    ) -> BaseResult:
        """Retrieve raw IMGW station data for this location.

        Acquires weather station measurements through the IMGW provider
        for Polish locations. Returns empty result for locations outside
        Poland or beyond station coverage range.

        Data availability issues return a zero-confidence result with
        warnings — this method never raises on missing data.

        Args:
            station_type: Type of weather station. Currently only
                ``"synop"`` (synoptic stations) is supported.
            last_days: Number of days to look back from today. Defaults to 30.
                Note: IMGW MVP only returns current observations.

        Returns:
            ``BaseResult`` with raw IMGW data, confidence, and metadata.

        Example:
            >>> field = location(lat=52.23, lon=21.01)  # Warsaw
            >>> result = field.data.imgw(station_type="synop")
            >>> result.confidence  # doctest: +SKIP
            0.85
        """
        from satellitehub._pipeline import _acquire_weather

        try:
            raw = _acquire_weather(
                location=self._location,
                provider_name="imgw",
                station_type=station_type,
                last_days=last_days,
            )
        except (ProviderError, ConfigurationError) as exc:
            logger.warning("IMGW data acquisition failed: %s", exc)
            return BaseResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="imgw"),
                warnings=[str(exc)],
            )

        # Empty result from pipeline (no catalog entries)
        if raw.data.size == 0:
            return BaseResult(
                data=raw.data,
                confidence=0.0,
                metadata=ResultMetadata(source="imgw"),
                warnings=[
                    "No IMGW data found for the requested location and time range"
                ],
            )

        # Build metadata from provider's download response
        meta = raw.metadata
        result_meta = ResultMetadata(
            source="imgw",
            timestamps=[meta.get("observation_time", "")],
            observation_count=1,
        )

        return BaseResult(
            data=raw.data,
            confidence=1.0,
            metadata=result_meta,
        )


class Location:
    """A geographic location for satellite data queries.

    Stores validated WGS84 coordinates, a frozen ``Config`` snapshot
    captured at creation time, and the computed UTM zone for internal
    CRS operations.

    Args:
        lat: Validated latitude.
        lon: Validated longitude.
        config: Frozen configuration snapshot.

    Example:
        >>> loc = Location(lat=51.25, lon=22.57, config=Config())
        >>> loc.utm_epsg
        32634
    """

    __slots__ = (
        "_lat",
        "_lon",
        "_config",
        "_utm_zone",
        "_utm_epsg",
        "_location_hash",
        "_providers",
        "_data_tier",
        "_cache_manager",
    )

    def __init__(self, lat: float, lon: float, config: Config) -> None:
        """Initialize from pre-validated coordinates and frozen config.

        Use the ``location()`` factory function for coordinate validation.
        """
        self._lat = lat
        self._lon = lon
        self._config = config
        self._utm_zone = _compute_utm_zone(lat, lon)
        self._utm_epsg = _compute_utm_epsg(lat, lon)
        canonical = f"{lat:.10f},{lon:.10f}"
        self._location_hash: LocationHash = hashlib.sha256(
            canonical.encode()
        ).hexdigest()
        self._providers: dict[str, DataProvider] = {}
        self._data_tier: DataTier | None = None
        self._cache_manager: Any = None

    # ── Read-only properties ─────────────────────────────────────

    @property
    def lat(self) -> float:
        """Latitude in WGS84 degrees."""
        return self._lat

    @property
    def lon(self) -> float:
        """Longitude in WGS84 degrees."""
        return self._lon

    @property
    def config(self) -> Config:
        """Frozen configuration snapshot captured at creation."""
        return self._config

    @property
    def utm_zone(self) -> int:
        """UTM zone number (1--60)."""
        return self._utm_zone

    @property
    def utm_epsg(self) -> int:
        """EPSG code for the UTM zone (326XX north, 327XX south)."""
        return self._utm_epsg

    @property
    def location_hash(self) -> LocationHash:
        """Deterministic SHA-256 hex digest for cache key generation.

        Uses canonical float formatting to ensure reproducibility
        (NFR20). Computed once at construction and cached.
        """
        return self._location_hash

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box as (minx, miny, maxx, maxy) in WGS84 coordinates.

        For a point location, returns the point itself as a degenerate
        bounding box. This is useful for weather queries where the
        exact point is sufficient.

        Returns:
            Tuple of (longitude_min, latitude_min, longitude_max, latitude_max).
        """
        return (self._lon, self._lat, self._lon, self._lat)

    @property
    def cache(self) -> Any:
        """Cached CacheManager instance for this location's config.

        Lazily instantiated on first access to avoid repeated SQLite
        database creation in the pipeline.
        """
        if self._cache_manager is None:
            from satellitehub.cache import CacheManager

            self._cache_manager = CacheManager(config=self._config)
        return self._cache_manager

    @property
    def data(self) -> DataTier:
        """Raw data access tier for this location.

        Returns a ``DataTier`` providing methods like ``sentinel2()``
        for direct data retrieval without semantic analysis.

        Example:
            >>> field = location(lat=51.25, lon=22.57)
            >>> tier = field.data
            >>> tier  # doctest: +SKIP
            <DataTier for Location(lat=51.25, lon=22.57)>
        """
        if self._data_tier is None:
            self._data_tier = DataTier(self)
        return self._data_tier

    # ── Provider access ────────────────────────────────────────

    def get_provider(self, name: str) -> DataProvider:
        """Return a cached provider instance by name.

        On the first call for a given provider name, the provider is
        instantiated via the provider registry with this Location's
        config snapshot. Subsequent calls return the cached instance.

        Args:
            name: Provider identifier (``"cdse"``, ``"cds"``, or ``"imgw"``).

        Returns:
            A configured ``DataProvider`` instance tied to this Location.

        Raises:
            ConfigurationError: If *name* does not match a registered provider.

        Example:
            >>> loc = location(lat=51.25, lon=22.57)
            >>> provider = loc.get_provider("cdse")
            >>> provider.name
            'cdse'
        """
        key = name.lower()
        if key not in self._providers:
            from satellitehub.providers import get_provider

            self._providers[key] = get_provider(key, self._config)
        return self._providers[key]

    # ── Data discovery ─────────────────────────────────────────

    def available_data(self, last_days: int = 30) -> dict[str, Any]:
        """Summarize available satellite and weather data for this location.

        Queries all configured providers for catalog entries within the
        specified time range. Provider errors are caught and added to
        warnings (graceful degradation).

        Args:
            last_days: Number of days to look back. Defaults to 30.

        Returns:
            Dictionary with keys ``"providers"`` (dict of provider name
            to list of passes), ``"total_passes"`` (int),
            ``"date_range"`` (tuple), and ``"warnings"`` (list).

        Example:
            >>> field = location(lat=51.25, lon=22.57)
            >>> summary = field.available_data(last_days=30)
            >>> summary["total_passes"]  # doctest: +SKIP
            12
        """
        now = datetime.now(timezone.utc)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=last_days)).strftime("%Y-%m-%d")
        time_range: TimeRange = (start_date, end_date)

        providers_data: dict[str, list[dict[str, Any]]] = {}
        warnings: list[str] = []
        total_passes = 0

        from satellitehub.providers import get_registered_names

        for provider_name in get_registered_names():
            try:
                provider = self.get_provider(provider_name)
                entries = provider.search(
                    location=self,
                    time_range=time_range,
                )
                passes = [
                    {
                        "date": e.timestamp,
                        "cloud_cover": e.cloud_cover,
                    }
                    for e in entries
                ]
                providers_data[provider_name] = passes
                total_passes += len(passes)
            except (ProviderError, ConfigurationError) as exc:
                logger.warning(
                    "Could not query %s for available data: %s",
                    provider_name,
                    exc,
                )
                warnings.append(f"{provider_name}: {exc}")
                providers_data[provider_name] = []

        return {
            "providers": providers_data,
            "total_passes": total_passes,
            "date_range": (start_date, end_date),
            "warnings": warnings,
        }

    def available_methods(self) -> list[Any]:
        """List available analysis methods for this location.

        Returns information about each semantic method including its name,
        description, required data sources, and whether it's currently available
        based on configured credentials.

        Returns:
            List of MethodInfo objects describing each available method.
            Each MethodInfo includes:
            - name: Method name to call
            - description: What the method does
            - data_sources: Required data providers
            - available: Whether credentials are configured
            - unavailable_reason: Why unavailable (if applicable)

        Example:
            >>> field = location(lat=51.25, lon=22.57)
            >>> methods = field.available_methods()
            >>> for m in methods:
            ...     print(m)  # doctest: +SKIP
            vegetation_health(): Compute NDVI vegetation health [sources: cdse] ✓
            vegetation_change(): Detect vegetation changes [sources: cdse] ✓
            weather(): Get weather summary [sources: cds, imgw] ✗ (CDS not configured)
        """
        from satellitehub._types import MethodInfo
        from satellitehub.config import resolve_credentials_path

        methods: list[MethodInfo] = []

        # Check provider availability
        cdse_available = resolve_credentials_path(
            self._config.copernicus_credentials,
        ) is not None
        cds_available = resolve_credentials_path(
            self._config.cds_credentials,
        ) is not None

        # Define semantic methods
        cdse_reason = "Copernicus credentials not configured"
        methods.append(
            MethodInfo(
                name="vegetation_health",
                description="Compute NDVI-based vegetation health analysis",
                data_sources=["cdse"],
                available=cdse_available,
                unavailable_reason="" if cdse_available else cdse_reason,
            )
        )

        methods.append(
            MethodInfo(
                name="vegetation_change",
                description="Detect vegetation changes between time periods",
                data_sources=["cdse"],
                available=cdse_available,
                unavailable_reason="" if cdse_available else cdse_reason,
            )
        )

        # Weather requires CDS, IMGW is optional (Poland only)
        weather_available = cds_available  # CDS is the primary source
        weather_reason = "" if weather_available else "CDS credentials not configured"
        methods.append(
            MethodInfo(
                name="weather",
                description="Get temperature and precipitation summary",
                data_sources=["cds", "imgw"],
                available=weather_available,
                unavailable_reason=weather_reason,
            )
        )

        return methods

    # ── Semantic analysis methods ─────────────────────────────────

    def vegetation_health(self, last_days: int = 30) -> VegetationResult:
        """Compute vegetation health index for this location.

        Retrieves Sentinel-2 imagery, applies cloud masking, computes NDVI,
        and assesses data quality for the specified time period.

        Args:
            last_days: Number of days to look back from today. Defaults to 30.

        Returns:
            VegetationResult with NDVI values, confidence score, and quality
            metadata. If no cloud-free data is available, returns a result
            with confidence=0.0 and descriptive warnings.

        Raises:
            ConfigurationError: If Copernicus credentials are not configured.
            ProviderError: If CDSE is unreachable after retries.

        Example:
            >>> import satellitehub as sh
            >>> sh.configure(copernicus_credentials="path/to/credentials.json")
            >>> field = sh.location(lat=51.25, lon=22.57)
            >>> result = field.vegetation_health(last_days=30)
            >>> result.confidence  # doctest: +SKIP
            0.78
            >>> result.mean_ndvi  # doctest: +SKIP
            0.42
        """
        from satellitehub._pipeline import (
            _acquire,
            _assess_quality,
            _build_result,
            _cloud_mask,
        )
        from satellitehub.analysis.vegetation import compute_ndvi

        # Step 1: Acquire raw Sentinel-2 data with required bands
        # B04 = red, B08 = NIR, SCL = scene classification for cloud masking
        raw = _acquire(
            location=self,
            provider_name="cdse",
            product="sentinel2-l2a",
            bands=["B04", "B08", "SCL"],
            cloud_max=_DEFAULT_CLOUD_MAX,
            last_days=last_days,
        )

        # Handle no-data case: return structured result with descriptive warnings
        if raw.data.size == 0:
            logger.info("No Sentinel-2 data available for vegetation_health query")
            return VegetationResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=[
                    "No satellite observations available for the requested period",
                    "Consider using field.data.sentinel2() with relaxed cloud_max "
                    "for manual analysis",
                ],
                mean_ndvi=float("nan"),
                ndvi_std=float("nan"),
                trend=None,
                observation_count=0,
                cloud_free_count=0,
            )

        # Step 2: Apply cloud masking
        # Band ordering is determined by the bands= list in _acquire():
        #   bands=["B04", "B08", "SCL"] → index 0=B04, 1=B08, 2=SCL
        # This hardcoded index assumes the bands list above is unchanged.
        scl_band_index = 2
        masked = _cloud_mask(raw, scl_band_index=scl_band_index)

        # Handle all-cloudy case
        if masked.cloud_free_ratio == 0.0:
            logger.info("All observations cloudy for vegetation_health query")
            quality = _assess_quality(
                observation_count=1,
                cloud_free_count=0,
                cloud_cover_percentages=[1.0],
            )
            # Add FR14 fallback suggestion to warnings
            warnings = list(quality.warnings)
            warnings.append(
                "Consider using field.data.sentinel2() with relaxed cloud_max "
                "for manual analysis"
            )
            return VegetationResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=warnings,
                mean_ndvi=float("nan"),
                ndvi_std=float("nan"),
                trend=None,
                observation_count=quality.observation_count,
                cloud_free_count=quality.cloud_free_count,
            )

        # Step 3: Compute NDVI from masked bands
        # masked.data shape is (2, H, W) after SCL removal: band 0=B04, band 1=B08
        red = masked.data[0]
        nir = masked.data[1]
        ndvi = compute_ndvi(red, nir)

        # Step 4: Assess quality
        # For single-acquisition pipeline, observation_count=1
        cloud_free_count = 1 if masked.cloud_free_ratio > 0 else 0
        quality = _assess_quality(
            observation_count=1,
            cloud_free_count=cloud_free_count,
            cloud_cover_percentages=[1.0 - masked.cloud_free_ratio],
        )

        # Step 5: Build result metadata
        meta = raw.metadata
        result_meta = ResultMetadata(
            source="cdse",
            timestamps=[meta.get("timestamp", "")],
            observation_count=1,
            cloud_cover_pct=(1.0 - masked.cloud_free_ratio) * 100,
            bounds=meta.get("bounds", {}),
            resolution_m=meta.get("resolution_m"),
            bands=["NDVI"],
        )

        # Step 6: Build and return VegetationResult
        return _build_result(ndvi, quality, result_meta)

    def _compute_vegetation_for_period(
        self,
        period: tuple[str, str],
    ) -> VegetationResult:
        """Compute vegetation health for a specific date range.

        Internal helper that reuses the vegetation_health pipeline logic
        but accepts explicit start/end dates instead of last_days.

        Args:
            period: Date range as (start_date, end_date) ISO strings.

        Returns:
            VegetationResult for the specified period.
        """
        from satellitehub._pipeline import (
            _acquire,
            _assess_quality,
            _build_result,
            _cloud_mask,
        )
        from satellitehub.analysis.vegetation import compute_ndvi

        start_date, end_date = period

        # Acquire raw Sentinel-2 data for the specific period
        raw = _acquire(
            location=self,
            provider_name="cdse",
            product="sentinel2-l2a",
            bands=["B04", "B08", "SCL"],
            cloud_max=_DEFAULT_CLOUD_MAX,
            time_range=(start_date, end_date),
        )

        # Handle no-data case
        if raw.data.size == 0:
            logger.info(
                "No Sentinel-2 data available for period %s to %s",
                start_date,
                end_date,
            )
            return VegetationResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=[
                    f"No satellite observations available for "
                    f"{start_date} to {end_date}",
                    "Consider using field.data.sentinel2() with relaxed cloud_max "
                    "for manual analysis",
                ],
                mean_ndvi=float("nan"),
                ndvi_std=float("nan"),
                trend=None,
                observation_count=0,
                cloud_free_count=0,
            )

        # Apply cloud masking
        scl_band_index = 2
        masked = _cloud_mask(raw, scl_band_index=scl_band_index)

        # Handle all-cloudy case
        if masked.cloud_free_ratio == 0.0:
            logger.info(
                "All observations cloudy for period %s to %s",
                start_date,
                end_date,
            )
            quality = _assess_quality(
                observation_count=1,
                cloud_free_count=0,
                cloud_cover_percentages=[1.0],
            )
            # Add FR14 fallback suggestion to warnings
            warnings = list(quality.warnings)
            warnings.append(
                "Consider using field.data.sentinel2() with relaxed cloud_max "
                "for manual analysis"
            )
            return VegetationResult(
                data=np.array([], dtype=np.float32),
                confidence=0.0,
                metadata=ResultMetadata(source="cdse"),
                warnings=warnings,
                mean_ndvi=float("nan"),
                ndvi_std=float("nan"),
                trend=None,
                observation_count=quality.observation_count,
                cloud_free_count=quality.cloud_free_count,
            )

        # Compute NDVI
        red = masked.data[0]
        nir = masked.data[1]
        ndvi = compute_ndvi(red, nir)

        # Assess quality
        cloud_free_count = 1 if masked.cloud_free_ratio > 0 else 0
        quality = _assess_quality(
            observation_count=1,
            cloud_free_count=cloud_free_count,
            cloud_cover_percentages=[1.0 - masked.cloud_free_ratio],
        )

        # Build result metadata
        meta = raw.metadata
        result_meta = ResultMetadata(
            source="cdse",
            timestamps=[meta.get("timestamp", "")],
            observation_count=1,
            cloud_cover_pct=(1.0 - masked.cloud_free_ratio) * 100,
            bounds=meta.get("bounds", {}),
            resolution_m=meta.get("resolution_m"),
            bands=["NDVI"],
        )

        return _build_result(ndvi, quality, result_meta)

    def vegetation_change(
        self,
        period_1: tuple[str, str],
        period_2: tuple[str, str],
    ) -> ChangeResult:
        """Compare vegetation health between two time periods.

        Computes NDVI for both periods and returns the change magnitude
        with plain-language interpretation for consulting deliverables.

        Args:
            period_1: Baseline period as (start_date, end_date) ISO strings.
            period_2: Comparison period as (start_date, end_date) ISO strings.

        Returns:
            ChangeResult with per-period NDVI, delta, direction, and confidence.
            If either period has no cloud-free data, returns a result with
            confidence=0.0 for that period and descriptive warnings.

        Raises:
            ConfigurationError: If Copernicus credentials are not configured.
            ProviderError: If CDSE is unreachable after retries.

        Example:
            >>> import satellitehub as sh
            >>> sh.configure(copernicus_credentials="path/to/credentials.json")
            >>> field = sh.location(lat=51.25, lon=22.57)
            >>> result = field.vegetation_change(
            ...     period_1=("2025-01-01", "2025-01-31"),
            ...     period_2=("2026-01-01", "2026-01-31"),
            ... )
            >>> result.delta  # doctest: +SKIP
            -0.15
            >>> result.direction  # doctest: +SKIP
            'declining'
        """
        # Step 1: Compute vegetation health for period 1
        result_1 = self._compute_vegetation_for_period(period_1)

        # Step 2: Compute vegetation health for period 2
        result_2 = self._compute_vegetation_for_period(period_2)

        # Step 3: Compute delta and direction
        delta = result_2.mean_ndvi - result_1.mean_ndvi
        direction = self._determine_change_direction(
            delta, result_1.confidence, result_2.confidence, _STABLE_THRESHOLD
        )

        # Step 4: Combine confidence (minimum of both periods)
        combined_confidence = min(result_1.confidence, result_2.confidence)

        # Step 5: Collect warnings from both periods
        warnings: list[str] = []
        if result_1.confidence == 0.0:
            warnings.append(
                f"Period 1 ({period_1[0]} to {period_1[1]}): insufficient data"
            )
        if result_2.confidence == 0.0:
            warnings.append(
                f"Period 2 ({period_2[0]} to {period_2[1]}): insufficient data"
            )

        # Add FR14 fallback warning if both periods have no data
        if result_1.confidence == 0.0 and result_2.confidence == 0.0:
            warnings.append(
                "Consider using field.data.sentinel2() with relaxed cloud_max "
                "for manual analysis"
            )

        # Include any specific warnings from individual periods
        for w in result_1.warnings:
            if w not in warnings:
                warnings.append(w)
        for w in result_2.warnings:
            if w not in warnings:
                warnings.append(w)

        # Step 6: Build combined metadata
        combined_meta = ResultMetadata(
            source="cdse",
            timestamps=(
                result_1.metadata.timestamps + result_2.metadata.timestamps
            ),
            observation_count=(
                result_1.metadata.observation_count
                + result_2.metadata.observation_count
            ),
        )

        # Step 7: Determine data array
        # Stack period data if both have valid arrays, else empty
        if result_1.data.size > 0 and result_2.data.size > 0:
            data = np.stack([result_1.data, result_2.data])
        else:
            data = np.array([], dtype=np.float32)

        return ChangeResult(
            data=data,
            confidence=combined_confidence,
            metadata=combined_meta,
            warnings=warnings,
            period_1_ndvi=result_1.mean_ndvi,
            period_2_ndvi=result_2.mean_ndvi,
            delta=delta if not math.isnan(delta) else float("nan"),
            direction=direction,
            period_1_confidence=result_1.confidence,
            period_2_confidence=result_2.confidence,
            period_1_range=period_1,
            period_2_range=period_2,
        )

    def weather(self, last_days: int = 30) -> WeatherResult:
        """Query weather data for this location.

        Retrieves weather data from ERA5 (global climate reanalysis) and
        IMGW (Polish weather stations) for the specified time period,
        aggregates it into daily summaries, and returns a WeatherResult
        with temperature and precipitation statistics.

        Args:
            last_days: Number of days to look back from today. Defaults to 30.

        Returns:
            WeatherResult with temperature, precipitation, confidence score,
            and quality metadata. If no weather data is available, returns
            a result with confidence=0.0 and descriptive warnings.

        Example:
            >>> import satellitehub as sh
            >>> sh.configure(cds_api_key="your-api-key")
            >>> field = sh.location(lat=51.25, lon=22.57)
            >>> result = field.weather(last_days=30)
            >>> result.mean_temperature  # doctest: +SKIP
            8.5
            >>> result.total_precipitation  # doctest: +SKIP
            45.2
        """
        from satellitehub._pipeline import _acquire_weather
        from satellitehub.analysis.weather import build_weather_result

        # Calculate time range
        now = datetime.now(timezone.utc)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=last_days)).strftime("%Y-%m-%d")
        time_range = (start_date, end_date)

        # Step 1: Acquire ERA5 data (global coverage)
        era5_raw = None
        try:
            era5_raw = _acquire_weather(
                location=self,
                provider_name="cds",
                variables=None,  # Default variables
                last_days=last_days,
            )
            if era5_raw.data.size == 0:
                era5_raw = None
        except (ProviderError, ConfigurationError) as exc:
            logger.warning("ERA5 data acquisition failed: %s", exc)

        # Step 2: Acquire IMGW data (Polish stations only)
        imgw_raw = None
        # Only try IMGW for locations roughly in Poland (lat: 49-55, lon: 14-24)
        if 49.0 <= self._lat <= 55.0 and 14.0 <= self._lon <= 24.0:
            try:
                imgw_raw = _acquire_weather(
                    location=self,
                    provider_name="imgw",
                    station_type="synop",
                    last_days=last_days,
                )
                if imgw_raw.data.size == 0:
                    imgw_raw = None
            except (ProviderError, ConfigurationError) as exc:
                logger.warning("IMGW data acquisition failed: %s", exc)

        # Step 3: Build WeatherResult from raw data
        result = build_weather_result(
            era5_data=era5_raw,
            imgw_data=imgw_raw,
            location=self,
            time_range=time_range,
        )

        return result

    @staticmethod
    def _determine_change_direction(
        delta: float,
        conf_1: float,
        conf_2: float,
        stable_threshold: float,
    ) -> str:
        """Determine vegetation change direction from NDVI delta.

        Args:
            delta: NDVI change (period_2 - period_1).
            conf_1: Confidence score for period 1.
            conf_2: Confidence score for period 2.
            stable_threshold: Delta below this absolute value is "stable".

        Returns:
            Direction string: "improving", "declining", "stable", or "unknown".
        """
        if conf_1 == 0.0 or conf_2 == 0.0:
            return "unknown"
        if math.isnan(delta):
            return "unknown"
        if abs(delta) < stable_threshold:
            return "stable"
        return "improving" if delta > 0 else "declining"

    # ── Dunder methods ───────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """Two locations are equal when their coordinates match."""
        if not isinstance(other, Location):
            return NotImplemented
        return self._lat == other._lat and self._lon == other._lon

    def __hash__(self) -> int:
        """Hash based on coordinate pair for use as dict keys."""
        return hash((self._lat, self._lon))

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Location(lat={self._lat}, lon={self._lon}, utm_zone={self._utm_zone})"
