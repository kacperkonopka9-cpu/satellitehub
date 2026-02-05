"""IMGW Polish meteorological station data access."""

from __future__ import annotations

import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
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
# IMGW API constants (Task 1)
# ---------------------------------------------------------------------------

_IMGW_API_URL = "https://danepubliczne.imgw.pl/api/data"
_IMGW_SYNOP_URL = f"{_IMGW_API_URL}/synop"

# ---------------------------------------------------------------------------
# Station type constants (MVP: only synoptic stations supported)
# ---------------------------------------------------------------------------

_SUPPORTED_STATION_TYPE = "synop"  # MVP supports synoptic stations only

# ---------------------------------------------------------------------------
# Timeout constants (NFR13: 30-second connection timeout)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # seconds for connection/request timeout
_STATUS_TIMEOUT = 10  # shorter timeout for status checks

# ---------------------------------------------------------------------------
# Retry constants (NFR18: up to 3 retries with backoff)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 60.0  # seconds
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 408})
_SUCCESS_STATUS_CODES = frozenset({200})

# ---------------------------------------------------------------------------
# Station search constants
# ---------------------------------------------------------------------------

_MAX_STATION_DISTANCE_KM = 50.0  # Maximum distance to search for stations
_EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers

# ---------------------------------------------------------------------------
# Polish synoptic stations metadata
# ---------------------------------------------------------------------------

_IMGW_STATIONS: dict[str, dict[str, Any]] = {
    "12375": {"name": "Warszawa-Okecie", "lat": 52.1656, "lon": 20.9667},
    "12566": {"name": "Krakow-Balice", "lat": 50.0778, "lon": 19.7986},
    "12424": {"name": "Wroclaw", "lat": 51.1027, "lon": 16.8859},
    "12330": {"name": "Poznan", "lat": 52.4211, "lon": 16.8263},
    "12135": {"name": "Gdansk-Rebiechowo", "lat": 54.3856, "lon": 18.4669},
    "12495": {"name": "Lublin-Radawiec", "lat": 51.2194, "lon": 22.3964},
    "12560": {"name": "Katowice", "lat": 50.2389, "lon": 19.0322},
    "12205": {"name": "Szczecin-Dabie", "lat": 53.3942, "lon": 14.6233},
    "12418": {"name": "Lodz-Lublinek", "lat": 51.7167, "lon": 19.3972},
    "12585": {"name": "Rzeszow-Jasionka", "lat": 50.1089, "lon": 22.0192},
    "12295": {"name": "Bydgoszcz", "lat": 53.0967, "lon": 17.9764},
    "12160": {"name": "Kolobrzeg", "lat": 54.1822, "lon": 15.5833},
    "12120": {"name": "Leba", "lat": 54.7536, "lon": 17.5342},
    "12100": {"name": "Hel", "lat": 54.6083, "lon": 18.8017},
    "12570": {"name": "Bielsko-Biala", "lat": 49.8058, "lon": 19.0028},
    "12500": {"name": "Kielce", "lat": 50.8133, "lon": 20.6267},
    "12540": {"name": "Tarnow", "lat": 50.0303, "lon": 20.9861},
    "12455": {"name": "Zamosc", "lat": 50.7083, "lon": 23.2500},
    "12385": {"name": "Siedlce", "lat": 52.2583, "lon": 22.2667},
    "12195": {"name": "Swinoujscie", "lat": 53.9167, "lon": 14.2333},
    "12105": {"name": "Ustka", "lat": 54.5833, "lon": 16.8667},
    "12250": {"name": "Pila", "lat": 53.1333, "lon": 16.7500},
    "12270": {"name": "Torun", "lat": 53.0417, "lon": 18.5833},
    "12280": {"name": "Wloclawek", "lat": 52.6167, "lon": 19.0667},
    "12360": {"name": "Plock", "lat": 52.5833, "lon": 19.7333},
    "12400": {"name": "Suwalki", "lat": 54.1333, "lon": 22.9500},
    "12410": {"name": "Bialystok", "lat": 53.1000, "lon": 23.1667},
    "12435": {"name": "Terespol", "lat": 52.0667, "lon": 23.6167},
    "12465": {"name": "Wlodawa", "lat": 51.5500, "lon": 23.5333},
    "12575": {"name": "Zakopane", "lat": 49.2931, "lon": 19.9506},
    "12580": {"name": "Kasprowy-Wierch", "lat": 49.2317, "lon": 19.9817},
    "12595": {"name": "Lesko", "lat": 49.4667, "lon": 22.3333},
}


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.

    Returns:
        Distance between points in kilometers.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_KM * c


class IMGWProvider(DataProvider):
    """IMGW Polish weather data provider.

    Provides access to Polish Institute of Meteorology and Water
    Management station data. IMGW data is publicly available and
    requires no authentication.

    Args:
        config: Frozen configuration snapshot from the Location.

    Example:
        >>> from satellitehub.config import Config
        >>> provider = IMGWProvider(config=Config())
        >>> provider.name
        'imgw'
    """

    _name: str = "imgw"

    def __init__(self, config: Config) -> None:
        """Initialize IMGW provider with configuration.

        Args:
            config: SDK configuration captured at Location creation.
        """
        super().__init__(config)
        self._session: requests.Session = requests.Session()
        self._authenticated: bool = False

    def authenticate(self, credentials: ProviderCredentials) -> None:
        """IMGW is public data, no authentication needed.

        This method succeeds as a no-op since IMGW provides public access
        to weather data without requiring credentials.

        Args:
            credentials: Ignored for IMGW (public API).

        Example:
            >>> provider = IMGWProvider(config=Config())
            >>> provider.authenticate(ProviderCredentials())
            >>> provider._authenticated
            True
        """
        # IMGW is public data, no authentication needed
        self._authenticated = True
        logger.debug("IMGW authentication skipped (public API)")

    def _find_nearest_stations(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = _MAX_STATION_DISTANCE_KM,
    ) -> list[tuple[str, str, float]]:
        """Find nearest IMGW stations to a given location.

        Args:
            lat: Latitude of target location.
            lon: Longitude of target location.
            max_distance_km: Maximum search radius in kilometers.

        Returns:
            List of (station_id, station_name, distance_km) tuples,
            sorted by distance (nearest first).
        """
        stations_with_distance: list[tuple[str, str, float]] = []

        for station_id, station_info in _IMGW_STATIONS.items():
            distance = _haversine_distance(
                lat, lon, station_info["lat"], station_info["lon"]
            )
            if distance <= max_distance_km:
                stations_with_distance.append(
                    (station_id, station_info["name"], distance)
                )

        # Sort by distance (nearest first)
        stations_with_distance.sort(key=lambda x: x[2])

        return stations_with_distance

    def search(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> list[CatalogEntry]:
        """Search for nearest IMGW stations with available data.

        IMGW data comes from physical weather stations, not gridded products.
        This method finds the nearest synoptic stations to the requested
        location and returns catalog entries for each.

        Note:
            MVP supports synoptic stations only. The ``station_type`` parameter
            is accepted for future compatibility but currently only "synop"
            stations are available.

        Args:
            location: Geographic location to search around.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Additional IMGW-specific search parameters.
                ``max_distance_km``: Maximum station search radius (default: 50km).

        Returns:
            List of catalog entries for nearby stations, empty if none found
            within the search radius.

        Example:
            >>> provider.search(location, ("2024-01-01", "2024-01-31"))
            [CatalogEntry(provider='imgw', ...)]
        """
        max_distance = params.get("max_distance_km", _MAX_STATION_DISTANCE_KM)
        station_type = _SUPPORTED_STATION_TYPE  # MVP: synoptic only

        start_date = time_range[0]
        end_date = time_range[1]

        # Find nearest stations
        nearby_stations = self._find_nearest_stations(
            location.lat, location.lon, max_distance
        )

        if not nearby_stations:
            logger.debug(
                "No IMGW stations found within %.1f km of (%.4f, %.4f)",
                max_distance,
                location.lat,
                location.lon,
            )
            return []

        # Build catalog entries for each nearby station
        entries: list[CatalogEntry] = []
        for station_id, station_name, distance in nearby_stations:
            station_info = _IMGW_STATIONS[station_id]
            entry = CatalogEntry(
                provider="imgw",
                product_id=f"{station_type}:{station_id}:{start_date}:{end_date}",
                timestamp=start_date,
                cloud_cover=0.0,  # Not applicable for weather stations
                geometry={
                    "type": "Point",
                    "coordinates": [station_info["lon"], station_info["lat"]],
                },
                bands_available=[
                    "temperature", "precipitation", "humidity", "pressure"
                ],
                metadata={
                    "station_id": station_id,
                    "station_name": station_name,
                    "station_type": station_type,
                    "distance_km": str(round(distance, 2)),
                    "start_date": start_date,
                    "end_date": end_date,
                    "lat": str(station_info["lat"]),
                    "lon": str(station_info["lon"]),
                },
            )
            entries.append(entry)

        logger.debug(
            "Found %d IMGW stations within %.1f km of (%.4f, %.4f)",
            len(entries),
            max_distance,
            location.lat,
            location.lon,
        )

        return entries

    def download(
        self,
        entry: CatalogEntry,
        bands: list[str] | None = None,
    ) -> RawData:
        """Download IMGW station measurements.

        Retrieves weather measurements from the specified IMGW station.
        IMGW provides point measurements (not gridded data), so the result
        contains measurement data for the station location.

        Note:
            MVP returns the **current** observation only. The IMGW public API
            provides real-time data; historical data retrieval will be
            implemented in a future story. The time_range in the catalog
            entry is stored in metadata but does not affect the API call.

        Args:
            entry: Catalog entry from ``search()`` to download.
            bands: Optional list of specific measurements to download.
                Available: "temperature", "precipitation", "humidity", "pressure".

        Returns:
            Raw data with numpy arrays and station metadata.

        Raises:
            ProviderError: If download fails after retries exhausted.

        Example:
            >>> provider.download(entry, bands=["temperature"])  # doctest: +SKIP
        """
        station_id = entry.metadata.get("station_id", "")
        station_name = entry.metadata.get("station_name", "Unknown")
        distance_km = float(entry.metadata.get("distance_km", "0.0"))

        if not station_id:
            raise ProviderError(
                what="IMGW download failed",
                cause="Missing station_id in catalog entry",
                fix="Use search() to get valid catalog entries",
            )

        # Build API URL for station data (MVP: current observation only)
        url = f"{_IMGW_SYNOP_URL}/station/{station_id}"

        # MVP: Log that we're fetching current data, not historical
        logger.info(
            "Downloading IMGW data from station %s (%s)...", station_id, station_name
        )
        logger.debug(
            "MVP: Fetching current observation only. "
            "Historical data retrieval not yet implemented."
        )

        # Fetch station data with retry
        resp = self._retry_request("get", url)

        # Parse response
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                what="IMGW download failed",
                cause="Invalid JSON response",
                fix="Try again; if persistent, check IMGW API status",
            ) from exc

        # Handle both single object and list responses
        if isinstance(data, list):
            if not data:
                raise ProviderError(
                    what="IMGW download failed",
                    cause="Empty response from station",
                    fix="Station may be temporarily unavailable",
                )
            data = data[0]

        # Extract measurements
        measurements = self._parse_measurements(data, bands)

        # Get station coordinates from metadata
        station_lat = float(entry.metadata.get("lat", "0"))
        station_lon = float(entry.metadata.get("lon", "0"))

        return RawData(
            data=measurements["data"],
            metadata={
                "provider": "imgw",
                "station_id": station_id,
                "station_name": station_name,
                "distance_km": distance_km,
                "measurement_type": "synoptic",
                "measurements": measurements["fields"],
                "timestamp": data.get("data_pomiaru", ""),
                "hour": data.get("godzina_pomiaru", ""),
                "coordinates": {"lat": station_lat, "lon": station_lon},
            },
        )

    def _parse_measurements(
        self,
        data: dict[str, Any],
        bands: list[str] | None,
    ) -> dict[str, Any]:
        """Parse IMGW API response into measurements.

        Args:
            data: Raw API response dictionary.
            bands: Optional filter for specific measurements.

        Returns:
            Dictionary with 'data' (numpy array) and 'fields' (list of field names).
        """
        # Map band names to IMGW API field names
        field_mapping = {
            "temperature": "temperatura",
            "precipitation": "suma_opadu",
            "humidity": "wilgotnosc_wzgledna",
            "pressure": "cisnienie",
            "wind_speed": "predkosc_wiatru",
            "wind_direction": "kierunek_wiatru",
        }

        # Determine which fields to extract
        if bands:
            fields_to_extract = [
                (band, field_mapping.get(band, band))
                for band in bands
                if band in field_mapping
            ]
        else:
            fields_to_extract = list(field_mapping.items())

        # Extract values
        values: list[float] = []
        extracted_fields: list[str] = []

        for band_name, api_field in fields_to_extract:
            raw_value = data.get(api_field)
            if raw_value is not None and raw_value != "":
                try:
                    values.append(float(raw_value))
                    extracted_fields.append(band_name)
                except (ValueError, TypeError):
                    # Skip fields that can't be converted to float
                    pass

        if not values:
            logger.warning(
                "No valid measurements extracted from IMGW response. "
                "All fields may be empty or null."
            )

        return {
            "data": np.array(values, dtype=np.float32),
            "fields": extracted_fields,
        }

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

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    raise ProviderError(
                        what="IMGW request failed",
                        cause=f"HTTP {resp.status_code}",
                        fix="Check IMGW API status at https://danepubliczne.imgw.pl",
                    )

                # Retryable status code - compute backoff
                backoff = self._compute_backoff(attempt)
                logger.warning(
                    "IMGW request failed (HTTP %d, attempt %d/%d), "
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
                        "IMGW request failed (%s, attempt %d/%d), retrying in %.1fs...",
                        type(exc).__name__,
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)

        # All retries exhausted
        if last_exc is not None:
            raise ProviderError(
                what="IMGW request failed after retries",
                cause=str(last_exc),
                fix="Check internet connection and try again",
            ) from last_exc

        raise ProviderError(
            what="IMGW request failed after retries",
            cause=f"HTTP {last_status} after {_MAX_RETRIES} retries",
            fix="Check IMGW API status at https://danepubliczne.imgw.pl",
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
        """Check IMGW API operational status.

        Makes a lightweight request to verify the IMGW API is reachable.
        Never raises - returns ``ProviderStatus`` with ``available=False``
        and a descriptive message on any failure.

        Returns:
            Current operational status of the IMGW API.

        Example:
            >>> provider = IMGWProvider(config=Config())
            >>> provider.check_status().available  # doctest: +SKIP
            True
        """
        try:
            resp = self._session.get(_IMGW_SYNOP_URL, timeout=_STATUS_TIMEOUT)
            if resp.status_code in _SUCCESS_STATUS_CODES:
                return ProviderStatus(available=True)
            return ProviderStatus(
                available=False,
                message=f"IMGW returned HTTP {resp.status_code}",
            )
        except requests.RequestException as exc:
            return ProviderStatus(
                available=False,
                message=f"IMGW API unreachable: {exc}",
            )
