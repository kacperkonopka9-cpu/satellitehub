"""Pipeline helpers for data acquisition and processing.

Implements AD-4 pipeline orchestration. The ``_acquire`` function is the
central data-fetching helper used by data-tier methods on ``Location``.
"""

from __future__ import annotations

import json
import logging
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from satellitehub._types import MaskedData, QualityAssessment, RawData, TimeRange
from satellitehub.config import load_credentials, resolve_credentials_path
from satellitehub.providers.base import ProviderCredentials
from satellitehub.results import ResultMetadata, VegetationResult

if TYPE_CHECKING:
    from satellitehub.location import Location

logger = logging.getLogger(__name__)

# ESA Sentinel-2 Scene Classification Layer (SCL) classes to mask.
# Values: 0=NoData, 1=Saturated, 3=CloudShadow, 8=CloudMedium,
#         9=CloudHigh, 10=ThinCirrus.
_SCL_MASK_CLASSES: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})
_SCL_MASK_ARRAY: npt.NDArray[np.int32] = np.array(
    sorted(_SCL_MASK_CLASSES), dtype=np.int32
)


# ── Serialization helpers ──────────────────────────────────────────


def _serialize_raw_data(raw: RawData) -> bytes:
    """Serialize a ``RawData`` instance to bytes for cache storage.

    Format: 4-byte header length (uint32 big-endian) + JSON header + array bytes.

    The JSON header stores array shape, dtype, and provider metadata so
    that reconstruction is lossless.

    Args:
        raw: The raw data to serialize.

    Returns:
        Opaque bytes suitable for ``CacheManager.store()``.
    """
    header: dict[str, Any] = {
        "shape": list(raw.data.shape),
        "dtype": str(raw.data.dtype),
        "metadata": raw.metadata,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    array_bytes = raw.data.tobytes()
    length_prefix = struct.pack(">I", len(header_bytes))
    return length_prefix + header_bytes + array_bytes


def _deserialize_raw_data(data: bytes) -> RawData:
    """Reconstruct a ``RawData`` instance from cache bytes.

    Inverse of ``_serialize_raw_data``. Reads the length prefix, extracts
    the JSON header, and rebuilds the numpy array with correct shape and dtype.

    Args:
        data: Bytes previously produced by ``_serialize_raw_data``.

    Returns:
        Reconstructed ``RawData`` with original array and metadata.
    """
    (header_len,) = struct.unpack(">I", data[:4])
    header_bytes = data[4 : 4 + header_len]
    array_bytes = data[4 + header_len :]

    header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
    shape = tuple(header["shape"])
    dtype = np.dtype(header["dtype"])
    metadata: dict[str, Any] = header["metadata"]

    array: npt.NDArray[np.floating[Any]] = np.frombuffer(
        array_bytes, dtype=dtype
    ).reshape(shape)

    return RawData(data=array.copy(), metadata=metadata)


# ── Cloud masking helper ───────────────────────────────────────────


def _cloud_mask(raw: RawData, scl_band_index: int) -> MaskedData:
    """Apply ESA SCL-based cloud masking to raw Sentinel-2 data.

    Pixels whose SCL class is in ``_SCL_MASK_CLASSES`` are set to NaN in
    every spectral band. The SCL band itself is removed from the output
    because it is classification metadata, not reflectance data.

    Args:
        raw: Raw Sentinel-2 data with an SCL band included.
        scl_band_index: Index of the SCL band in ``raw.data``
            (axis 0 of the ``(bands, height, width)`` array).

    Returns:
        ``MaskedData`` with cloudy pixels set to NaN, the boolean
        validity mask, and the cloud-free pixel ratio.
    """
    data = raw.data

    # Remove SCL band dimension from output regardless of path
    keep_indices = [i for i in range(data.shape[0]) if i != scl_band_index]

    # Handle empty / zero-sized arrays
    if data.size == 0:
        spectral_shape = (len(keep_indices), *data.shape[1:])
        return MaskedData(
            data=np.empty(spectral_shape, dtype=data.dtype),
            mask=None,
            cloud_free_ratio=0.0,
        )

    # Extract SCL band (integer classification values)
    scl_band = data[scl_band_index]

    # Build boolean validity mask: True = cloud-free (keep)
    valid_mask: npt.NDArray[np.bool_] = ~np.isin(
        scl_band.astype(np.int32), _SCL_MASK_ARRAY
    )

    # Compute cloud-free ratio
    total_pixels = valid_mask.size
    cloud_free_ratio = (
        float(np.count_nonzero(valid_mask)) / total_pixels if total_pixels > 0 else 0.0
    )

    # Keep only spectral bands; use float for NaN support
    out_dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float64
    spectral_data = data[keep_indices].astype(out_dtype).copy()

    # Apply mask: set invalid pixels to NaN in all spectral bands
    spectral_data[:, ~valid_mask] = np.nan

    return MaskedData(
        data=spectral_data,
        mask=valid_mask,
        cloud_free_ratio=cloud_free_ratio,
    )


# ── Quality assessment helper ──────────────────────────────────────

# Confidence scoring weights.
_RATIO_WEIGHT: float = 0.7
_COUNT_WEIGHT: float = 0.3
_MIN_ADEQUATE_OBSERVATIONS: int = 3


def _assess_quality(
    observation_count: int,
    cloud_free_count: int,
    cloud_cover_percentages: list[float] | None = None,
) -> QualityAssessment:
    """Compute quality assessment for satellite analysis results.

    Produces a ``QualityAssessment`` with a confidence score, observation
    metadata, and human-readable warnings for limited-data scenarios.

    The confidence score combines two factors:

    * **Cloud-free ratio** (weight 0.7): proportion of usable observations.
    * **Observation adequacy** (weight 0.3): whether enough passes exist
      for reliable temporal coverage (minimum 3).

    When cloud-free observations are very sparse (≤ 2), confidence is
    capped to prevent over-estimation from high adequacy scores.

    If ``cloud_free_count`` exceeds ``observation_count``, it is clamped
    to ``observation_count`` to prevent invalid confidence values.

    Args:
        observation_count: Total satellite passes available (≥ 0).
        cloud_free_count: Passes with usable cloud-free data (≥ 0).
        cloud_cover_percentages: Per-observation cloud cover (0.0–1.0 each).
            Defaults to an empty list when ``None``.

    Returns:
        ``QualityAssessment`` with confidence in [0.0, 1.0] and warnings.
    """
    if cloud_cover_percentages is None:
        cloud_cover_percentages = []
    else:
        cloud_cover_percentages = list(cloud_cover_percentages)

    # Clamp invalid inputs to sane values
    observation_count = max(observation_count, 0)
    cloud_free_count = max(min(cloud_free_count, observation_count), 0)

    warnings: list[str] = []

    # ── Confidence computation ────────────────────────────────────
    if observation_count == 0:
        confidence = 0.0
        warnings.append("No satellite observations available for the requested period")
    elif cloud_free_count == 0:
        confidence = 0.0
        warnings.append(
            f"Insufficient cloud-free observations: "
            f"0 of {observation_count} passes usable"
        )
    else:
        ratio_score = cloud_free_count / observation_count
        count_score = min(observation_count / _MIN_ADEQUATE_OBSERVATIONS, 1.0)
        confidence = _RATIO_WEIGHT * ratio_score + _COUNT_WEIGHT * count_score

        # Penalise very sparse cloud-free data (≤ 2 usable passes)
        if cloud_free_count <= 2:
            confidence = min(confidence, cloud_free_count * 0.15)

        confidence = min(max(confidence, 0.0), 1.0)

    # ── Warning generation ────────────────────────────────────────
    if 0 < cloud_free_count <= 2:
        warnings.append(
            f"Limited cloud-free observations: "
            f"{cloud_free_count} of {observation_count} passes usable"
        )

    excluded = observation_count - cloud_free_count
    if excluded > 0 and cloud_free_count > 0:
        warnings.append(f"{excluded} passes excluded (cloud cover >80%)")

    return QualityAssessment(
        confidence=confidence,
        observation_count=observation_count,
        cloud_free_count=cloud_free_count,
        cloud_cover_percentages=cloud_cover_percentages,
        warnings=warnings,
    )


# ── Result construction helper ─────────────────────────────────────


def _build_result(
    ndvi: npt.NDArray[np.floating[Any]],
    quality: QualityAssessment,
    metadata: ResultMetadata,
) -> VegetationResult:
    """Assemble a ``VegetationResult`` from pipeline outputs.

    Combines the NDVI array, quality assessment, and metadata into a
    complete result object. Handles empty and all-NaN arrays gracefully,
    returning NaN statistics instead of raising.

    Args:
        ndvi: NDVI array from ``compute_ndvi()``. May contain NaN for
            cloud-masked pixels.
        quality: Quality assessment from ``_assess_quality()``.
        metadata: Result metadata (source, timestamps, bounds, etc.).

    Returns:
        ``VegetationResult`` with NDVI statistics, confidence, and warnings.
    """
    if ndvi.size == 0 or np.all(np.isnan(ndvi)):
        mean_ndvi = float("nan")
        ndvi_std = float("nan")
    else:
        mean_ndvi = float(np.nanmean(ndvi))
        ndvi_std = float(np.nanstd(ndvi))

    return VegetationResult(
        data=ndvi,
        confidence=quality.confidence,
        metadata=metadata,
        warnings=list(quality.warnings),
        mean_ndvi=mean_ndvi,
        ndvi_std=ndvi_std,
        trend=None,
        observation_count=quality.observation_count,
        cloud_free_count=quality.cloud_free_count,
    )


# ── Pipeline acquire helper ────────────────────────────────────────


def _acquire(
    location: Location,
    provider_name: str,
    product: str,
    bands: list[str] | None,
    cloud_max: float,
    last_days: int | None = None,
    time_range: TimeRange | None = None,
) -> RawData:
    """Acquire raw data through the cache-provider pipeline.

    Implements the AD-4 pipeline flow: cache check → authenticate →
    search catalog → download → cache store → return ``RawData``.

    Infrastructure errors (``ProviderError``, ``ConfigurationError``)
    propagate to the caller. The data-tier methods on ``Location`` are
    responsible for graceful degradation.

    Args:
        location: Geographic location for the query.
        provider_name: Provider identifier (e.g., ``"cdse"``).
        product: Data product name (e.g., ``"sentinel2-l2a"``).
        bands: Specific bands to download, or ``None`` for all.
        cloud_max: Maximum acceptable cloud cover (0.0--1.0).
        last_days: Number of days to look back from today. Either this or
            ``time_range`` must be provided.
        time_range: Explicit (start_date, end_date) ISO strings. If provided,
            takes precedence over ``last_days``.

    Returns:
        ``RawData`` with numpy array and provider metadata.

    Raises:
        ProviderError: If the provider fails after retries.
        ConfigurationError: If credentials are missing or invalid.
        ValueError: If neither ``last_days`` nor ``time_range`` is provided.
    """
    config = location.config

    # Compute time range from parameters
    if time_range is not None:
        # Use explicit time range
        resolved_time_range: TimeRange = time_range
    elif last_days is not None:
        # Compute from last_days
        now = datetime.now(timezone.utc)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=last_days)).strftime("%Y-%m-%d")
        resolved_time_range = (start_date, end_date)
    else:
        raise ValueError("Either 'last_days' or 'time_range' must be provided")

    # Build cache key (reuse Location's cached CacheManager)
    cache = location.cache
    params: dict[str, str] = {}
    if bands is not None:
        params["bands"] = ",".join(bands)
    params["cloud_max"] = str(cloud_max)
    cache_key = cache._build_cache_key(
        provider=provider_name,
        product=product,
        location_hash=location.location_hash,
        time_range=resolved_time_range,
        params=params,
    )

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for %s", cache_key)
        return _deserialize_raw_data(cached)

    logger.debug("Cache miss for %s — fetching from provider", cache_key)

    # Get provider and authenticate
    provider = location.get_provider(provider_name)
    _credential_map: dict[str, Path | None] = {
        "cdse": config.copernicus_credentials,
        "cds": config.cds_credentials,
        "landsat": None,  # Planetary Computer is public, no auth needed
    }
    creds_path = resolve_credentials_path(
        explicit=_credential_map.get(provider_name),
    )
    if creds_path is not None:
        creds_data = load_credentials(creds_path)
        provider_section = creds_data.get(provider_name, {})
        provider.authenticate(
            ProviderCredentials(**provider_section)
            if provider_section
            else ProviderCredentials()
        )

    # Search catalog
    entries = provider.search(
        location=location,
        time_range=resolved_time_range,
        cloud_cover_max=cloud_max,
    )

    if not entries:
        logger.info("No catalog entries found for %s", provider_name)
        return RawData(data=np.array([], dtype=np.float32), metadata={})

    # Download first matching entry
    entry = entries[0]
    logger.info(
        "Downloading %s product %s...",
        provider_name,
        entry.product_id,
    )
    raw = provider.download(entry, bands=bands)

    # Store in cache
    serialized = _serialize_raw_data(raw)
    cache.store(
        cache_key=cache_key,
        provider=provider_name,
        product=product,
        location_hash=location.location_hash,
        data=serialized,
    )

    return raw


def _acquire_weather(
    location: Location,
    provider_name: str,
    variables: list[str] | None = None,
    station_type: str = "synop",
    last_days: int = 30,
) -> RawData:
    """Acquire raw weather data through the cache-provider pipeline.

    Implements weather-specific pipeline flow for CDS (ERA5) and IMGW
    providers: cache check → authenticate → search catalog → download →
    cache store → return ``RawData``.

    Args:
        location: Geographic location for the query.
        provider_name: Provider identifier (``"cds"`` or ``"imgw"``).
        variables: Climate variables to download (CDS only).
        station_type: Station type for IMGW (default: ``"synop"``).
        last_days: Number of days to look back from today. Defaults to 30.

    Returns:
        ``RawData`` with numpy array and provider metadata.

    Raises:
        ProviderError: If the provider fails after retries.
        ConfigurationError: If credentials are missing or invalid.
    """
    config = location.config

    # Compute time range
    now = datetime.now(timezone.utc)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=last_days)).strftime("%Y-%m-%d")
    time_range: TimeRange = (start_date, end_date)

    # Build cache key
    cache = location.cache
    params: dict[str, str] = {}
    if variables is not None:
        params["variables"] = ",".join(variables)
    if provider_name == "imgw":
        params["station_type"] = station_type
    cache_key = cache._build_cache_key(
        provider=provider_name,
        product="weather",
        location_hash=location.location_hash,
        time_range=time_range,
        params=params,
    )

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Weather cache hit for %s", cache_key)
        return _deserialize_raw_data(cached)

    logger.debug("Weather cache miss for %s — fetching from provider", cache_key)

    # Get provider and authenticate
    provider = location.get_provider(provider_name)

    # CDS requires API key authentication
    if provider_name == "cds":
        creds_path = resolve_credentials_path(explicit=config.cds_credentials)
        if creds_path is not None:
            creds_data = load_credentials(creds_path)
            provider_section = creds_data.get("cds", {})
            if provider_section:
                provider.authenticate(ProviderCredentials(**provider_section))
    # IMGW doesn't require authentication

    # Search catalog
    search_params: dict[str, Any] = {}
    if variables is not None:
        search_params["variables"] = variables
    if provider_name == "imgw":
        search_params["station_type"] = station_type

    entries = provider.search(
        location=location,
        time_range=time_range,
        **search_params,
    )

    if not entries:
        logger.info("No weather catalog entries found for %s", provider_name)
        return RawData(data=np.array([], dtype=np.float32), metadata={})

    # Download first matching entry
    entry = entries[0]
    logger.info(
        "Downloading %s weather data %s...",
        provider_name,
        entry.product_id,
    )
    raw = provider.download(entry)

    # Store in cache (6-hour TTL for weather data)
    serialized = _serialize_raw_data(raw)
    cache.store(
        cache_key=cache_key,
        provider=provider_name,
        product="weather",
        location_hash=location.location_hash,
        data=serialized,
    )

    return raw
