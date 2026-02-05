"""Provider interface contract and shared types.

Defines the ``DataProvider`` abstract base class (AD-2) and the
provider-domain types used across all data source implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import requests
from pydantic import BaseModel, ConfigDict

from satellitehub._types import RawData, TimeRange
from satellitehub.config import Config

if TYPE_CHECKING:
    from satellitehub.location import Location


class ProviderCredentials(BaseModel):
    """Credentials for authenticating with a data provider.

    Immutable container for provider authentication details.
    Different providers use different credential fields: CDSE uses
    ``username``/``password``, CDS uses ``api_key``, and IMGW
    requires no authentication.

    Args:
        username: Provider account username.
        password: Provider account password.
        api_key: API key for key-based authentication.

    Example:
        >>> creds = ProviderCredentials(username="placeholder", password="placeholder")
        >>> creds.username
        'placeholder'
    """

    model_config = ConfigDict(frozen=True)

    username: str = ""
    password: str = ""
    api_key: str = ""


@dataclass
class CatalogEntry:
    """A data product entry from a provider catalog search.

    Represents a single discoverable data product (e.g., one Sentinel-2
    scene) returned by ``DataProvider.search()``.

    Args:
        provider: Provider name (e.g., ``"cdse"``, ``"cds"``).
        product_id: Provider-specific unique product identifier.
        timestamp: ISO-8601 acquisition timestamp.
        cloud_cover: Cloud cover fraction (0.0--1.0).
        geometry: GeoJSON-like spatial footprint of the product.
        bands_available: List of spectral bands available in this product.
        metadata: Additional provider-specific metadata.

    Example:
        >>> entry = CatalogEntry(
        ...     provider="cdse",
        ...     product_id="S2A_MSIL2A_20240101",
        ...     timestamp="2024-01-01T10:30:00Z",
        ...     cloud_cover=0.15,
        ...     bands_available=["B4", "B8"],
        ... )
    """

    provider: str = ""
    product_id: str = ""
    timestamp: str = ""
    cloud_cover: float = 0.0
    geometry: dict[str, Any] = field(default_factory=dict)
    bands_available: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderStatus:
    """Operational status of a data provider.

    Returned by ``DataProvider.check_status()`` to indicate whether
    the provider's API is reachable and functioning.

    Args:
        available: ``True`` if the provider is operational.
        message: Human-readable status message (empty when healthy).
        last_checked: ISO-8601 timestamp of the last status check.

    Example:
        >>> status = ProviderStatus(available=True)
        >>> status.message
        ''
    """

    available: bool = False
    message: str = ""
    last_checked: str = ""


class DataProvider(ABC):
    """Abstract base class for satellite and weather data providers.

    All data sources (CDSE, CDS, IMGW) implement this contract to
    provide a uniform interface for authentication, catalog search,
    data download, and status checking (AD-2).

    Subclasses must implement all four abstract methods and set the
    ``_name`` class attribute to a unique provider identifier.

    Args:
        config: Frozen configuration snapshot for this provider instance.

    Example:
        >>> from satellitehub.providers.cdse import CDSEProvider
        >>> provider = CDSEProvider(config=Config())
        >>> provider.name
        'cdse'
    """

    _name: str = ""

    def __init__(self, config: Config) -> None:
        """Initialize with frozen configuration snapshot.

        Args:
            config: SDK configuration captured at Location creation.
        """
        self._config = config
        self._session: requests.Session | None = None

    @property
    def name(self) -> str:
        """Provider identifier used in registry and cache keys."""
        return self._name

    @abstractmethod
    def authenticate(self, credentials: ProviderCredentials) -> None:
        """Validate and store provider credentials.

        Args:
            credentials: Provider-specific authentication credentials.

        Raises:
            ConfigurationError: If credentials are invalid or rejected.
        """
        ...

    @abstractmethod
    def search(
        self,
        location: Location,
        time_range: TimeRange,
        **params: Any,
    ) -> list[CatalogEntry]:
        """Search provider catalog for available data.

        Returns an empty list when no data matches the query.
        Never raises on missing data — only on infrastructure failures.

        Args:
            location: Geographic location to search around.
            time_range: ISO-8601 date pair ``(start, end)``.
            **params: Provider-specific search parameters.

        Returns:
            List of matching catalog entries, empty if none found.
        """
        ...

    @abstractmethod
    def download(
        self,
        entry: CatalogEntry,
        bands: list[str] | None = None,
    ) -> RawData:
        """Download data for a catalog entry.

        Handles retries internally. Raises ``ProviderError`` only
        after all retry attempts are exhausted.

        Args:
            entry: Catalog entry to download.
            bands: Optional list of specific bands to download.
                Downloads all available bands if ``None``.

        Returns:
            Raw data with numpy arrays and provider metadata.

        Raises:
            ProviderError: If download fails after retries exhausted.
        """
        ...

    @abstractmethod
    def check_status(self) -> ProviderStatus:
        """Check provider operational status.

        Never raises — returns a ``ProviderStatus`` with
        ``available=False`` and a descriptive message on failure.

        Returns:
            Current operational status of the provider.
        """
        ...
