"""SatelliteHub exception hierarchy.

All exceptions follow a three-part message pattern: what failed,
likely cause, and suggested fix (NFR29).
"""

from __future__ import annotations


class SatelliteHubError(Exception):
    """Base exception for all SatelliteHub errors.

    All SatelliteHub exceptions use a three-part message pattern
    providing structured error context for developers.

    Args:
        what: Description of what failed.
        cause: Likely cause of the failure.
        fix: Suggested action to resolve the issue.

    Example:
        >>> raise SatelliteHubError(
        ...     what="Operation failed",
        ...     cause="Unexpected internal state",
        ...     fix="Please report this issue",
        ... )
    """

    def __init__(
        self,
        what: str,
        cause: str = "",
        fix: str = "",
    ) -> None:
        """Initialize with structured error context.

        Args:
            what: Description of what failed.
            cause: Likely cause of the failure.
            fix: Suggested action to resolve the issue.
        """
        self.what = what
        self.cause = cause
        self.fix = fix
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Build the multi-line error message from parts.

        Returns:
            Formatted message with optional Cause and Fix lines.
        """
        parts = [self.what]
        if self.cause:
            parts.append(f"Cause: {self.cause}")
        if self.fix:
            parts.append(f"Fix: {self.fix}")
        return "\n".join(parts)


class ConfigurationError(SatelliteHubError):
    """Raised for configuration and credential errors.

    Example:
        >>> raise ConfigurationError(
        ...     what="Cannot read credentials file",
        ...     cause="File not found: ~/.satellitehub/credentials.json",
        ...     fix="Run 'satellitehub configure' or set env var",
        ... )
    """


class ProviderError(SatelliteHubError):
    """Raised for data provider failures after retries exhausted.

    Example:
        >>> raise ProviderError(
        ...     what="CDSE download failed",
        ...     cause="HTTP 503 after 3 retries",
        ...     fix="Check CDSE status at https://dataspace.copernicus.eu/",
        ... )
    """


class CacheError(SatelliteHubError):
    """Raised for cache subsystem errors.

    Example:
        >>> raise CacheError(
        ...     what="Cache database corrupted",
        ...     cause="SQLite integrity check failed",
        ...     fix="Delete cache directory and re-download data",
        ... )
    """
