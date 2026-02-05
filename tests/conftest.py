"""Shared test fixtures for SatelliteHub test suite."""

from __future__ import annotations

import pytest

from satellitehub.config import Config
from satellitehub.location import Location, location


@pytest.fixture
def test_config() -> Config:
    """Return a fresh default Config instance for test isolation."""
    return Config()


@pytest.fixture
def test_location(test_config: Config) -> Location:
    """Return a Location for Warsaw using an explicit test config."""
    return location(lat=51.25, lon=22.57, config=test_config)
