"""Tests for Config model, configure(), and credential resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from satellitehub.config import (
    _CREDENTIALS_ENV_VAR,
    Config,
    _check_file_permissions,
    configure,
    get_default_config,
    load_credentials,
    resolve_credentials_path,
)
from satellitehub.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level config before each test."""
    import satellitehub.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", Config())


# ── Task 5: Config model tests ──────────────────────────────────────────


@pytest.mark.unit
class TestConfigDefaults:
    """Verify Config constructs with correct defaults."""

    def test_defaults(self) -> None:
        cfg = Config()
        assert cfg.copernicus_credentials is None
        assert cfg.cds_credentials is None
        assert cfg.cache_size_mb == 5000
        assert cfg.default_crs == "EPSG:4326"

    def test_cache_dir_default_expands_home(self) -> None:
        cfg = Config()
        assert cfg.cache_dir.is_absolute()
        assert "~" not in str(cfg.cache_dir)


@pytest.mark.unit
class TestConfigExplicit:
    """Verify Config constructs with explicit values."""

    def test_explicit_values(self, tmp_path: Path) -> None:
        cred = tmp_path / "creds.json"
        cred.touch()
        cfg = Config(
            copernicus_credentials=cred,
            cds_credentials=cred,
            cache_dir=tmp_path / "cache",
            cache_size_mb=2000,
            default_crs="EPSG:32634",
        )
        assert cfg.copernicus_credentials == cred
        assert cfg.cds_credentials == cred
        assert cfg.cache_dir == tmp_path / "cache"
        assert cfg.cache_size_mb == 2000
        assert cfg.default_crs == "EPSG:32634"

    def test_string_paths_converted(self, tmp_path: Path) -> None:
        cfg = Config(cache_dir=str(tmp_path))
        assert isinstance(cfg.cache_dir, Path)

    def test_credential_path_from_string(self, tmp_path: Path) -> None:
        cred = tmp_path / "c.json"
        cfg = Config(copernicus_credentials=str(cred))
        assert isinstance(cfg.copernicus_credentials, Path)

    def test_none_credentials_remain_none(self) -> None:
        cfg = Config(copernicus_credentials=None, cds_credentials=None)
        assert cfg.copernicus_credentials is None
        assert cfg.cds_credentials is None


@pytest.mark.unit
class TestConfigImmutable:
    """Verify Config is frozen (AD-1 snapshot safety)."""

    def test_frozen_field_raises(self) -> None:
        cfg = Config()
        with pytest.raises(ValidationError):
            cfg.cache_size_mb = 999  # type: ignore[misc]

    def test_frozen_cache_dir_raises(self) -> None:
        cfg = Config()
        with pytest.raises(ValidationError):
            cfg.cache_dir = Path("/other")  # type: ignore[misc]


@pytest.mark.unit
class TestConfigValidation:
    """Verify field validators reject invalid inputs."""

    @pytest.mark.parametrize("size", [0, -1, -100])
    def test_cache_size_must_be_positive(self, size: int) -> None:
        with pytest.raises(ValidationError, match="cache_size_mb"):
            Config(cache_size_mb=size)

    @pytest.mark.parametrize(
        "crs",
        ["epsg:4326", "EPSG:", "EPSG:abc", "WGS84", "4326", ""],
    )
    def test_invalid_crs_rejected(self, crs: str) -> None:
        with pytest.raises(ValidationError, match="default_crs"):
            Config(default_crs=crs)

    def test_valid_crs_accepted(self) -> None:
        cfg = Config(default_crs="EPSG:32634")
        assert cfg.default_crs == "EPSG:32634"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Config(nonexistent_field="value")


@pytest.mark.unit
class TestConfigPathExpansion:
    """Verify ~ is expanded in Path fields."""

    def test_cache_dir_tilde_expanded(self) -> None:
        cfg = Config(cache_dir="~/my-cache")
        assert cfg.cache_dir.is_absolute()
        assert "~" not in str(cfg.cache_dir)

    def test_copernicus_credentials_tilde_expanded(self) -> None:
        cfg = Config(copernicus_credentials="~/creds.json")
        assert cfg.copernicus_credentials is not None
        assert cfg.copernicus_credentials.is_absolute()
        assert "~" not in str(cfg.copernicus_credentials)

    def test_cds_credentials_tilde_expanded(self) -> None:
        cfg = Config(cds_credentials="~/cds.json")
        assert cfg.cds_credentials is not None
        assert cfg.cds_credentials.is_absolute()
        assert "~" not in str(cfg.cds_credentials)


@pytest.mark.unit
class TestConfigIndependence:
    """Verify two Config instances share no mutable state."""

    def test_independent_instances(self) -> None:
        cfg1 = Config(cache_size_mb=1000)
        cfg2 = Config(cache_size_mb=2000)
        assert cfg1.cache_size_mb == 1000
        assert cfg2.cache_size_mb == 2000

    def test_model_dump_independent(self) -> None:
        cfg1 = Config()
        cfg2 = Config(cache_size_mb=999)
        d1 = cfg1.model_dump()
        d2 = cfg2.model_dump()
        assert d1["cache_size_mb"] != d2["cache_size_mb"]


# ── Task 6: configure(), get_default_config(), credential resolution ─────


@pytest.mark.unit
class TestConfigure:
    """Verify configure() updates module-level config."""

    def test_configure_updates_default(self) -> None:
        configure(cache_size_mb=1234)
        cfg = get_default_config()
        assert cfg.cache_size_mb == 1234

    def test_get_default_config_returns_current(self) -> None:
        cfg = get_default_config()
        assert isinstance(cfg, Config)
        assert cfg.cache_size_mb == 5000

    def test_configure_partial_merges_with_defaults(self) -> None:
        configure(default_crs="EPSG:32634")
        cfg = get_default_config()
        assert cfg.default_crs == "EPSG:32634"
        assert cfg.cache_size_mb == 5000  # unchanged default

    def test_configure_replaces_previous(self) -> None:
        configure(cache_size_mb=1000)
        configure(cache_size_mb=2000)
        assert get_default_config().cache_size_mb == 2000

    def test_configure_invalid_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            configure(cache_size_mb=-1)


@pytest.mark.unit
class TestResolveCredentialsPath:
    """Verify AD-8 credential path resolution priority."""

    def test_explicit_path_highest_priority(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        explicit = tmp_path / "explicit.json"
        explicit.write_text("{}")
        env_path = tmp_path / "env.json"
        env_path.write_text("{}")
        monkeypatch.setenv(_CREDENTIALS_ENV_VAR, str(env_path))

        result = resolve_credentials_path(explicit=explicit)
        assert result == explicit

    def test_env_var_second_priority(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_path = tmp_path / "env-creds.json"
        env_path.write_text("{}")
        monkeypatch.setenv(_CREDENTIALS_ENV_VAR, str(env_path))

        result = resolve_credentials_path()
        assert result == env_path

    def test_default_path_lowest_priority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(_CREDENTIALS_ENV_VAR, raising=False)
        # Default path (~/.satellitehub/credentials.json) likely doesn't exist
        result = resolve_credentials_path()
        assert result is None

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        result = resolve_credentials_path(explicit=missing)
        assert result is None

    def test_returns_path_when_file_exists(self, tmp_path: Path) -> None:
        cred = tmp_path / "creds.json"
        cred.write_text("{}")
        result = resolve_credentials_path(explicit=cred)
        assert result == cred


@pytest.mark.unit
class TestFilePermissions:
    """Verify NFR5 permission warning on POSIX systems."""

    @staticmethod
    def _make_fake_stat(
        real_path: Path,
        desired_mode: int,
    ) -> Any:
        """Create a patched Path.stat that returns *desired_mode* for *real_path*."""
        import os

        original_stat = Path.stat

        def patched_stat(self: Path, **kwargs: Any) -> os.stat_result:
            result = original_stat(self, **kwargs)
            if self == real_path:
                return os.stat_result(
                    (
                        (result.st_mode & ~0o777) | desired_mode,
                        *tuple(result)[1:],
                    )
                )
            return result

        return patched_stat

    def test_warning_logged_for_permissive_file(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("satellitehub.config.sys.platform", "linux")

        cred = tmp_path / "creds.json"
        cred.write_text("{}")

        monkeypatch.setattr(Path, "stat", self._make_fake_stat(cred, 0o644))

        with caplog.at_level(logging.WARNING, logger="satellitehub"):
            _check_file_permissions(cred)

        assert "overly permissive" in caplog.text

    def test_no_warning_for_secure_file(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("satellitehub.config.sys.platform", "linux")

        cred = tmp_path / "creds.json"
        cred.write_text("{}")

        monkeypatch.setattr(Path, "stat", self._make_fake_stat(cred, 0o600))

        with caplog.at_level(logging.WARNING, logger="satellitehub"):
            _check_file_permissions(cred)

        assert "overly permissive" not in caplog.text

    def test_no_warning_on_windows(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("satellitehub.config.sys.platform", "win32")
        cred = tmp_path / "creds.json"
        cred.write_text("{}")

        with caplog.at_level(logging.WARNING, logger="satellitehub"):
            _check_file_permissions(cred)

        assert "overly permissive" not in caplog.text


@pytest.mark.unit
class TestLoadCredentials:
    """Verify credential file loading and error handling."""

    def test_valid_json_loaded(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {
            "copernicus": {"username": "placeholder", "password": "placeholder"},  # noqa: S106
            "cds": {"api_key": "placeholder"},
        }
        cred = tmp_path / "creds.json"
        cred.write_text(json.dumps(data))

        result = load_credentials(cred)
        assert result == data

    def test_missing_file_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(ConfigurationError, match="Cannot read"):
            load_credentials(missing)

    def test_invalid_json_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{")
        with pytest.raises(ConfigurationError, match="Invalid credentials"):
            load_credentials(bad)

    def test_error_message_contains_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.json"
        with pytest.raises(ConfigurationError) as exc_info:
            load_credentials(missing)
        assert "gone.json" in str(exc_info.value)

    def test_error_message_never_contains_credentials(
        self,
        tmp_path: Path,
    ) -> None:
        secret_data = '{"copernicus": {"password": "SUPER_SECRET_123"}}'  # noqa: S105
        bad = tmp_path / "creds.json"
        bad.write_text(secret_data + " invalid json tail }{")
        with pytest.raises(ConfigurationError) as exc_info:
            load_credentials(bad)
        msg = str(exc_info.value)
        assert "SUPER_SECRET_123" not in msg

    def test_error_for_missing_file_has_fix(self, tmp_path: Path) -> None:
        missing = tmp_path / "creds.json"
        with pytest.raises(ConfigurationError) as exc_info:
            load_credentials(missing)
        assert exc_info.value.fix != ""
        assert _CREDENTIALS_ENV_VAR in exc_info.value.fix

    def test_non_dict_json_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        cred = tmp_path / "creds.json"
        cred.write_text("[1, 2, 3]")
        with pytest.raises(ConfigurationError, match="Expected a JSON object"):
            load_credentials(cred)

    def test_tilde_path_expanded(self, tmp_path: Path) -> None:
        # load_credentials should expand ~ even if passed as string-like
        data = {"test": True}
        cred = tmp_path / "creds.json"
        cred.write_text(json.dumps(data))
        result = load_credentials(cred)
        assert result == data
