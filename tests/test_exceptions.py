"""Tests for the SatelliteHub exception hierarchy."""

from __future__ import annotations

import pytest

from satellitehub.exceptions import (
    CacheError,
    ConfigurationError,
    ProviderError,
    SatelliteHubError,
)

ALL_EXCEPTION_CLASSES = [
    SatelliteHubError,
    ConfigurationError,
    ProviderError,
    CacheError,
]

SUBCLASS_EXCEPTION_CLASSES = [
    ConfigurationError,
    ProviderError,
    CacheError,
]


@pytest.mark.unit
class TestExceptionInheritance:
    """Verify the exception inheritance chain."""

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(SatelliteHubError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        SUBCLASS_EXCEPTION_CLASSES,
        ids=lambda c: c.__name__,
    )
    def test_subclass_inherits_from_base(
        self, exc_cls: type[SatelliteHubError]
    ) -> None:
        assert issubclass(exc_cls, SatelliteHubError)
        assert issubclass(exc_cls, Exception)


@pytest.mark.unit
class TestThreePartMessage:
    """Verify the three-part message pattern (what, cause, fix)."""

    @pytest.mark.parametrize(
        "exc_cls",
        ALL_EXCEPTION_CLASSES,
        ids=lambda c: c.__name__,
    )
    def test_full_message(self, exc_cls: type[SatelliteHubError]) -> None:
        exc = exc_cls(
            what="Operation failed",
            cause="Bad input",
            fix="Check your data",
        )
        msg = str(exc)
        assert "Operation failed" in msg
        assert "Cause: Bad input" in msg
        assert "Fix: Check your data" in msg

    @pytest.mark.parametrize(
        "exc_cls",
        ALL_EXCEPTION_CLASSES,
        ids=lambda c: c.__name__,
    )
    def test_what_only_message(self, exc_cls: type[SatelliteHubError]) -> None:
        exc = exc_cls(what="Something broke")
        msg = str(exc)
        assert msg == "Something broke"
        assert "Cause:" not in msg
        assert "Fix:" not in msg

    def test_message_omits_empty_cause(self) -> None:
        exc = SatelliteHubError(what="Failed", fix="Retry")
        msg = str(exc)
        assert "Cause:" not in msg
        assert "Fix: Retry" in msg

    def test_message_omits_empty_fix(self) -> None:
        exc = SatelliteHubError(what="Failed", cause="Timeout")
        msg = str(exc)
        assert "Cause: Timeout" in msg
        assert "Fix:" not in msg

    def test_multiline_format(self) -> None:
        exc = SatelliteHubError(
            what="Download failed",
            cause="HTTP 503",
            fix="Retry later",
        )
        lines = str(exc).split("\n")
        assert lines[0] == "Download failed"
        assert lines[1] == "Cause: HTTP 503"
        assert lines[2] == "Fix: Retry later"

    def test_repr_includes_formatted_message(self) -> None:
        exc = SatelliteHubError(
            what="Download failed",
            cause="HTTP 503",
            fix="Retry later",
        )
        r = repr(exc)
        assert "Download failed" in r
        assert "HTTP 503" in r


@pytest.mark.unit
class TestExceptionAttributes:
    """Verify attributes are stored and accessible."""

    @pytest.mark.parametrize(
        "exc_cls",
        ALL_EXCEPTION_CLASSES,
        ids=lambda c: c.__name__,
    )
    def test_attributes_stored(self, exc_cls: type[SatelliteHubError]) -> None:
        exc = exc_cls(what="W", cause="C", fix="F")
        assert exc.what == "W"
        assert exc.cause == "C"
        assert exc.fix == "F"

    def test_default_cause_and_fix_are_empty(self) -> None:
        exc = SatelliteHubError(what="W")
        assert exc.cause == ""
        assert exc.fix == ""


@pytest.mark.unit
class TestExceptionCatchability:
    """Verify exceptions can be caught by base class."""

    @pytest.mark.parametrize(
        "exc_cls",
        SUBCLASS_EXCEPTION_CLASSES,
        ids=lambda c: c.__name__,
    )
    def test_catch_by_base_class(self, exc_cls: type[SatelliteHubError]) -> None:
        with pytest.raises(SatelliteHubError):
            raise exc_cls(what="test")

    def test_cache_error_caught_as_base(self) -> None:
        with pytest.raises(SatelliteHubError):
            raise CacheError(what="Cache corrupted")

    def test_raise_and_catch_preserves_message(self) -> None:
        try:
            raise ProviderError(what="CDSE failed", cause="HTTP 503", fix="Retry")
        except SatelliteHubError as exc:
            assert exc.what == "CDSE failed"
            assert "Cause: HTTP 503" in str(exc)
