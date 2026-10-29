"""Tests for connectors/registry.py."""
import pytest

from connectors.base import ConnectorType
from connectors.registry import get_connector, known_connector_types
from connectors.wikipedia.spec import WikipediaSpec


def test_get_connector_returns_the_wikipedia_spec():
    assert isinstance(get_connector(ConnectorType.WIKIPEDIA), WikipediaSpec)


def test_get_connector_accepts_the_plain_string_value():
    assert isinstance(get_connector("wikipedia"), WikipediaSpec)


def test_get_connector_returns_a_fresh_instance_each_call():
    """Specs carry per-run state (bind_run), so two runs must never share one."""
    assert get_connector("wikipedia") is not get_connector("wikipedia")


def test_get_connector_raises_on_an_unknown_name():
    with pytest.raises(ValueError, match="unknown connector_type"):
        get_connector("slack")


def test_get_connector_raises_on_a_declared_but_unregistered_type(monkeypatch):
    """A type in the enum with no registered package must fail the lookup
    rather than fall back to some default connector."""
    from connectors import registry

    trimmed = {k: v for k, v in registry._REGISTRY.items()
               if k is not ConnectorType.ARXIV}
    monkeypatch.setattr(registry, "_REGISTRY", trimmed)
    with pytest.raises(ValueError, match="not registered"):
        get_connector(ConnectorType.ARXIV)


def test_known_connector_types_matches_the_registry():
    assert known_connector_types() == ["arxiv", "wikipedia"]
