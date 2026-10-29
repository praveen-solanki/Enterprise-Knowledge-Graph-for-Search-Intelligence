"""Connector lookup by type.

Mirrors `services/acl/builder.py`'s DocumentAclBuilder: a dict of
implementations, a normalising lookup, and a hard failure on an unknown key
rather than a silent default. Adding a connector means adding a directory and
one line here — never editing anything in `core/`.
"""
from __future__ import annotations

from typing import Callable

from connectors.arxiv.spec import ArxivSpec
from connectors.base import ConnectorSpec, ConnectorType
from connectors.wikipedia.spec import WikipediaSpec

# Factories, not instances: a spec carries per-run state (bind_run),
# so two concurrent runs must never share one.
_REGISTRY: dict[ConnectorType, Callable[[], ConnectorSpec]] = {
    ConnectorType.ARXIV: ArxivSpec,
    ConnectorType.WIKIPEDIA: WikipediaSpec,
}


def get_connector(connector_type: ConnectorType | str) -> ConnectorSpec:
    """The spec for this connector type. Raises on anything unregistered —
    a run that cannot name its source must stop, not guess."""
    try:
        key = ConnectorType(connector_type)
    except ValueError:
        raise ValueError(
            f"unknown connector_type {connector_type!r} — known: "
            f"{sorted(c.value for c in _REGISTRY)}"
        ) from None
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"connector_type {key.value!r} is defined but not registered — known: "
            f"{sorted(c.value for c in _REGISTRY)}"
        )
    return factory()


def known_connector_types() -> list[str]:
    """The `--connector` choices, so the CLI never drifts from the registry."""
    return sorted(c.value for c in _REGISTRY)
