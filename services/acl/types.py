"""Shared ACL types, used by every connector strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AclDecision:
    """Result of deriving ACL tokens for one document.

    tokens is None when the document must be dropped (fail-closed).
    """

    tokens: Optional[list[str]]
    dropped: bool
    reason: str


@dataclass
class AclContext:
    """Inputs needed to derive ACL beyond the document payload itself.

    connector_type is the one name for the concept (connectors/base.py's
    ConnectorType); it selects which strategy the builder dispatches to.
    """

    connector_type: str = "wikipedia"
    site_audiences: Optional[dict[str, list[str]]] = None
    es_document: Optional[dict] = None
    tenant_id: Optional[str] = None
