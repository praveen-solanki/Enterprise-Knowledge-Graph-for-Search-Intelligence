"""Connector-agnostic document ACL builder for bulk and incremental ingestion.

`build_document_acl(document, context)` selects a strategy by connector_type.
A connector that derives real ABAC tokens registers its own strategy from its
own package; everything else passes its document tokens through. Fail-closed —
no derivable token means the document is dropped.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

from services.acl.types import AclContext, AclDecision

logger = logging.getLogger("graphrag.services.acl")


def _normalize_connector_type(connector_type: Optional[str]) -> str:
    """Empty/None → "zeus" (the bulk native index carries no explicit value)."""
    return (connector_type or "").strip().lower() or "wikipedia"


def _normalize_tokens(tokens) -> Optional[list[str]]:
    """Sorted, de-duplicated non-empty tokens; None if input is not list-like."""
    if not isinstance(tokens, (list, tuple, set)):
        return None
    return sorted({
        token.strip()
        for token in tokens
        if isinstance(token, str) and token.strip()
    })


@runtime_checkable
class AclStrategy(Protocol):
    """Per-connector ACL builder. Return dropped when no tokens apply."""

    def build(self, document: dict, context: AclContext) -> AclDecision: ...


class ConnectorStrategy:
    """Pass through connector document tokens (no re-derivation).

    Reads, in order: `object_type_metadata.allow_access_control`,
    `_allow_access_control`, then top-level `allow_access_control` if it is a list.
    """

    def build(self, document: dict, context: AclContext) -> AclDecision:
        tokens = _normalize_tokens(self._document_tokens(document))
        if not tokens:
            return AclDecision(
                None, True, f"connector[{context.connector_type}]:no-document-tokens"
            )
        return AclDecision(
            tokens, False, f"connector[{context.connector_type}]:passthrough"
        )

    @staticmethod
    def _document_tokens(document: dict) -> list:
        metadata = document.get("object_type_metadata") or {}
        candidates = [
            metadata.get("allow_access_control"),
            document.get("_allow_access_control"),
            document.get("allow_access_control"),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                return candidate
        return []


class DocumentAclBuilder:
    """Registry of per-connector strategies; unknown ones pass through.

    Registration happens from the connector's own package, so this registry is
    only populated for connectors something has imported. Falling back to
    pass-through for a connector that does have a real strategy would drop
    every one of its documents fail-closed, with nothing but a per-document
    INFO to show for it — so the fallback says so, once per name. At INFO, not
    WARNING: pass-through is the correct path for a connector that supplies its
    own tokens, and warning on the normal case teaches people to ignore it.

    Prefer asking the connector for its strategy directly (ConnectorSpec.acl)
    over dispatching by name here. This path exists for callers that only have
    a string, such as the incremental worker reading a Kafka event's
    connector_type field.
    """

    _default: AclStrategy = ConnectorStrategy()
    _registry: dict[str, AclStrategy] = {}
    _warned_default: set = set()

    @classmethod
    def register(cls, connector_types, strategy: AclStrategy) -> None:
        for connector_type in connector_types:
            cls._registry[_normalize_connector_type(connector_type)] = strategy

    @classmethod
    def strategy_for(cls, connector_type: Optional[str]) -> AclStrategy:
        key = _normalize_connector_type(connector_type)
        strategy = cls._registry.get(key)
        if strategy is not None:
            return strategy
        if key not in cls._warned_default:
            cls._warned_default.add(key)
            logger.info(
                "connector_type=%r has no registered ACL strategy — using document-token "
                "pass-through. Expected for a connector that supplies its own tokens; if "
                "that connector is supposed to DERIVE tokens, its package has not been "
                "imported and every one of its documents will be dropped fail-closed.", key,
            )
        return cls._default

    @classmethod
    def build(cls, document: dict, context: Optional[AclContext] = None) -> AclDecision:
        context = context or AclContext()
        return cls.strategy_for(context.connector_type).build(document, context)



def build_document_acl(
    document: dict,
    context: Optional[AclContext] = None,
) -> AclDecision:
    """Shared ACL entry point for bulk and incremental paths."""
    return DocumentAclBuilder.build(document, context)
