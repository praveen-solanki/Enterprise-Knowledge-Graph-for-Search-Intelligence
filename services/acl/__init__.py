"""Document ACL package — the shared, connector-agnostic contract.

Public API:
    build_document_acl, AclContext, AclDecision, AclStrategy, DocumentAclBuilder

Per-connector strategies live in their own connector package and register
themselves on import (see connectors/zeus/acl.py).
"""
from services.acl.builder import AclStrategy, DocumentAclBuilder, build_document_acl
from services.acl.types import AclContext, AclDecision

__all__ = [
    "AclContext",
    "AclDecision",
    "AclStrategy",
    "DocumentAclBuilder",
    "build_document_acl",
]
