"""Deterministic naming for per-run Neo4j databases / Mongo collections /
Qdrant collections (AIS-4086). One name works unmodified across all three
targets — nothing needs a per-target reformat step.

Format: g-<tenant_id>-<connector_type>[-<kind>]-<suffix>

- Leading "g-" guarantees the name starts with a letter no matter what
  tenant_id looks like — required by Neo4j (database names must start with
  an alphabetic character); a tenant_id that's a UUID (e.g. "191e03de-...")
  would otherwise start with a digit and be rejected outright.
- "-" is the separator, not "_": Neo4j database names may only contain
  letters, digits, dots, and dashes — underscores are rejected there. Using
  "-" everywhere (instead of "_" for Mongo/Qdrant and "-" only for Neo4j)
  keeps one name valid for all three stores, no per-target reformat.
- <suffix> (see generate_run_suffix) is the actual isolation key — the one
  segment the length-fitting below never shortens, since two different runs
  of the same (tenant_id, connector_type) must never collide onto the same
  name just because their tenant_id/connector_type got truncated the same way.
- <kind> (e.g. "entities", "chunks") distinguishes multiple Qdrant
  collections within one run; omitted for the single Neo4j database that
  holds everything for that run.

Neo4j's 63-character name limit is the binding constraint (Mongo/Qdrant are
far more permissive) — names are fitted to it by shortening tenant_id/
connector_type, never the suffix.
"""
from __future__ import annotations

import re
import secrets
import string

_MAX_LENGTH = 63  # Neo4j database name hard limit
_PREFIX = "g"
_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits
_SUFFIX_LENGTH = 4


def generate_run_suffix() -> str:
    """A short random per-run token — generate once per bulk run and persist
    it (e.g. in the active-pointer registry); never re-derive it from
    tenant_id/connector_type alone, or two runs of the same (tenant,
    connector) would collide onto the same name."""
    return "".join(secrets.choice(_SUFFIX_ALPHABET) for _ in range(_SUFFIX_LENGTH))


def _sanitize(segment: str) -> str:
    """Lowercases and collapses anything outside [a-z0-9] into a single '-'
    — keeps a segment safe for Neo4j/Mongo/Qdrant regardless of what a real
    tenant_id/connector_type contains (a UUID's dashes pass through
    untouched; anything else gets normalized)."""
    return re.sub(r"[^a-z0-9]+", "-", segment.lower()).strip("-")


def build_resource_name(tenant_id: str, connector_type: str, suffix: str, kind: str | None = None) -> str:
    """One deterministic name valid as a Neo4j database name, a Mongo
    collection name, and a Qdrant collection name — same string, no
    per-target reformatting.

    `suffix` is never shortened: if the assembled name would exceed Neo4j's
    63-character limit, tenant_id and connector_type are trimmed (longest
    first) until it fits — the suffix and kind always survive intact.
    """
    if not tenant_id or not connector_type or not suffix:
        raise ValueError("tenant_id, connector_type, and suffix must all be non-empty")

    tenant = _sanitize(tenant_id)
    connector = _sanitize(connector_type)
    tail = [_sanitize(kind), _sanitize(suffix)] if kind else [_sanitize(suffix)]

    def assemble(t: str, c: str) -> str:
        return "-".join(part for part in [_PREFIX, t, c, *tail] if part)

    name = assemble(tenant, connector)
    overflow = len(name) - _MAX_LENGTH
    while overflow > 0 and (len(tenant) > 1 or len(connector) > 1):
        if len(tenant) >= len(connector) and len(tenant) > 1:
            tenant = tenant[:-1]
        elif len(connector) > 1:
            connector = connector[:-1]
        overflow -= 1
        name = assemble(tenant, connector)

    return name
