"""
Stable entity identity, tenant-scoped.

link_or_mint gives every entity a stable entity_id ("ent_<uuid4>") the first
time it's seen, and returns the SAME id every later time the same
(normalized_name, entity_type, tenant_id) is looked up — used by both the
bulk path (core/pipeline.py) and the incremental path
(services/incremental_canonicalization.py) so both mint through the same
registry and never diverge.

Identity is exact-normalized-name-based, which does not disambiguate two
different real-world entities sharing a name. PERSON is the exception:
services/people/ resolves PERSON entities by identity instead of name — on
(connector_type, source_key) when a source named them with a key
(materialize.py), or via find_all_by_alias/enrich below when it did not
(resolution.py) — keeping two different real people who share a name from
colliding into one node.

The registry is never wiped by a graph rebuild, so a full bulk re-run reuses
the same entity_id for any entity whose identity triple hasn't changed.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pymongo.errors import DuplicateKeyError

from repository.mongo_repository import _get_db

_COLLECTION = "entity_registry"

_index_ensured = False


def _new_entity_id() -> str:
    return f"ent_{uuid.uuid4().hex}"


def _ensure_index() -> None:
    """The two indexes this collection is queried by. Cached process-wide since
    this runs per-entity, potentially thousands of times per bulk run, and
    create_index is a network round trip even when idempotent.

    1. Unique on (normalized_name, entity_type, tenant_id) — makes link_or_mint's
       DuplicateKeyError race handling meaningful (two concurrent minters of the
       same identity can only leave one row behind).

    2. (aliases, entity_type, tenant_id) for find_all_by_alias, the PERSON ambiguity
       guard. Index 1 can't serve it — normalized_name leads and is absent from the
       filter — so without this every bare-name PERSON mention was a full collection
       scan (3,811 docs examined per lookup on a 50-document corpus) of a registry
       that is never wiped, getting slower every run.

       Multikey, since aliases is an array (and the only one, per MongoDB's
       one-array-per-compound-index limit). Deliberately NOT unique: two real people
       sharing an alias is precisely what the guard exists to resolve."""
    global _index_ensured
    if _index_ensured:
        return
    collection = _get_db()[_COLLECTION]
    collection.create_index(
        [("normalized_name", 1), ("entity_type", 1), ("tenant_id", 1)], unique=True,
    )
    collection.create_index([("aliases", 1), ("entity_type", 1), ("tenant_id", 1)])
    _index_ensured = True


def find(normalized_name: str, entity_type: str, tenant_id: str) -> Optional[dict]:
    """The registry record for this exact identity, or None if it's never been minted."""
    return _get_db()[_COLLECTION].find_one({
        "normalized_name": normalized_name, "entity_type": entity_type, "tenant_id": tenant_id,
    })


def get(entity_id: str) -> Optional[dict]:
    return _get_db()[_COLLECTION].find_one({"_id": entity_id})


def link_or_mint(canonical_name: str, entity_type: str, normalized_name: str, tenant_id: str,
                 aliases: Optional[list] = None) -> dict:
    """Return the existing registry record for this identity, or mint a new one.

    Concurrency-safe: a racing duplicate insert collapses to the surviving
    record via the unique index — the loser's insert raises DuplicateKeyError,
    caught here and re-read as a link instead."""
    _ensure_index()
    existing = find(normalized_name, entity_type, tenant_id)
    if existing:
        if aliases:
            add_aliases(existing["_id"], aliases)
        return existing
    doc = {
        "_id": _new_entity_id(),
        "canonical_name": canonical_name,
        "normalized_name": normalized_name,
        "entity_type": entity_type,
        "tenant_id": tenant_id,
        "aliases": sorted({a for a in (aliases or []) if a}),
    }
    try:
        _get_db()[_COLLECTION].insert_one(doc)
        return doc
    except DuplicateKeyError:
        return find(normalized_name, entity_type, tenant_id) or doc


def find_all_by_alias(alias: str, entity_type: str, tenant_id: str) -> list[dict]:
    """Every registry record of this type/tenant whose alias list contains
    this exact normalized alias. Used by the person ambiguity guard
    (services/people/resolution.py): 0 records -> mint; 1 -> link; more than 1
    -> genuinely ambiguous (two different people sharing a name), needing an
    LLM tie-break rather than a blind merge.

    Ensures the indexes itself rather than relying on link_or_mint running first: on
    a corpus with no People documents this is the FIRST registry call of the process,
    so the alias index would not exist yet for exactly the queries that need it."""
    _ensure_index()
    return list(_get_db()[_COLLECTION].find({
        "aliases": alias, "entity_type": entity_type, "tenant_id": tenant_id,
    }))


def add_aliases(entity_id: str, aliases: list) -> None:
    aliases = sorted({a for a in aliases if a})
    if not aliases:
        return
    _get_db()[_COLLECTION].update_one(
        {"_id": entity_id}, {"$addToSet": {"aliases": {"$each": aliases}}},
    )


def delete_merged(entity_id: str) -> None:
    """Removes a registry row whose Neo4j node was just merged away into a
    different canonical entity_id. No tombstone/redirect stub is kept, so
    leaving this row behind would let a later bare-name mention re-link to
    the now-nonexistent entity_id — undoing the merge. Callers must merge
    the Neo4j graph state first, then call this."""
    _get_db()[_COLLECTION].delete_one({"_id": entity_id})


def enrich(entity_id: str, enrichment: dict) -> None:
    """Stores structured attributes (job_title/department/team/organization/
    manager) on a registry record, so the ambiguity guard can describe
    candidates by more than just a name. Silently drops falsy values rather
    than overwriting a known attribute with nothing."""
    fields = {k: v for k, v in (enrichment or {}).items() if v}
    if not fields:
        return
    _get_db()[_COLLECTION].update_one(
        {"_id": entity_id}, {"$set": {f"enrichment.{k}": v for k, v in fields.items()}},
    )
