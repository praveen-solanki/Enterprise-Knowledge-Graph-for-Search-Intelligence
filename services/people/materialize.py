"""MATERIALIZE — the shared middle of the people phase.

Identical for every connector, and connector-blind by construction: it reads a
PersonRecord's `source_key`, `display_name` and `email`, and nothing else.
`attributes` is carried through to the node write untouched — the connector
that filled it is the only thing that knows what is in it.

Identity is keyed on (connector_type, source_key). Two consequences worth
naming:

  A rename no longer mints a second person. The old key folded the normalized
  name in ("jane doe\\x1fperson:user-99"), so an entity whose display name
  changed got a new identity and a new node.

  A source that has ids at all can now use them. The old key's disambiguator
  was an HR user_id falling back to an email, which is why authorship only
  resolved for the intranet: nothing else populates those fields.

The display name and the email go in as registry aliases. That is what lets a
bare-name mention of the same person elsewhere in the corpus resolve to this
entity (services/people/resolution.py) instead of minting a second one.

DEPLOYING THIS IS A MIGRATION, NOT A RELEASE. The old key mints a different
entity_id, so a person already in the graph is not found and is duplicated —
the old node keeps its authorship/org edges and no query filtering on
source_key can see it. Clear the tenant's graph AND its registry rows, then run
bulk from scratch: not incremental, and not bulk over the existing graph.
"""
from __future__ import annotations

import logging
from typing import Iterable

from repository import entity_registry
from services.clustering_utils import normalize_name

logger = logging.getLogger("graphrag.services.people.materialize")


def person_identity_norm(connector_type: str, source_key: str) -> str:
    """The registry identity for a person their source named with a key."""
    return f"{connector_type}\x1fperson:{source_key}"


def materialize_persons(records: Iterable, connector_type: str, tenant_id: str) -> dict:
    """{entity_id: corpus entry} for every record with a usable key and name.

    records: PersonRecords (connectors/base.py — duck-typed, not imported;
    nothing in services/ imports a connector package).

    The entry is the shape services/person_corpus_resolver.py already builds
    for a bare-name mention, plus `person_fields` — the properties
    repository/neo4j_repository.py's write_entities stamps onto the node. Four
    of those are this stage's own (`source_key`, `connector_type`, `email`,
    `url` — email is what makes a person reachable by the standard traversal
    for every connector); the rest are the connector's `attributes`, prefixed
    and passed through unread.

    Records sharing an entity_id (the same person named by two documents)
    collapse into one entry, latest attributes winning — same as the People-doc
    loop this replaces.
    """
    entries: dict = {}
    for record in records:
        name = (record.display_name or "").strip()
        if not record.source_key or not name:
            # Identified by where it came from, never by its contents — a
            # PersonRecord carries someone's email and job title, and a log line
            # is the wrong place for either.
            logger.warning("Skipping a person record with no source_key or no display_name "
                           "(from doc(s) %s)", record.source_doc_ids or "none")
            continue
        plain = normalize_name(name) or name.lower()
        email = (record.email or "").strip().lower()
        rec = entity_registry.link_or_mint(
            name, "PERSON", person_identity_norm(connector_type, record.source_key), tenant_id,
            aliases=[plain] + ([email] if email else []),
        )
        entity_id = rec["_id"]
        # Structured attributes for the bare-mention ambiguity guard's prompt
        # (services/people/disambiguation.py's attr_text picks the keys it knows).
        entity_registry.enrich(entity_id, dict(record.attributes))
        entry = entries.setdefault(entity_id, {
            "type": "PERSON", "name": name,
            "source_docs": [], "descriptions": [], "aliases": [],
        })
        entry["source_docs"] += [d for d in record.source_doc_ids if d not in entry["source_docs"]]
        entry["person_fields"] = {
            "source_key": record.source_key,
            "connector_type": connector_type,
            # Three names, same value: `email` is what this pipeline has always
            # written; ai-graph-retrieval reads `person_email` then `user_email`
            # (PEOPLE_RESOLVE / DOCUMENT_CONTRIBUTORS coalesce). Kept as additive
            # keys rather than a rename so nothing already reading `email` breaks.
            **({"email": email, "person_email": email, "user_email": email} if email else {}),
            # Same shape as Document.url/Chunk.url — a plain `url`, not
            # `person_url`, so a citation-hover-card can read it identically
            # regardless of node type.
            **({"url": record.url} if record.url else {}),
            # Falsy values are dropped rather than written: write_entities' `SET
            # e += person_fields` would otherwise blank a property a previous
            # run knew, which the old per-field coalesce() never did.
            **{f"person_{k}": v for k, v in record.attributes.items() if v},
        }
    return entries
