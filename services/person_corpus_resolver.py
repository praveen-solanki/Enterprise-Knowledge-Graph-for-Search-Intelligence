"""
Corpus-stage integration for PERSON disambiguation.
services/people/ does the actual resolution; this module keeps PERSON entities
out of graph_canonicalization_service.py's normal cross-document name
clustering.

PERSON entities are split out before that stage runs because its cheap
exact-name-match pass would silently merge two different real people who
share a name. Resolved persons are keyed by entity_id rather than canonical
name (since two people can share a display name), with the real display name
carried in a "name" field that write_entities reads when present. Because of
this, core/pipeline.py must add canon_to_id[entity_id] = entity_id directly
rather than via resolve_entity_ids, which would otherwise try to re-mint an
entity_id from the entity_id string as if it were a name.
"""
from __future__ import annotations

from typing import Optional

from services.embedding_service import get_embedding_service
from services.graph_canonicalization_service import merge_corpus_descriptions
from services.people import resolve_bare_persons


def _new_person_entry(entity_type: str, name: str) -> dict:
    return {"type": entity_type, "name": name, "source_docs": [], "descriptions": [], "aliases": []}


def _person_embed_text(name: str, description: Optional[str]) -> str:
    """The text a PERSON is embedded from: name, plus a description merged across
    every document that mentioned them (never just the first one)."""
    return f"{name}: {description}" if description else name


async def resolve_persons_for_corpus(
    canonical_entities_by_doc: dict,
    materialized_persons: dict,
    tenant_id: str,
) -> tuple[dict, dict, dict]:
    """canonical_entities_by_doc: {doc_id: {name: {type, description}}} — Step 2's
    raw per-document output, every entity type mixed together.
    materialized_persons: {entity_id: entry} — MATERIALIZE's output
    (services/people/materialize.py), already minted in the registry, so a bare
    mention of the same person below resolves onto it instead of minting a
    second node. This used to be {doc_id: raw_people_doc} and this module used
    to do the minting itself off HR field names; the connector's DISCOVER hands
    over PersonRecords now and this stage never learns that a source can have a
    people directory.

    Returns:
      non_person_entities_by_doc: same shape as the input, PERSON entries removed —
        safe to feed into graph_canonicalization_service.canonicalize_corpus unchanged.
      person_final_entities: {entity_id: {type, name, source_docs, descriptions,
        aliases, embedding, person_fields}} — one entry per resolved real person.
        person_fields is only present on a materialized record; a bare-name
        mention entry has no person_fields key.
      person_final_merge_map_by_doc: {doc_id: {doc_canonical_name: entity_id}} — merge
        into the corpus stage's own final_merge_map_by_doc so
        edge_pool_service's two-hop name resolution still reaches a person's
        entity_id like any other entity.
    """
    person_final_entities: dict = dict(materialized_persons)

    # Split PERSON out. No awaits here, so input order survives however resolve_bare_persons
    # below interleaves its own concurrency.
    non_person_entities_by_doc: dict = {}
    mentions: list[tuple[str, str, dict]] = []  # (doc_id, name, meta), in input order
    for doc_id, entities in canonical_entities_by_doc.items():
        non_person: dict = {}
        for name, meta in entities.items():
            if meta.get("type") != "PERSON":
                non_person[name] = meta
                continue
            mentions.append((doc_id, name, meta))
        non_person_entities_by_doc[doc_id] = non_person

    # ONE batched call: registry reads collapse to one per distinct name, and the ambiguity
    # guard packs several mentions per LLM call. Replaces the old per-mention loop.
    resolved_ids = await resolve_bare_persons(
        [(name, meta.get("description")) for _doc_id, name, meta in mentions], tenant_id)

    # Apply in input order, so the winning surface form and the source_docs/descriptions/
    # aliases order don't depend on resolve_bare_persons' internal completion order.
    person_final_merge_map_by_doc: dict = {}
    for index, (doc_id, name, meta) in enumerate(mentions):
        entity_id = resolved_ids[index]
        person_final_merge_map_by_doc.setdefault(doc_id, {})[name] = entity_id
        entry = person_final_entities.setdefault(entity_id, _new_person_entry("PERSON", name))
        if doc_id not in entry["source_docs"]:
            entry["source_docs"].append(doc_id)
        if meta.get("description"):
            entry["descriptions"].append({"text": meta["description"], "doc_id": doc_id})
        entry["aliases"].append({"name": name, "doc_id": doc_id})

    # Embed each resolved person once, from a description merged across every document
    # that mentioned them (never just the first one) — same corpus-wide merge
    # graph_canonicalization_service.py's own registry uses. write_entities pushes this
    # to Qdrant the same as any other entity.
    if person_final_entities:
        merged_by_entity = await merge_corpus_descriptions({
            entity_id: [d["text"] for d in entry["descriptions"]]
            for entity_id, entry in person_final_entities.items()
        })
        embedder = get_embedding_service()
        entity_ids = list(person_final_entities.keys())
        texts = [_person_embed_text(person_final_entities[eid]["name"], merged_by_entity.get(eid))
                 for eid in entity_ids]
        vectors = embedder.embed_batch(texts)
        for entity_id, vector in zip(entity_ids, vectors):
            person_final_entities[entity_id]["embedding"] = vector
            # The same merged text the vector was built from IS the person's one
            # readable description — persisted as e.description by write_entities,
            # exactly like canonicalize_corpus does for every other entity.
            person_final_entities[entity_id]["description"] = merged_by_entity.get(entity_id)

    return non_person_entities_by_doc, person_final_entities, person_final_merge_map_by_doc
