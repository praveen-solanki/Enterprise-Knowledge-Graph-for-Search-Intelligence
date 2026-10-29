"""
Step 4 — edge pool. Re-points every relationship/claim endpoint from its raw
per-chunk entity name to its FINAL canonical name (doc_merge_map ->
final_merge_map_by_doc from Steps 2-3), and merges duplicate edges: the same
(source, type, target) relationship found across N chunks becomes one edge.

Relationship types are normalized to UPPER_SNAKE_CASE only — no embedding-based
folding of near-duplicate verbs (e.g. "ADDS_FEATURE" vs "ADDED_A_FEATURE" stay
distinct types).

ACL: an edge merges across documents, so its acl is the UNION of every
contributing document's acl ("visible if any source is visible"). A claim
never merges (one chunk_id each), so its acl is just its source document's acl.

Descriptions and dates are never blended across that union — each contributing
chunk keeps its own description + documented_on entry, so a more sensitive
private-document description can never silently become "the" description shown
to a viewer who can only see a public source. Picking which entry to show a
given viewer is retrieval's job, not this one.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from services.edge_gate import is_junk_endpoint

# Bump whenever this file's remapping/normalization logic changes, so
# core/pipeline.py's cache key changes with it and a stale cached edge pool
# is never silently reused (no LLM prompt here to hash against).
#
# v7-edge-description: edges carry a singular `description` (LLM text) alongside
# the ACL-tagged `descriptions` list — same split as Entity.description vs
# descriptions_json. Cached v6 pools lack the singular field, so bump the key.
EDGE_POOL_VERSION = "v7-edge-description"

logger = logging.getLogger("graphrag.services.edge_pool")


def _normalize_rel_type(rel_type: str) -> str:
    """UPPER_SNAKE_CASE, no folding of similar verbs — see module docstring."""
    if not rel_type or not rel_type.strip():
        return "RELATED_TO"
    return re.sub(r"[^\w]+", "_", rel_type.strip().upper()).strip("_") or "RELATED_TO"


def _resolve_endpoint(raw_name: str, doc_merge_map: dict, final_map_for_doc: dict) -> Optional[str]:
    """raw entity name -> doc-level canonical -> final canonical name (None if it
    never resolves). Returns the NAME, not entity_id — the junk-endpoint gate
    below needs the name string for its regex checks."""
    doc_canon = doc_merge_map.get(raw_name)
    if doc_canon is None:
        return None
    return final_map_for_doc.get(doc_canon)


def _to_entity_id(final_name: Optional[str], canon_to_id: dict) -> Optional[str]:
    """Final canonical name -> stable entity_id. Must be called AFTER the
    junk-endpoint gate — gating needs the name string, not an opaque id."""
    if final_name is None:
        return None
    return canon_to_id.get(final_name)


def _build_doc_type_registry(chunk_results: list) -> dict:
    """{raw_name: type}, spanning EVERY chunk in the document, not just one.

    A claim's subject can be an entity extracted in an EARLIER chunk of the
    same document (unlike a relationship, whose endpoints are always in its
    own chunk) — a per-chunk-only lookup would return "" for that case and
    silently fail write_claims' MATCH (subj:Entity {name, type})."""
    registry: dict = {}
    for result in chunk_results:
        for entity in result.entities:
            registry.setdefault(entity.name, entity.type)
    return registry


def merge_edge_pools(target: dict, source: dict) -> None:
    """Fold `source`'s edges into `target` in place, MERGING any edge both sides
    already hold rather than letting one replace the other.

    An edge key is (src_entity_id, rel_type, tgt_entity_id), and entity_ids are
    corpus-wide, not per-document — so two different documents routinely produce
    the same key. That is the whole point of Step 3's canonicalization: "Globex
    ADDS_FEATURE Enterprise Search" found in fifty documents is one edge with
    fifty contributors.

    build_edge_pool merges those itself for every document it is given in ONE
    call, which is how core/pipeline.py now calls it (a batch of
    _EDGE_POOL_DOC_BATCH documents). This function covers what that cannot: the
    same edge key coming back from a LATER batch, since no single call can see
    the whole corpus without breaking the module's O(batch) memory contract.
    Accumulating those results with a plain dict.update() would silently keep only
    the last batch's descriptions, chunk_ids, raw_types and acl. The acl case is an
    access-control under-grant: an edge that should be visible to a viewer of any
    contributing document would become visible only to viewers of whichever batch
    happened to be processed last.

    Merges exactly as build_edge_pool merges its own per-chunk contributions:
    chunk_ids/raw_types/acl union, descriptions concatenate (never blended into
    one string — see the module docstring's ACL reasoning), singular description
    keeps the first non-empty value (same independence as Entity.description),
    and src_type/tgt_type keep the first contributor's value.

    Copies on insert rather than storing `source`'s dict by reference: callers
    accumulate the same per-document result into two collections (a pending write
    batch and a whole-run total), and sharing one mutable dict between them would
    apply the next document's merge twice to it.
    """
    for key, edge in source.items():
        existing = target.get(key)
        if existing is None:
            target[key] = {
                **edge,
                "descriptions": list(edge["descriptions"]),
                "chunk_ids": list(edge["chunk_ids"]),
                "raw_types": list(edge["raw_types"]),
                "acl": list(edge["acl"]),
            }
            continue
        existing["descriptions"] = existing["descriptions"] + edge["descriptions"]
        if not (existing.get("description") or "").strip():
            if text := (edge.get("description") or "").strip():
                existing["description"] = text
        for field in ("chunk_ids", "raw_types", "acl"):
            existing[field] = sorted(set(existing[field]) | set(edge[field]))


def build_edge_pool(
    results_by_doc: dict,
    chunk_ids_by_doc: dict,
    doc_merge_maps: dict,
    final_merge_map_by_doc: dict,
    canon_to_id: dict,
    doc_acl_by_doc: Optional[dict] = None,
    doc_dates_by_doc: Optional[dict] = None,
) -> tuple[dict, list, list]:
    """Build the deduplicated relationship pool and re-point every claim's
    subject at its final canonical name's stable entity_id.

    results_by_doc:         {doc_id: [ChunkExtractionResult, ...]} — Step 1 output, grouped by doc.
    chunk_ids_by_doc:       {doc_id: [chunk_id, ...]} — same order as results_by_doc[doc_id].
    doc_merge_maps:         {doc_id: {raw_name: doc_canonical_name}} — Step 2 output.
    final_merge_map_by_doc: {doc_id: {doc_canonical_name: final_canonical_name}} — Step 3 output.
    canon_to_id:            {final_canonical_name: entity_id}. Every relationship/claim/mention
                            below is keyed by entity_id, never by name.
    doc_acl_by_doc:         {doc_id: [acl_token, ...]} (services/acl/). Optional,
                            defaults to {}.
    doc_dates_by_doc:       {doc_id: "YYYY-MM-DD" or ""} — each doc's own published/modified
                            date. Optional, defaults to {} (documented_on comes back "").

    Returns (final_edges, final_claims, mentions):
      final_edges: {(src_id, rel_type, tgt_id): {description: str (singular LLM text),
                     descriptions: [{text, acl, chunk_id, documented_on}, ...],
                     chunk_ids: [...], raw_types: [...], acl: [...] (union across contributors),
                     src_type, tgt_type (informational only — MATCH is by entity_id alone)}}
      final_claims: [{subject (entity_id), predicate, object, statement_type, temporal_type,
                       valid_from, valid_to, chunk_id, acl}, ...] — acl is this claim's OWN
                       document's acl, never merged.
      mentions: [(chunk_id, entity_id, entity_type), ...] — for Step 5's
                Chunk -[MENTIONS]-> Entity edges.
    """
    doc_acl_by_doc = doc_acl_by_doc or {}
    doc_dates_by_doc = doc_dates_by_doc or {}
    final_edges: dict = {}
    final_claims: list = []
    mentions: list = []
    total_rel = bound = self_loops = unbound = junk_dropped = 0
    total_claims = claims_bound = 0
    unbound_claim_subjects: set = set()

    for doc_id, chunk_results in results_by_doc.items():
        doc_merge_map = doc_merge_maps.get(doc_id, {})
        final_map_for_doc = final_merge_map_by_doc.get(doc_id, {})
        chunk_ids = chunk_ids_by_doc.get(doc_id, [])
        doc_acl = doc_acl_by_doc.get(doc_id) or []
        doc_date = doc_dates_by_doc.get(doc_id) or ""
        doc_type_registry = _build_doc_type_registry(chunk_results)

        for chunk_id, result in zip(chunk_ids, chunk_results):
            for entity in result.entities:
                final_name = _resolve_endpoint(entity.name, doc_merge_map, final_map_for_doc)
                entity_id = _to_entity_id(final_name, canon_to_id)
                if entity_id:
                    mentions.append((chunk_id, entity_id, entity.type))

            for rel in result.relationships:
                total_rel += 1
                src_name = _resolve_endpoint(rel.source_entity, doc_merge_map, final_map_for_doc)
                tgt_name = _resolve_endpoint(rel.target_entity, doc_merge_map, final_map_for_doc)
                if not src_name or not tgt_name:
                    unbound += 1
                    continue
                if src_name == tgt_name:
                    self_loops += 1
                    continue
                src_type = doc_type_registry.get(rel.source_entity, "")
                tgt_type = doc_type_registry.get(rel.target_entity, "")
                # Junk-endpoint gate (pronouns, generic refs, bare values) must run on the
                # NAME before translating to entity_id — an id string carries no signal for it.
                if is_junk_endpoint(src_name, src_type) or is_junk_endpoint(tgt_name, tgt_type):
                    junk_dropped += 1
                    continue
                src = _to_entity_id(src_name, canon_to_id)
                tgt = _to_entity_id(tgt_name, canon_to_id)
                if not src or not tgt:
                    unbound += 1
                    continue
                bound += 1
                rel_type = _normalize_rel_type(rel.relationship_type)
                key = (src, rel_type, tgt)
                if key not in final_edges:
                    final_edges[key] = {
                        # Singular LLM description — written to r.description the same
                        # way Entity.description is; independent of the per-chunk list.
                        "description": "",
                        "descriptions": [], "chunk_ids": set(), "raw_types": set(), "acl": set(),
                        # Informational only now — write_relationships MATCHes by entity_id alone.
                        "src_type": src_type, "tgt_type": tgt_type,
                    }
                final_edges[key]["chunk_ids"].add(chunk_id)
                final_edges[key]["raw_types"].add(rel.relationship_type)
                # ACL is unioned across contributors ("visible if any source is visible") —
                # descriptions stay per-source, see module docstring.
                final_edges[key]["acl"].update(doc_acl)
                rel_text = (rel.description or "").strip()
                if rel_text and not final_edges[key]["description"]:
                    final_edges[key]["description"] = rel_text
                final_edges[key]["descriptions"].append(
                    {"text": rel.description or "", "acl": doc_acl, "chunk_id": chunk_id,
                     "documented_on": doc_date}
                )

            for claim in result.claims:
                total_claims += 1
                subject_name = _resolve_endpoint(claim.subject, doc_merge_map, final_map_for_doc)
                subject = _to_entity_id(subject_name, canon_to_id)
                if not subject:
                    unbound_claim_subjects.add(claim.subject)
                    continue
                claims_bound += 1
                # Two claims differing only in validity window are different facts —
                # valid_from/valid_to must be part of claim identity, or the second
                # silently overwrites the first.
                final_claims.append({
                    "subject": subject,
                    "subject_type": doc_type_registry.get(claim.subject, ""),
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "statement_type": claim.statement_type,
                    "temporal_type": claim.temporal_type,
                    "valid_from": claim.valid_from,
                    "valid_to": claim.valid_to,
                    "chunk_id": chunk_id,
                    # This claim's own document's date (same doc_date used above for
                    # the edge description entry).
                    "documented_on": doc_date,
                    # Claims never merge across documents, so nothing to union here
                    # (unlike edges).
                    "acl": doc_acl,
                })

    pct_rel = 100.0 * bound / max(1, total_rel)
    pct_claims = 100.0 * claims_bound / max(1, total_claims)
    logger.info("Edge pool: %d/%d relationships bound (%.1f%%) -> %d unique edges | "
                "%d self-loops dropped, %d unbound, %d junk-endpoint dropped",
                bound, total_rel, pct_rel, len(final_edges), self_loops, unbound, junk_dropped)
    logger.info("Edge pool: %d/%d claims bound to a final entity (%.1f%%)", claims_bound, total_claims, pct_claims)
    if unbound_claim_subjects:
        # Subjects resolve by exact extracted-entity name, so a claim the model
        # wrote about an alias ("DUGNN" for "Deep Universal Graph Embedding") or
        # about a generic noun phrase ("the article", "models") binds to nothing
        # and the claim is dropped. Both are expected, but only the first is a
        # loss worth acting on, and the bare percentage above cannot tell them
        # apart — so name them. Sampled: one document can produce dozens.
        sample = sorted(unbound_claim_subjects)[:15]
        logger.info("Edge pool: %d distinct claim subject(s) resolved to no entity, e.g. %s",
                    len(unbound_claim_subjects), ", ".join(repr(s) for s in sample))

    for edge in final_edges.values():  # sets -> sorted lists, for JSON/Mongo friendliness
        edge["chunk_ids"] = sorted(c for c in edge["chunk_ids"] if c)
        edge["raw_types"] = sorted(edge["raw_types"])
        edge["acl"] = sorted(edge["acl"])

    logger.info("Edge pool: %d entity mention(s) resolved to a final entity", len(mentions))
    return final_edges, final_claims, mentions
