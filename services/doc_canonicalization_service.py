"""
Step 2 — doc-level canonicalization. Within ONE document, fold duplicate
entity mentions (e.g. "Globex" appearing in 3 of its 4 chunks) into a single
canonical entity per real-world thing.

Approach, matching the proven pipeline's Stage 1: normalize names -> embed
names only (name-only is deliberate here; descriptions over-merge at this
scale) -> group entities whose embeddings are similar -> ask the LLM to
confirm which of those similar-looking entities are actually the same thing.
Clustering + LLM-resolution mechanics live in services/clustering_utils.py,
shared with Step 3.

Description merging (_merge_mention_descriptions): chunks are extracted
independently, so one entity arrives as N mentions each with its OWN description.
All of them are folded into one capped synthesis per document before clustering,
since that text is both the cluster-merge LLM's identity signal and what Step 3
persists. Calls are packed by character budget (_DESCRIPTION_MERGE_CHAR_BUDGET)
and bounded per entity at _DESCRIPTIONS_PER_GROUP_MAX descriptions.

Multi-round clustering (experiment-parity, mirrors the proven pipeline's
core/batch_canonicalization.py:_canonicalize_iterative): the clustering+merge
pass above runs to a FIXPOINT via services/clustering_utils.cluster_to_fixpoint
— a merged entity's fresh embedding + folded description re-enters clustering
on the next round, catching multi-hop merges a single pass would miss (round
1 merges A+B; round 2 the merged AB should also merge with C, which never
gets the chance if clustering only ever runs once). No Mongo checkpointing of
its own either way (the caller in core/pipeline.py checkpoints the whole
result).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from pydantic import BaseModel, Field

from config.settings import settings

from models.extraction import ChunkExtractionResult
from services.clustering_utils import (
    CLUSTERING_VERSION,
    DEFAULT_LLM_BATCH_MAX,
    ClusterResolution,
    cluster_to_fixpoint,
    is_precision,
    normalize_name,
    resolve_cluster_batched,
    shortest_clean,
    structured_llm_call,
)
from services.llm_profiles import CallKind
from services.embedding_service import get_embedding_service
from services.description_pack_utils import pack_by_char_budget

logger = logging.getLogger("graphrag.services.doc_canonicalization")

# Experiment-parity review #30: the old value (0.85) traced to a DEAD config field
# in the experiment (config.py's doc_threshold, defined but never actually read by
# any clustering call site) — the REAL, actually-wired value, confirmed by tracing
# core/batch_canonicalization.py's real call sites, is cfg.doc_similarity_threshold
# = 0.82. The stale "matches" comment this replaces was simply wrong, not a
# deliberate divergence.
DOC_SIMILARITY_THRESHOLD = 0.82
# Anti-chaining (tight_partition) runs at a LOOSER cut than the clustering pass above:
# below DOC_SIMILARITY_THRESHOLD, so a cluster is only split where a member is clearly
# unrelated rather than merely less similar than its neighbours.
ANTI_CHAIN_THRESHOLD = min(0.78, DOC_SIMILARITY_THRESHOLD - 0.05)

# Payload budget per description-merge/reduce call, in characters. Packed greedily
# rather than by a fixed entity/piece count: per-entity size varies widely, so a
# fixed count would overflow on the heavy ones or waste calls on the light ones.
# Deliberately smaller than the corpus-stage budget — doc descriptions are short
# profiles (~250-400 chars), not already-merged corpus summaries.
_DESCRIPTION_MERGE_CHAR_BUDGET = settings.doc_description_merge_char_budget

# Cap on ONE entity's descriptions per prompt — the other axis, bounding how much text
# each entity contributes rather than how much total text shares the call. An entity
# mentioned in hundreds of chunks would otherwise overflow the response and fail its
# whole batch back to _no_llm_fallback.
_DESCRIPTIONS_PER_GROUP_MAX = settings.doc_descriptions_per_group_max

# Both prompts cap the description at 2-3 lines: a doc-level description is a short
# general profile, while dated/relational facts belong to relationships and claims.
# Stated in lines AND characters, since "lines" alone drifts for a model.
_DESCRIPTION_RULE = (
    "The description must be a GENERAL profile of 2-3 lines (roughly 250-350 characters, "
    "and never more than 400): what the entity is, what it does or is used for, and its "
    "most notable attribute. Do NOT list every detail — specific figures, dates, and "
    "relationships are captured elsewhere. Never invent anything that is not supported "
    "by the text you were given."
)

_DESCRIPTION_MERGE_SYSTEM = (
    "Each item below is ONE entity that was mentioned several times in a single document. "
    "Because each mention was extracted independently, you are given several surface forms of "
    "its name and several partial descriptions of it. These are NOT different entities — do "
    "not try to split them.\n\n"
    "For each item, return:\n"
    "- id: echo the item's id back unchanged.\n"
    "- canonical_name: the clearest, best-formed of the given surface forms (fix obvious casing "
    "or punctuation garble, but do not rename the entity into something different).\n"
    "- description: ONE description that combines the information from all of the given "
    f"descriptions. {_DESCRIPTION_RULE}\n\n"
    "Respond with strict JSON: {\"entities\": [{\"id\": ..., \"canonical_name\": ..., "
    "\"description\": ...}]}"
)

_MERGE_SYSTEM = (
    "You are deduplicating entity mentions extracted from ONE document. Given a list of "
    "candidate entity names (with their type and description) that look similar, decide which "
    "ones refer to the SAME real-world thing and which are genuinely different. Group members "
    "that are the same into one group; keep genuinely different entities in their own group. "
    "For each group, pick the clearest natural name as canonical_name, and write one merged "
    f"description combining what's known. {_DESCRIPTION_RULE} "
    "When uncertain, keep them SEPARATE — under-merging is "
    "cheap to fix later, over-merging corrupts the graph.\n\n"
    "Respond with strict JSON: {\"groups\": [{\"canonical_name\": ..., \"canonical_type\": ..., "
    "\"merged_description\": ..., \"members\": [...]}]}"
)

# Changes whenever the merge prompt, similarity threshold, the clustering
# algorithm, OR the checkpoint's own output schema change, so a stale cached
# result (keyed with this version, in core/pipeline.py) is never silently
# reused after the logic has moved on.
#
# v2-chunk-ids: core/pipeline.py's streaming rewrite added a "chunk_ids" field
# to this checkpoint (Step 4 needs it to reload extraction results per doc
# without holding them in memory). Checkpoints saved before that change don't
# have this field. Without this bump, load_checkpoint() would still return a
# cache HIT for those old entries — canon_data.get("chunk_ids", []) then
# silently returns [], so Step 4 sees zero chunks for that doc and produces
# zero edges/claims/mentions for it, with no error and no log line. Bumping
# this forces exactly those stale entries to miss and recompute (cheap — reuses
# already-cached extraction results, just re-runs canonicalization) so they
# get saved with the current schema.
# v3-multi-round: clustering now runs to a fixpoint (services/clustering_utils.
# cluster_to_fixpoint) instead of a single pass — a stale v2 cached result may be
# missing a multi-hop merge the new logic would have caught, and must be
# recomputed, not reused.
# v4-merged-descriptions: every mention's description is folded into one capped
# synthesis, and the LLM's own name/description are trusted. A v3 entry holds a
# description picked arbitrarily from one chunk, so it must recompute.
# v5-batched-cluster-merge: a cluster above DEFAULT_LLM_BATCH_MAX is sliced and
# resolved recursively instead of sent as one prompt that failed closed to "no
# merges". The cap is hashed in too, so re-tuning it invalidates on its own.
# v6-reduced-descriptions: a group's descriptions are tree-reduced to
# _DESCRIPTIONS_PER_GROUP_MAX before the batch prompt is built. A v5 entry may hold
# the all-groups-fell-back outcome an overflowing prompt caused.
# v7-char-budget-packing: description-merge/reduce calls are packed by character
# budget rather than a fixed entity/piece count. A v6 entry may have been produced
# under a different packing (and thus different prompt boundaries).
# v8-reduce-to-one: reduce continues until exactly one description per entity, not
# merely under the per-group cap. A v7 entry may hold a multi-description pre-merge
# signal the new path would have folded first.
_OUTPUT_SCHEMA_VERSION = "v8-reduce-to-one"
DOC_CANON_VERSION = hashlib.sha1(
    f"{_MERGE_SYSTEM}|{_DESCRIPTION_MERGE_SYSTEM}|{DOC_SIMILARITY_THRESHOLD}|"
    f"{CLUSTERING_VERSION}|{DEFAULT_LLM_BATCH_MAX}|{_DESCRIPTIONS_PER_GROUP_MAX}|"
    f"{_DESCRIPTION_MERGE_CHAR_BUDGET}|{_OUTPUT_SCHEMA_VERSION}".encode("utf-8")
).hexdigest()[:12]


class _MergedMention(BaseModel):
    """One entity's folded name + description, as returned by the per-document
    description-merge call."""
    id: int = Field(..., description="The id of the input entity this result is for, echoed back "
                    "exactly as given.")
    canonical_name: str = Field(
        ..., description="The single most representative surface-form name among this entity's "
        "names (not a garbled casing variant or an invented blend).")
    description: str = Field(
        ..., description="One synthesized description covering every given description of this "
        "entity, preserving concrete facts (numbers, dates, versions) rather than a vague "
        "generic summary.")


class _DocumentDescriptionMerge(BaseModel):
    entities: list[_MergedMention] = Field(
        ..., description="One entry per input entity, each carrying its echoed id.")


def _build_mention_groups(chunk_results: list[ChunkExtractionResult]) -> list[dict]:
    """Collapse every chunk's raw entities into one group per real entity, keyed by
    normalized name, keeping EVERY surface form and EVERY description. Grouping in a
    single pass, not build-by-raw-name-then-regroup: the two-stage version dropped all
    but one description per raw name before the normalized grouping ever ran.

    Returns {key, names, type, descriptions} per group, in first-seen order (so the
    stage is deterministic) with names and descriptions exact-deduped.
    """
    groups: dict[str, dict] = {}
    for result in chunk_results:
        for entity in result.entities:
            group = groups.setdefault(
                normalize_name(entity.name),
                {"key": normalize_name(entity.name), "names": [], "type": entity.type,
                 "descriptions": []},
            )
            if entity.name not in group["names"]:
                group["names"].append(entity.name)
            # First non-empty type wins, matching the previous first-seen behaviour.
            if not group["type"] and entity.type:
                group["type"] = entity.type
            if entity.description and entity.description not in group["descriptions"]:
                group["descriptions"].append(entity.description)
    return list(groups.values())


def _needs_llm_decision(group: dict) -> bool:
    """True when there's something for the LLM to decide: more than one surface form
    to name it by, or more than one description to fold. A single mention with a
    single description has neither, and must not cost a call."""
    return len(group["names"]) > 1 or len(group["descriptions"]) > 1


def _validated_canonical_name(llm_name: str, group: dict) -> str:
    """Accept the LLM's name only if it still normalizes to THIS group's form. That
    lets the model fix casing/punctuation garble ("GLoBEX" -> "Globex") while
    rejecting a hallucinated name for a different entity. A hallucination guard, not
    a naming heuristic — shortest_clean is only the fallback."""
    cleaned = (llm_name or "").strip()
    if cleaned and normalize_name(cleaned) == group["key"]:
        return cleaned
    if cleaned:
        logger.debug("Description merge: rejected off-group canonical_name %r for group %r",
                     cleaned, group["names"][:2])
    return shortest_clean(group["names"])


def _no_llm_fallback(group: dict) -> dict:
    """The answer when no LLM call happens (or one fails): the deterministic name
    pick, and the longest description — with nothing to synthesize the merged text,
    the richest single one is the least-lossy answer available."""
    return {
        "name": shortest_clean(group["names"]),
        "description": max(group["descriptions"], key=len) if group["descriptions"] else None,
    }


async def _reduce_one_piece_batch(
    pieces: list[tuple[str, int, dict, list[str]]],
    depth: int,
) -> dict[tuple[str, int], str]:
    """One LLM call folding several description pieces into one partial each. Pieces may
    belong to DIFFERENT entities — each payload item is independent, so packing across
    entities keeps the call count proportional to slice volume, not entity count.

    Returns {(key, ordinal): partial} for pieces that got a non-empty description back.
    Missing entries mean that piece contributed nothing (same empty-filter as the old
    per-entity reduce)."""
    payload = [
        {"id": i, "names": group["names"], "type": group["type"], "descriptions": descs}
        for i, (_key, _ordinal, group, descs) in enumerate(pieces)
    ]
    ok, parsed = await structured_llm_call(
        _DESCRIPTION_MERGE_SYSTEM, json.dumps(payload), _DocumentDescriptionMerge,
        step="doc_description_reduce",
        context=f"pieces={[(key, ordinal) for key, ordinal, _g, _d in pieces][:5]} depth={depth}",
        calls_metric="description_reduce_calls",
        failures_metric="description_reduce_call_failures",
        kind=CallKind.DOC_DESCRIPTION_MERGE,
    )
    by_id = {m.id: m for m in parsed.entities} if ok and parsed is not None else {}
    out: dict[tuple[str, int], str] = {}
    for i, (key, ordinal, _group, _descs) in enumerate(pieces):
        merged = by_id.get(i)
        if merged is None:
            continue
        description = (merged.description or "").strip()
        if description:
            out[(key, ordinal)] = description
    return out


async def _reduce_pending_to_cap(groups: list[dict], depth: int = 0) -> dict[str, str]:
    """Tree-reduce every entity's descriptions to exactly ONE so the merge call that
    follows is a final naming/synthesis pass over a single description each.

    Entities with > _DESCRIPTIONS_PER_GROUP_MAX descriptions are sliced into pieces of
    that size; entities with 2..cap descriptions submit their whole list as one piece.
    Pieces are pooled across entities into shared calls packed by
    _DESCRIPTION_MERGE_CHAR_BUDGET. Recurses until every entity has one description.
    Fails closed PER ENTITY to its single richest description. Never raises."""
    result: dict[str, str] = {}
    pending = []
    for group in groups:
        descriptions = group["descriptions"]
        if len(descriptions) <= 1:
            result[group["key"]] = descriptions[0] if descriptions else ""
        else:
            pending.append(group)
    if not pending:
        return result

    # Flat list of (key, ordinal, group, piece), in batch order — deterministic.
    # Over-cap: slice into <=cap pieces. Under-cap but >1: whole list as one piece.
    pieces: list[tuple[str, int, dict, list[str]]] = []
    for group in pending:
        descriptions = group["descriptions"]
        if len(descriptions) > _DESCRIPTIONS_PER_GROUP_MAX:
            slices = [descriptions[i:i + _DESCRIPTIONS_PER_GROUP_MAX]
                      for i in range(0, len(descriptions), _DESCRIPTIONS_PER_GROUP_MAX)]
        else:
            slices = [descriptions]
        for ordinal, chunk in enumerate(slices):
            pieces.append((group["key"], ordinal, group, chunk))

    chunks = pack_by_char_budget(
        pieces, lambda p: sum(len(d) for d in p[3]), _DESCRIPTION_MERGE_CHAR_BUDGET)
    logger.info("Description merge reduce (depth %d): %d entit(y/ies) with >1 description -> "
                "%d piece(s) in %d LLM call(s)", depth, len(pending), len(pieces), len(chunks))

    partials_by_key: dict[str, list[tuple[int, str]]] = {}
    for batch_result in await asyncio.gather(*[_reduce_one_piece_batch(c, depth) for c in chunks]):
        for (key, ordinal), text in batch_result.items():
            partials_by_key.setdefault(key, []).append((ordinal, text))

    still_pending: list[dict] = []
    for group in pending:
        key = group["key"]
        descriptions = group["descriptions"]
        # By piece ordinal, not completion order — next round stays deterministic.
        partials = [text for _ordinal, text in sorted(partials_by_key.get(key, []))]
        # Progress guard: recurse only on a strictly shorter list. PER ENTITY.
        if not partials or len(partials) >= len(descriptions):
            logger.warning("Description merge: reducing %r (%d descriptions) made no progress at "
                           "depth %d — keeping its single richest description instead",
                           group["names"][0], len(descriptions), depth)
            result[key] = max(descriptions, key=len)
            continue
        if len(partials) == 1:
            result[key] = partials[0]
        else:
            still_pending.append({**group, "descriptions": partials})

    if still_pending:
        result.update(await _reduce_pending_to_cap(still_pending, depth + 1))
    return result


async def _merge_one_description_batch(batch: list[dict]) -> dict[str, dict]:
    """One LLM call for a char-budget-packed batch of entity groups. Each group has
    already been reduced to one description; this call synthesizes the final named
    profile. Fails closed to _no_llm_fallback per group — degraded, never wrong, never
    an exception; losing a whole document to a description-merge blip would be a bad
    trade."""
    # Fold every group to exactly one description BEFORE the shared merge prompt:
    # over-cap entities are sliced then folded; under-cap multi-desc entities fold in
    # one extra pooled round; single-desc entities pass through free.
    reduced_by_key = await _reduce_pending_to_cap(batch)
    payload = [
        {"id": i, "names": group["names"], "type": group["type"],
         "descriptions": [reduced_by_key[group["key"]]]}
        for i, group in enumerate(batch)
    ]
    ok, parsed = await structured_llm_call(
        _DESCRIPTION_MERGE_SYSTEM, json.dumps(payload), _DocumentDescriptionMerge,
        step="doc_description_merge",
        context=f"entities={[g['names'][0] for g in batch][:5]}",
        calls_metric="description_merge_calls",
        failures_metric="description_merge_call_failures",
        kind=CallKind.DOC_DESCRIPTION_MERGE,
    )

    by_id = {m.id: m for m in parsed.entities} if ok and parsed is not None else {}
    if not by_id:
        logger.warning("Description merge returned nothing usable for %d entity group(s) — "
                       "falling back to the richest single description per entity", len(batch))
    resolved: dict[str, dict] = {}
    for i, group in enumerate(batch):
        merged = by_id.get(i)
        if merged is None:
            # A group the response skipped: fall back for that one group only,
            # never discard it.
            resolved[group["key"]] = _no_llm_fallback(group)
            continue
        description = (merged.description or "").strip()
        resolved[group["key"]] = {
            "name": _validated_canonical_name(merged.canonical_name, group),
            "description": description or _no_llm_fallback(group)["description"],
        }
    return resolved


async def _merge_mention_descriptions(groups: list[dict]) -> dict[str, dict]:
    """Fold every group's repeated mentions into one canonical name and one capped
    description, in ONE LLM call per document when they fit the char budget —
    per-entity calls would multiply Step 2's LLM volume for no extra signal. Above
    _DESCRIPTION_MERGE_CHAR_BUDGET it splits into concurrent packed calls, with no
    cross-slice reconciliation needed (each group is already one decided entity).

    Returns {group_key: {"name", "description"}} for every group, including the ones
    that never needed a call.
    """
    resolved: dict[str, dict] = {}
    pending: list[dict] = []
    for group in groups:
        if _needs_llm_decision(group):
            pending.append(group)
        else:
            resolved[group["key"]] = {
                "name": group["names"][0],
                "description": group["descriptions"][0] if group["descriptions"] else None,
            }

    if not pending:
        return resolved

    batches = pack_by_char_budget(
        pending, lambda g: sum(len(d) for d in g["descriptions"]),
        _DESCRIPTION_MERGE_CHAR_BUDGET)
    if len(batches) > 1:
        logger.info("Description merge: %d entity group(s) split into %d concurrent call(s)",
                    len(pending), len(batches))
    for batch_result in await asyncio.gather(*[_merge_one_description_batch(b) for b in batches]):
        resolved.update(batch_result)
    return resolved


async def _resolve_doc_cluster(cluster: list[str], registry: dict,
                               system_prompt: str) -> ClusterResolution:
    """Doc-level binding of the shared large-cluster splitter. Uses the shared
    DEFAULT_LLM_BATCH_MAX rather than a Step-2-specific constant — no reason for the
    two stages' prompt caps to differ, and one number is one thing to calibrate."""
    return await resolve_cluster_batched(
        cluster, registry, system_prompt,
        batch_max=DEFAULT_LLM_BATCH_MAX, log_prefix="Doc canonicalization",
        max_depth=settings.doc_cluster_resolve_max_depth,
    )


async def canonicalize_document(chunk_results: list[ChunkExtractionResult]) -> tuple[dict, dict]:
    """Fold duplicate entity mentions within one document's chunks into canonical entities.

    Returns (canonical_entities, merge_map):
      canonical_entities: {canonical_name: {type, description}}
      merge_map: {raw_name: canonical_name} — every raw entity name this document produced,
                 pointing at its canonical name (itself, if it wasn't merged with anything).
                 Every raw name must appear: services/edge_pool_service.py resolves each
                 relationship endpoint and claim subject through this map, and a missing
                 key silently drops that edge.
    """
    groups = _build_mention_groups(chunk_results)
    if not groups:
        return {}, {}
    all_raw_names = [name for group in groups for name in group["names"]]

    # Fold each entity's repeated mentions into one name + one description BEFORE
    # anything downstream reads a description: the cluster-merge LLM below is handed
    # this synthesis as its identity signal, and Step 3 persists it into the graph.
    merged_by_key = await _merge_mention_descriptions(groups)

    # From here on a group is addressed by its chosen representative name, which is
    # also the doc-level canonical name for every raw surface form in that group.
    registry: dict = {}
    raw_names_by_rep: dict = {}
    for group in groups:
        merged = merged_by_key[group["key"]]
        rep = merged["name"]
        registry[rep] = {"type": group["type"], "description": merged["description"]}
        # Two groups can only collide on `rep` if they had different normalized
        # names, which _validated_canonical_name rules out — but fold rather than
        # overwrite so a future prompt change can't silently drop raw names here.
        raw_names_by_rep.setdefault(rep, []).extend(group["names"])

    representatives = list(registry.keys())

    canonical_entities: dict = {}
    merge_map: dict = {}

    def _accept_as_own_canonical(rep: str) -> None:
        canonical_entities[rep] = {"type": registry[rep]["type"], "description": registry[rep].get("description")}
        for raw in raw_names_by_rep[rep]:
            merge_map[raw] = rep

    if len(representatives) < 2:
        for rep in representatives:
            _accept_as_own_canonical(rep)
        return canonical_entities, merge_map

    # Precision values (money, counts, dates, versions, IPs, standard clause IDs) carry no
    # synonymy — "$82k" and "$1.2k" must never merge just because they look similar under an
    # embedding. The normalized-name grouping above already folded exact duplicates, so any
    # precision representative reaching this point is a genuinely distinct value: held out of
    # similarity clustering and the LLM merge entirely (never just peeled back out after),
    # since holding out BEFORE clustering also stops a precision value from acting as a
    # single-linkage bridge that chains two unrelated prose entities together. (Their
    # descriptions are still merged above — that's about text quality, not identity.)
    precision_reps = [r for r in representatives if is_precision(r, registry[r]["type"])]
    prose_reps = [r for r in representatives if r not in set(precision_reps)]
    for rep in precision_reps:
        _accept_as_own_canonical(rep)

    if len(prose_reps) < 2:
        for rep in prose_reps:
            _accept_as_own_canonical(rep)
        return canonical_entities, merge_map

    embedder = get_embedding_service()
    prose_registry = {r: {"type": registry[r]["type"], "description": registry[r].get("description")}
                      for r in prose_reps}
    # resolve_fn: without it cluster_to_fixpoint sends a cluster of ANY size as one
    # prompt, and an oversized one fails closed to "every candidate its own group" —
    # merging silently stops for the document's largest entity. Naming comes from the
    # LLM as returned (cluster_to_fixpoint's trust_llm_naming default).
    _fixpoint_merge_map, fixpoint_final, _first_round_embeddings = await cluster_to_fixpoint(
        prose_reps, prose_registry, lambda n, _reg: n, embedder,
        DOC_SIMILARITY_THRESHOLD, ANTI_CHAIN_THRESHOLD, _MERGE_SYSTEM,
        resolve_fn=_resolve_doc_cluster,
    )
    for final_name, meta in fixpoint_final.items():
        canonical_entities[final_name] = {"type": meta["type"], "description": meta["description"]}
        for original_rep in meta["members"]:
            for raw in raw_names_by_rep.get(original_rep, [original_rep]):
                merge_map[raw] = final_name

    # Defensive completion: if an LLM response ever dropped a member, don't silently lose
    # it — fall back to mapping it to itself as its own canonical entity.
    for rep in representatives:
        for raw in raw_names_by_rep[rep]:
            if raw not in merge_map:
                merge_map[raw] = rep
                canonical_entities.setdefault(
                    rep, {"type": registry[rep]["type"], "description": registry[rep].get("description")})

    logger.info("Doc canonicalization: %d raw entities -> %d mention groups -> %d canonical",
                len(all_raw_names), len(groups), len(canonical_entities))
    return canonical_entities, merge_map
