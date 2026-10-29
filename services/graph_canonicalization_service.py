"""
Step 3 — corpus-wide canonicalization. Dedupes entities across every
document in the run (Step 2 only dedupes within a single document).

Embeds NAME + DESCRIPTION (unlike Step 2's name-only), since corpus scale has
more unrelated entities sharing a name, and uses a more conservative merge
threshold since over-merging across the whole corpus is harder to undo.

Clustering runs to a fixpoint (services/clustering_utils.cluster_to_fixpoint)
so multi-hop merges across rounds aren't missed. Each cluster's LLM
resolution is checkpointed to MongoDB as it completes, so a crash mid-run
only re-resolves clusters that weren't finished yet.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from config.settings import settings

from services.clustering_utils import (
    CLUSTERING_VERSION, ClusterResolution, MergeGroup,
    cluster_to_fixpoint, embedding_as_list, is_precision, normalize_name, resolve_cluster,
    resolve_cluster_batched, shortest_clean, structured_llm_call,
)
from services.embedding_service import get_embedding_service
from services.llm_profiles import CallKind
from services.description_pack_utils import pack_by_char_budget

logger = logging.getLogger("graphrag.services.graph_canonicalization")


# Matches core/batch_canonicalization.py's cfg.graph_similarity_threshold
# (not config.py's unused graph_threshold=0.85 field).
GRAPH_SIMILARITY_THRESHOLD = 0.82

# Looser cut than the clustering threshold above: below GRAPH_SIMILARITY_THRESHOLD, so
# a cluster is only split where a member is clearly unrelated rather than merely less
# similar than its neighbours.
ANTI_CHAIN_THRESHOLD = min(0.78, GRAPH_SIMILARITY_THRESHOLD - 0.05)

# Caps candidates per LLM merge call. Clusters here are already complete-linkage
# cliques (tight_partition ran first), so slicing preserves coherence without
# needing to re-cluster.
_LLM_BATCH_MAX = settings.corpus_cluster_merge_batch_max


def _cluster_key(cluster: list[str]) -> str:
    """A stable, short hash that uniquely identifies a specific cluster of names.
    Same names in any order → same key. Used as the per-cluster MongoDB cache key."""
    return hashlib.sha1("|".join(sorted(cluster)).encode("utf-8")).hexdigest()[:12]


def _load_cluster_cache(checkpoint_key: str) -> Optional[ClusterResolution]:
    """Load a single cluster's already-resolved LLM result from MongoDB.
    Returns None if this cluster hasn't been resolved yet (cache miss)."""
    from repository.mongo_repository import load_checkpoint  # lazy import — avoids circular deps
    data = load_checkpoint("corpus_cluster_cache", checkpoint_key)
    if data is None:
        return None
    try:
        return ClusterResolution(**data)
    except Exception:
        return None  # corrupted cache entry — treat as a miss, re-resolve


def _save_cluster_cache(checkpoint_key: str, resolution: ClusterResolution) -> None:
    """Save one cluster's LLM resolution to MongoDB the moment it completes.
    If saving fails (transient network error etc.), log and continue — the
    cluster result is still used for this run, it just won't be cached."""
    from repository.mongo_repository import save_checkpoint
    try:
        save_checkpoint("corpus_cluster_cache", checkpoint_key, resolution.model_dump())
    except Exception:
        logger.warning("Could not cache cluster resolution for key %s — will re-resolve on restart",
                       checkpoint_key)


async def _resolve_cluster_batched(cluster: list[str], registry: dict, system_prompt: str,
                                   depth: int = 0) -> ClusterResolution:
    """Corpus binding of clustering_utils.resolve_cluster_batched, which now holds the
    slice/recurse/expand logic and shares it with Step 2. resolve_cluster and
    _LLM_BATCH_MAX are passed explicitly so they stay THIS module's names — the corpus
    tests' patch points, and a cap independent of Step 2's."""
    return await resolve_cluster_batched(
        cluster, registry, system_prompt,
        resolve_fn=resolve_cluster, batch_max=_LLM_BATCH_MAX,
        log_prefix="Corpus canonicalization",
        max_depth=settings.corpus_cluster_resolve_max_depth, depth=depth,
    )


# ---------------------------------------------------------------------------
# Corpus-level description merge — builds each entity's internal similarity signal
# from EVERY contributing document's description instead of one of them.
# ---------------------------------------------------------------------------

# An entity's per-document descriptions per merge call. A common organisation name
# accumulates thousands, so anything above this is tree-reduced
# (_reduce_descriptions_to_cap), not truncated to the richest N — which built the
# identity signal from a few percent of the evidence, chosen by length. Truncation
# survives only as the fail-closed path.
_MERGE_INPUT_MAX_DESCRIPTIONS = settings.corpus_description_merge_max_descriptions

# Payload budget per merge call, in characters. Packed greedily rather than by a fixed
# entity count: per-entity size varies by an order of magnitude (1 description vs 12),
# so a fixed count would overflow on the heavy ones or waste calls on the light ones.
_DESCRIPTION_MERGE_CHAR_BUDGET = settings.corpus_description_merge_char_budget

# Hard ceiling on the merged signal (~15 lines at ~90 chars), for a model that ignores
# _DESCRIPTION_LENGTH_RULE: nothing downstream truncates, and resolve_cluster puts up
# to _LLM_BATCH_MAX of these in one prompt.
_DESCRIPTION_SIGNAL_MAX_CHARS = settings.corpus_description_signal_max_chars

# Hard cap on the per-entity PERSISTED "descriptions" list (separate from the
# merge-signal caps above) — guards a single entity's Mongo document against
# the 16MB BSON limit when a name is mentioned in an extreme number of
# documents. Applied once, in one pass, right before canonicalize_corpus
# returns, so it covers every code path uniformly.
_MAX_PERSISTED_DESCRIPTIONS = settings.corpus_entity_max_persisted_descriptions

# One length rule for every reduce level and the final merge, so a depth-2 partial is
# bounded like the final answer — otherwise each level re-expands what the one below
# compressed. Given in lines AND characters (neither unit is fixed for a model), with
# the budget derived from the line count so an obedient response never gets clipped.
_DESCRIPTION_LENGTH_RULE = (
    "Keep the description to 2-3 paragraphs spanning 10-15 lines "
    "(roughly 900-1300 characters, and never more than 1400). Prefer what the documents "
    "state consistently; keep a detail only one document gives if it is concrete and "
    "identifying. Drop repetition rather than lengthening the text. Never invent anything "
    "the given text does not support."
)

_DESCRIPTION_MERGE_SYSTEM = (
    "Each item below is ONE entity as several different documents in a corpus described it "
    "independently. These are NOT different entities — do not split them. Some items are "
    "themselves already-merged summaries of other descriptions of that same entity; treat "
    "them exactly the same way.\n\n"
    "For each item, return:\n"
    "- id: echo the item's id back unchanged.\n"
    "- description: ONE description combining the information from all of the given "
    f"descriptions. {_DESCRIPTION_LENGTH_RULE}\n\n"
    "Respond with strict JSON: {\"entities\": [{\"id\": ..., \"description\": ...}]}"
)


class _MergedCorpusDescription(BaseModel):
    id: int = Field(..., description="The id of the input item this result is for, echoed back "
                    "exactly as given.")
    description: str = Field(
        ..., description="One synthesized description covering every given description of this "
        "item, preserving concrete facts (numbers, dates, versions) rather than a vague "
        "generic summary.")


class _CorpusDescriptionMerge(BaseModel):
    entities: list[_MergedCorpusDescription] = Field(
        ..., description="One entry per input item, each carrying its echoed id.")


def _richest(descriptions: list[str], limit: Optional[int] = None) -> list[str]:
    """Distinct descriptions, richest first — the `limit` longest if given, all of
    them otherwise. Sorted by (-length, text) so equal-length texts order stably.
    Called unlimited to gather the reduce's input, limited for fail-closed paths."""
    ordered = sorted(dict.fromkeys(d for d in descriptions if d), key=lambda d: (-len(d), d))
    return ordered[:limit] if limit is not None else ordered


def _deal_into_slices(descriptions: list[str], slice_size: int) -> list[list[str]]:
    """Slices of at most `slice_size`, DEALT round-robin rather than cut in order. Input
    arrives richest-first, so cutting would pool every long description in one slice and
    every thin one in another — and since the next level treats each slice's summary as
    an equal peer, that promotes thin passing mentions and dilutes the rich ones."""
    count = -(-len(descriptions) // slice_size)  # ceil, so no slice exceeds slice_size
    slices: list[list[str]] = [[] for _ in range(count)]
    for i, description in enumerate(descriptions):
        slices[i % count].append(description)
    return slices


def _cap_signal(text: Optional[str]) -> Optional[str]:
    """Trim a merged description to _DESCRIPTION_SIGNAL_MAX_CHARS on a word boundary. ONLY
    the internal signal — a persisted final_entities["descriptions"] entry is never cut."""
    if not text or len(text) <= _DESCRIPTION_SIGNAL_MAX_CHARS:
        return text or None
    cut = text[:_DESCRIPTION_SIGNAL_MAX_CHARS]
    return cut[:cut.rfind(" ")].rstrip() if " " in cut else cut


def _cap_persisted_descriptions(name: str, descriptions: list) -> list:
    """Trim the persisted per-doc "descriptions" list to _MAX_PERSISTED_DESCRIPTIONS.
    Only the pathological tail (a name mentioned in an extreme number of
    documents) is ever affected — everything else passes through unchanged."""
    if len(descriptions) <= _MAX_PERSISTED_DESCRIPTIONS:
        return descriptions
    logger.warning(
        "Corpus canonicalization: %r has %d per-document descriptions — capping the "
        "persisted list to %d to stay well under Mongo's 16MB document limit",
        name, len(descriptions), _MAX_PERSISTED_DESCRIPTIONS,
    )
    return descriptions[:_MAX_PERSISTED_DESCRIPTIONS]


async def _reduce_one_slice_batch(slices: list[tuple[str, int, list[str]]]) -> dict:
    """One LLM call folding several slices into one partial description each. Slices may
    belong to DIFFERENT entities — each payload item is independent, so packing across
    entities keeps the call count proportional to description volume, not entity count.

    Returns {(key, ordinal): [description, ...]} — the partial on success, the slice's own
    descriptions unchanged on failure. Passing a failure through UNREDUCED is deliberate:
    collapsing it would look like progress and skip the truncation fallback."""
    payload = [{"id": i, "descriptions": descs} for i, (_key, _ordinal, descs) in enumerate(slices)]
    ok, parsed = await structured_llm_call(
        _DESCRIPTION_MERGE_SYSTEM, json.dumps(payload), _CorpusDescriptionMerge,
        step="corpus_description_reduce",
        context=f"slices={[(key, ordinal) for key, ordinal, _d in slices][:5]}",
        calls_metric="corpus_description_reduce_calls",
        failures_metric="corpus_description_reduce_call_failures",
        kind=CallKind.CORPUS_DESCRIPTION_MERGE,
    )
    by_id = ({m.id: (m.description or "").strip() for m in parsed.entities}
             if ok and parsed is not None else {})
    if not by_id:
        logger.warning("Corpus description reduce returned nothing usable for %d slice(s) — "
                       "passing their descriptions through unreduced", len(slices))
    out: dict = {}
    for i, (key, ordinal, descs) in enumerate(slices):
        partial = by_id.get(i)
        out[(key, ordinal)] = [_cap_signal(partial) or partial] if partial else list(descs)
    return out


async def _reduce_descriptions_to_cap(pending: list[tuple[str, list[str]]],
                                      depth: int = 0) -> list[tuple[str, list[str]]]:
    """Tree-reduce every entity's descriptions to exactly ONE so the merge call that
    follows is a final synthesis over a single description each.

    Entities with > _MERGE_INPUT_MAX_DESCRIPTIONS are dealt into slices of that size;
    entities with 2..cap descriptions submit their whole list as one slice. Slices are
    pooled across entities into shared calls packed by _DESCRIPTION_MERGE_CHAR_BUDGET.
    Recurses until every entity has one description. Fails closed per entity to its
    single richest description. Never raises."""
    needs_fold = [(key, descs) for key, descs in pending if len(descs) > 1]
    if not needs_fold:
        return pending

    slices: list[tuple[str, int, list[str]]] = []
    for key, descs in needs_fold:
        if len(descs) > _MERGE_INPUT_MAX_DESCRIPTIONS:
            for ordinal, chunk in enumerate(
                    _deal_into_slices(descs, _MERGE_INPUT_MAX_DESCRIPTIONS)):
                slices.append((key, ordinal, chunk))
        else:
            # Already fits one prompt item — fold the whole list to one in this round.
            slices.append((key, 0, descs))

    batches = pack_by_char_budget(
        slices, lambda s: sum(len(d) for d in s[2]), _DESCRIPTION_MERGE_CHAR_BUDGET)
    logger.info("Corpus description reduce (depth %d): %d entit(y/ies) with >1 description -> "
                "%d slice(s) in %d LLM call(s)", depth, len(needs_fold),
                len(slices), len(batches))

    partials_by_key: dict = {}
    for result in await asyncio.gather(*[_reduce_one_slice_batch(b) for b in batches]):
        for (key, ordinal), texts in result.items():
            partials_by_key.setdefault(key, []).append((ordinal, texts))

    reduced: list[tuple[str, list[str]]] = []
    for key, descs in pending:
        if len(descs) <= 1:
            reduced.append((key, descs))
            continue
        # By slice ordinal, not completion order, so the next round's input is identical.
        new_descs = [text for _ordinal, texts in sorted(partials_by_key.get(key, []))
                     for text in texts]
        if not new_descs or len(new_descs) >= len(descs):
            logger.warning("Corpus description reduce: %r made no progress at depth %d "
                           "(%d -> %d) — keeping its single richest description instead",
                           key, depth, len(descs), len(new_descs))
            reduced.append((key, [max(descs, key=len)]))
        else:
            reduced.append((key, new_descs))

    return await _reduce_descriptions_to_cap(reduced, depth + 1)


async def merge_corpus_descriptions(descriptions_by_key: dict) -> dict:
    """Fold each key's per-document descriptions into ONE merged description.

    descriptions_by_key: {key: [description text, ...]} for one entity.
    Returns {key: merged description or None} — none gives None, exactly one gives that
    one unchanged (no call, nothing to decide).

    Used for the corpus registry's similarity signal below, and by person_corpus_resolver.py:
    PERSON entities skip this stage's clustering but need the same signal. Batched, not
    per-entity, since at corpus scale per-entity calls would dominate LLM volume for no
    extra signal; concurrency is bounded by the throttle inside structured_llm_call."""
    merged: dict = {}
    pending: list[tuple[str, list[str]]] = []
    for key, descriptions in descriptions_by_key.items():
        # No input cap: _reduce_descriptions_to_cap folds every multi-desc entity
        # down to exactly one description before the final merge call.
        distinct = _richest(descriptions)
        if not distinct:
            merged[key] = None
        elif len(distinct) == 1:
            merged[key] = _cap_signal(distinct[0])
        else:
            pending.append((key, distinct))

    if not pending:
        return merged

    pending = await _reduce_descriptions_to_cap(pending)
    # _reduce_descriptions_to_cap guarantees every entry above is now exactly one
    # description — same "nothing left to decide" shortcut as the len(distinct) == 1
    # case above, so no merge call is needed: it would only ask the LLM to "combine"
    # a list of one.
    merged.update({key: _cap_signal(descs[0]) for key, descs in pending})
    return merged


_MERGE_SYSTEM = (
    "You are deduplicating entities across an ENTIRE document corpus (not just one document). "
    "Given a list of candidate entity names (with type and description) that look similar, decide "
    "which ones refer to the SAME real-world thing. Merge ONLY if you are confident — a shared or "
    "similar name is NOT enough on its own (e.g. a 'Q1' financial quarter and a 'Q1' product code "
    "are different entities). This is corpus-wide: an incorrect merge here is much harder to notice "
    "and undo than a document-level one. When in doubt, keep them SEPARATE.\n\n"
    "Respond with strict JSON: {\"groups\": [{\"canonical_name\": ..., \"canonical_type\": ..., "
    "\"merged_description\": ..., \"members\": [...]}]}"
)

# Bump whenever this stage's prompt, threshold, clustering algorithm, or output
# schema changes, so core/pipeline.py never serves a cached result in a shape
# the current code doesn't expect.
#
# v7-merged-signal: the singular description is now a synthesis of every contributing
# document, a merged entity's vector is re-embedded from its FINAL name + description, and
# large-cluster consolidation recurses. None touch a hashed value, so v6 would be reused.
# v8-llm-naming: trust_llm_naming defaults on, so the canonical NAME and between-round
# description are the LLM's choice, not shortest_clean's and best_description's — changing
# which entities merge and what they end up called.
# v9-reduced-descriptions: descriptions are tree-reduced to _MERGE_INPUT_MAX_DESCRIPTIONS
# rather than truncated to the richest N, changing the signal, the embedding, and therefore
# the merges. The four caps are hashed in too, so re-tuning one invalidates on its own.
# v10-reduce-to-one: reduce continues until exactly one description per entity, not merely
# under the per-call input cap. A v9 entry may hold a multi-description pre-merge signal
# the new path would have folded first.
_OUTPUT_SCHEMA_VERSION = "v10-reduce-to-one"
GRAPH_CANON_VERSION = hashlib.sha1(
    f"{_MERGE_SYSTEM}|{_DESCRIPTION_MERGE_SYSTEM}|{GRAPH_SIMILARITY_THRESHOLD}|"
    f"{CLUSTERING_VERSION}|{_MERGE_INPUT_MAX_DESCRIPTIONS}|"
    f"{_DESCRIPTION_MERGE_CHAR_BUDGET}|{_DESCRIPTION_SIGNAL_MAX_CHARS}|{_LLM_BATCH_MAX}|"
    f"{_OUTPUT_SCHEMA_VERSION}".encode("utf-8")
).hexdigest()[:12]


async def canonicalize_corpus(canonical_by_doc: dict,
                              cluster_checkpoint_key: Optional[str] = None) -> tuple[dict, dict]:
    """Dedupe entities across every document's Step-2 output.

    canonical_by_doc: {doc_id: (doc_canonical_entities, doc_merge_map)} — Step 2's output.

    Returns (final_entities, final_merge_map_by_doc):
      final_entities: {final_canonical_name: {type, source_docs: [...], embedding: [...],
                        description, descriptions: [{text, doc_id}, ...],
                        aliases: [{name, doc_id}, ...]}}
        embedding is persisted (not discarded) so Neo4j can build a vector index
        for the query agent to search the graph.
        description (singular) is the LLM's final merged description for the
        entity — the fixpoint's folded text for a merged group, the norm-group
        synthesis otherwise — persisted to e.description by decision: the graph
        must carry one readable description per canon entity, ACL notwithstanding
        (the per-doc entries below remain the ACL-filterable form).
        descriptions/aliases keep one entry per contributing document, never
        blended into a single string/list — core/pipeline.py Step 5 ACL-tags each
        entry by its own source document, since collapsing them could let a
        viewer with access to only one source read text that came from a
        document they can't see.
      final_merge_map_by_doc: {doc_id: {doc_canonical_name: final_canonical_name}} — lets Step 4
        re-point each document's relationships/claims at the FINAL name, not just the doc-level one.
    """
    # One corpus-wide registry: each name maps to every doc it came from and every doc's own
    # description text. registry[n]["description"] (singular) is the LLM-merged text — a
    # similarity signal for clustering_utils.py's embedding/prompt, and ALSO what
    # final_entities carries as "description" (persisted to e.description); the per-doc
    # "descriptions" list is written alongside it as descriptions_json.
    registry: dict = {}
    for doc_id, (doc_entities, _merge_map) in canonical_by_doc.items():
        for name, meta in doc_entities.items():
            entry = registry.setdefault(
                name, {"type": meta["type"], "source_docs": [], "descriptions": [], "description": None}
            )
            entry["source_docs"].append(doc_id)
            desc_text = meta.get("description")
            if desc_text:
                entry["descriptions"].append({"text": desc_text, "doc_id": doc_id})

    names = list(registry.keys())

    # Grouped by normalized name so surface-form variants the exact-name pass collapses
    # anyway ("Globex" / "globex ") share ONE merged description. Built here, not in the
    # clustered branch below, because the description merge needs it too.
    groups_by_norm: dict = {}
    for n in names:
        groups_by_norm.setdefault(normalize_name(n), []).append(n)

    # Was the FIRST non-empty description seen — whichever document Mongo's cursor returned
    # first — so the merge decision rested on one arbitrary blurb: under-merging on a thin
    # passing mention, over-merging on a generic one. Now a synthesis of all of them.
    merged_by_norm = await merge_corpus_descriptions({
        norm: [d["text"] for member in members for d in registry[member]["descriptions"]]
        for norm, members in groups_by_norm.items()
    })
    for norm, members in groups_by_norm.items():
        for member in members:
            registry[member]["description"] = merged_by_norm.get(norm)
    del merged_by_norm  # folded into registry above; not needed for the rest of this call

    final_entities: dict = {}
    name_to_final: dict = {}
    embedder = get_embedding_service()

    def _embed_text(n: str) -> str:
        desc = registry[n].get("description")
        return f"{n}: {desc}" if desc else n

    if len(names) < 2:
        for n in names:
            final_entities[n] = dict(registry[n])
            final_entities[n]["embedding"] = embedder.embed_batch([_embed_text(n)])[0] if names else []
            name_to_final[n] = n
    else:
        # Cheap pass: collapse exact-normalized-name duplicates before any embedding/LLM
        # calls (groups_by_norm was built above, for the description merge). shortest_clean
        # picks the representative so a casing garble like "GLoBEX" can't win by position.
        representatives = [shortest_clean(members) for members in groups_by_norm.values()]

        # One embedding per representative, populated by either branch below.
        # Kept in scope so the defensive-completion pass at the bottom can reuse
        # a real embedding for a straggler instead of defaulting to [].
        embedding_by_rep: dict = {}

        def _accept_as_own_final(rep: str, embedding: Optional[list] = None) -> None:
            final_entities[rep] = dict(registry[rep])
            final_entities[rep]["embedding"] = embedding if embedding is not None \
                else embedder.embed_batch([_embed_text(rep)])[0]
            embedding_by_rep[rep] = final_entities[rep]["embedding"]
            for raw in groups_by_norm[normalize_name(rep)]:
                name_to_final[raw] = rep
                if raw != rep:
                    final_entities[rep]["source_docs"] += registry[raw]["source_docs"]

        # Precision values (money, counts, dates, versions, IDs) carry no synonymy —
        # e.g. "$82k" and "$1.2k" must never merge just because they look similar
        # under an embedding. Held out of clustering/LLM merge entirely, and held out
        # BEFORE clustering (not peeled back out after), so a precision value can't
        # act as a single-linkage bridge chaining two unrelated prose entities together.
        precision_reps = [r for r in representatives if is_precision(r, registry[r].get("type"))]
        representatives = [r for r in representatives if r not in set(precision_reps)]
        # Batch-embed all precision reps in one call rather than one at a time.
        if precision_reps:
            precision_vectors = embedder.embed_batch([_embed_text(r) for r in precision_reps])
            for rep, vector in zip(precision_reps, precision_vectors):
                _accept_as_own_final(rep, embedding=vector)

        if len(representatives) < 2:
            for rep in representatives:
                _accept_as_own_final(rep)
        else:
            # Name + description this time — the deliberate difference from Step 2.
            def _embed_text_fn(n: str, round_registry: dict) -> str:
                desc = round_registry.get(n, {}).get("description")
                return f"{n}: {desc}" if desc else n

            prose_registry = {r: {"type": registry[r]["type"], "description": registry[r].get("description")}
                              for r in representatives}

            # Injected as cluster_to_fixpoint's resolve_fn so per-cluster Mongo caching
            # and large-cluster batching apply on every round, not just the first.
            # reg_view is that round's own registry — never the outer corpus-wide
            # `registry`, whose keys wouldn't include a round-2-only merged name.
            async def _resolve_one(cluster: list[str], reg_view: dict, system_prompt: str) -> ClusterResolution:
                if cluster_checkpoint_key:
                    ck = f"{cluster_checkpoint_key}:{_cluster_key(cluster)}"
                    hit = _load_cluster_cache(ck)
                    if hit is not None:
                        logger.debug("Corpus cluster cache hit: members=%s", cluster[:2])
                        return hit
                result = await _resolve_cluster_batched(cluster, reg_view, system_prompt)
                if cluster_checkpoint_key:
                    ck = f"{cluster_checkpoint_key}:{_cluster_key(cluster)}"
                    _save_cluster_cache(ck, result)
                return result

            _fixpoint_merge_map, fixpoint_final, round1_embeddings = await cluster_to_fixpoint(
                representatives, prose_registry, _embed_text_fn, embedder,
                GRAPH_SIMILARITY_THRESHOLD, ANTI_CHAIN_THRESHOLD, _MERGE_SYSTEM,
                resolve_fn=_resolve_one,
            )
            del _fixpoint_merge_map, prose_registry  # unused; drop before the LLM re-embed pass below
            embedding_by_rep.update(round1_embeddings)
            del round1_embeddings  # values now live on in embedding_by_rep too

            # The PERSISTED vector must describe the merged entity, not a pre-merge member:
            # cluster_to_fixpoint returns only round-1 vectors, so a merged entity used to
            # ship one member's blurb — and incremental_canonicalization.py auto-links off
            # that vector at 0.93 with no LLM confirm, compounding the error across runs.
            # Re-embed from the final name + folded description: one extra batched call.
            #
            # Covers multi-member entities plus any final name with no round-1 vector, which
            # trust_llm_naming allows even for a single-member group. A single-member group
            # keeping its own name is skipped: already exact.
            merged_names = [n for n, meta in fixpoint_final.items()
                            if len(meta["members"]) > 1 or n not in embedding_by_rep]
            if merged_names:
                merged_vectors = await asyncio.to_thread(
                    embedder.embed_batch,
                    [_embed_text_fn(n, fixpoint_final) for n in merged_names],
                )
                embedding_by_rep.update(zip(merged_names, merged_vectors))

            for final_name, meta in fixpoint_final.items():
                # final_name is the LLM's own canonical_name (trust_llm_naming) and need not
                # match any member or registry key, so everything below keys off
                # meta["members"].
                source_docs: list = []
                descriptions: list = []
                for member in meta["members"]:
                    for raw in groups_by_norm.get(normalize_name(member), [member]):
                        name_to_final[raw] = final_name
                        source_docs.extend(registry.get(raw, {}).get("source_docs", []))
                        # Pool each member's own per-doc descriptions (the ACL-taggable
                        # form); the fixpoint's folded text goes in as "description" below.
                        descriptions.extend(registry.get(raw, {}).get("descriptions", []))
                # final_name may not match any single representative (multi-hop merges) —
                # reuse whichever member's round-1 embedding is available as an
                # approximation of the merged concept.
                rep_embedding = embedding_by_rep.get(final_name)
                if rep_embedding is None:
                    for member in meta["members"]:
                        if member in embedding_by_rep:
                            rep_embedding = embedding_by_rep[member]
                            break
                final_entities[final_name] = {
                    "type": meta["type"],
                    "source_docs": sorted(set(source_docs)),
                    "embedding": embedding_as_list(rep_embedding),
                    "description": meta.get("description") or None,
                    "descriptions": descriptions,
                }

        # Defensive completion: never silently drop a name. Reached when an LLM
        # cluster-merge call doesn't cover every input name (e.g. a failed/empty
        # resolve_cluster). Must never hard-code embedding=[] — that would make the
        # entity unsearchable — so reuse the representative's precomputed embedding.
        for n in names:
            if n not in name_to_final:
                name_to_final[n] = n
                rep = groups_by_norm.get(normalize_name(n), [n])[0]
                embedding = embedding_by_rep.get(rep)
                if embedding is None:
                    # Should be unreachable since every representative is embedded
                    # up front — last-resort guard against an unsearchable entity.
                    logger.warning("Corpus canonicalization: no pre-computed embedding for "
                                   "representative %r of straggler %r — embedding individually", rep, n)
                    embedding = embedder.embed_batch([_embed_text(n)])[0]
                final_entities.setdefault(
                    n, {**registry[n], "embedding": embedding_as_list(embedding)},
                )

    # Translate: for each doc, compose (doc raw name -> doc canonical name -> final canonical name)
    # into one direct (doc canonical name -> final canonical name) map.
    final_merge_map_by_doc: dict = {}
    for doc_id, (_doc_entities, _doc_merge_map) in canonical_by_doc.items():
        final_merge_map_by_doc[doc_id] = {
            doc_canon: name_to_final[doc_canon]
            for doc_canon in _doc_entities
        }

    # Aliases: every raw surface form that resolved to this final entity, tagged with
    # its source doc_id like descriptions above — an alias is surface-form text that
    # could otherwise leak from a document the viewer can't see.
    aliases_by_final: dict = {}
    for doc_id, (_doc_entities, doc_merge_map) in canonical_by_doc.items():
        for raw_name, doc_canon in doc_merge_map.items():
            final_name = name_to_final.get(doc_canon)
            if final_name:
                aliases_by_final.setdefault(final_name, set()).add((raw_name, doc_id))
    for final_name, entity in final_entities.items():
        pairs = aliases_by_final.get(final_name) or {(final_name, entity["source_docs"][0])
                                                      if entity.get("source_docs") else (final_name, "")}
        entity["aliases"] = [{"name": n, "doc_id": d} for n, d in sorted(pairs)]
        entity["descriptions"] = _cap_persisted_descriptions(final_name, entity["descriptions"])

    # registry/groups_by_norm/embedding_by_rep (when the clustered branch ran) are
    # corpus-sized and no longer needed — drop them before returning so this stage's
    # own peak doesn't carry into the Mongo persistence + Neo4j write that follow.
    del registry, groups_by_norm
    gc.collect()

    logger.info("Corpus canonicalization: %d doc-level canonical entities -> %d final entities",
                len(names), len(final_entities))
    return final_entities, final_merge_map_by_doc


async def resolve_entity_ids(final_entities: dict, tenant_id: str) -> dict:
    """Resolve/mint a stable entity_id for each final canonical name via
    repository/entity_registry.py — the same registry
    services/incremental_canonicalization.py uses, so both paths never diverge
    on identity. Pure translation from name to entity_id; identity itself was
    already decided by the clustering above.

    Returns {canonical_name: entity_id} for core/pipeline.py to hand to
    services/edge_pool_service.py, so relationships/claims/mentions can be
    keyed by entity_id instead of name.

    Concurrent, bounded by settings.corpus_entity_id_resolution_concurrency —
    the same asyncio.Semaphore + asyncio.to_thread shape services/people/
    resolution.py's resolve_bare_persons already uses against this exact
    registry. Safe because link_or_mint's unique index makes a racing insert
    collapse onto one surviving record (see repository/entity_registry.py);
    each canon_name here is also already its own distinct identity, so no two
    calls in this loop contend for the same document in the first place.

    PERSON entries are skipped entirely: services/people/ already resolved them
    by identity before this runs (materialize.py on (connector_type,
    source_key), resolution.py by name for a bare mention), keyed by their own
    entity_id rather than a display name (two different people can share a
    name) — re-running link_or_mint on that key would mint a spurious new
    identity instead. core/pipeline.py adds their identity mapping directly."""
    from repository import entity_registry  # lazy import — mirrors this file's
                                             # existing Mongo access (_load/_save_cluster_cache above)
    semaphore = asyncio.Semaphore(settings.corpus_entity_id_resolution_concurrency)

    async def _link_or_mint(canon_name: str, meta: dict) -> tuple[str, str]:
        plain = normalize_name(canon_name) or canon_name.strip().lower() or "∅"
        alias_names = [a.get("name") for a in (meta.get("aliases") or []) if a.get("name")]
        async with semaphore:
            rec = await asyncio.to_thread(
                entity_registry.link_or_mint,
                canon_name, meta.get("type") or "", plain, tenant_id,
                aliases=[normalize_name(a) for a in alias_names] + [plain],
            )
        return canon_name, rec["_id"]

    targets = [(canon_name, meta) for canon_name, meta in final_entities.items()
              if meta.get("type") != "PERSON"]
    total = len(targets)
    logger.info("resolve_entity_ids: minting/linking %d non-PERSON entities via Mongo registry", total)
    tasks = [asyncio.create_task(_link_or_mint(canon_name, meta)) for canon_name, meta in targets]
    canon_to_id: dict = {}
    done = 0
    for coro in asyncio.as_completed(tasks):
        canon_name, entity_id = await coro
        canon_to_id[canon_name] = entity_id
        done += 1
        if total and (done % 10000 == 0 or done == total):
            logger.info(
                "resolve_entity_ids progress %d/%d (%.1f%%)",
                done, total, 100.0 * done / total,
            )
    return canon_to_id
