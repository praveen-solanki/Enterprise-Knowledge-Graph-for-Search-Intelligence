"""
Incremental entity resolution: for each newly extracted entity, vector-search
the graph as it exists right now for likely duplicates, ask an LLM a yes/no
"same entity?" per candidate in similarity order, and merge on the first
confident yes. Used to reflect a single changed document within minutes,
unlike the corpus-wide bulk canonicalization services.

Fail-closed by design: an exhausted or non-retryable error is treated as
"not a match" rather than raised or retried forever — over-merging silently
corrupts the graph and doesn't self-correct, while under-merging is cheap to
fix later. Such failures are still logged to the call_failures ledger
(repository/mongo_repository.py's record_call_failure) for later auditing.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncAzureOpenAI

from repository.neo4j_repository import Neo4jRepository
from services import llm_client
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore

logger = logging.getLogger("graphrag.services.incremental_canonicalization")

DEFAULT_SIMILARITY_THRESHOLD = 0.83

# A candidate at or above this similarity is auto-confirmed without an LLM
# call — the extra confirmation would be pure cost, not a real safety check.
AUTO_LINK_THRESHOLD = 0.93

_MERGE_SYSTEM = (
    "You decide whether a NEWLY EXTRACTED entity is the SAME real-world entity as an "
    "EXISTING one already in a knowledge graph, or a genuinely different entity that only "
    "looks similar. Merge only if they refer to the same concept — same name pattern is NOT "
    "enough on its own (e.g. a 'Q12' survey and a 'Q12' team are different entities). When "
    "uncertain, prefer NOT merging: under-merging is cheap to fix later, over-merging "
    "silently corrupts the graph and cannot be undone automatically.\n\n"
    'Respond with strict JSON: {"merge": true/false}. No other text.'
)


def _embed_text(embedder: EmbeddingService, name: str, description: str) -> list[float]:
    text = f"{name}: {description}" if description else name
    return embedder.embed_batch([text])[0]


def _candidate_description(descriptions_json: Optional[str]) -> str:
    """Picks one representative description out of descriptions_json (one entry
    per contributing document) as a signal for the LLM merge-confirm prompt.
    Only ever used internally, never displayed to a user."""
    if not descriptions_json:
        return ""
    try:
        entries = json.loads(descriptions_json)
    except (TypeError, ValueError):
        return ""
    return entries[0].get("text", "") if entries else ""


def _get_client() -> AsyncAzureOpenAI:
    """Shared client (services/llm_client.py), not a fresh one per
    call. Thin wrapper kept so tests can patch this module's _get_client."""
    return llm_client.get_client()


async def _llm_confirms_merge(new_name: str, new_desc: str, cand_name: str, cand_desc: str) -> bool:
    """Fail-closed ("not a match") on a non-retryable error or exhausted
    retries. Retry/backoff/logging is shared via confirm_with_retry."""
    user = (f"NEW entity: {new_name} — {new_desc or '(no description)'}\n"
            f"EXISTING entity: {cand_name} — {cand_desc or '(no description)'}\n"
            "Are these the same real-world entity?")
    return await llm_client.confirm_with_retry(
        _MERGE_SYSTEM, user, "merge",
        step="cluster_merge_confirm_incremental",
        context=f"new={new_name!r} candidate={cand_name!r}",
        log_label="Merge-confirmation",
    )


async def resolve_canonical_name(
    repo: Neo4jRepository,
    embedder: EmbeddingService,
    vector_store: VectorStore,
    name: str,
    entity_type: str,
    description: str,
    tenant_id: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[str, Optional[list]]:
    """Returns (canonical_name, embedding). canonical_name is `name` itself if
    no existing entity was confirmed as a match. embedding is None on the
    exact-name fast path below; the caller pushes it to Qdrant only when one
    was actually computed, leaving an untouched entity's existing vector alone.

    tenant_id scopes find_similar_entities so an entity can only ever match
    and merge into another entity from the same tenant.

    Exact-name fast path: if an Entity node for this exact (name, type,
    tenant_id) already exists, `name` is already canonical — skip the
    embedding call, ANN query, and LLM confirms entirely. This is the common
    case for incremental updates, so it's a real cost/latency win.

    A candidate at or above AUTO_LINK_THRESHOLD skips the LLM confirm."""
    if repo.entity_exists_exact(name, entity_type, tenant_id):
        return name, None
    embedding = await asyncio.to_thread(_embed_text, embedder, name, description)
    candidates = repo.find_similar_entities(vector_store, embedding, entity_type, tenant_id,
                                            k=5, min_similarity=threshold)
    for cand in candidates:
        score = cand.get("score", 0.0)
        if score >= AUTO_LINK_THRESHOLD:
            logger.info("Auto-linked %r into existing entity %r (score=%.3f >= %.2f, no LLM call)",
                       name, cand["name"], score, AUTO_LINK_THRESHOLD)
            return cand["name"], embedding
        cand_description = _candidate_description(cand.get("descriptions_json"))
        if await _llm_confirms_merge(name, description, cand["name"], cand_description):
            logger.info("Merged %r into existing entity %r (score=%.3f)", name, cand["name"], score)
            return cand["name"], embedding
    return name, embedding
