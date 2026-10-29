"""
Neo4j + Qdrant dual-write safety (manager requirement, full-traceability
project): the two stores don't share a transaction, so true atomicity isn't
literally possible — there's no single commit that covers both. What we CAN
guarantee is the practical equivalent: retry the Qdrant side up to
settings.qdrant_max_retries times, and if it still fails, correct Neo4j's state so it never
keeps claiming a write that didn't actually happen — never leave one store
saying "yes, I have this" while the other doesn't, even silently.

Used by repository/neo4j_repository.py at every point it pushes to Qdrant
right after a Neo4j write. call_with_retry_or_rollback is the primitive;
three thin wrappers around it cover the three actual shapes every call site
in that file needs, so the loop/batching/context-string logic exists once
each instead of being repeated at every call site:

  - push_vectors_in_batches: a batched entity/chunk vector push, chunked at
    Qdrant's own payload-size limit, with an optional rollback per batch
    (write_entities/write_chunks — the vector itself, so Neo4j's has_vector
    flag would otherwise lie about a write Qdrant never received).
  - delete_vectors: delete a list of ids, no rollback (the Neo4j node is
    already gone either way — orphan/retraction cleanup).
  - sync_acl_for_each: push the same (or per-row) new acl to each id's Qdrant
    payload, no rollback (Neo4j's own acl is already correct; only Qdrant's
    copy would go stale on failure).

Every exhausted attempt is logged to the same call_failures ledger every
other retry-then-log call site in this repo uses (repository/mongo_repository.
py's record_call_failure) — full traceability means this shows up in the
same queryable place as an exhausted LLM call, not a separate mechanism.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional

from config.settings import settings
from repository.mongo_repository import record_call_failure
from utils.error_classification import classify_error

logger = logging.getLogger("graphrag.services.dual_write_guard")

# Retry policy: settings.qdrant_max_retries / qdrant_base_backoff_seconds /
# qdrant_max_backoff_seconds.

# A single huge Qdrant upsert can exceed Qdrant's own request-body limit regardless
# of Neo4j's batch size — confirmed against the real dev server (a 1280-entity batch
# serialized to ~50MB, over Qdrant's 32MB limit). Fixed, not calibrated, since it's a
# hard protocol limit, not a soft timing thing — same value the reference
# implementation uses for this exact problem (storage/vector_store.py's own B = 256).
_QDRANT_VECTOR_PUSH_BATCH = 256


def call_with_retry_or_rollback(fn: Callable[[], None], *, step: str, context: str,
                                rollback: Optional[Callable[[], None]] = None) -> bool:
    """Calls fn() (a zero-arg callable wrapping one Qdrant operation) up to
    settings.qdrant_max_retries times with exponential backoff + jitter. Returns
    True the moment it succeeds.

    If every attempt fails: logs one durable call_failures record (so an
    outage that broke a dual write is queryable later, not just an app log
    line), then — if given — calls rollback() to correct the Neo4j side.
    rollback itself is never allowed to raise into the caller: a rollback
    failure is logged loudly (this now needs manual reconciliation) but
    never masks the original failure or crashes whatever write path called
    this. Returns False.

    step/context: same shape as every other record_call_failure call site —
    step names the call site (e.g. "qdrant_entity_upsert"), context is a
    short human-readable description of what was being written (e.g. the
    affected entity_ids)."""
    delay = 0.0
    for attempt in range(1, settings.qdrant_max_retries + 1):
        if delay > 0:
            time.sleep(delay)
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 — classified below, not swallowed
            error_class = classify_error(exc)
            if error_class != "transient" or attempt == settings.qdrant_max_retries:
                logger.error("%s: failed after %d attempt(s) — %s: %s",
                            step, attempt, context, exc)
                record_call_failure(step, context, error_class, str(exc), attempts=attempt)
                if rollback is not None:
                    try:
                        rollback()
                    except Exception:  # noqa: BLE001 — never let a rollback failure hide the original
                        logger.exception("%s: rollback ALSO failed for %s — manual reconciliation "
                                         "needed (Neo4j and Qdrant are now genuinely inconsistent)",
                                         step, context)
                return False
            delay = min(settings.qdrant_base_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.3),
                       settings.qdrant_max_backoff_seconds)
            logger.warning("%s: attempt %d/%d failed (transient), retrying in %.1fs: %s",
                           step, attempt, settings.qdrant_max_retries, delay, exc)
    return False


def push_vectors_in_batches(store, rows: list[dict], *, step: str, id_field: str = "entity_id",
                            rollback: Optional[Callable[[list[str]], None]] = None) -> None:
    """Chunks rows at _QDRANT_VECTOR_PUSH_BATCH and pushes each chunk via
    store.upsert_entities_batch, retry-then-log(-or-rollback) per chunk —
    shared by every batched entity/chunk vector push in
    repository/neo4j_repository.py (write_entities, write_chunks), which
    otherwise repeated this exact loop with only the step name and rollback
    differing. rollback, when given, is called with the failed chunk's own
    ids (e.g. mark_vector_missing) — never the whole rows list, so a partial
    batch failure only corrects the ids that actually failed."""
    for i in range(0, len(rows), _QDRANT_VECTOR_PUSH_BATCH):
        batch = rows[i:i + _QDRANT_VECTOR_PUSH_BATCH]
        ids = [row[id_field] for row in batch]
        call_with_retry_or_rollback(
            lambda batch=batch: store.upsert_entities_batch(batch),
            step=step, context=f"{id_field}s={ids[:5]} (n={len(ids)})",
            rollback=(lambda ids=ids: rollback(ids)) if rollback is not None else None,
        )


def delete_vectors(store, ids: list[str], *, step: str, id_label: str = "entity_ids") -> None:
    """Deletes ids from store if there are any to delete, retry-then-log --
    shared by every orphan/retraction vector-delete call site in
    repository/neo4j_repository.py. No rollback: the Neo4j node this
    corresponds to is already gone either way, so there's nothing to correct
    back — an exhausted retry just needs a durable, queryable record for
    manual reconciliation."""
    if not ids:
        return
    call_with_retry_or_rollback(
        lambda: store.delete_entities(ids),
        step=step, context=f"{id_label}={ids[:5]} (n={len(ids)})",
    )


def sync_acl_for_each(store, id_acl_pairs: list[tuple[str, list[str]]], *, step: str,
                      id_label: str = "entity_id") -> None:
    """Per-id ACL sync, retry-then-log -- shared by every entity/chunk ACL-sync
    loop in repository/neo4j_repository.py. No rollback: Neo4j's own acl is
    already correct at this point (this call is only pushing the SAME acl to
    Qdrant's payload); Qdrant is the one side that would go stale on failure,
    so an exhausted retry just needs a durable record, not a Neo4j correction."""
    for entity_id, acl in id_acl_pairs:
        call_with_retry_or_rollback(
            lambda entity_id=entity_id, acl=acl: store.set_acl(entity_id, acl),
            step=step, context=f"{id_label}={entity_id!r}",
        )
