"""
Checkpoint storage for resumable runs. Each expensive step's result (a
chunk's extraction, a document's canonical entities, a corpus's final
entities) gets saved here under a stable key. Re-running the pipeline checks
here FIRST — if a key's already saved, that step is skipped entirely instead
of re-paying for the same Azure calls.

Two functions, nothing else: save one thing, load one thing.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

from bson import encode as bson_encode
from pymongo import MongoClient, ReturnDocument, UpdateOne

from config.db_endpoints import DbEndpoints
from config.settings import settings

logger = logging.getLogger("graphrag.repository.mongo")

# One client per (uri, db) — the incremental worker may talk to two MongoDBs
# (primary and EC2) over the life of the process, so a single global client no
# longer suffices. Reused across messages that resolve to the same endpoint.
_clients: dict[tuple[str, str], MongoClient] = {}

# The MongoDB the CURRENT unit of work targets, bound per Kafka message from its
# `use_ec2_database` header (core/kafka_worker_entrypoint.py). A ContextVar, not
# a plain global, for the same reason services/run_context.py's _run_config_ctx
# is one: each event runs in its own asyncio context, so a binding never leaks
# between events. Unbound (the bulk path, or any direct caller) falls back to the
# process-global settings — i.e. the exact pre-per-message behaviour.
_db_endpoints_ctx: ContextVar[Optional[DbEndpoints]] = ContextVar("mongo_db_endpoints", default=None)


def bind_db_endpoints(endpoints: Optional[DbEndpoints]) -> None:
    """Route every _get_db() call in this context at the given MongoDB. Pass None
    to fall back to the process-global settings. Pair with a clear (bind None) in
    a finally block so it never leaks onto the next message."""
    _db_endpoints_ctx.set(endpoints)

# Mongo's hard per-document limit is 16 MiB. Reject earlier with a clear error
# so callers shard (corpus_map / edge_pool_map) instead of failing mid-bulk.
_MONGO_BSON_SOFT_LIMIT_BYTES = 15 * 1024 * 1024


def _get_db():
    ep = _db_endpoints_ctx.get()
    uri = ep.mongo_uri if ep else settings.mongo_uri
    db_name = ep.mongo_db if ep else settings.mongo_db
    # Fail fast on a missing URI instead of a slow ServerSelectionTimeout later.
    if not uri:
        raise ValueError(
            "MongoDB URI is empty — set MONGO_URI (or EC2_MONGO_URI when routing to EC2) "
            "in .env or the environment before running the pipeline"
        )
    key = (uri, db_name)
    client = _clients.get(key)
    if client is None:
        client = MongoClient(uri)
        _clients[key] = client
    return client[db_name]


def _assert_bson_under_limit(key: str, data: dict) -> None:
    """Fail before write if ``{_key, data}`` would exceed Mongo's 16MB doc cap."""
    size = len(bson_encode({"_key": key, "data": data}))
    if size > _MONGO_BSON_SOFT_LIMIT_BYTES:
        raise ValueError(
            f"Checkpoint {key!r} encodes to {size} bytes "
            f"(soft limit {_MONGO_BSON_SOFT_LIMIT_BYTES}); "
            "shard/split the payload — Mongo rejects documents over 16MB"
        )


def save_checkpoint(collection: str, key: str, data: dict) -> None:
    """Save (or overwrite) a checkpoint. `collection` groups checkpoints by step
    (e.g. 'extraction', 'doc_canonical'); `key` identifies this one result within it
    (e.g. a chunk_id or doc_id)."""
    _assert_bson_under_limit(key, data)
    db = _get_db()
    db[collection].update_one({"_key": key}, {"$set": {"_key": key, "data": data}}, upsert=True)
    logger.info("Checkpoint saved: %s/%s", collection, key)


def delete_checkpoint(collection: str, key: str) -> None:
    """Remove a checkpoint document by ``_key``. No-op if the key is absent.

    Used to invalidate sharded map roots (corpus_map / edge_pool_map) before
    rewriting chunks so a crash cannot leave stale chunk counts pointing at a
    mix of old and new shard documents.
    """
    db = _get_db()
    db[collection].delete_one({"_key": key})
    logger.debug("Checkpoint deleted: %s/%s", collection, key)


def save_checkpoints_batch(
    collection: str,
    items: list[tuple[str, dict]],
    *,
    batch_size: Optional[int] = None,
) -> None:
    """Upsert many checkpoints via ``bulk_write`` instead of one ``update_one`` each.

    Step 4's edge-pool warm-start cache writes ~10^5 docs; serial ``save_checkpoint``
    calls took multi-hour wall clock on a real production-scale run. Unordered bulk chunks
    keep the same ``{_key, data}`` document shape so loaders are unchanged.
    ``items`` is a list of ``(key, data)`` pairs.
    """
    if not items:
        return
    chunk_size = batch_size if batch_size is not None else settings.mongo_checkpoint_bulk_size
    if chunk_size <= 0:
        raise ValueError("mongo checkpoint bulk size must be positive")
    db = _get_db()
    coll = db[collection]
    total = len(items)
    # Progress every chunk once the job is large enough that silence looks like a hang
    # (Step 3 entity persist / Step 4 edge-pool were multi-hour with one log at end).
    log_progress = total >= chunk_size
    for start in range(0, total, chunk_size):
        chunk = items[start:start + chunk_size]
        for key, data in chunk:
            _assert_bson_under_limit(key, data)
        ops = [
            UpdateOne({"_key": key}, {"$set": {"_key": key, "data": data}}, upsert=True)
            for key, data in chunk
        ]
        coll.bulk_write(ops, ordered=False)
        done = min(start + chunk_size, total)
        if log_progress:
            logger.info(
                "Checkpoints bulk-save progress: %s — %d/%d (%.1f%%)",
                collection, done, total, 100.0 * done / total,
            )
    logger.info("Checkpoints bulk-saved: %s — %d doc(s) in chunks of %d",
                collection, total, chunk_size)


def load_checkpoint(collection: str, key: str) -> Optional[dict]:
    """Return the saved data for `key`, or None if this step hasn't run for it yet."""
    db = _get_db()
    doc = db[collection].find_one({"_key": key})
    if doc:
        logger.debug("Checkpoint hit: %s/%s", collection, key)
        return doc["data"]
    logger.debug("Checkpoint miss: %s/%s", collection, key)
    return None


def load_checkpoints_batch(collection: str, keys: list) -> dict:
    """Load many checkpoints in one MongoDB round-trip via $in, instead of one
    find_one() per key. Returns {key: data} for every key that had a hit."""
    if not keys:
        return {}
    db = _get_db()
    docs = db[collection].find({"_key": {"$in": list(keys)}})
    result = {doc["_key"]: doc["data"] for doc in docs}
    logger.debug("Checkpoint batch load: %s — %d/%d hits", collection, len(result), len(keys))
    return result


def iter_checkpoints(collection: str, key_prefix: str):
    """Iterate over all checkpoint documents whose key starts with `key_prefix`.
    Used by Step 3 to page through doc_canonical results from MongoDB without
    loading all of them into memory at once."""
    db = _get_db()
    cursor = db[collection].find({"_key": {"$regex": f"^{key_prefix}"}}, {"_key": 1, "data": 1})
    for doc in cursor:
        yield doc["_key"], doc["data"]


def record_failure(run_id: str, item_id: str, step: str, error_class: str, message: str,
                   index: Optional[str] = None) -> None:
    """Ledger of per-item bulk-run failures, queryable and re-drivable via
    `core/entrypoint.py --retry-failed <run_id>` (unlike a log line).

    item_id is always a doc_id, even for chunk-level extraction failures —
    retries re-process at the document level. Sentinel item_id "__corpus__"
    covers whole-step failures not attributable to one document.

    Keyed by (run_id, item_id, step) so re-recording overwrites in place
    rather than accumulating duplicates."""
    db = _get_db()
    key = f"{run_id}:{item_id}:{step}"
    db["ingestion_failures"].update_one(
        {"_key": key},
        {"$set": {"_key": key, "run_id": run_id, "item_id": item_id, "step": step,
                  "error_class": error_class, "message": message, "index": index}},
        upsert=True,
    )
    logger.warning("Failure recorded: run_id=%s item_id=%s step=%s error_class=%s: %s",
                   run_id, item_id, step, error_class, message)


def list_failures(run_id: str) -> list[dict]:
    """Every currently-unresolved failure for one bulk run, across all steps —
    used by --retry-failed to know which doc_ids to re-drive."""
    db = _get_db()
    return list(db["ingestion_failures"].find({"run_id": run_id}))


def clear_failure(run_id: str, item_id: str, step: str) -> None:
    """Remove one failure record once its item succeeds on retry. Safe to call
    even if the record no longer exists (e.g. already cleared)."""
    db = _get_db()
    db["ingestion_failures"].delete_one({"_key": f"{run_id}:{item_id}:{step}"})


def record_call_failure(step: str, context: str, error_class: str, message: str,
                        attempts: Optional[int] = None) -> None:
    """Append-only audit log of individual operations (an AI call, a DB write)
    that exhausted retries and fell back to a safe default. Separate from
    ingestion_failures: that's a per-(run_id, item_id, step) ledger meant to
    be cleared and re-driven, while a single call site here (e.g. a merge
    confirmation) has no such natural key and runs far too often to track
    that way — this just keeps every exhausted call for later querying.

    step: which call site failed. context: short description of the input
    that failed. attempts: how many tries were made before giving up."""
    from datetime import datetime, timezone
    db = _get_db()
    db["call_failures"].insert_one({
        "step": step, "context": context, "error_class": error_class,
        "message": message, "attempts": attempts,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.warning("Call failure recorded: step=%s error_class=%s attempts=%s context=%s: %s",
                   step, error_class, attempts, context, message)


def increment_retry_count(key: str) -> int:
    """Atomically increment and return the delivery-attempt count for `key`
    (typically a Kafka event_id), persisted so it survives worker restarts
    and stays correct under concurrent replicas."""
    db = _get_db()
    doc = db["kafka_retry_counts"].find_one_and_update(
        {"_key": key}, {"$inc": {"count": 1}}, upsert=True, return_document=ReturnDocument.AFTER,
    )
    return doc["count"]


def increment_rolling_window_count(collection: str, key: str, window_seconds: float, now: float) -> int:
    """Atomically increment a per-key rolling-window counter, resetting it to 1
    if more than window_seconds has elapsed since the window last started.
    Returns the count after this call. Uses a single aggregation-pipeline
    update so the reset-or-increment decision and the write stay atomic —
    a read-then-write in Python would let concurrent callers undercount."""
    db = _get_db()
    expired = {"$or": [
        {"$eq": [{"$ifNull": ["$window_started_at", None]}, None]},
        {"$gt": [{"$subtract": [now, "$window_started_at"]}, window_seconds]},
    ]}
    doc = db[collection].find_one_and_update(
        {"_key": key},
        [{"$set": {
            "_key": key,
            "window_started_at": {"$cond": [expired, now, "$window_started_at"]},
            "count": {"$cond": [expired, 1, {"$add": [{"$ifNull": ["$count", 0]}, 1]}]},
        }}],
        upsert=True, return_document=ReturnDocument.AFTER,
    )
    return doc["count"]


def clear_retry_count(key: str) -> None:
    """Remove a retry-count record once its message succeeds or is DLQ'd, so a
    reused key starts fresh instead of inheriting a stale count."""
    db = _get_db()
    db["kafka_retry_counts"].delete_one({"_key": key})
