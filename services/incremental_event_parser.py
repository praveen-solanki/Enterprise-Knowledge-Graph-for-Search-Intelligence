"""
Parses the GraphExtractRequest Kafka event and resolves its payload. See
GRAPH_INCREMENTAL_SYNC_ORCHESTRATOR_DESIGN.md §4 for the exact schema this
follows.

This module reports whatever `operation` the event carries ("upsert" or
"delete") and leaves the decision of what to do with it to the caller (see
core/kafka_worker_entrypoint.py's handle_event).

Adapted for the new producer payload shape in search-utility-agents PR #679
(feat(AIS-3687)): tenant_id/connector_type/ingestion_type moved under a
nested `response_metadata` object; `index_name` renamed to `index` and moved
under `data`; `operation` moved under `data`; one Kafka message is now one
document, not a `{"docs": [...]}` batch; an upsert's heavy content (chunks +
embeddings) is fetched from S3 separately from the Kafka body's own metadata
+ ACL and needs merging. All of that is normalized away in load_payload()
below, at this one boundary, so nothing downstream (core/kafka_worker_
entrypoint.py, services/incremental_pipeline.py) needed to change.

STILL OPEN, not guessed at (see the message sent to the producer team):
completed_at (an int, used by services/incremental_pipeline.py's _is_stale
as a strict per-document ordering watermark -- an older update arriving late
must never overwrite a newer one already applied) doesn't exist in the new
payload at all. It's parsed here from the new `event_timestamp` field
(ISO-8601, set at publish time) so the existing watermark logic keeps
running unchanged -- but whether event_timestamp actually preserves the same
strict per-document ordering guarantee completed_at did, even across
Kafka retries/redelivery, has NOT been confirmed by the producer team as of
this writing. Do not remove this comment (or the one on
_event_timestamp_to_watermark) until that's answered.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from services.s3_client import download_json


@dataclass
class GraphExtractEvent:
    event_id: str
    tenant_id: str
    connector_type: str
    index_name: str
    ingestion_type: str
    operation: str
    completed_at: int


def _event_timestamp_to_watermark(event: dict) -> int:
    """PROVISIONAL -- see this module's docstring. Converts the new payload's
    `event_timestamp` (ISO-8601 string) into the epoch-MILLISECONDS int
    _is_stale/_advance_watermark already compare on, so that logic didn't
    need to change shape. Returns 0 (never stale) if the field is missing
    entirely, matching the old code's behavior when a watermark comparison
    has nothing to compare against.

    UNIT: milliseconds, matching the existing completed_at convention exactly
    (real values look like 1784192331000 -- 13 digits, epoch ms). Getting
    this wrong would compare a fresh value against a previously-saved
    millisecond watermark and ALWAYS look older regardless of the real
    dates, silently dropping every future update for an already-touched
    document. datetime.timestamp() returns SECONDS -- must multiply by 1000."""
    raw_ts = event.get("event_timestamp")
    if not raw_ts:
        return 0
    return int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp() * 1000)


def parse_event(event: dict) -> GraphExtractEvent:
    response_metadata = event.get("response_metadata") or {}
    data = event.get("data") or {}
    return GraphExtractEvent(
        event_id=event.get("event_id", ""),
        # Required, same fail-loud contract as before (raises if truly absent from
        # both the new nested location and the old top-level one) -- a missing
        # tenant_id must never be silently defaulted, per repository/mongo_
        # repository.py's tenant-isolation rationale elsewhere in this codebase.
        tenant_id=response_metadata.get("tenant_id") or event["tenant_id"],
        connector_type=response_metadata.get("connector_type", ""),
        index_name=data.get("index", "") or event.get("index_name", ""),
        ingestion_type=response_metadata.get("ingestion_type", ""),
        operation=data.get("operation") or event.get("operation", "upsert"),
        completed_at=event.get("completed_at", _event_timestamp_to_watermark(event)),
    )


def load_payload(event: dict) -> dict[str, Any]:
    """Return the {"docs": [...]} payload — the SAME shape every existing
    downstream caller expects, even though the new producer payload is one
    document per event, not a batch. Wrapping it in a single-item list HERE
    means core/kafka_worker_entrypoint.py's handle_event (delete/acl_update/
    upsert branches, all written as `for doc in docs`) needed no changes.

    The new payload splits one document's fields across two places: metadata
    + ACL always live in the Kafka body (`data`); for an upsert, the heavy
    content (chunks + embeddings) is fetched separately from S3 and merged
    in here. Chunks are also reshaped from the new flat `chunks` key into
    the `_chunks: {"v1": [...]}` shape dedup_chunks/_extract_chunks already
    expect, so nothing downstream of this function needed to change for
    that either.

    KNOWN GAP, not silently patched over: the new payload carries no document
    title field anywhere (confirmed by reading the producer's diff — neither
    `data` nor the S3 object has one). Every incremental document's title
    will be blank (_to_source_document's title fallback chain has nothing
    to read) until the producer adds one back or points us at where else to
    find it.
    """
    location = event.get("event_data_location")
    data = event.get("data") or {}

    if location == "EMBEDDED":
        # Deletes and acl_updates carry no S3 ref at all — data alone is the
        # whole document (matches the old EMBEDDED case exactly, just without
        # the old {"docs": [...]} batch wrapper the producer used to send).
        if "docs" in data:  # backward compat: an old-shape EMBEDDED payload
            return data
        if not data:
            return {"docs": []}
        return {"docs": [dict(data)]}

    if location == "S3":
        bucket, key = data.get("s3_bucket"), data.get("s3_key")
        if not bucket or not key:
            raise ValueError(f"event_data_location=S3 but data is missing s3_bucket/s3_key: {data!r}")
        s3_payload = download_json(bucket, key)
        if "docs" in s3_payload:  # backward compat: an old-shape S3 payload was already {"docs": [...]}
            return s3_payload
        chunks = s3_payload.pop("chunks", None)
        doc: dict[str, Any] = dict(data)
        doc.update(s3_payload)
        if chunks is not None:
            doc["_chunks"] = {"v1": chunks}
        return {"docs": [doc]}

    raise ValueError(f"Unknown event_data_location: {location!r}")
