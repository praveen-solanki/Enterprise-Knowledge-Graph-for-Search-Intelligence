"""Unit tests for services/incremental_event_parser.py (AIS-3634)."""
from unittest.mock import patch

import pytest

from services.incremental_event_parser import load_payload, parse_event


def _event(**overrides) -> dict:
    event = {
        "event_name": "GraphExtractRequest",
        "event_source": "enterprise-ingestion-daemon",
        "event_version": "v1",
        "event_id": "evt-1",
        "tenant_id": "tenant-1",
        "connector_type": "confluence",
        "index_name": "gs-prod-tenant-1-v9",
        "ingestion_type": "INCREMENTAL",
        "operation": "upsert",
        "completed_at": 1784192331000,
        "event_data_location": "EMBEDDED",
        "data": {"docs": [{"doc_id": "d1"}]},
    }
    event.update(overrides)
    return event


def test_parse_event_extracts_envelope_fields():
    parsed = parse_event(_event())
    assert parsed.event_id == "evt-1"
    assert parsed.tenant_id == "tenant-1"
    assert parsed.operation == "upsert"
    assert parsed.completed_at == 1784192331000


def test_parse_event_defaults_operation_to_upsert_when_missing():
    event = _event()
    del event["operation"]
    assert parse_event(event).operation == "upsert"


def test_load_payload_embedded_returns_data_directly():
    payload = load_payload(_event())
    assert payload == {"docs": [{"doc_id": "d1"}]}


def test_load_payload_s3_downloads_pointer():
    event = _event(event_data_location="S3", data={"s3_bucket": "b", "s3_key": "k.json"})
    with patch("services.incremental_event_parser.download_json", return_value={"docs": []}) as dl:
        payload = load_payload(event)
    dl.assert_called_once_with("b", "k.json")
    assert payload == {"docs": []}


def test_load_payload_s3_missing_pointer_raises():
    event = _event(event_data_location="S3", data={})
    with pytest.raises(ValueError):
        load_payload(event)


def test_load_payload_unknown_location_raises():
    event = _event(event_data_location="WEIRD")
    with pytest.raises(ValueError):
        load_payload(event)


# --- New payload shape, search-utility-agents PR #679 (feat(AIS-3687)) ---
# tenant_id/connector_type/ingestion_type under response_metadata; index_name
# renamed "index" and operation moved under data; one event = one document,
# not a {"docs": [...]} batch; upsert content fetched from S3 separately from
# the Kafka body's own metadata + ACL.

def _new_shape_event(**overrides) -> dict:
    event = {
        "event_id": "evt-2",
        "request_id": "req-1",
        "event_name": "GraphExtractRequest",
        "event_source": "enterprise-ingestion-daemon",
        "event_version": "v2",
        "event_timestamp": "2026-07-20T11:00:05+00:00",
        "response_metadata": {
            "tenant_id": "tenant-1",
            "connector_type": "confluence",
            "ingestion_type": "INCREMENTAL",
        },
        "event_data_location": "S3",
        "data": {
            "tenant_id": "tenant-1",
            "doc_id": "d1",
            "index": "gs-prod-tenant-1-v9",
            "source": "confluence",
            "operation": "upsert",
            "object_type": "content",
            "allow_access_control": ["site_type:PUBLIC"],
            "s3_bucket": "b",
            "s3_key": "k.json",
        },
    }
    event.update(overrides)
    return event


def test_parse_event_reads_new_nested_shape():
    parsed = parse_event(_new_shape_event())
    assert parsed.tenant_id == "tenant-1"
    assert parsed.connector_type == "confluence"
    assert parsed.ingestion_type == "INCREMENTAL"
    assert parsed.index_name == "gs-prod-tenant-1-v9"
    assert parsed.operation == "upsert"


def test_parse_event_derives_completed_at_from_event_timestamp_when_absent():
    parsed = parse_event(_new_shape_event())
    # 2026-07-20T11:00:05+00:00 -- exact epoch-MILLISECONDS value (matching
    # the real completed_at convention -- 1784192331000-style, 13 digits),
    # not just "non-zero", so a future change to the parsing logic can't
    # silently drift into the wrong unit.
    assert parsed.completed_at == 1784545205000


def test_parse_event_prefers_top_level_completed_at_when_both_present():
    # Backward/forward compat overlap: if the producer ever sends BOTH (e.g.
    # during a transition), the real watermark field wins over the derived one.
    event = _new_shape_event(completed_at=999)
    assert parse_event(event).completed_at == 999


def test_load_payload_new_shape_merges_s3_chunks_and_reshapes_them():
    s3_object = {
        "object_type": "content", "doc_id": "d1", "index": "gs-prod-tenant-1-v9",
        "tenant_id": "tenant-1", "parent_embeddings": {"title": [0.1]},
        "chunks": [{"chunk_id": "c1", "text": "hello"}],
    }
    with patch("services.incremental_event_parser.download_json", return_value=s3_object) as dl:
        payload = load_payload(_new_shape_event())
    dl.assert_called_once_with("b", "k.json")
    docs = payload["docs"]
    assert len(docs) == 1
    doc = docs[0]
    # ACL from the Kafka body is preserved onto the merged doc.
    assert doc["allow_access_control"] == ["site_type:PUBLIC"]
    # Chunks reshaped into the _chunks.v1 shape dedup_chunks/_extract_chunks expect.
    assert doc["_chunks"] == {"v1": [{"chunk_id": "c1", "text": "hello"}]}
    assert "chunks" not in doc  # the flat key is consumed, not left lying around too
    assert doc["parent_embeddings"] == {"title": [0.1]}


def test_load_payload_new_shape_delete_has_no_s3_ref():
    event = _new_shape_event(event_data_location="EMBEDDED",
                             data={"tenant_id": "tenant-1", "doc_id": "d1",
                                   "index": "gs-prod-tenant-1-v9", "operation": "delete"})
    payload = load_payload(event)
    assert payload == {"docs": [{"tenant_id": "tenant-1", "doc_id": "d1",
                                 "index": "gs-prod-tenant-1-v9", "operation": "delete"}]}
