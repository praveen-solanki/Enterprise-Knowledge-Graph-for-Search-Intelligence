"""Unit tests for core/kafka_worker_entrypoint.py's message dispatch (AIS-3634).

The consumer loop itself (run()) needs a real Kafka broker and isn't unit
tested here — only handle_event's dispatch logic (operation gating, payload
loading, looping over docs) is covered.
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from botocore.exceptions import ClientError

import core.kafka_worker_entrypoint as kafka_worker_entrypoint
from config.settings import settings
from core.kafka_worker_entrypoint import (
    _classify_error,
    _fetch_acl_source_docs,
    _handle_failure,
    _handle_message,
    handle_event,
    _request_shutdown,
)
from services.run_context import RunConfig

_RUN_CFG = RunConfig(
    embedding_backend="azure", embedding_model="text-embedding-3-large",
    embedding_dimension=1536, consume_upstream_vector=True,
    es_chunk_vector_field="embeddings_openai_large_1536",
    llm_backend="ministral", llm_model="ministral-3b",
    build_authorship_edges=True, contributor_entity_types=("PROJECT",), people_org_graph_enabled=True,
)


def _event(**overrides) -> dict:
    event = {
        "event_id": "evt-1", "tenant_id": "tenant-1", "connector_type": "confluence",
        "index_name": "gs-prod-tenant-1-v9", "ingestion_type": "INCREMENTAL",
        "operation": "upsert", "completed_at": 1000,
        "event_data_location": "EMBEDDED", "data": {"docs": [{"doc_id": "d1"}, {"doc_id": "d2"}]},
    }
    event.update(overrides)
    return event


def _spec():
    from connectors.registry import get_connector
    return get_connector("wikipedia")


def _patched_fetch(docs_by_id: dict):
    """Patches fetch_documents so _fetch_acl_source_docs returns a real-looking
    ES doc for each doc_id in docs_by_id, and nothing for any other doc_id."""
    def _fake_fetch_documents(es, index, content_ids=None, query=None, normalize=None):
        return [docs_by_id[cid] for cid in (content_ids or []) if cid in docs_by_id]
    return patch("core.kafka_worker_entrypoint.fetch_documents", side_effect=_fake_fetch_documents)


def test_handle_event_processes_every_doc_in_payload():
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         _patched_fetch({"d1": {"id": "d1"}, "d2": {"id": "d2"}}):
        asyncio.run(handle_event(_event(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    assert process_doc.await_count == 2


# --- log context ---

def test_handle_event_binds_and_clears_log_context():
    """Every log line during this event must carry its tenant_id/connector_type
    (utils/log_context.py) -- and it must be cleared afterward so it doesn't
    leak onto the next message's log lines in this long-running worker."""
    from utils.log_context import RunContextLogFilter

    seen = {}

    async def _fake_process_document(*args, **kwargs):
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "m", (), None)
        RunContextLogFilter().filter(record)
        seen["during"] = (record.tenant_id, record.connector_type)

    with patch("core.kafka_worker_entrypoint.process_document", side_effect=_fake_process_document), \
         _patched_fetch({"d1": {"id": "d1"}, "d2": {"id": "d2"}}):
        asyncio.run(handle_event(
            _event(tenant_id="tenant-1", response_metadata={"connector_type": "confluence"}),
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))

    assert seen["during"] == ("tenant-1", "confluence")
    after = logging.LogRecord("t", logging.INFO, __file__, 1, "m", (), None)
    RunContextLogFilter().filter(after)
    assert (after.tenant_id, after.connector_type) == ("-", "-")


def test_handle_event_passes_the_matching_es_doc_to_process_document():
    """Code review must-fix #1: ACL must be derived from a real ES lookup, not
    the payload's own (never-populated-by-the-producer) ACL fields."""
    es_doc_1, es_doc_2 = {"id": "d1", "content_status": "published"}, {"id": "d2", "content_status": "draft"}
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         _patched_fetch({"d1": es_doc_1, "d2": es_doc_2}):
        asyncio.run(handle_event(_event(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    calls = {c.args[1]["doc_id"]: c.args[6] for c in process_doc.await_args_list}
    assert calls["d1"] == es_doc_1
    assert calls["d2"] == es_doc_2


def test_handle_event_passes_none_when_es_lookup_has_no_match():
    """A doc_id with no matching ES document must reach process_document as
    es_doc=None so it drops the document (fail-closed), never as public."""
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         _patched_fetch({"d1": {"id": "d1"}}):  # d2 has no match
        asyncio.run(handle_event(_event(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    calls = {c.args[1]["doc_id"]: c.args[6] for c in process_doc.await_args_list}
    assert calls["d1"] == {"id": "d1"}
    assert calls["d2"] is None


def test_fetch_acl_source_docs_fails_closed_when_es_lookup_raises():
    """If the ES lookup itself errors (network, index missing, etc.), every
    doc_id in this event is treated as unresolved — never falls back to public."""
    with patch("core.kafka_worker_entrypoint.fetch_documents", side_effect=RuntimeError("boom")):
        es_doc_by_id, site_audiences_by_id = _fetch_acl_source_docs(
            MagicMock(), MagicMock(event_id="e1", index_name="idx"),
            [{"doc_id": "d1"}, {"doc_id": "d2"}], _spec())
    assert es_doc_by_id == {}
    assert site_audiences_by_id == {}


def test_fetch_acl_source_docs_drops_fail_closed_when_event_has_no_index():
    """PR-41 review point 3: there is no default index anymore. An event without
    index_name is dropped fail-closed rather than looked up against a fabricated
    single-tenant default that would derive ACL from the wrong tenant's data."""
    with patch("core.kafka_worker_entrypoint.fetch_documents") as fetch_docs:
        result = _fetch_acl_source_docs(
            MagicMock(), MagicMock(event_id="e1", index_name=""), [{"doc_id": "d1"}], _spec())

    assert result == ({}, {})
    fetch_docs.assert_not_called()


def test_fetch_acl_source_docs_skips_es_for_a_doc_with_embedded_site_audiences():
    """A doc carrying its own site_audiences_by_id (the fully-embedded payload
    shape) is trusted as-is -- no ES lookup for it, and its own map is returned."""
    with patch("core.kafka_worker_entrypoint.fetch_documents") as fetch_docs:
        es_doc_by_id, site_audiences_by_id = _fetch_acl_source_docs(
            MagicMock(), MagicMock(event_id="e1", index_name="idx"),
            [{"doc_id": "d1", "site_audiences_by_id": {"site-1": ["aud-1"]}}], _spec())
    fetch_docs.assert_not_called()
    assert es_doc_by_id == {}
    assert site_audiences_by_id == {"site-1": ["aud-1"]}


def test_fetch_acl_source_docs_mixed_batch_only_looks_up_the_non_embedded_doc():
    """A batch with one fully-embedded doc and one legacy (no site_audiences_by_id)
    doc must only send the legacy one through the live ES lookup."""
    with _patched_fetch({"d2": {"id": "d2", "content_site_id": "site-2"}}):
        es_doc_by_id, site_audiences_by_id = _fetch_acl_source_docs(
            MagicMock(), MagicMock(event_id="e1", index_name="idx"),
            [{"doc_id": "d1", "site_audiences_by_id": {"site-1": ["aud-1"]}},
             {"doc_id": "d2"}], _spec())
    assert "d1" not in es_doc_by_id
    assert es_doc_by_id["d2"] == {"id": "d2", "content_site_id": "site-2"}
    assert site_audiences_by_id == {"site-1": ["aud-1"]}


def test_handle_event_routes_delete_to_delete_document():
    repo, vector_store = MagicMock(), MagicMock()
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         patch("core.kafka_worker_entrypoint.delete_document") as delete_doc, \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=False), \
         patch("core.kafka_worker_entrypoint._advance_watermark"):
        asyncio.run(handle_event(_event(operation="delete"), repo, MagicMock(), MagicMock(), MagicMock(), vector_store))
    process_doc.assert_not_called()
    assert delete_doc.call_count == 2
    delete_doc.assert_any_call("d1", repo, "tenant-1", vector_store, None)


def test_handle_event_delete_skips_stale_event_and_never_retracts():
    """Code review must-fix #7: a redelivered/out-of-order delete older than
    the watermark must be skipped, just like the upsert path already does —
    otherwise it could retract a document legitimately re-added afterward."""
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.delete_document") as delete_doc, \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=True) as is_stale, \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="delete"), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    delete_doc.assert_not_called()
    advance_watermark.assert_not_called()
    assert is_stale.call_count == 2


def test_handle_event_delete_advances_watermark_after_successful_retraction():
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.delete_document") as delete_doc, \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=False), \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="delete"), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    assert delete_doc.call_count == 2
    advance_watermark.assert_any_call("tenant-1", "d1", 1000)
    advance_watermark.assert_any_call("tenant-1", "d2", 1000)


def test_handle_event_routes_acl_update_to_update_acl_only():
    """Prove handle_event calls resolve_incremental_acl and passes its result to update_acl_only."""
    docs = [{"doc_id": "d1", "allow_access_control": ["raw-token-1"]},
            {"doc_id": "d2", "allow_access_control": ["raw-token-2"]}]
    repo, vector_store = MagicMock(), MagicMock()
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         patch("core.kafka_worker_entrypoint.update_acl_only") as update_acl, \
         patch("core.kafka_worker_entrypoint.resolve_incremental_acl",
               side_effect=lambda doc, es_doc, sites, connector_type="zeus", tenant_id=None: doc["allow_access_control"]) as resolve, \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=False), \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="acl_update", data={"docs": docs}),
                                 repo, MagicMock(), MagicMock(), MagicMock(), vector_store))
    process_doc.assert_not_called()
    assert resolve.call_count == 2
    assert update_acl.call_count == 2
    update_acl.assert_any_call("d1", repo, "tenant-1", ["raw-token-1"], vector_store, None)
    advance_watermark.assert_any_call("tenant-1", "d1", 1000)
    advance_watermark.assert_any_call("tenant-1", "d2", 1000)


def test_handle_event_acl_update_skips_stale_event():
    docs = [{"doc_id": "d1", "allow_access_control": ["raw-token-1"]}]
    with patch("core.kafka_worker_entrypoint.update_acl_only") as update_acl, \
         patch("core.kafka_worker_entrypoint.resolve_incremental_acl") as resolve, \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=True), \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="acl_update", data={"docs": docs}),
                                 MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    resolve.assert_not_called()
    update_acl.assert_not_called()
    advance_watermark.assert_not_called()


def test_handle_event_acl_update_uses_es_doc_lastmodifieddate_as_the_watermark():
    """resolve_watermark (services/incremental_pipeline.py) prefers the live
    ES document's own lastmodifieddate over event.completed_at whenever an
    es_doc is available -- acl_update always has one (fetched for ACL
    resolution), unlike delete. Confirms handle_event actually passes it
    through, not just that resolve_watermark itself works in isolation."""
    docs = [{"doc_id": "d1", "allow_access_control": ["raw-token-1"]}]
    es_doc = {"id": "d1", "lastmodifieddate": "2025-12-18T02:02:49.872Z"}  # -> 1766023369872 (ms)
    repo, vector_store = MagicMock(), MagicMock()
    with patch("core.kafka_worker_entrypoint.update_acl_only") as update_acl, \
         patch("core.kafka_worker_entrypoint.resolve_incremental_acl",
              side_effect=lambda doc, es_doc, sites, connector_type="zeus", tenant_id=None: doc["allow_access_control"]), \
         patch("core.kafka_worker_entrypoint._fetch_acl_source_docs",
              return_value=({"d1": es_doc}, {})), \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=False) as is_stale, \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="acl_update", data={"docs": docs}),
                                 repo, MagicMock(), MagicMock(), MagicMock(), vector_store))
    is_stale.assert_called_once_with("tenant-1", "d1", 1766023369872)
    update_acl.assert_called_once()
    advance_watermark.assert_called_once_with("tenant-1", "d1", 1766023369872)


def test_handle_event_acl_update_retracts_when_acl_resolves_to_none():
    """resolve_incremental_acl returning None must retract, not leave a stale ACL."""
    docs = [{"doc_id": "d1", "allow_access_control": []}]
    repo, vector_store = MagicMock(), MagicMock()
    with patch("core.kafka_worker_entrypoint.update_acl_only") as update_acl, \
         patch("core.kafka_worker_entrypoint.delete_document") as delete_doc, \
         patch("core.kafka_worker_entrypoint.resolve_incremental_acl", return_value=None), \
         patch("core.kafka_worker_entrypoint._is_stale", return_value=False), \
         patch("core.kafka_worker_entrypoint._advance_watermark") as advance_watermark:
        asyncio.run(handle_event(_event(operation="acl_update", data={"docs": docs}),
                                 repo, MagicMock(), MagicMock(), MagicMock(), vector_store))
    update_acl.assert_not_called()
    delete_doc.assert_called_once_with("d1", repo, "tenant-1", vector_store, None)
    advance_watermark.assert_called_once_with("tenant-1", "d1", 1000)
    delete_doc.assert_called_once_with("d1", repo, "tenant-1", vector_store, None)
    advance_watermark.assert_called_once_with("tenant-1", "d1", 1000)


def test_handle_event_skips_unhandled_operations():
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         patch("core.kafka_worker_entrypoint.delete_document") as delete_doc:
        asyncio.run(handle_event(_event(operation="rename"), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    process_doc.assert_not_called()
    delete_doc.assert_not_called()


def test_handle_event_processes_docs_concurrently_but_bounded():
    """Code review must-fix #6 (B1): docs in one event must overlap (not one
    strict for/await loop), but never exceed settings.incremental_doc_concurrency
    in flight."""
    doc_ids = [f"d{i}" for i in range(settings.incremental_doc_concurrency + 5)]
    event = _event(data={"docs": [{"doc_id": d} for d in doc_ids]})
    in_flight = 0
    max_in_flight = 0

    async def _fake_process_document(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    with patch("core.kafka_worker_entrypoint.process_document", side_effect=_fake_process_document), \
         _patched_fetch({d: {"id": d} for d in doc_ids}):
        asyncio.run(handle_event(event, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))

    assert max_in_flight > 1, "docs should process concurrently, not strictly one at a time"
    assert max_in_flight <= settings.incremental_doc_concurrency


def test_handle_event_downloads_s3_payload_when_pointed():
    event = _event(event_data_location="S3", data={"s3_bucket": "b", "s3_key": "k.json"})
    with patch("core.kafka_worker_entrypoint.process_document", new=AsyncMock()) as process_doc, \
         patch("services.incremental_event_parser.download_json", return_value={"docs": [{"doc_id": "d3"}]}), \
         _patched_fetch({"d3": {"id": "d3"}}):
        asyncio.run(handle_event(event, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()))
    assert process_doc.await_count == 1


# --- DLQ: error classification (code review must-fix #5) ---

def test_classify_error_json_decode_error_is_permanent():
    import json
    try:
        json.loads("not json")
    except json.JSONDecodeError as exc:
        assert _classify_error(exc) == "permanent"


def test_classify_error_key_error_is_permanent():
    assert _classify_error(KeyError("tenant_id")) == "permanent"


def test_classify_error_generic_value_error_is_transient():
    assert _classify_error(ValueError("Unknown event_data_location: None")) == "transient"


def test_classify_error_s3_404_is_permanent():
    exc = ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")
    assert _classify_error(exc) == "permanent"


def test_classify_error_s3_403_is_permanent():
    exc = ClientError({"Error": {"Code": "AccessDenied"},
                       "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")
    assert _classify_error(exc) == "permanent"


def test_classify_error_s3_500_is_transient():
    exc = ClientError({"Error": {"Code": "InternalError"},
                       "ResponseMetadata": {"HTTPStatusCode": 500}}, "GetObject")
    assert _classify_error(exc) == "transient"


def test_classify_error_generic_exception_is_transient():
    assert _classify_error(RuntimeError("connection reset")) == "transient"


def test_classify_error_timeout_is_transient():
    assert _classify_error(TimeoutError("timed out")) == "transient"


# --- DLQ: _handle_failure / _handle_message (code review must-fix #5) ---

def _msg(offset=42, topic="t", partition=0, value=b'{"event_id": "evt-1"}',
         headers=None, key=None) -> MagicMock:
    msg = MagicMock()
    msg.offset.return_value = offset
    msg.topic.return_value = topic
    msg.partition.return_value = partition
    msg.value.return_value = value
    # confluent-kafka returns a list of (str, bytes) tuples or None — default to a
    # header-less record so tests exercise the backward-compatible (no-header) path
    # unless they explicitly pass routing headers.
    msg.headers.return_value = headers
    msg.key.return_value = key
    return msg


def _dlq_producer(delivered: bool = True) -> MagicMock:
    """A mock Producer that simulates confluent_kafka's real async delivery
    contract: produce() takes an on_delivery callback (invoked here
    synchronously, standing in for the broker's ack), and flush() reports
    how many messages are still undelivered (0 = all confirmed)."""
    producer = MagicMock()

    def _fake_produce(*args, **kwargs):
        on_delivery = kwargs.get("on_delivery")
        if on_delivery:
            on_delivery(None if delivered else Exception("delivery failed"), MagicMock())

    producer.produce.side_effect = _fake_produce
    producer.flush.return_value = 0 if delivered else 1
    return producer


def test_handle_failure_sends_permanent_errors_straight_to_dlq_and_commits():
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=True), _msg()
    with patch("core.kafka_worker_entrypoint.record_failure") as record_call:
        _handle_failure(msg, consumer, dlq_producer, KeyError("tenant_id"), {"event_id": "evt-1"})
    dlq_producer.produce.assert_called_once()
    dlq_producer.flush.assert_called_once()
    consumer.commit.assert_called_once_with(message=msg)
    record_call.assert_called_once()


def test_handle_failure_does_not_commit_when_dlq_send_is_not_confirmed_delivered():
    """Regression test (Round-2 review New-2): produce() is fire-and-forget and
    flush() can time out with the message still queued — treating either as
    success would commit the source offset for a message that was never
    actually written to the DLQ, losing it for good instead of retrying."""
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=False), _msg()
    with patch("core.kafka_worker_entrypoint.time.sleep"), \
         patch("core.kafka_worker_entrypoint.record_failure"), \
         patch("core.kafka_worker_entrypoint._dlq_topic_exists", return_value=True):
        _handle_failure(msg, consumer, dlq_producer, KeyError("tenant_id"), {"event_id": "evt-1"})
    consumer.commit.assert_not_called()
    consumer.seek.assert_called_once()


def test_handle_failure_commits_when_dlq_topic_is_missing():
    """A missing DLQ topic must not wedge the partition forever: Mongo already
    has the durable failure record, so commit and move on to later messages."""
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=False), _msg()
    with patch("core.kafka_worker_entrypoint.record_failure"), \
         patch("core.kafka_worker_entrypoint._dlq_topic_exists", return_value=False), \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings:
        mock_settings.kafka_topic_graph_extract_dlq = "AI_Graph_Extract_Req_DLQ_DEV"
        _handle_failure(msg, consumer, dlq_producer, KeyError("tenant_id"), {"event_id": "evt-1"})
    consumer.commit.assert_called_once_with(message=msg)
    consumer.seek.assert_not_called()


def test_handle_failure_does_not_commit_when_dlq_flush_times_out_with_pending_messages():
    consumer, dlq_producer, msg = MagicMock(), MagicMock(), _msg()
    dlq_producer.flush.return_value = 1  # one message still undelivered after the timeout
    with patch("core.kafka_worker_entrypoint.time.sleep"), \
         patch("core.kafka_worker_entrypoint.record_failure"), \
         patch("core.kafka_worker_entrypoint._dlq_topic_exists", return_value=True):
        _handle_failure(msg, consumer, dlq_producer, KeyError("tenant_id"), {"event_id": "evt-1"})
    consumer.commit.assert_not_called()


def test_handle_failure_crashes_on_fatal_error_instead_of_dlqing():
    """Regression test (Round-2 review New-4): a global/config problem (bad
    Neo4j credentials) would fail identically for every future message too —
    draining the whole topic into the DLQ one message at a time would turn a
    one-line config fix into a mass data-loss event, so this must crash
    loudly instead."""
    from requests.exceptions import HTTPError
    consumer, dlq_producer, msg = MagicMock(), MagicMock(), _msg()
    exc = HTTPError("401 Unauthorized")
    exc.response = type("Resp", (), {"status_code": 401})()
    try:
        _handle_failure(msg, consumer, dlq_producer, exc, {"event_id": "evt-1"})
        assert False, "expected the fatal error to propagate"
    except HTTPError:
        pass
    dlq_producer.produce.assert_not_called()
    consumer.commit.assert_not_called()


def test_handle_failure_leaves_transient_errors_uncommitted_below_the_limit():
    consumer, dlq_producer, msg = MagicMock(), MagicMock(), _msg()
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=1), \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings, \
         patch("core.kafka_worker_entrypoint.time.sleep") as sleep:
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_topic_graph_extract_dlq = "dlq-topic"
        mock_settings.kafka_redelivery_backoff_cap_seconds = 10.0
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), {"event_id": "evt-1"})
    dlq_producer.produce.assert_not_called()
    consumer.commit.assert_not_called()
    sleep.assert_called_once_with(1)


def test_handle_failure_seeks_back_to_the_failed_offset_on_transient_retry():
    """Regression test for a real gap found on re-review (code review
    must-fix #5 follow-up): leaving a message merely "uncommitted" does
    nothing by itself — poll() has already moved past it internally, so
    without an explicit seek(), a LATER message on the same partition
    succeeding and committing would silently advance the partition's one
    committed offset past this failed one, losing it for good rather than
    redelivering it. The consumer must be told to rewind to this exact
    offset so the very next poll() hands it back."""
    consumer, dlq_producer, msg = MagicMock(), MagicMock(), _msg(offset=99, topic="t1", partition=3)
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=2), \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings, \
         patch("core.kafka_worker_entrypoint.time.sleep") as sleep:
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_redelivery_backoff_cap_seconds = 10.0
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), {"event_id": "evt-1"})
    consumer.seek.assert_called_once()
    tp = consumer.seek.call_args[0][0]
    assert (tp.topic, tp.partition, tp.offset) == ("t1", 3, 99)
    consumer.commit.assert_not_called()
    sleep.assert_called_once_with(2)


def test_handle_failure_does_not_seek_once_attempts_are_exhausted_and_dlqd():
    """Once a transient error is given up on and sent to the DLQ, the offset
    is committed and moved past deliberately — seeking back here would
    reintroduce the exact infinite-retry loop the DLQ exists to end."""
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=True), _msg()
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=5), \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.record_failure"), \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings:
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_topic_graph_extract_dlq = "dlq-topic"
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), {"event_id": "evt-1"})
    consumer.seek.assert_not_called()


def test_handle_failure_sends_transient_errors_to_dlq_once_attempts_exhausted():
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=True), _msg()
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=5), \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.record_failure") as record_call, \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings:
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_topic_graph_extract_dlq = "dlq-topic"
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), {"event_id": "evt-1"})
    dlq_producer.produce.assert_called_once()
    consumer.commit.assert_called_once_with(message=msg)
    record_call.assert_called_once()


def test_handle_failure_clears_retry_count_once_exhausted_and_dlqd():
    """Regression test: without clearing, kafka_retry_counts grows unbounded
    and a reused key would inherit a stale count instead of starting fresh."""
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=True), _msg()
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=5), \
         patch("core.kafka_worker_entrypoint.clear_retry_count") as clear, \
         patch("core.kafka_worker_entrypoint.record_failure"), \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings:
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_topic_graph_extract_dlq = "dlq-topic"
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), {"event_id": "evt-1"})
    clear.assert_called_once_with("evt-1")


def test_handle_failure_retry_key_falls_back_to_topic_partition_offset_without_event_id():
    consumer, dlq_producer, msg = MagicMock(), MagicMock(), _msg(offset=7, topic="t1", partition=2)
    with patch("core.kafka_worker_entrypoint.increment_retry_count", return_value=1) as inc, \
         patch("core.kafka_worker_entrypoint.settings") as mock_settings, \
         patch("core.kafka_worker_entrypoint.time.sleep"):
        mock_settings.kafka_max_delivery_attempts = 5
        mock_settings.kafka_redelivery_backoff_cap_seconds = 10.0
        _handle_failure(msg, consumer, dlq_producer, RuntimeError("boom"), None)
    inc.assert_called_once_with("t1:2:7")


def _fake_run_ctx() -> MagicMock:
    """A stand-in RunContext (services/run_context.py) -- the store-name fields
    _handle_message reads plus the run_config it binds for the event."""
    run_ctx = MagicMock()
    run_ctx.neo4j_database = "g-tenant-1-confluence-ab12"
    run_ctx.qdrant_entities_collection = "g-tenant-1-confluence-entities-ab12"
    run_ctx.qdrant_chunks_collection = "g-tenant-1-confluence-chunks-ab12"
    run_ctx.run_config = _RUN_CFG
    return run_ctx


def test_handle_message_binds_the_events_run_config_before_processing():
    """Every event must be pinned to its bulk run's recorded config, so its
    embedding/LLM/flag decisions match the run it updates regardless of this
    worker's own settings/probe. Captured at bind time because the config is
    read (via effective_run_config) deep inside handle_event."""
    from services.run_context import active_run_config

    bound = {}
    consumer, dlq_producer = MagicMock(), MagicMock()
    msg = _msg(value=b'{"event_id": "evt-1", "tenant_id": "tenant-1"}')

    def _capture(*_a, **_k):
        bound["cfg"] = active_run_config()

    with patch("core.kafka_worker_entrypoint.handle_event", new=AsyncMock(side_effect=_capture)), \
         patch("core.kafka_worker_entrypoint.get_latest_run", return_value=_fake_run_ctx()), \
         patch("core.kafka_worker_entrypoint.Neo4jRepository"), \
         patch("core.kafka_worker_entrypoint.VectorStore"), \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.clear_failure"):
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())

    assert bound["cfg"] is _RUN_CFG


def test_handle_message_commits_on_success():
    consumer, dlq_producer = MagicMock(), MagicMock()
    msg = _msg(value=b'{"event_id": "evt-1", "tenant_id": "tenant-1"}')
    with patch("core.kafka_worker_entrypoint.handle_event", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.get_latest_run", return_value=_fake_run_ctx()), \
         patch("core.kafka_worker_entrypoint.Neo4jRepository"), \
         patch("core.kafka_worker_entrypoint.VectorStore"), \
         patch("core.kafka_worker_entrypoint.clear_retry_count") as clear, \
         patch("core.kafka_worker_entrypoint.clear_failure") as clear_fail:
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    consumer.commit.assert_called_once_with(message=msg)
    dlq_producer.produce.assert_not_called()
    clear.assert_called_once_with("evt-1")
    # Full-traceability requirement: a message that succeeds (even after
    # previously failing) must not leave a stale failure record behind.
    clear_fail.assert_called_once_with("kafka_incremental", "evt-1", "kafka_dlq")


# --- Kafka routing headers (use_ec2_database + runIdSuffix) ---

def test_parse_kafka_headers_reads_use_ec2_and_run_id_suffix():
    msg = _msg(headers=[("use_ec2_database", b"true"),
                        ("runIdSuffix", b'{"zeus":"ab12","confluence":""}')])
    use_ec2, suffixes = kafka_worker_entrypoint._parse_kafka_headers(msg)
    assert use_ec2 is True
    assert suffixes == {"zeus": "ab12", "confluence": ""}


def test_parse_kafka_headers_false_and_absent():
    assert kafka_worker_entrypoint._parse_kafka_headers(
        _msg(headers=[("use_ec2_database", b"false")]))[0] is False
    # Absent header -> None (caller falls back to the global setting) and {}.
    use_ec2, suffixes = kafka_worker_entrypoint._parse_kafka_headers(_msg(headers=None))
    assert use_ec2 is None and suffixes == {}


def test_parse_kafka_headers_malformed_is_ignored():
    use_ec2, suffixes = kafka_worker_entrypoint._parse_kafka_headers(
        _msg(headers=[("use_ec2_database", b"maybe"), ("runIdSuffix", b"not json")]))
    assert use_ec2 is None
    assert suffixes == {}


def test_handle_message_routes_to_ec2_and_resolves_run_by_suffix():
    """use_ec2_database=true + runIdSuffix pins the event to the EC2 environment
    and one specific run: the run is resolved by suffix (not "latest"), and both
    Neo4j and Qdrant follow the SAME EC2 choice as Mongo."""
    consumer, dlq_producer = MagicMock(), MagicMock()
    value = b'{"event_id":"evt-1","response_metadata":{"tenant_id":"tenant-1","connector_type":"zeus"}}'
    msg = _msg(value=value, headers=[("use_ec2_database", b"true"),
                                     ("runIdSuffix", b'{"zeus":"ab12","confluence":""}')])
    run_ctx = _fake_run_ctx()
    with patch("core.kafka_worker_entrypoint.handle_event", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.get_run_by_suffix", return_value=run_ctx) as by_suffix, \
         patch("core.kafka_worker_entrypoint.get_latest_run") as latest, \
         patch("core.kafka_worker_entrypoint.Neo4jRepository") as repo_cls, \
         patch("core.kafka_worker_entrypoint.VectorStore"), \
         patch("core.kafka_worker_entrypoint.get_qdrant_client", return_value=MagicMock()) as qc, \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.clear_failure"):
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    by_suffix.assert_called_once_with("tenant-1", "zeus", "ab12")
    latest.assert_not_called()
    assert repo_cls.call_args.kwargs["http_url"] == settings.ec2_neo4j_http_url
    qc.assert_called_once_with(settings.ec2_qdrant_url, settings.ec2_qdrant_api_key)


def test_handle_message_without_headers_uses_primary_and_latest_run():
    """Backward compatibility: a header-less message follows the global setting
    (primary here) and resolves the latest run, exactly as before this routing."""
    consumer, dlq_producer = MagicMock(), MagicMock()
    msg = _msg(value=b'{"event_id":"evt-1","tenant_id":"tenant-1"}')  # no headers
    with patch.object(settings, "use_ec2_databases", False), \
         patch("core.kafka_worker_entrypoint.handle_event", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.get_latest_run", return_value=_fake_run_ctx()) as latest, \
         patch("core.kafka_worker_entrypoint.get_run_by_suffix") as by_suffix, \
         patch("core.kafka_worker_entrypoint.Neo4jRepository") as repo_cls, \
         patch("core.kafka_worker_entrypoint.VectorStore"), \
         patch("core.kafka_worker_entrypoint.get_qdrant_client", return_value=MagicMock()) as qc, \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.clear_failure"):
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    latest.assert_called_once_with("tenant-1", "")
    by_suffix.assert_not_called()
    assert repo_cls.call_args.kwargs["http_url"] == settings.neo4j_http_url
    qc.assert_called_once_with(settings.qdrant_url, settings.qdrant_api_key)


def test_handle_message_resolves_this_events_own_run_and_closes_it():
    """repo/vector_store must come from the event's own (tenant_id,
    connector_type)'s latest completed run, built fresh and closed afterward --
    not a single shared repo built once at worker startup."""
    consumer, dlq_producer = MagicMock(), MagicMock()
    msg = _msg(value=b'{"event_id": "evt-1", "tenant_id": "tenant-1"}')
    run_ctx = _fake_run_ctx()
    repo_instance = MagicMock()
    with patch("core.kafka_worker_entrypoint.handle_event", new=AsyncMock()), \
         patch("core.kafka_worker_entrypoint.get_latest_run", return_value=run_ctx) as get_latest, \
         patch("core.kafka_worker_entrypoint.Neo4jRepository", return_value=repo_instance) as repo_cls, \
         patch("core.kafka_worker_entrypoint.VectorStore") as vector_store_cls, \
         patch("core.kafka_worker_entrypoint.get_qdrant_client", return_value=MagicMock()), \
         patch("core.kafka_worker_entrypoint.clear_retry_count"), \
         patch("core.kafka_worker_entrypoint.clear_failure"):
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    get_latest.assert_called_once_with("tenant-1", "")  # "" -- _msg() carries no response_metadata
    # database still comes from the event's own run; the endpoint kwargs are the per-message
    # (header-resolved) routing added alongside it.
    assert repo_cls.call_args.kwargs["database"] == run_ctx.neo4j_database
    entity_collections = [c.kwargs.get("collection") for c in vector_store_cls.call_args_list]
    assert run_ctx.qdrant_entities_collection in entity_collections
    assert run_ctx.qdrant_chunks_collection in entity_collections
    repo_instance.close.assert_called_once()


def test_handle_message_sends_to_dlq_when_no_completed_run_exists():
    """An event for a (tenant, connector) with no COMPLETED bulk run
    yet has nothing to write into -- NoCompletedRunError classifies as
    permanent, so it's DLQ'd immediately rather than redelivered forever."""
    consumer, dlq_producer = MagicMock(), _dlq_producer(delivered=True)
    msg = _msg(value=b'{"event_id": "evt-1", "tenant_id": "tenant-1"}')
    with patch("core.kafka_worker_entrypoint.get_latest_run", return_value=None), \
         patch("core.kafka_worker_entrypoint.record_failure") as record_call:
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    dlq_producer.produce.assert_called_once()
    consumer.commit.assert_called_once_with(message=msg)
    assert record_call.call_args.args[3] == "permanent"


def test_handle_message_routes_failure_to_handle_failure():
    consumer, dlq_producer, msg = MagicMock(), _dlq_producer(delivered=True), _msg(value=b'not valid json')
    with patch("core.kafka_worker_entrypoint.record_failure") as record_call:
        _handle_message(msg, consumer, dlq_producer, MagicMock(), MagicMock(), MagicMock())
    # malformed JSON -> permanent -> straight to DLQ, committed, never redelivered.
    dlq_producer.produce.assert_called_once()
    consumer.commit.assert_called_once_with(message=msg)
    # Full-traceability requirement: a permanent error is logged to the
    # database ledger the instant it's classified, not just sent to the DLQ.
    record_call.assert_called_once()
    assert record_call.call_args.args[0] == "kafka_incremental"
    assert record_call.call_args.args[2] == "kafka_dlq"
    assert record_call.call_args.args[3] == "permanent"


def test_request_shutdown_sets_module_flag():
    kafka_worker_entrypoint._shutdown_requested = False
    try:
        _request_shutdown(15, None)
        assert kafka_worker_entrypoint._shutdown_requested is True
    finally:
        kafka_worker_entrypoint._shutdown_requested = False


def test_run_registers_signal_handlers():
    """Doesn't exercise the real poll loop (needs a live broker) — just proves
    run() wires up graceful-shutdown signal handlers and closes the consumer
    on the way out."""
    with patch("core.kafka_worker_entrypoint.setup_logging"), \
         patch("core.kafka_worker_entrypoint.signal.signal") as signal_signal, \
         patch("core.kafka_worker_entrypoint._make_consumer") as make_consumer, \
         patch("core.kafka_worker_entrypoint._make_dlq_producer"), \
         patch("core.kafka_worker_entrypoint.ensure_dlq_topic", return_value=True), \
         patch("core.kafka_worker_entrypoint.resolve_llm_backends"), \
         patch("core.kafka_worker_entrypoint.ExtractionService"), \
         patch("core.kafka_worker_entrypoint.EmbeddingService"):
        consumer = MagicMock()

        def fake_poll(_timeout):
            kafka_worker_entrypoint._shutdown_requested = True  # exit after one iteration
            return None

        consumer.poll.side_effect = fake_poll
        make_consumer.return_value = consumer

        kafka_worker_entrypoint._shutdown_requested = False
        try:
            kafka_worker_entrypoint.run()
        finally:
            kafka_worker_entrypoint._shutdown_requested = False

    assert signal_signal.call_count == 2
    consumer.close.assert_called_once()


def test_run_resolves_llm_backends_once_before_building_extractor_and_embedder():
    """This process is always-on and shares one resolved backend for its whole
    lifetime (services/llm_backend.py) -- must resolve before ExtractionService/
    EmbeddingService are built, not after, and never re-resolve per event."""
    call_order = []
    with patch("core.kafka_worker_entrypoint.setup_logging"), \
         patch("core.kafka_worker_entrypoint.signal.signal"), \
         patch("core.kafka_worker_entrypoint._make_consumer") as make_consumer, \
         patch("core.kafka_worker_entrypoint._make_dlq_producer"), \
         patch("core.kafka_worker_entrypoint.ensure_dlq_topic", return_value=True), \
         patch("core.kafka_worker_entrypoint.resolve_llm_backends",
              side_effect=lambda: call_order.append("resolve_llm_backends")), \
         patch("core.kafka_worker_entrypoint.ExtractionService",
              side_effect=lambda: call_order.append("ExtractionService")), \
         patch("core.kafka_worker_entrypoint.EmbeddingService",
              side_effect=lambda: call_order.append("EmbeddingService")):
        consumer = MagicMock()

        def fake_poll(_timeout):
            kafka_worker_entrypoint._shutdown_requested = True
            return None

        consumer.poll.side_effect = fake_poll
        make_consumer.return_value = consumer

        kafka_worker_entrypoint._shutdown_requested = False
        try:
            kafka_worker_entrypoint.run()
        finally:
            kafka_worker_entrypoint._shutdown_requested = False

    assert call_order[0] == "resolve_llm_backends"
    assert set(call_order[1:]) == {"ExtractionService", "EmbeddingService"}


def test_run_does_not_build_a_shared_repo_or_vector_store_at_startup():
    """Unlike before, run() must NOT construct a single shared
    Neo4jRepository/VectorStore (or call create_constraints/ensure_collection)
    at startup -- each event resolves its own via _handle_message's
    run_registry lookup instead, since this worker serves every tenant/
    connector, not one fixed pair."""
    with patch("core.kafka_worker_entrypoint.setup_logging"), \
         patch("core.kafka_worker_entrypoint.signal.signal"), \
         patch("core.kafka_worker_entrypoint._make_consumer") as make_consumer, \
         patch("core.kafka_worker_entrypoint._make_dlq_producer"), \
         patch("core.kafka_worker_entrypoint.ensure_dlq_topic", return_value=True), \
         patch("core.kafka_worker_entrypoint.resolve_llm_backends"), \
         patch("core.kafka_worker_entrypoint.Neo4jRepository") as repo_cls, \
         patch("core.kafka_worker_entrypoint.ExtractionService"), \
         patch("core.kafka_worker_entrypoint.EmbeddingService"), \
         patch("core.kafka_worker_entrypoint.VectorStore") as vector_store_cls:
        consumer = MagicMock()

        def fake_poll(_timeout):
            kafka_worker_entrypoint._shutdown_requested = True
            return None

        consumer.poll.side_effect = fake_poll
        make_consumer.return_value = consumer

        kafka_worker_entrypoint._shutdown_requested = False
        try:
            kafka_worker_entrypoint.run()
        finally:
            kafka_worker_entrypoint._shutdown_requested = False

    repo_cls.assert_not_called()
    vector_store_cls.assert_not_called()
