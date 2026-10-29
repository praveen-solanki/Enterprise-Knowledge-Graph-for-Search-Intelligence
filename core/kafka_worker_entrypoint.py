"""
Always-on Kafka consumer entrypoint — the "worker" image, as opposed to
core/entrypoint.py's one-shot "job" image. Consumes GraphExtractRequest
events (see GRAPH_INCREMENTAL_SYNC_ORCHESTRATOR_DESIGN.md §4), resolves the
payload (inline or S3), and processes each document through
services/incremental_pipeline.py.

Manual commit only after a message's documents all process successfully — a
crash mid-processing leaves the message uncommitted, so it gets redelivered
on restart rather than silently dropped.

DLQ: a poison message (malformed JSON, a missing required field, an S3
claim-check pointer that 404s) could otherwise block every later message on
its partition indefinitely. Errors are classified (see _classify_error) into:
  - permanent: retrying would fail identically forever (bad JSON, a missing/
    malformed event field, S3 403/404) — sent straight to the DLQ topic and
    committed immediately, so the partition is never blocked by these.
  - transient: worth retrying (network blips, Neo4j 5xx, timeouts) — the
    consumer seeks back to this exact offset so the next poll() redelivers
    it immediately (a plain "leave it uncommitted" does nothing on its own,
    since poll() has already moved past it internally — without the seek, a
    LATER message on the same partition succeeding and committing would
    silently advance the partition's one committed offset past this one,
    losing it for good, not just delaying it), UNTIL a per-event attempt
    counter (persisted in Mongo, survives worker restarts) exceeds
    kafka_max_delivery_attempts, at which point it's DLQ'd too rather than
    retried forever.

Full traceability (manager requirement): the moment a message is given up on
(permanent, or transient-exhausted) and heads to the DLQ, it also gets a
durable record in the SAME ingestion_failures MongoDB collection the bulk
pipeline's own failure ledger uses (repository/mongo_repository.py's
record_failure) — one place to query "what failed, anywhere in the system",
not a bulk-only view plus a Kafka-only DLQ view with no shared identity. A
message that eventually succeeds after retrying clears that record, same
"clear-on-success" treatment the retry-count ledger already gets.

Usage:
    python -m core.kafka_worker_entrypoint [--run-id RUN_ID]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Optional

import time

# Must run before any import that touches config.settings — Settings is built
# eagerly the moment it's first imported, so Vault's values need to already be
# in os.environ by then. See connectors/vault_manager.py's module docstring.
from connectors.vault_manager import setup_environment_variables

setup_environment_variables()

from confluent_kafka import Consumer, Producer, TopicPartition

from config.db_endpoints import resolve_db_endpoints
from config.settings import settings
from connectors.base import ConnectorSpec, ConnectorType, SourceDocument
from connectors.registry import get_connector, known_connector_types
from repository.mongo_repository import (
    bind_db_endpoints, clear_failure, clear_retry_count, increment_retry_count, record_failure,
)
from repository.neo4j_repository import Neo4jRepository
from repository.run_registry import NoCompletedRunError, get_latest_run, get_run, get_run_by_suffix
from services.run_context import bind_run_config
from services.embedding_service import EmbeddingService
from utils.error_classification import classify_error as _classify_error
from services.extraction_service import ExtractionService
from services.incremental_event_parser import GraphExtractEvent, load_payload, parse_event
from services.incremental_pipeline import (
    _advance_watermark,
    _is_stale,
    _resolve_spec,
    delete_document,
    process_document,
    resolve_incremental_acl,
    resolve_watermark,
    update_acl_only,
)
from services.llm_backend import resolve_llm_backends
from services.vector_store import VectorStore, get_qdrant_client
from connectors.logger_setup import setup_logging
from utils import flow_log
from utils.log_context import clear_log_context, set_log_context

logger = logging.getLogger("graphrag.core.kafka_worker")

# Docs within one event are processed concurrently, bounded by settings.
# incremental_doc_concurrency — most events carry exactly one doc anyway, but a
# rare multi-doc event shouldn't wait for doc 1's entire pipeline to finish
# before doc 2 even starts, nor should a pathological event fire unbounded
# concurrent Azure calls at once. This overlap is real for the async LLM/
# embedding calls, but NOT for Neo4j writes: repo.* calls are synchronous
# `requests` calls, not wrapped in asyncio.to_thread, so one doc's Neo4j write
# blocks the whole event loop (including other docs' in-flight async work)
# until it returns — writes across docs never truly run in parallel, only
# their async phases do. Not a correctness issue (it keeps the shared
# requests.Session, and upsert_entity_incremental's two-round-trip
# read-then-write, race-free), just worth knowing the concurrency is I/O-only.

# Set by _request_shutdown on SIGTERM/SIGINT; checked once per poll iteration so a
# Kubernetes-issued termination (redeploy, scale-down) finishes whatever message is
# already in flight and exits cleanly, instead of being killed mid-processing. Not a
# correctness requirement — nothing commits until success either way, so an abrupt
# kill is still safe — just avoids every routine deploy looking like a crash in logs.
_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    logger.info("Received signal %s — finishing in-flight message, then shutting down", signum)
    _shutdown_requested = True


def _make_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "security.protocol": settings.kafka_security_protocol,
        "group.id": settings.kafka_consumer_group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,  # commit only after successful dispatch
        # Without this, librdkafka's own connection/SSL-handshake errors happen in
        # a background thread and never surface anywhere — a silently-failed broker
        # connection and a healthy-but-idle one look identical (poll() just returns
        # None either way). Routing them through our logger makes a real connection
        # failure actually visible instead of indistinguishable from "no messages yet".
        "logger": logger,
    })


def _on_assign(consumer: Consumer, partitions: list) -> None:
    """Partition-assign callback. Optionally jumps to the tip so a local/demo
    worker does not drain hundreds of historical poison messages (tenants with
    no COMPLETED bulk run) before it can see newly produced events."""
    if settings.kafka_seek_to_end_on_start:
        from confluent_kafka import OFFSET_END

        for tp in partitions:
            tp.offset = OFFSET_END
        logger.info(
            "KAFKA_SEEK_TO_END_ON_START=true — seeking %d partition(s) to tip; "
            "only messages produced after this start will be consumed",
            len(partitions),
        )
    consumer.assign(partitions)

def _make_dlq_producer() -> Producer:
    return Producer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "security.protocol": settings.kafka_security_protocol,
    })


def _make_admin_client():
    from confluent_kafka.admin import AdminClient

    return AdminClient({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "security.protocol": settings.kafka_security_protocol,
    })


def _dlq_topic_exists(admin=None) -> bool:
    """True when the configured DLQ topic is present on the broker."""
    client = admin or _make_admin_client()
    md = client.list_topics(topic=settings.kafka_topic_graph_extract_dlq, timeout=15)
    topic = md.topics.get(settings.kafka_topic_graph_extract_dlq)
    return topic is not None and topic.error is None


def ensure_dlq_topic() -> bool:
    """Create the DLQ topic if missing. Returns True once it exists (created
    or already present). A missing DLQ otherwise turns every permanent failure
    into an infinite seek-back loop that wedges the source partition forever.
    """
    from confluent_kafka.admin import NewTopic

    admin = _make_admin_client()
    topic = settings.kafka_topic_graph_extract_dlq
    if _dlq_topic_exists(admin):
        logger.info("DLQ topic %s already exists", topic)
        return True
    logger.warning("DLQ topic %s missing — attempting to create it", topic)
    futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=3)])
    fut = futures[topic]
    try:
        fut.result(timeout=30)
        logger.info("Created DLQ topic %s", topic)
        return True
    except Exception as exc:  # noqa: BLE001 — any create failure falls through to the exists check
        # TOPIC_ALREADY_EXISTS (race with another worker) is fine; anything else
        # may still leave the topic usable if a peer created it concurrently.
        logger.warning("DLQ topic create for %s returned: %s", topic, exc)
        return _dlq_topic_exists(admin)


def _send_to_dlq(dlq_producer: Producer, msg, reason: str) -> bool:
    """Returns True only once the DLQ message is CONFIRMED delivered.
    produce() is fire-and-forget — the broker's ack comes back later via a
    delivery callback — and flush() can time out with messages still queued.
    Treating either as success would let a DLQ send that actually failed
    (DLQ topic misconfigured, broker unreachable) still commit the source
    offset, silently losing the original message for good instead of just
    delaying it. produce() can also raise BufferError if the local queue is
    full; caught here rather than left to propagate and crash the worker."""
    delivered = {"ok": False}

    def _on_delivery(err, _msg) -> None:
        delivered["ok"] = err is None
        if err:
            logger.error("DLQ delivery failed: %s", err)

    try:
        dlq_producer.produce(
            settings.kafka_topic_graph_extract_dlq, value=msg.value(),
            headers=[("dlq_reason", reason.encode("utf-8", errors="replace"))],
            on_delivery=_on_delivery,
        )
    except BufferError:
        logger.exception("DLQ producer's local queue is full — cannot send to DLQ right now")
        return False

    pending = dlq_producer.flush(timeout=settings.kafka_dlq_flush_timeout_seconds)
    if pending > 0:
        logger.error("DLQ flush timed out with %d message(s) still undelivered", pending)
        return False
    return delivered["ok"]


def _default_spec() -> ConnectorSpec:
    """The connector this worker reads.

    settings.incremental_connector (env INCREMENTAL_CONNECTOR) picks it; zeus
    is the default, which is what this worker has always run. Read through
    settings rather than os.environ — config/settings.py is the only place
    that touches the environment. Known values: see known_connector_types().
    """
    name = (settings.incremental_connector or ConnectorType.WIKIPEDIA.value).strip().lower()
    if name not in known_connector_types():
        raise ValueError(f"INCREMENTAL_CONNECTOR={name!r} is not a known connector — "
                         f"known: {known_connector_types()}")
    return get_connector(name)


def fetch_documents(transport, index, content_ids=None, query=None, normalize=None):
    """Lookup by id through the connector that owns `normalize` (a bound
    spec.acl_lookup_document). Tests patch this name."""
    spec = getattr(normalize, "__self__", None)
    if spec is None or not content_ids:
        return []
    found = []
    for batch, _count in spec.fetch(
        transport, index, spec.build_query(list(content_ids)), batch_size=10**9,
    ):
        for doc in batch:
            if isinstance(doc, SourceDocument):
                raw = dict(doc.raw)
                raw["id"] = doc.doc_id
            else:
                raw = doc
            item = normalize(raw) if normalize else raw
            if item:
                found.append(item)
    return found


def _fetch_acl_source_docs(
    es, event: GraphExtractEvent, docs: list[dict], spec: ConnectorSpec,
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Batch-fetch the real Elasticsearch documents for every doc_id in this
    event's payload, in one ES call, so process_document can derive ACL from
    real site/status/owner fields instead of trusting the payload's own ACL
    fields (the producer doesn't populate them). Also fetches the ABAC
    site-audience map for every site referenced by those documents, in a
    second bounded call.

    Returns (es_doc_by_id, site_audiences_by_id). A doc_id missing from
    es_doc_by_id (lookup failed entirely, or that specific doc_id has no
    matching ES document — e.g. not yet indexed, or genuinely gone) means
    process_document must drop it, fail-closed, rather than guess at its
    visibility. If the site-audience lookup itself fails, EVERY doc that
    still needed it is dropped too — without knowing whether a "public" site
    is actually audience-gated, deriving ACL for any of them would risk
    over-granting.

    A doc that already carries its own "site_audiences_by_id" key (the
    fully-embedded payload shape — present, even as {}, only there) is trusted
    as-is and skipped here entirely: no ES lookup for it, and its own
    site_audiences_by_id is merged into the returned map instead. Only docs
    WITHOUT that key go through the live ES lookup below, exactly as before
    this existed — an ES failure only affects those, never the embedded ones.

    index_name: event.index_name is required — there is no default index
    anymore (PR-41 review point 3). An event without one is dropped fail-closed
    here rather than looked up against a fabricated, single-tenant default that
    would derive ACL from the wrong tenant's data."""
    embedded_site_audiences: dict[str, list[str]] = {}
    needs_es_lookup = []
    for d in docs:
        if "site_audiences_by_id" in d:
            embedded_site_audiences.update(d.get("site_audiences_by_id") or {})
        else:
            needs_es_lookup.append(d)

    doc_ids = [d["doc_id"] for d in needs_es_lookup if d.get("doc_id")]
    if not doc_ids:
        return {}, embedded_site_audiences
    index_name = event.index_name
    if not index_name:
        logger.error("event_id=%s: no index_name on the event and no default index — "
                     "dropping all %d doc(s) this pass (fail-closed; ACL lookup needs an "
                     "explicit index).", event.event_id, len(doc_ids))
        return {}, embedded_site_audiences
    try:
        es_docs = fetch_documents(es, index_name, content_ids=doc_ids,
                                  normalize=spec.acl_lookup_document)
    except Exception:
        logger.exception("event_id=%s: ES ACL lookup failed for %d doc(s) against index=%s — "
                         "dropping all of them this pass (fail-closed, no ACL can be derived "
                         "without a working ES lookup)", event.event_id, len(doc_ids), index_name)
        return {}, embedded_site_audiences
    es_doc_by_id = {d["id"]: d for d in es_docs if d.get("id")}

    site_ids = spec.acl_scope_ids(list(es_doc_by_id.values()))
    if not site_ids:
        return es_doc_by_id, embedded_site_audiences
    try:
        site_audiences_by_id = spec.acl_scope_audiences(es, index_name, site_ids)
    except Exception:
        logger.exception("event_id=%s: site-audience lookup failed for %d site(s) against "
                         "index=%s — dropping every doc that needed it (fail-closed: can't "
                         "safely derive ACL for a 'public' site without knowing whether it's "
                         "audience-gated)", event.event_id, len(site_ids), index_name)
        return {}, embedded_site_audiences
    return es_doc_by_id, {**embedded_site_audiences, **site_audiences_by_id}


async def handle_event(raw_event: dict, repo: Neo4jRepository, extractor: ExtractionService,
                       embedder: EmbeddingService, es, vector_store: VectorStore,
                       chunks_vector_store: Optional[VectorStore] = None,
                       spec: Optional[ConnectorSpec] = None) -> None:
    """Binds tenant_id/connector_type/run_id to every log line for this event, cleared
    in finally so it never leaks onto the next message's log lines. run_id is this
    worker's own _KAFKA_RUN_ID (--run-id/INCREMENTAL_RUN_ID, or the fixed
    "kafka_incremental" sentinel if neither was given) — every line logged while
    processing this event, not just the one startup line, now carries it."""
    event: GraphExtractEvent = parse_event(raw_event)
    set_log_context(event.tenant_id, event.connector_type, run_id=_KAFKA_RUN_ID)
    try:
        await _handle_parsed_event(event, raw_event, repo, extractor, embedder, es, vector_store,
                                   chunks_vector_store, spec=spec)
    finally:
        clear_log_context()


async def _handle_parsed_event(event: GraphExtractEvent, raw_event: dict, repo: Neo4jRepository,
                               extractor: ExtractionService, embedder: EmbeddingService, es,
                               vector_store: VectorStore,
                               chunks_vector_store: Optional[VectorStore] = None,
                               spec: Optional[ConnectorSpec] = None) -> None:
    # The worker always passes one; this default is for direct callers. Resolved
    # here rather than deeper down so it stays inside _handle_message's own
    # try/except, where a misconfigured connector is classified like any other
    # failure instead of escaping every handler.
    spec = spec if spec is not None else _default_spec()
    payload = load_payload(raw_event)
    docs = payload.get("docs", [])

    if event.operation == "delete":
        # Staleness watermark: without this, a redelivered/out-of-order delete
        # could retract a document that was legitimately re-added by a LATER
        # event already processed. Same check, same watermark collection, as
        # process_document (services/incremental_pipeline.py) uses for upserts.
        # No es_doc here (the document is gone — nothing to look up); doc
        # itself is still passed through in case a future delete payload
        # ever carries doc_updated_at too — resolve_watermark falls through
        # cleanly to event.completed_at if it doesn't.
        for doc in docs:
            doc_id = doc["doc_id"]
            watermark = resolve_watermark(event, doc, None)
            if _is_stale(event.tenant_id, doc_id, watermark):
                flow_log.step_skipped(logger, "Delete document", reason="stale event",
                                      doc_id=doc_id, watermark=watermark)
                continue
            flow_log.step_start(logger, "Delete document", doc_id=doc_id)
            delete_document(doc_id, repo, event.tenant_id, vector_store, chunks_vector_store)
            _advance_watermark(event.tenant_id, doc_id, watermark)
            flow_log.step_done(logger, "Delete document", doc_id=doc_id)
        return
    if event.operation == "acl_update":
        # search-utility-agents' confirmed lightweight event: content/body
        # unchanged, only permissions changed — refresh ACL on the existing
        # graph nodes without any re-extraction. Same staleness-watermark
        # treatment as delete above. resolve_watermark prefers doc's own
        # doc_updated_at (the producer's dedicated field for this), then the
        # live es_doc's lastmodifieddate (already fetched for ACL resolution
        # below, so free), over event.completed_at — see that function's
        # docstring for why: event.completed_at/event_timestamp can get
        # reshuffled by a retried publish, neither of the other two can.
        es_doc_by_id, site_audiences_by_id = _fetch_acl_source_docs(es, event, docs, spec=spec)
        for doc in docs:
            doc_id = doc["doc_id"]
            watermark = resolve_watermark(event, doc, es_doc_by_id.get(doc_id))
            if _is_stale(event.tenant_id, doc_id, watermark):
                flow_log.step_skipped(logger, "ACL update", reason="stale event",
                                      doc_id=doc_id, watermark=watermark)
                continue
            flow_log.step_start(logger, "ACL update", doc_id=doc_id)
            # Same producer-tokens-then-live-ES-fallback resolution as the upsert path
            # (services/incremental_pipeline.py's resolve_incremental_acl) -- unlike
            # upsert, an acl_update that resolves to nothing must not just skip the
            # document: leaving it untouched would fail OPEN (still visible under
            # permissions that no longer apply), so it retracts instead, mirroring the
            # reference's own acl_update-resolves-to-dropped -> delete fallthrough.
            acl = resolve_incremental_acl(doc, es_doc_by_id.get(doc_id), site_audiences_by_id,
                                          connector_type=event.connector_type, tenant_id=event.tenant_id)
            if acl is None:
                logger.warning(
                    "doc_id=%s: acl_update resolved to zero visibility (no producer "
                    "tokens, no live ES ACL) — retracting rather than leaving a stale "
                    "ACL in place", doc_id)
                delete_document(doc_id, repo, event.tenant_id, vector_store, chunks_vector_store)
                _advance_watermark(event.tenant_id, doc_id, watermark)
                flow_log.step_done(logger, "ACL update", doc_id=doc_id, result="retracted (zero visibility)")
                continue
            update_acl_only(doc_id, repo, event.tenant_id, acl, vector_store, chunks_vector_store)
            _advance_watermark(event.tenant_id, doc_id, watermark)
            flow_log.step_done(logger, "ACL update", doc_id=doc_id, tokens=len(acl))
        return
    if event.operation != "upsert":
        logger.warning("event_id=%s: operation=%r not handled — skipping",
                       event.event_id, event.operation)
        return

    es_doc_by_id, site_audiences_by_id = _fetch_acl_source_docs(es, event, docs, spec=spec)
    doc_sem = asyncio.Semaphore(settings.incremental_doc_concurrency)

    async def _process_one(doc: dict) -> None:
        async with doc_sem:
            await process_document(event, doc, repo, extractor, embedder, vector_store,
                                   es_doc_by_id.get(doc["doc_id"]), site_audiences_by_id,
                                   chunks_vector_store)

    await asyncio.gather(*[_process_one(doc) for doc in docs])


# Full-traceability requirement: every Kafka failure lands in the SAME
# database ledger (repository/mongo_repository.py's ingestion_failures) the
# bulk pipeline already uses — one place to query "what failed, anywhere in
# the system", not a bulk-only view plus a separate Kafka-only DLQ view.
# This sentinel run_id is what distinguishes incremental failures from a
# real bulk run_id when browsing the collection; item_id is the same
# per-event retry_key already used for the Mongo retry-count ledger, so a
# redelivered message's failure record and its retry count share one
# identity.
#
# Overridable via --run-id/INCREMENTAL_RUN_ID (testing/debugging only, never in
# production), set once in run() before the poll loop starts.
_KAFKA_RUN_ID = "kafka_incremental"

# Testing/debugging only — never set in production. Production always writes to
# the latest run for a (tenant, connector); this pins to one specific run instead,
# for isolating your own test run when several exist for the same tenant. None
# (the default) keeps production behavior unchanged. Set together with
# _KAFKA_RUN_ID in run(), from the same --run-id/INCREMENTAL_RUN_ID value.
_TARGET_RUN_ID: Optional[str] = None


def _retry_key(msg, raw_event: Optional[dict]) -> str:
    return (raw_event or {}).get("event_id") or f"{msg.topic()}:{msg.partition()}:{msg.offset()}"


def _msg_key(msg) -> Optional[str]:
    """The message key as text, for logging — None when the record has no key."""
    key = msg.key()
    if key is None:
        return None
    return key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)


def _coords(msg) -> str:
    """Greppable Kafka coordinates for one record, shared by every important
    processing/failure log so a message can be traced end-to-end."""
    return (f"topic={msg.topic()} | partition={msg.partition()} | offset={msg.offset()} | "
            f"key={_msg_key(msg)}")


def _header_text(value) -> str:
    return (value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray))
            else str(value)).strip()


def _parse_kafka_headers(msg) -> tuple[Optional[bool], dict]:
    """Read the producer's routing headers off a confluent-kafka message.

    Returns (use_ec2_database, run_id_suffix_by_connector):
      - use_ec2_database: True/False parsed from the `use_ec2_database` header,
        or None when the header is absent or unparseable — the caller then falls
        back to the process-global setting, so a header-less message behaves
        exactly as before this routing existed.
      - run_id_suffix_by_connector: the `runIdSuffix` JSON object
        (e.g. {"zeus": "", "confluence": ""}), or {} when absent/unparseable.
    """
    raw = msg.headers() or []
    values = {k: v for k, v in raw}

    use_ec2: Optional[bool] = None
    if "use_ec2_database" in values:
        text = _header_text(values["use_ec2_database"]).lower()
        if text in ("true", "1"):
            use_ec2 = True
        elif text in ("false", "0"):
            use_ec2 = False
        else:
            logger.warning("Unparseable use_ec2_database header %r — falling back to the global "
                           "setting", text)

    suffixes: dict = {}
    if "runIdSuffix" in values:
        try:
            parsed = json.loads(_header_text(values["runIdSuffix"]))
            if isinstance(parsed, dict):
                suffixes = parsed
            else:
                logger.warning("runIdSuffix header is not a JSON object (%r) — ignoring", parsed)
        except (ValueError, TypeError):
            logger.warning("Unparseable runIdSuffix header — ignoring", exc_info=True)
    return use_ec2, suffixes


def _handle_message(msg, consumer: Consumer, dlq_producer: Producer,
                    extractor: ExtractionService, embedder: EmbeddingService, es) -> None:
    """Resolves this event's own Neo4j database + Qdrant collections fresh,
    per message — this worker serves every tenant/connector, so there's no
    single repo/vector_store to build once at startup. Whichever bulk run
    created that database/collections already ran create_constraints()/
    ensure_collection(), so this doesn't repeat either."""
    parsed_for_error: Optional[dict] = None
    repo: Optional[Neo4jRepository] = None
    try:
        # Routing headers decide which environment this message targets. Resolved
        # and bound FIRST, before any Mongo access, so the run-registry lookup,
        # retry-count ledger, and failure ledger all hit the same MongoDB — and
        # do so consistently across redeliveries of this same message. use_ec2 is
        # None when the header is absent, so resolve_db_endpoints falls back to the
        # global setting (pre-header behaviour unchanged).
        use_ec2, run_id_suffixes = _parse_kafka_headers(msg)
        endpoints = resolve_db_endpoints(use_ec2)
        bind_db_endpoints(endpoints)
        logger.debug("Kafka transport | %s | use_ec2_database=%s", _coords(msg), endpoints.use_ec2)

        raw_event: dict = json.loads(msg.value())
        parsed_for_error = raw_event
        event = parse_event(raw_event)
        # Tolerant resolution, same as process_document's own: live legacy
        # events carry free-form connector_type values ("", legacy spellings) that
        # predate the enum and have always meant the intranet — a strict
        # get_connector here would DLQ every one of them.
        spec = _resolve_spec(event.connector_type)
        # _TARGET_RUN_ID: testing/debugging only, never set in production.
        if _TARGET_RUN_ID:
            run_ctx = get_run(_TARGET_RUN_ID)
            if run_ctx is None:
                raise NoCompletedRunError(
                    f"--run-id/INCREMENTAL_RUN_ID={_TARGET_RUN_ID!r} does not match any registered run"
                )
        else:
            # runIdSuffix pins the event to one specific run when the producer sends it for
            # this connector; an empty/absent suffix keeps the original "latest COMPLETED run"
            # behaviour. connector_type is the key into the header's per-connector object.
            run_suffix = (run_id_suffixes.get(event.connector_type) or "").strip()
            if run_suffix:
                run_ctx = get_run_by_suffix(event.tenant_id, event.connector_type, run_suffix)
                if run_ctx is None:
                    raise NoCompletedRunError(
                        f"no COMPLETED run found for tenant_id={event.tenant_id!r} "
                        f"connector_type={event.connector_type!r} run_suffix={run_suffix!r} "
                        f"(from the runIdSuffix header) — nothing for this incremental event "
                        f"to write into"
                    )
            else:
                run_ctx = get_latest_run(event.tenant_id, event.connector_type)
                if run_ctx is None:
                    raise NoCompletedRunError(
                        f"no COMPLETED bulk run found for tenant_id={event.tenant_id!r} "
                        f"connector_type={event.connector_type!r} — nothing for this incremental "
                        f"event to write into yet"
                    )
        flow_log.event_received(logger, event_id=event.event_id, op=event.operation,
                                topic=msg.topic(), partition=msg.partition(),
                                offset=msg.offset(), key=_msg_key(msg))
        # Log every datastore this event resolved to, up front — Neo4j + Qdrant
        # follow the SAME environment choice as Mongo (the run was registered in
        # this environment's registry and its graph/vectors live in its Neo4j/Qdrant).
        flow_log.databases_selected(
            logger,
            env=("EC2" if endpoints.use_ec2 else "primary"),
            run_suffix=run_ctx.run_suffix,
            mongo_uri=endpoints.mongo_uri, mongo_db=endpoints.mongo_db,
            neo4j_http_url=endpoints.neo4j_http_url, neo4j_database=run_ctx.neo4j_database,
            qdrant_url=endpoints.qdrant_url,
            qdrant_entities=run_ctx.qdrant_entities_collection,
            qdrant_chunks=run_ctx.qdrant_chunks_collection,
        )
        repo = Neo4jRepository(database=run_ctx.neo4j_database, http_url=endpoints.neo4j_http_url,
                               username=endpoints.neo4j_username, password=endpoints.neo4j_password)
        qdrant = get_qdrant_client(endpoints.qdrant_url, endpoints.qdrant_api_key)
        vector_store = VectorStore(collection=run_ctx.qdrant_entities_collection, client=qdrant)
        chunks_vector_store = VectorStore(collection=run_ctx.qdrant_chunks_collection, client=qdrant)
        # Pin this event to the model/space/flags its bulk run recorded, so it
        # lands in the same collections at the same dimension and uses the same
        # LLM regardless of what THIS worker's process settings/probe resolved to.
        # Bound in the calling context right before asyncio.run so the coroutine's
        # task inherits it; every event rebinds and each message has its own
        # asyncio context, so it never leaks across events.
        bind_run_config(run_ctx.run_config)
        asyncio.run(handle_event(raw_event, repo, extractor, embedder, es, vector_store,
                                 chunks_vector_store, spec=spec))
        consumer.commit(message=msg)
        flow_log.event_done(logger, event_id=event.event_id, op=event.operation, offset=msg.offset())
        retry_key = _retry_key(msg, parsed_for_error)
        # No-op if this message never failed before (the common case) — but if it DID
        # retry a few times before eventually succeeding, this stops the ledger entry
        # from lingering forever and (were the same key ever reused) being inherited
        # as a stale starting count by an unrelated future message.
        clear_retry_count(retry_key)
        # Same clear-on-success treatment for the failure ledger — a message that
        # failed once, retried, and eventually succeeded must not leave a stale
        # "still failing" record behind for someone querying call_failures later.
        clear_failure(_KAFKA_RUN_ID, retry_key, "kafka_dlq")
    except Exception as exc:  # noqa: BLE001 — every failure mode is handled explicitly below
        # Still inside the bound-endpoints context, so the retry-count / failure
        # ledger writes _handle_failure makes land in the same MongoDB this
        # message resolved to.
        _handle_failure(msg, consumer, dlq_producer, exc, parsed_for_error)
    finally:
        if repo is not None:
            repo.close()
        # Clear the per-message endpoint binding so it never leaks onto the next
        # message handled on this thread.
        bind_db_endpoints(None)


def _seek_back_and_backoff(consumer: Consumer, msg, attempts: int) -> None:
    """Without this seek, leaving the message "uncommitted" does nothing useful: poll()
    already advanced past this offset internally, so the very next poll() would just
    hand back the NEXT message, not this one. If that next message then succeeds and
    calls consumer.commit(), Kafka's single per-partition high-water mark advances past
    THIS offset too — silently losing it forever, since a restart resumes from the
    committed offset, not from "the first offset that ever failed." Seeking back here
    guarantees the next poll() re-delivers this exact message, so nothing later can ever
    commit past it until it succeeds or exhausts its attempts and gets DLQ'd."""
    consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
    # Bounded backoff so a persistently-failing message (e.g. Neo4j down) doesn't spin
    # the poll loop at full CPU retrying the same offset with zero delay.
    time.sleep(min(attempts, settings.kafka_redelivery_backoff_cap_seconds))


def _commit_poison_after_dlq_attempt(
    consumer: Consumer, dlq_producer: Producer, msg, reason: str, *, clear_retry_key: Optional[str] = None,
) -> None:
    """DLQ + commit on success. If the DLQ topic itself is missing, still commit:
    the failure is already in Mongo (caller recorded it), and seeking forever
    wedges every later message on the partition behind one poison event — worse
    than advancing after a durable ledger write. Other DLQ failures (broker
    blip, full queue) still seek-back so we don't silently drop the message."""
    coords = _coords(msg)
    if _send_to_dlq(dlq_producer, msg, reason=reason):
        consumer.commit(message=msg)
        if clear_retry_key is not None:
            clear_retry_count(clear_retry_key)
        return
    if not _dlq_topic_exists():
        logger.critical(
            "Kafka message | %s: DLQ topic %s is missing — committing after Mongo "
            "failure ledger write so later messages on this partition can proceed. "
            "Create the DLQ topic (or fix ensure_dlq_topic) to restore Kafka-side DLQ.",
            coords, settings.kafka_topic_graph_extract_dlq,
        )
        consumer.commit(message=msg)
        if clear_retry_key is not None:
            clear_retry_count(clear_retry_key)
        return
    logger.error("Kafka message | %s: DLQ send failed — leaving uncommitted and "
                 "retrying rather than losing the message", coords)
    _seek_back_and_backoff(consumer, msg, attempts=1)


def _handle_failure(msg, consumer: Consumer, dlq_producer: Producer, exc: Exception,
                    raw_event: Optional[dict]) -> None:
    coords = _coords(msg)
    error_class = _classify_error(exc)

    if error_class == "fatal":
        # A global, config-level problem (bad credentials, wrong database, etc.) that
        # will fail identically for every future message too — draining the entire
        # topic into the DLQ one message at a time would be far worse than crashing
        # loudly here so someone notices and fixes the underlying config. run()'s
        # finally block still closes the consumer/repo cleanly before this propagates.
        logger.critical("Kafka message | %s: fatal error (likely bad credentials/config) — "
                        "crashing the worker instead of draining the topic into the DLQ",
                        coords)
        raise exc

    if error_class == "permanent":
        # NoCompletedRunError is expected for backlog events whose tenants never
        # got a bulk run — log once at WARNING without a full traceback, DLQ+commit,
        # and move on. Other permanent errors still get the full stack.
        if isinstance(exc, NoCompletedRunError):
            logger.warning("Kafka message | %s: %s — sending to DLQ and skipping", coords, exc)
        else:
            logger.exception("Kafka message | %s: permanent error — sending straight to the DLQ "
                             "(retrying would fail identically forever)", coords)
        # Full-traceability requirement: a permanent error is, by definition, never
        # retried (retrying would fail identically forever) — it goes straight to the
        # durable ledger the instant it's classified, same as every other retry-then-log
        # call site's exhaustion point.
        record_failure(_KAFKA_RUN_ID, _retry_key(msg, raw_event), "kafka_dlq",
                       error_class, str(exc))
        _commit_poison_after_dlq_attempt(
            consumer, dlq_producer, msg, reason=f"permanent: {exc}",
        )
        return

    # transient: count delivery attempts per event_id (persisted in Mongo, so it
    # survives worker restarts, not just in-process redeliveries within one run) —
    # once exhausted, this stops being "worth retrying" and gets DLQ'd too, instead
    # of blocking this partition's later messages forever.
    retry_key = _retry_key(msg, raw_event)
    attempts = increment_retry_count(retry_key)
    if attempts >= settings.kafka_max_delivery_attempts:
        logger.exception("Kafka message | %s: transient error persisted for %d attempt(s) "
                         "(limit=%d) — giving up, sending to the DLQ",
                         coords, attempts, settings.kafka_max_delivery_attempts)
        # Full-traceability requirement: retries first (this branch is only reached
        # after kafka_max_delivery_attempts real redeliveries), THEN a durable
        # database record once genuinely exhausted — never just the DLQ message
        # and an app log line.
        record_failure(_KAFKA_RUN_ID, retry_key, "kafka_dlq", error_class, str(exc))
        _commit_poison_after_dlq_attempt(
            consumer, dlq_producer, msg,
            reason=f"transient, exhausted {attempts} attempt(s): {exc}",
            clear_retry_key=retry_key,
        )
        return

    logger.exception("Kafka message | %s: transient error (attempt %d/%d) — rewinding to retry, "
                     "will not let any later message on this partition commit past it",
                     coords, attempts, settings.kafka_max_delivery_attempts)
    _seek_back_and_backoff(consumer, msg, attempts)


def run(run_id: Optional[str] = None) -> int:
    setup_logging()
    # Testing/debugging only — see _TARGET_RUN_ID's docstring. Resolved once,
    # before the poll loop, same as every other setting this always-on process
    # reads exactly once at startup.
    global _KAFKA_RUN_ID, _TARGET_RUN_ID
    resolved_run_id = run_id or settings.incremental_run_id or None
    if resolved_run_id:
        _KAFKA_RUN_ID = resolved_run_id
        _TARGET_RUN_ID = resolved_run_id
    # Resolved once, here, before ExtractionService/EmbeddingService are built below —
    # this whole (always-on) process shares one resolved backend for its entire
    # lifetime, never re-checked per event. See services/llm_backend.py's docstring.
    resolve_llm_backends()
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    # A missing DLQ topic used to turn every permanent failure into an infinite
    # seek-back loop (UNKNOWN_TOPIC_OR_PART → leave uncommitted → same offset
    # forever). Create it up front when possible; _commit_poison_after_dlq_attempt
    # still advances past poison if create is denied by ACLs.
    if not ensure_dlq_topic():
        logger.critical(
            "DLQ topic %s is missing and could not be created — permanent failures "
            "will still advance (Mongo failure ledger only) so the partition is not wedged",
            settings.kafka_topic_graph_extract_dlq,
        )
    consumer = _make_consumer()
    consumer.subscribe([settings.kafka_topic_graph_extract], on_assign=_on_assign)
    dlq_producer = _make_dlq_producer()
    es = _default_spec().open_transport()
    # Validate the configured connector once, loudly, before consuming anything.
    # A bad INCREMENTAL_CONNECTOR must stop the worker here rather than turn every
    # message into a per-message failure. The instance is not kept —
    # _handle_message builds its own per message, because a spec carries per-run
    # state and this worker serves every tenant on the topic.
    connector_name = _default_spec().connector_type.value
    extractor = ExtractionService()
    embedder = EmbeddingService()
    # No shared Neo4jRepository/VectorStore built here — _handle_message resolves
    # each event's own database/collections fresh via get_latest_run().

    logger.info("Kafka worker started — connector=%s topic=%s group=%s dlq_topic=%s run_id=%s",
               connector_name, settings.kafka_topic_graph_extract,
               settings.kafka_consumer_group, settings.kafka_topic_graph_extract_dlq, _KAFKA_RUN_ID)
    try:
        last_heartbeat = time.monotonic()
        while not _shutdown_requested:
            msg = consumer.poll(settings.kafka_poll_timeout_seconds)
            if msg is None:
                # Idle is the expected steady state (no event on the topic yet) —
                # this heartbeat is just so that's visibly "connected, waiting" in
                # the logs rather than indistinguishable silence from a stuck consumer.
                now = time.monotonic()
                if now - last_heartbeat >= settings.kafka_heartbeat_interval_seconds:
                    logger.info("Still polling %s (group=%s) — no new messages",
                               settings.kafka_topic_graph_extract, settings.kafka_consumer_group)
                    last_heartbeat = now
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                continue
            _handle_message(msg, consumer, dlq_producer, extractor, embedder, es)
    finally:
        logger.info("Kafka worker shutting down")
        consumer.close()
    return 0


def _resolve_run_id(args: argparse.Namespace) -> Optional[str]:
    # Same CLI-flag-then-settings pattern as core/entrypoint.py's --retry-failed:
    # a CLI flag wins over the env var when both are given.
    run_id = args.run_id if args.run_id is not None else settings.incremental_run_id
    return (run_id or "").strip() or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the always-on incremental Kafka worker.")
    parser.add_argument("--run-id", default=None, metavar="RUN_ID",
                        help="TESTING/DEBUGGING ONLY — never set in production. Pins this worker "
                        "to one specific run's graph instead of the latest, and labels its log/"
                        "failure-ledger entries. Falls back to env var INCREMENTAL_RUN_ID.")
    args = parser.parse_args()
    sys.exit(run(run_id=_resolve_run_id(args)))
