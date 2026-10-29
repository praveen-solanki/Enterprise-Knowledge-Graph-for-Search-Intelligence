"""Unit tests for services/incremental_pipeline.py (AIS-3634)."""
import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import settings
from connectors.base import PeopleContext, PeopleResult, PersonRecord
from connectors.registry import get_connector
from models.extraction import ChunkExtractionResult, ExtractedClaim, ExtractedEntity
from services.incremental_event_parser import GraphExtractEvent
from services.incremental_pipeline import (
    _to_source_document,
    _extract_chunks,
    _retire_merged_person_mentions,
    delete_document,
    process_document,
    resolve_incremental_acl,
    resolve_watermark,
    update_acl_only,
)
from services.run_context import RunConfig, bind_run_config


def _run_cfg(*, build_authorship_edges: bool = True,
             contributor_entity_types: tuple = ("PROJECT",)) -> RunConfig:
    """A RunConfig pinning the graph-shaping flags process_document reads through
    effective_run_config(); embedding/LLM fields are filler this path doesn't use."""
    return RunConfig(
        embedding_backend="azure", embedding_model="text-embedding-3-large",
        embedding_dimension=1536, consume_upstream_vector=True,
        es_chunk_vector_field="embeddings_openai_large_1536",
        llm_backend="azure", llm_model="gpt",
        build_authorship_edges=build_authorship_edges,
        contributor_entity_types=contributor_entity_types, people_org_graph_enabled=True,
    )


def _event(**overrides) -> GraphExtractEvent:
    event = GraphExtractEvent(
        event_id="evt-1", tenant_id="tenant-1", connector_type="wikipedia",
        index_name="gs-prod-tenant-1-v9", ingestion_type="INCREMENTAL",
        operation="upsert", completed_at=1000,
    )
    for k, v in overrides.items():
        setattr(event, k, v)
    return event


def _payload_doc(**overrides) -> dict:
    """Matches the REAL producer's shape (search-utility-agents AIS-3687,
    verified against its actual code) — chunks nested under _chunks.v1
    exactly like Elasticsearch, each chunk carrying extra fields (chunk_type,
    embeddings, ...) beyond chunk_id/text that we deliberately ignore."""
    doc = {
        "doc_id": "doc-1", "title": "Some doc", "content_is_restricted": False,
        "_chunks": {"v1": [{"chunk_id": "c1", "text": "Globex acquired Initech.",
                            "chunk_type": "SEMANTIC", "_embeddings": {"unused": [0.1]}}]},
    }
    doc.update(overrides)
    return doc


def _es_doc(**overrides) -> dict:
    """Live ES document used for Zeus ACL derivation (not the Kafka payload)."""
    doc = {
        "id": "doc-1", "object_type": "content", "content_status": "published",
        "content_site_type": "public", "content_site_id": "site-1",
        "content_site_is_deleted": False, "is_deleted": False,
        "createdbyid": "user-1", "lastmodifiedbyid": "user-2",
    }
    doc.update(overrides)
    return doc


# --- resolve_incremental_acl (Zeus derives from live ES doc) ---

def test_resolve_incremental_acl_is_public():
    """Wikipedia has no site ACL. Every document is public."""
    doc = _payload_doc(allow_access_control=["stale:daemon:token"])
    acl = resolve_incremental_acl(doc, es_doc=_es_doc(), site_audiences_by_id={})
    assert acl == ["public"]


def test_resolve_incremental_acl_ignores_audience_maps():
    acl = resolve_incremental_acl(_payload_doc(), es_doc=_es_doc(),
                                  site_audiences_by_id={"site-1": ["aud-x"]})
    assert acl == ["public"]


def test_resolve_incremental_acl_is_public_without_a_source_doc():
    acl = resolve_incremental_acl(_payload_doc(), es_doc=None, site_audiences_by_id=None)
    assert acl == ["public"]


def test_extract_chunks_reads_the_real_nested_es_style_shape():
    chunks = _extract_chunks(_payload_doc())
    assert chunks == [{"chunk_id": "c1", "text": "Globex acquired Initech.",
                       "chunk_type": "SEMANTIC", "_embeddings": {"unused": [0.1]}}]


def test_extract_chunks_dedupes_overlapping_recursive_chunks():
    """Experiment-parity review C1: same dedup the bulk path gets — without
    it, the upstream chunker's overlapping windows (sharing one
    recursive_chunk_id) would each get extracted separately."""
    doc = _payload_doc(_chunks={"v1": [
        {"chunk_id": "c1", "recursive_chunk_id": "rid-1", "text": "hello"},
        {"chunk_id": "c2", "recursive_chunk_id": "rid-1", "text": "hello"},
    ]})
    assert [c["chunk_id"] for c in _extract_chunks(doc)] == ["c1"]


def test_extract_chunks_empty_when_missing():
    assert _extract_chunks({"doc_id": "doc-1"}) == []


def _acl_ok(tokens=("public",)):
    from services.acl import AclDecision
    return AclDecision(list(tokens), False, "incremental")


def test_to_source_document_types_the_payload_under_its_own_names():
    """This replaces _adapt_for_bulk_write, which fabricated content_title /
    content_publish_on_date / createddate so the bulk write methods could be
    reused. Those methods take SourceDocuments now, so nothing is renamed."""
    from connectors.base import ConnectorType

    chunks = _extract_chunks(_payload_doc())
    doc = _to_source_document(_payload_doc(), chunks, "wikipedia", _acl_ok())
    assert doc.doc_id == "doc-1"
    assert doc.title == "Some doc"
    assert doc.connector_type is ConnectorType.WIKIPEDIA
    assert [c.chunk_id for c in doc.chunks] == [c["chunk_id"] for c in chunks]
    assert [c.text for c in doc.chunks] == [c["text"] for c in chunks]


def test_to_source_document_records_the_connector_on_the_document():
    """write_documents used to read a raw connector_type field off the adapted
    dict to populate Document.source_type. The document now says which
    connector it came from, as a ConnectorType, and the write reads that."""
    doc = _to_source_document(_payload_doc(), [], "wikipedia", _acl_ok())
    assert doc.connector_type.value == "wikipedia"


def test_wikipedia_resolves_to_wikipedia_spec():
    from connectors.base import ConnectorType
    from services.incremental_pipeline import _resolve_spec

    doc = _to_source_document(_payload_doc(), [], "wikipedia", _acl_ok())
    assert doc.connector_type is ConnectorType.WIKIPEDIA
    assert _resolve_spec("wikipedia").connector_type is ConnectorType.WIKIPEDIA


def test_to_source_document_falls_back_to_wikipedia_for_an_unknown_connector_type():
    from connectors.base import ConnectorType

    doc = _to_source_document(_payload_doc(), [], "intranet", _acl_ok())
    assert doc.connector_type is ConnectorType.WIKIPEDIA


def test_to_source_document_leaves_dates_empty_rather_than_inventing_them():
    """A thin payload carries no dates; the old code wrote
    content_publish_on_date=None and createddate="" for the same reason."""
    doc = _to_source_document(_payload_doc(), [], "zeus", _acl_ok())
    assert doc.published_at == ""
    assert doc.updated_at == ""


def test_to_source_document_takes_dates_from_the_live_es_document_when_there_is_one():
    doc = _to_source_document(_payload_doc(), [], "wikipedia", _acl_ok(),
                              {"touched": "2026-02-01T00:00:00Z"})
    assert doc.published_at == "2026-02-01"
    assert doc.updated_at == "2026-02-01T00:00:00Z"


def test_to_source_document_prefers_doc_title_over_the_legacy_title_field():
    """The fully-embedded payload's title field is doc_title, not title."""
    doc = _to_source_document(_payload_doc(doc_title="Real Title"), [], "zeus", _acl_ok())
    assert doc.title == "Real Title"


def test_to_source_document_falls_back_to_the_legacy_title_field_when_no_doc_title():
    doc = _to_source_document(_payload_doc(), [], "zeus", _acl_ok())
    assert doc.title == "Some doc"


def test_to_source_document_takes_dates_author_and_object_type_from_doc_when_no_es_doc():
    """Fully-embedded payload, no ES doc fetched at all (es_doc=None): dates,
    author and object_type must come straight off the payload doc itself,
    under the same field names a live ES document would have used."""
    doc = _to_source_document(
        _payload_doc(object_type="article", touched="2026-01-15T00:00:00Z"),
        [], "wikipedia", _acl_ok(), es_doc=None)
    assert doc.published_at == "2026-01-15"
    assert doc.updated_at == "2026-01-15T00:00:00Z"
    assert doc.object_type == "article"
    assert doc.author is None


def _run(coro):
    return asyncio.run(coro)


# --- resolve_watermark (confirmed live-parity finding: search-utility-agents
# PR #679 sets event_timestamp at Kafka PUBLISH time, inside buildGraphEvent --
# a retried publish gets a fresh, later event_timestamp even though the
# underlying edit itself didn't get any newer, which could make an older edit
# look newer than one that published cleanly the first time. The producer
# team confirmed this and added doc.doc_updated_at specifically for this --
# sourced from the same ES lastmodifieddate our own fallback already reads,
# so both ultimately agree; doc_updated_at is preferred when present since it
# costs nothing extra (already in the Kafka payload, no ES round-trip).) ---

def test_resolve_watermark_prefers_doc_updated_at_when_present():
    event = _event(completed_at=1000)
    doc = {"doc_updated_at": "2025-12-18T02:02:49.872Z"}
    # Milliseconds, matching completed_at's own convention (13-digit values
    # like 1784192331000) -- not epoch seconds. Getting the unit wrong here
    # would make every resolved watermark look "older" than an existing
    # saved (millisecond) one regardless of the real dates involved.
    assert resolve_watermark(event, doc, _es_doc()) == 1766023369872


def test_resolve_watermark_falls_back_to_es_doc_lastmodifieddate_when_doc_updated_at_absent():
    event = _event(completed_at=1000)
    es_doc = _es_doc(lastmodifieddate="2025-12-18T02:02:49.872Z")
    assert resolve_watermark(event, {}, es_doc) == 1766023369872


def test_resolve_watermark_falls_back_to_completed_at_when_neither_source_has_a_value():
    event = _event(completed_at=1000)
    assert resolve_watermark(event, {}, _es_doc()) == 1000


def test_resolve_watermark_falls_back_to_completed_at_when_doc_and_es_doc_are_both_none():
    # The delete path: the document may carry nothing usable, and there's
    # nothing in ES left to look up either.
    event = _event(completed_at=1000)
    assert resolve_watermark(event, None, None) == 1000


def test_resolve_watermark_falls_back_on_an_unparseable_doc_updated_at():
    event = _event(completed_at=1000)
    doc = {"doc_updated_at": "not-a-real-date"}
    es_doc = _es_doc(lastmodifieddate="2025-12-18T02:02:49.872Z")
    # An unparseable PRIMARY source falls through to the next one (es_doc),
    # not straight to completed_at -- only exhausting every source lands there.
    assert resolve_watermark(event, doc, es_doc) == 1766023369872


def test_resolve_watermark_falls_back_on_an_unparseable_lastmodifieddate():
    event = _event(completed_at=1000)
    es_doc = _es_doc(lastmodifieddate="not-a-real-date")
    assert resolve_watermark(event, {}, es_doc) == 1000


def test_process_document_prefers_es_doc_lastmodifieddate_for_its_staleness_check():
    """The actual bug this closes: a document whose event.completed_at LOOKS
    stale against the saved watermark must still be processed if the real ES
    lastmodifieddate is genuinely newer -- and vice versa, an event.completed_at
    that looks fresh must still be rejected if lastmodifieddate says the real
    edit is actually older than what's already been applied."""
    repo = MagicMock()
    extractor = MagicMock()
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=ChunkExtractionResult(),
    )
    es_doc = _es_doc(lastmodifieddate="2025-12-18T02:02:49.872Z")  # -> 1766023369872 (ms)
    with patch("services.incremental_pipeline.load_checkpoint",
               return_value={"completed_at": 1766023369872 - 1000}), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"):  # saved watermark is one second older
        _run(process_document(_event(completed_at=1), _payload_doc(), repo, extractor, MagicMock(),
                              MagicMock(), es_doc))
    # event.completed_at=1 alone would have looked stale against nothing here,
    # but the real check is against the resolved (lastmodifieddate-based) value,
    # which is newer than the saved watermark -- so this must NOT be skipped.
    repo.write_documents.assert_called_once()


def test_process_document_skips_when_acl_dropped():
    """Fail-closed drop: a connector_type with no registered ACL strategy falls
    back to passing the document's own tokens through, and a document carrying
    none of those resolves to no ACL at all — which must drop it rather than
    default it to public (see services/acl/builder.py)."""
    repo = MagicMock()
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None):
        _run(process_document(_event(connector_type="not-a-registered-connector"), _payload_doc(),
                              repo, MagicMock(), MagicMock(), MagicMock(), _es_doc()))
    repo.write_documents.assert_not_called()


def test_process_document_skips_when_no_chunks():
    repo = MagicMock()
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None):
        _run(process_document(_event(), _payload_doc(_chunks={"v1": []}), repo, MagicMock(), MagicMock(),
                              MagicMock(), _es_doc()))
    repo.write_documents.assert_not_called()


async def _fake_extract_chunks_batch(prepared_texts, chunk_ids, on_result=None, *, result=None):
    """Stand-in for ExtractionService.extract_chunks_batch that actually invokes
    on_result — _extract_with_cache (core/pipeline.py) populates its cache
    entirely via that callback, not via this function's return value, so a
    plain AsyncMock(return_value=...) silently looks like an empty extraction."""
    result = result or ChunkExtractionResult()
    for cid in chunk_ids:
        if on_result:
            await on_result(cid, result)
    return [result for _ in chunk_ids]


def test_process_document_happy_path_writes_and_advances_watermark():
    repo = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
        relationships=[],
        claims=[ExtractedClaim(subject="Globex", predicate="acquired", object="Initech",
                               statement_type="FACT", temporal_type="STATIC")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint") as save_checkpoint, \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))):
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    repo.write_documents.assert_called_once()
    repo.write_chunks.assert_called_once()
    repo.upsert_entity_incremental.assert_called_once()
    _, kwargs = repo.upsert_entity_incremental.call_args
    # Wikipedia has no site/audience model — every document is public, and the
    # entity inherits that same token (the same builder the bulk path uses).
    assert kwargs["acl"] == ["public"]
    assert kwargs["doc_id"] == "doc-1"
    repo.write_claims.assert_called_once()
    save_checkpoint.assert_called_once_with(
        "incremental_watermark", "tenant-1:doc-1", {"completed_at": 1000},
    )

    # Experiment-parity review #34: chunk-diff looks up this doc_id's existing
    # chunks BEFORE writing the new ones, so it diffs against the OLD state,
    # not the state write_chunks is about to overwrite it with.
    repo.get_document_chunk_rids.assert_called_once_with("doc-1", "tenant-1")
    diff_call_order = repo.method_calls.index(next(c for c in repo.method_calls
                                                    if c[0] == "get_document_chunk_rids"))
    write_docs_call_order = repo.method_calls.index(next(c for c in repo.method_calls
                                                         if c[0] == "write_documents"))
    assert diff_call_order < write_docs_call_order
    # This test's mocked repo has no pre-existing chunks for this doc_id (a
    # brand-new document), so there's nothing to retract.
    repo.retract_chunks.assert_not_called()

    # Code review must-fix #4: tenant_id threaded through every repo write call.
    write_docs_args = repo.write_documents.call_args.args
    assert write_docs_args[1] == "tenant-1"
    _, upsert_kwargs = repo.upsert_entity_incremental.call_args
    assert upsert_kwargs["tenant_id"] == "tenant-1"

    # experiment-parity review #35: degree recomputed for this doc's entities only.
    repo.recompute_degree.assert_called_once()
    degree_args = repo.recompute_degree.call_args.args
    assert degree_args[0] == "tenant-1"
    assert degree_args[1] == [repo.upsert_entity_incremental.return_value]


# --- chunk-diff on update (experiment-parity review #34) ---

def test_process_document_skips_extraction_entirely_when_no_chunk_is_new():
    """Every chunk in this update was already known (same recursive_chunk_id
    as before) -- nothing to extract, but Document/Chunk acl must still get
    refreshed and the watermark must still advance."""
    repo = MagicMock()
    repo.get_document_chunk_rids.return_value = {"c1": "c1"}  # rid falls back to chunk_id
    extractor = MagicMock()
    extractor.extract_chunks_batch = MagicMock(side_effect=AssertionError(
        "extraction must not run when chunk-diff finds nothing new"))

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint") as save_checkpoint:
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    repo.write_documents.assert_called_once()
    repo.write_chunks.assert_called_once()
    repo.retract_chunks.assert_not_called()
    repo.upsert_entity_incremental.assert_not_called()
    repo.write_claims.assert_not_called()
    save_checkpoint.assert_called_once_with(
        "incremental_watermark", "tenant-1:doc-1", {"completed_at": 1000},
    )


def test_process_document_retracts_only_the_chunks_that_actually_dropped():
    """A chunk present in the OLD graph state but absent from the new payload
    (its recursive_chunk_id disappeared) must be retracted individually --
    surviving/kept chunks (still present under the same rid) must not be
    touched at all."""
    repo = MagicMock()
    repo.get_document_chunk_rids.return_value = {"c1": "c1", "rid-gone": "c-old-gone"}
    extractor = MagicMock()
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=ChunkExtractionResult(),
    )

    vector_store = MagicMock()
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"):
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), vector_store, _es_doc()))

    repo.retract_chunks.assert_called_once_with("doc-1", "tenant-1", ["c-old-gone"], vector_store, None)


def test_process_document_extracts_only_genuinely_new_chunks():
    """A payload with one already-known chunk (same rid) and one genuinely
    new chunk must only send the new chunk through extraction -- the known
    one is left completely untouched."""
    repo = MagicMock()
    repo.get_document_chunk_rids.return_value = {"c-old": "c-old"}
    doc = _payload_doc(_chunks={"v1": [
        {"chunk_id": "c-old", "text": "Unchanged text."},
        {"chunk_id": "c-new", "text": "Brand new text."},
    ]})
    extractor = MagicMock()
    seen_chunk_ids = []

    async def _capture(texts, ids, on_result=None):
        seen_chunk_ids.extend(ids)
        return await _fake_extract_chunks_batch(texts, ids, on_result=on_result,
                                                result=ChunkExtractionResult())
    extractor.extract_chunks_batch = _capture

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"):
        _run(process_document(_event(), doc, repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    assert seen_chunk_ids == ["c-new"]
    repo.retract_chunks.assert_not_called()


def test_process_document_writes_the_resolved_acl_onto_the_document():
    """Whatever the connector's strategy resolved is what lands on the Document
    node — Wikipedia's is a single public token, and there is no ES doc to
    derive anything else from."""
    repo = MagicMock()
    extractor = MagicMock()
    extractor.extract_chunks_batch = AsyncMock(return_value=[ChunkExtractionResult()])

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"):
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), es_doc=None))

    args, _ = repo.write_documents.call_args
    assert args[1] == "tenant-1"
    assert args[2] == {"doc-1": ["public"]}


def test_process_document_skips_llm_call_for_already_cached_chunk():
    """Regression test for the missing extraction cache — a chunk already
    extracted (e.g. a redelivered Kafka message reprocessing the same doc)
    must not pay for a second LLM call."""
    repo = MagicMock()
    extractor = MagicMock()
    extractor.extract_chunks_batch = AsyncMock(return_value=[])
    cached_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    ).model_dump()

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={"c1:v1": cached_result}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.EXTRACTION_VERSION", "v1"), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))):
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    # to_extract_ids ended up empty -> extract_chunks_batch is called with an empty list,
    # never actually invoking the LLM for chunk c1 since it was already cached.
    extractor.extract_chunks_batch.assert_not_called()
    repo.upsert_entity_incremental.assert_called_once()


def test_process_document_resolves_entities_concurrently_but_bounded():
    """Code review must-fix #6 (B1): resolve_canonical_name calls for different
    entities in the same document must overlap (not one strict for/await loop),
    but never exceed settings.incremental_entity_resolve_concurrency in flight
    at once."""
    repo = MagicMock()
    extractor = MagicMock()
    entity_names = [f"Entity{i}" for i in range(settings.incremental_entity_resolve_concurrency + 5)]
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name=n, type="ORGANIZATION", description="d") for n in entity_names],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )

    in_flight = 0
    max_in_flight = 0

    async def _fake_resolve(repo, embedder, vector_store, name, entity_type, description, tenant_id):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return name, [0.1]

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name", side_effect=_fake_resolve):
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    assert max_in_flight > 1, "entities should resolve concurrently, not strictly one at a time"
    assert max_in_flight <= settings.incremental_entity_resolve_concurrency
    assert repo.upsert_entity_incremental.call_count == len(entity_names)


def test_process_document_dedups_repeated_entity_names_before_resolving():
    """A raw entity name repeated across chunks must only be resolved once —
    same intent as the old loop's `if entity.name in merge_map: continue`."""
    repo = MagicMock()
    extractor = MagicMock()
    result_c1 = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="d")],
    )
    result_c2 = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="d")],
    )
    doc = _payload_doc(_chunks={"v1": [
        {"chunk_id": "c1", "text": "Globex does things."},
        {"chunk_id": "c2", "text": "Globex does more things."},
    ]})

    async def _fake_extract(prepared_texts, chunk_ids, on_result=None):
        results = [result_c1, result_c2]
        for cid, res in zip(chunk_ids, results):
            if on_result:
                await on_result(cid, res)
        return results

    extractor.extract_chunks_batch = _fake_extract
    resolve_mock = AsyncMock(return_value=("Globex", [0.1]))

    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name", new=resolve_mock):
        _run(process_document(_event(), doc, repo, extractor, MagicMock(), MagicMock(), _es_doc()))

    resolve_mock.assert_called_once()
    repo.upsert_entity_incremental.assert_called_once()


def test_delete_document_delegates_to_repo_retract_document():
    repo = MagicMock()
    repo.retract_document.return_value = {"doc_deleted": True}
    repo.count_documents.return_value = 100
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=1):
        stats = delete_document("doc-1", repo, "tenant-1")
    repo.retract_document.assert_called_once_with("doc-1", "tenant-1", None, None)
    assert stats == {"doc_deleted": True}


# --- update_acl_only (acl_update event, search-utility-agents confirmed) ---

def test_update_acl_only_delegates_to_repo_update_document_acl():
    repo = MagicMock()
    repo.update_document_acl.return_value = {"document_updated": True, "chunks_updated": 3}
    stats = update_acl_only("doc-1", repo, "tenant-1", ["group:site:site-1"])
    repo.update_document_acl.assert_called_once_with("doc-1", "tenant-1", ["group:site:site-1"], None, None)
    assert stats == {"document_updated": True, "chunks_updated": 3}


# --- mass-retraction safety guard (experiment-parity review #9) ---

def test_delete_document_blocked_when_it_would_exceed_the_mass_retraction_fraction():
    """A tenant with 100 docs: the 51st delete attempt this window would push
    it past the default 50% threshold — must block, never call
    retract_document."""
    repo = MagicMock()
    repo.count_documents.return_value = 100
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=51) as inc:
        stats = delete_document("doc-51", repo, "tenant-1")
    repo.retract_document.assert_not_called()
    assert stats["guarded"] is True
    assert stats["attempted_count"] == 51
    assert stats["total_docs"] == 100
    inc.assert_called_once()


def test_delete_document_proceeds_when_under_the_mass_retraction_fraction():
    repo = MagicMock()
    repo.count_documents.return_value = 100
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=11):
        stats = delete_document("doc-11", repo, "tenant-1")
    repo.retract_document.assert_called_once_with("doc-11", "tenant-1", None, None)
    assert stats == {"doc_deleted": True}


# --- People/org graph: a departed employee is marked inactive, never retracted ---

def test_delete_document_marks_a_people_doc_departed_instead_of_retracting():
    repo = MagicMock()
    repo.count_documents.return_value = 100
    repo.doc_object_type.return_value = "people"
    repo.mark_person_departed.return_value = 1
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=1):
        stats = delete_document("doc-99", repo, "tenant-1")
    repo.doc_object_type.assert_called_once_with("doc-99", "tenant-1")
    repo.mark_person_departed.assert_called_once()
    args, kwargs = repo.mark_person_departed.call_args
    assert args[:2] == ("tenant-1", "doc-99")
    assert "when" in kwargs
    repo.retract_document.assert_not_called()
    assert stats == {"departed": 1}


def test_delete_document_retracts_normally_for_a_non_people_doc():
    repo = MagicMock()
    repo.count_documents.return_value = 100
    repo.doc_object_type.return_value = "content"
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=1):
        stats = delete_document("doc-1", repo, "tenant-1")
    repo.mark_person_departed.assert_not_called()
    repo.retract_document.assert_called_once_with("doc-1", "tenant-1", None, None)
    assert stats == {"doc_deleted": True}


def test_delete_document_skips_people_check_when_org_graph_disabled():
    repo = MagicMock()
    repo.count_documents.return_value = 100
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=1), \
         patch("services.incremental_pipeline.settings.people_org_graph_enabled", False):
        stats = delete_document("doc-1", repo, "tenant-1")
    repo.doc_object_type.assert_not_called()
    repo.retract_document.assert_called_once_with("doc-1", "tenant-1", None, None)
    assert stats == {"doc_deleted": True}


def test_mass_retraction_guard_never_trips_below_the_minimum_corpus_floor():
    """A tenant with only 5 documents legitimately clearing all of them isn't
    the runaway-burst scenario this guard protects against."""
    repo = MagicMock()
    repo.count_documents.return_value = 5
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.increment_rolling_window_count") as inc:
        stats = delete_document("doc-1", repo, "tenant-1")
    inc.assert_not_called()  # short-circuits before even touching the counter
    repo.retract_document.assert_called_once()
    assert stats == {"doc_deleted": True}


def test_mass_retraction_guard_disabled_via_settings_skips_the_check_entirely():
    repo = MagicMock()
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.mass_retraction_guard_enabled = False
        stats = delete_document("doc-1", repo, "tenant-1")
    repo.count_documents.assert_not_called()
    repo.retract_document.assert_called_once()
    assert stats == {"doc_deleted": True}


def test_mass_retraction_guard_increments_the_counter_atomically_with_window_and_time():
    """The guard must delegate window-reset-vs-increment entirely to
    increment_rolling_window_count (a single atomic Mongo operation) rather
    than reimplementing a read-then-write version itself, which would
    reintroduce the lost-update race that function exists to close."""
    repo = MagicMock()
    repo.count_documents.return_value = 100
    repo.retract_document.return_value = {"doc_deleted": True}
    with patch("services.incremental_pipeline.increment_rolling_window_count", return_value=1) as inc, \
         patch("services.incremental_pipeline.time") as mock_time:
        mock_time.time.return_value = 12345.0
        delete_document("doc-1", repo, "tenant-1")
    inc.assert_called_once_with("mass_retraction_window", "tenant-1", 3600.0, 12345.0)


# --- People phase on the upsert path: the same three stages bulk runs ---

class _StubPeopleProvider:
    """Stands in for a connector whose source HAS people. Both connectors this
    repo ships deliberately discover nobody (Wikipedia pages have no single
    author, arXiv has no identity directory), so a stub is the only way to
    exercise the orchestrator's side of the three people stages. What a real
    provider does inside enrich is its own tests' business — these only prove
    process_document calls the stages and hands them the right scope."""

    def __init__(self, records: Optional[list[PersonRecord]] = None) -> None:
        self.records = records or []
        self.enrich_calls: list[tuple[list, PeopleContext]] = []

    def observe(self, docs) -> None:
        return None

    async def discover(self, ctx) -> list[PersonRecord]:
        return self.records

    async def enrich(self, records, ctx) -> PeopleResult:
        self.enrich_calls.append((records, ctx))
        return PeopleResult()


def _spec_with_people(provider: _StubPeopleProvider):
    """`spec.people` is a read-only property over `_people`, so swap the backing
    attribute on a real spec rather than faking the whole ConnectorSpec."""
    spec = get_connector("wikipedia")
    spec._people = provider
    return patch("services.incremental_pipeline._resolve_spec", return_value=spec)


def _mint_person(entity_id: str = "ent-person-1"):
    return patch("repository.entity_registry.link_or_mint", return_value={"_id": entity_id})


def test_process_document_materializes_a_keyed_person_from_a_people_event():
    """A person arriving over Kafka gets the identity bulk gives them —
    (connector_type, source_key), what materialize_authorship_edges joins on.
    This path skipped MATERIALIZE, so a new hire landed as a bare name."""
    repo = MagicMock()
    extractor = MagicMock()
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=ChunkExtractionResult(entities=[]),
    )
    provider = _StubPeopleProvider([PersonRecord(
        source_key="hr-9", display_name="Rahul Sharma", email="Rahul.Sharma@Acme.COM",
        attributes={"job_title": "Engineer", "department": "R&D", "manager_id": "hr-1"},
        # Provenance: which of this run's documents the record came from, which
        # is also what _tag_entity_acls attributes the person's own acl from.
        source_doc_ids=["hr-9"],
    )])
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         _spec_with_people(provider), \
         _mint_person(), patch("repository.entity_registry.enrich"), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        _run(process_document(_event(), _payload_doc(doc_id="hr-9"), repo, extractor,
                              MagicMock(), MagicMock(), _es_doc()))
    entries, tenant_id, canon_to_id, _vector_store = repo.write_entities.call_args[0]
    entry = entries["ent-person-1"]
    assert entry["person_fields"]["source_key"] == "hr-9"
    assert entry["person_fields"]["connector_type"] == "wikipedia"
    assert entry["person_fields"]["email"] == "rahul.sharma@acme.com"
    assert entry["name"] == "Rahul Sharma"
    # Keyed by entity_id already, so canon_to_id is identity — and the person's
    # acl is this document's own.
    assert canon_to_id == {"ent-person-1": "ent-person-1"}
    assert entry["acl"] == ["public"]
    assert tenant_id == "tenant-1"


def test_incremental_document_carries_the_live_documents_author_and_editor():
    """The other half of authorship: created_by_id/modified_by_id come from the
    live document via the connector, not the thin Kafka payload. Hard-coded None
    before, so such a document had nothing for an AUTHORED edge to join from."""
    from services.acl.types import AclDecision as _AclDecision
    source_doc = _to_source_document(
        _payload_doc(), [{"chunk_id": "c1", "text": "x"}], "wikipedia",
        _AclDecision(["public"], False, "incremental"), _es_doc(),
    )
    assert source_doc.author is None
    assert source_doc.editor is None


def test_process_document_runs_connector_enrich_over_this_documents_scope():
    """ENRICH is the connector's stage, not repo calls inlined here. What it
    does with the scope is the provider's own tests' business — this proves the
    wiring only: it runs once, over this document's persons plus the entities
    this document resolved, and over this doc_id alone."""
    repo = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )
    repo.upsert_entity_incremental.return_value = "ent-globex"
    provider = _StubPeopleProvider()
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         _spec_with_people(provider), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))
    assert len(provider.enrich_calls) == 1
    _records, ctx = provider.enrich_calls[0]
    assert ctx.tenant_id == "tenant-1"
    assert ctx.entity_ids == ["ent-globex"]
    assert ctx.page_doc_ids == ["doc-1"]
    assert ctx.repo is repo
    # write_entities labels a keyed person as it writes it, so this path no
    # longer relabels afterwards.
    repo.label_person_records.assert_not_called()


def test_process_document_hands_enrich_the_mention_retirement_callback():
    """Parity-audit gap: a provider that merges person mentions only migrates
    Neo4j edges — the Mongo registry row and Qdrant vector for the merged-away
    mention must also be cleaned up, or a later bare mention could re-link to
    the now-nonexistent entity_id and undo the merge. That cleanup is shared,
    so core hands it to ENRICH rather than each provider reimplementing it."""
    repo = MagicMock()
    vector_store = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )
    provider = _StubPeopleProvider()
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         _spec_with_people(provider), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), vector_store, _es_doc()))
    _records, ctx = provider.enrich_calls[0]
    assert ctx.retire_merged_mentions is _retire_merged_person_mentions
    assert ctx.vector_store is vector_store


def test_process_document_materializes_authorship_edges_when_enabled():
    repo = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        # Authorship flags now come from the run's config, not process settings.
        bind_run_config(_run_cfg(build_authorship_edges=True, contributor_entity_types=("PROJECT",)))
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))
    repo.materialize_authorship_edges.assert_called_once_with("tenant-1", ["doc-1"], ("PROJECT",))


def test_process_document_skips_authorship_edges_when_disabled():
    repo = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        bind_run_config(_run_cfg(build_authorship_edges=False))
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))
    repo.materialize_authorship_edges.assert_not_called()


def test_process_document_skips_people_org_graph_when_disabled():
    """The org-graph flag is the connector's own gate inside ENRICH now
    (connectors/zeus/people.py), not a branch in the orchestrator."""
    repo = MagicMock()
    extractor = MagicMock()
    extraction_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION", description="A company")],
    )
    extractor.extract_chunks_batch = lambda texts, ids, on_result=None: _fake_extract_chunks_batch(
        texts, ids, on_result=on_result, result=extraction_result,
    )
    with patch("services.incremental_pipeline.load_checkpoint", return_value=None), \
         patch("services.incremental_pipeline.save_checkpoint"), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("services.incremental_pipeline.resolve_canonical_name",
              new=AsyncMock(return_value=("Globex", [0.1, 0.2]))), \
         patch("services.incremental_pipeline.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        # Real (non-MagicMock) value — process_document bounds entity resolution
        # with asyncio.Semaphore(settings.incremental_entity_resolve_concurrency),
        # which requires an actual int.
        mock_settings.incremental_entity_resolve_concurrency = 10
        _run(process_document(_event(), _payload_doc(), repo, extractor, MagicMock(), MagicMock(), _es_doc()))
    repo.materialize_reporting_edges.assert_not_called()
    repo.find_person_mention_merges.assert_not_called()
