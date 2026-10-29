"""Unit tests for core/pipeline.py's pure helpers, and its per-item fault
isolation / failure ledger integration (code review must-fix #13)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from connectors.registry import get_connector
from core.pipeline import (
    DOC_CANON_VERSION,
    _CORPUS_MAP_MERGE_DOC_CHUNK_SIZE,
    _CORPUS_MAP_NAME_CHUNK_SIZE,
    _EDGE_POOL_MAP_CHUNK_SIZE,
    _assert_upstream_vector_space,
    _build_and_write_edges,
    _canonicalize_corpus_with_cache,
    _canonicalize_document_isolated,
    _canonicalize_documents_in_batch,
    _corpus_map_entity_names_chunk_key,
    _corpus_map_merge_chunk_key,
    _edge_pool_map_chunk_key,
    _load_corpus_map_payload,
    _load_edge_key_records,
    _resolve_run_context,
    _retire_merged_person_mentions,
    _save_corpus_map,
    _save_edge_pool_map,
    _tag_entity_acls,
    run_retry_failed_async,
)
from services.edge_pool_service import merge_edge_pools
from services.run_context import RunConfig, RunContext


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# merge_edge_pools (edge overwrite bug: two documents contributing to the same
# (src, rel_type, tgt) edge must combine, not silently replace each other)
# ---------------------------------------------------------------------------

def test_merge_edge_pools_combines_two_documents_contribution_to_the_same_edge():
    target = {
        ("A", "REL", "B"): {
            "descriptions": [{"text": "from doc 1", "chunk_id": "c1"}],
            "chunk_ids": ["c1"], "raw_types": ["rel"], "acl": ["public"],
            "src_type": "X", "tgt_type": "Y",
        },
    }
    source = {
        ("A", "REL", "B"): {
            "descriptions": [{"text": "from doc 2", "chunk_id": "c2"}],
            "chunk_ids": ["c2"], "raw_types": ["related-to"], "acl": ["group:site:s1"],
            "src_type": "X", "tgt_type": "Y",
        },
    }
    merge_edge_pools(target, source)
    merged = target[("A", "REL", "B")]
    assert merged["descriptions"] == [{"text": "from doc 1", "chunk_id": "c1"},
                                      {"text": "from doc 2", "chunk_id": "c2"}]
    assert merged["chunk_ids"] == ["c1", "c2"]
    assert merged["raw_types"] == ["rel", "related-to"]
    assert merged["acl"] == ["group:site:s1", "public"]


def test_merge_edge_pools_inserts_a_new_key_untouched():
    target = {("A", "REL", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": [], "acl": []}}
    source = {("C", "REL", "D"): {"descriptions": [], "chunk_ids": ["c9"], "raw_types": [], "acl": []}}
    merge_edge_pools(target, source)
    assert target[("C", "REL", "D")] == source[("C", "REL", "D")]
    assert target[("A", "REL", "B")] == {"descriptions": [], "chunk_ids": [], "raw_types": [], "acl": []}


def test_merge_edge_pools_pending_and_all_edges_stay_independent_after_shared_first_insert():
    """PR #31 regression: _build_and_write_edges calls merge_edge_pools twice per doc
    -- once into pending_edges, once into all_edges, both from the same doc_edges
    source. A brand-new edge's first insertion must give each target dict its OWN
    copy, or doc 2's merge into pending_edges (mutating the shared object) would
    already be visible when merging doc 2 into all_edges right after, duplicating
    doc 2's description there."""
    pending_edges: dict = {}
    all_edges: dict = {}
    key = ("Alice", "WORKS_AT", "Globex")

    doc1_edges = {key: {"descriptions": [{"text": "Alice joined in 2020"}],
                        "chunk_ids": ["c1"], "raw_types": ["works_at"], "acl": ["public"]}}
    merge_edge_pools(pending_edges, doc1_edges)
    merge_edge_pools(all_edges, doc1_edges)

    doc2_edges = {key: {"descriptions": [{"text": "Alice is on Platform"}],
                        "chunk_ids": ["c2"], "raw_types": ["works_at"], "acl": ["public"]}}
    merge_edge_pools(pending_edges, doc2_edges)
    merge_edge_pools(all_edges, doc2_edges)

    assert pending_edges[key] is not all_edges[key]
    assert [d["text"] for d in pending_edges[key]["descriptions"]] == [
        "Alice joined in 2020", "Alice is on Platform",
    ]
    assert [d["text"] for d in all_edges[key]["descriptions"]] == [
        "Alice joined in 2020", "Alice is on Platform",
    ]


def test_merge_edge_pools_does_not_duplicate_a_repeated_chunk_id():
    """If the same chunk_id somehow appears in both sides, chunk_ids/raw_types/acl
    stay deduplicated (they're unioned as sets) -- only descriptions can repeat,
    since each is a distinct per-chunk entry, not a plain scalar."""
    target = {("A", "REL", "B"): {"descriptions": [], "chunk_ids": ["c1"], "raw_types": ["rel"], "acl": ["public"]}}
    source = {("A", "REL", "B"): {"descriptions": [], "chunk_ids": ["c1"], "raw_types": ["rel"], "acl": ["public"]}}
    merge_edge_pools(target, source)
    merged = target[("A", "REL", "B")]
    assert merged["chunk_ids"] == ["c1"]
    assert merged["raw_types"] == ["rel"]
    assert merged["acl"] == ["public"]


# ---------------------------------------------------------------------------
# _resolve_run_context (mint vs. resume a run's per-run stores)
# ---------------------------------------------------------------------------

_CFG = RunConfig(
    embedding_backend="azure", embedding_model="text-embedding-3-large",
    embedding_dimension=1536, consume_upstream_vector=True,
    es_chunk_vector_field="embeddings_openai_large_1536",
    llm_backend="azure", llm_model="gpt-5.4-nano",
    build_authorship_edges=True, contributor_entity_types=("PROJECT",),
    people_org_graph_enabled=True,
)


def _fake_run_ctx(suffix="ab12"):
    return RunContext(
        tenant_id="acme", connector_type="intranet", run_suffix=suffix,
        neo4j_database=f"g-acme-intranet-{suffix}",
        qdrant_entities_collection=f"g-acme-intranet-entities-{suffix}",
        qdrant_chunks_collection=f"g-acme-intranet-chunks-{suffix}",
        mongo_collections={"extraction": f"g-acme-intranet-extraction-{suffix}"},
        run_config=_CFG,
    )


def test_resolve_run_context_mints_a_new_run_when_no_run_id_given():
    # The new-run branch snapshots the resolved config once, then hands it to
    # register_run — patch the snapshot so the test doesn't probe a real backend.
    with patch("core.pipeline.register_run", return_value=("idx-1:acme:intranet:ab12", _fake_run_ctx())) \
         as register_run, \
         patch("services.llm_backend.snapshot_run_config", return_value=_CFG), \
         patch("core.pipeline.get_run") as get_run:
        run_id, run_ctx = _resolve_run_context(None, "idx-1", "acme", "intranet")
    register_run.assert_called_once_with("idx-1", "acme", "intranet", _CFG)
    get_run.assert_not_called()
    assert run_id == "idx-1:acme:intranet:ab12"
    assert run_ctx.run_suffix == "ab12"


def test_resolve_run_context_resumes_an_existing_run_id_without_minting():
    with patch("core.pipeline.get_run", return_value=_fake_run_ctx("zz99")) as get_run, \
         patch("core.pipeline.register_run") as register_run:
        run_id, run_ctx = _resolve_run_context("idx-1:acme:intranet:zz99", "idx-1", "acme", "intranet")
    get_run.assert_called_once_with("idx-1:acme:intranet:zz99")
    register_run.assert_not_called()
    assert run_id == "idx-1:acme:intranet:zz99"
    assert run_ctx.run_suffix == "zz99"


def test_resolve_run_context_raises_when_resuming_an_unknown_run_id():
    with patch("core.pipeline.get_run", return_value=None), \
         patch("core.pipeline.register_run") as register_run, \
         pytest.raises(ValueError, match="no run_registry entry"):
        _resolve_run_context("some-unknown-run-id", "idx-1", "acme", "intranet")
    register_run.assert_not_called()


# --- _retire_merged_person_mentions (parity-audit gap: registry/vector cleanup on merge) ---

def test_retire_merged_person_mentions_deletes_registry_row_and_vector():
    pairs = [{"m": "ent-mention1", "c": "ent-canonical1"}, {"m": "ent-mention2", "c": "ent-canonical1"}]
    vector_store = MagicMock()
    with patch("core.pipeline.entity_registry") as mock_registry:
        _retire_merged_person_mentions(pairs, vector_store)
    mock_registry.delete_merged.assert_any_call("ent-mention1")
    mock_registry.delete_merged.assert_any_call("ent-mention2")
    assert mock_registry.delete_merged.call_count == 2
    vector_store.delete_entities.assert_called_once_with(["ent-mention1", "ent-mention2"])


def test_retire_merged_person_mentions_empty_pairs_is_a_no_op():
    vector_store = MagicMock()
    with patch("core.pipeline.entity_registry") as mock_registry:
        _retire_merged_person_mentions([], vector_store)
    mock_registry.delete_merged.assert_not_called()
    vector_store.delete_entities.assert_not_called()


def test_retire_merged_person_mentions_skips_vector_cleanup_when_no_vector_store():
    pairs = [{"m": "ent-mention1", "c": "ent-canonical1"}]
    with patch("core.pipeline.entity_registry") as mock_registry:
        _retire_merged_person_mentions(pairs, None)
    mock_registry.delete_merged.assert_called_once_with("ent-mention1")


def test_retire_merged_person_mentions_vector_cleanup_failure_is_non_fatal():
    pairs = [{"m": "ent-mention1", "c": "ent-canonical1"}]
    vector_store = MagicMock()
    vector_store.delete_entities.side_effect = RuntimeError("qdrant down")
    with patch("core.pipeline.entity_registry") as mock_registry:
        _retire_merged_person_mentions(pairs, vector_store)  # must not raise
    mock_registry.delete_merged.assert_called_once_with("ent-mention1")


def test_tag_entity_acls_unions_source_docs_acl_at_entity_level():
    final_entities = {
        "Globex": {"source_docs": ["doc-1", "doc-2"], "descriptions": [], "aliases": []},
    }
    doc_acl_by_doc = {"doc-1": ["public"], "doc-2": ["group:site:s1"]}
    _tag_entity_acls(final_entities, doc_acl_by_doc)
    assert final_entities["Globex"]["acl"] == sorted({"public", "group:site:s1"})


def test_tag_entity_acls_tags_each_description_and_alias_with_its_own_source_acl_not_the_union():
    """Code review must-fix #8: two documents with different ACLs contribute
    descriptions/aliases to the same entity — each entry must carry its OWN
    source's acl, never the entity-level union, and neither entry is dropped
    or blended into the other's token grant."""
    final_entities = {
        "Globex": {
            "source_docs": ["doc-1", "doc-2"],
            "descriptions": [
                {"text": "public blurb", "doc_id": "doc-1"},
                {"text": "private detail", "doc_id": "doc-2"},
            ],
            "aliases": [
                {"name": "Globex Inc.", "doc_id": "doc-1"},
                {"name": "codename-x", "doc_id": "doc-2"},
            ],
        },
    }
    doc_acl_by_doc = {"doc-1": ["public"], "doc-2": ["group:site:s1"]}
    _tag_entity_acls(final_entities, doc_acl_by_doc)

    entity = final_entities["Globex"]
    assert entity["acl"] == sorted({"public", "group:site:s1"})  # entity-level union, unchanged

    descriptions_by_doc = {d["doc_id"]: d for d in entity["descriptions"]}
    assert descriptions_by_doc["doc-1"]["allow_access_control"] == ["public"]
    assert descriptions_by_doc["doc-2"]["allow_access_control"] == ["group:site:s1"]
    assert len(entity["descriptions"]) == 2  # neither dropped

    aliases_by_doc = {a["doc_id"]: a for a in entity["aliases"]}
    assert aliases_by_doc["doc-1"]["allow_access_control"] == ["public"]
    assert aliases_by_doc["doc-2"]["allow_access_control"] == ["group:site:s1"]
    assert len(entity["aliases"]) == 2  # neither dropped


def test_tag_entity_acls_handles_missing_descriptions_and_aliases_gracefully():
    final_entities = {"Globex": {"source_docs": ["doc-1"]}}
    _tag_entity_acls(final_entities, {"doc-1": ["public"]})
    assert final_entities["Globex"]["acl"] == ["public"]
    assert final_entities["Globex"]["descriptions"] == []
    assert final_entities["Globex"]["aliases"] == []


# ---------------------------------------------------------------------------
# Step 2 per-document fault isolation (code review must-fix #13)
# ---------------------------------------------------------------------------

def test_canonicalize_document_isolated_records_failure_and_returns_false_on_error():
    with patch("core.pipeline._canonicalize_document_with_cache", side_effect=RuntimeError("boom")), \
         patch("core.pipeline.record_failure") as record_failure, \
         patch("core.pipeline.clear_failure") as clear_failure:
        ok = asyncio.run(_canonicalize_document_isolated("doc-1", ["c1"], [], "run-1", "idx-1"))

    assert ok is False
    record_failure.assert_called_once()
    args, kwargs = record_failure.call_args
    assert args[0:3] == ("run-1", "doc-1", "doc_canonical")
    assert kwargs.get("index") == "idx-1"
    clear_failure.assert_not_called()


def test_canonicalize_document_isolated_clears_ledger_entry_on_success():
    with patch("core.pipeline._canonicalize_document_with_cache", return_value=None), \
         patch("core.pipeline.record_failure") as record_failure, \
         patch("core.pipeline.clear_failure") as clear_failure:
        ok = asyncio.run(_canonicalize_document_isolated("doc-1", ["c1"], [], "run-1", "idx-1"))

    assert ok is True
    record_failure.assert_not_called()
    clear_failure.assert_called_once_with("run-1", "doc-1", "doc_canonical")


def test_canonicalize_document_isolated_skips_ledger_when_run_id_is_none():
    with patch("core.pipeline._canonicalize_document_with_cache", side_effect=RuntimeError("boom")), \
         patch("core.pipeline.record_failure") as record_failure:
        ok = asyncio.run(_canonicalize_document_isolated("doc-1", ["c1"], [], None, None))
    assert ok is False
    record_failure.assert_not_called()


def test_canonicalize_documents_in_batch_runs_documents_concurrently():
    """Step 2's documents are independent, so the batch must overlap them
    instead of awaiting one before starting the next. Asserted by observing
    that a second document starts before the first has finished."""
    started, order, seen_colls = [], [], []

    async def _slow_canonicalize(doc_id, chunk_ids, doc_results, mongo_collections=None):
        started.append(doc_id)
        seen_colls.append(mongo_collections)
        await asyncio.sleep(0)  # yield: a sequential loop would resume the SAME doc
        order.append((doc_id, list(started)))

    results_by_doc = {"doc-1": [], "doc-2": [], "doc-3": []}
    chunk_ids_by_doc = {"doc-1": ["c1"], "doc-2": ["c2"], "doc-3": ["c3"]}
    colls = {"doc_canonical": "doc_canonical_run_abc"}
    with patch("core.pipeline._canonicalize_document_with_cache", side_effect=_slow_canonicalize), \
         patch("core.pipeline.clear_failure"):
        ok_ids = asyncio.run(_canonicalize_documents_in_batch(
            results_by_doc, chunk_ids_by_doc, "run-1", "idx-1", mongo_collections=colls))

    assert ok_ids == ["doc-1", "doc-2", "doc-3"]  # order preserved for all_doc_ids
    # By the time doc-1 resumed, every document had already started.
    assert order[0][1] == ["doc-1", "doc-2", "doc-3"]
    # This run's per-run collections must reach every document. Dropping them here
    # would checkpoint Step 2 to the shared default while Step 3/4 read the per-run
    # collection — a silent zero-entity run with no error anywhere.
    assert seen_colls == [colls, colls, colls]


def test_canonicalize_documents_in_batch_excludes_only_the_failed_document():
    """One document's failure is isolated and ledgered; its siblings still
    complete and are still returned."""
    async def _fail_doc_2(doc_id, chunk_ids, doc_results, mongo_collections=None):
        if doc_id == "doc-2":
            raise RuntimeError("boom")

    with patch("core.pipeline._canonicalize_document_with_cache", side_effect=_fail_doc_2), \
         patch("core.pipeline.record_failure") as record_failure, \
         patch("core.pipeline.clear_failure"):
        ok_ids = asyncio.run(_canonicalize_documents_in_batch(
            {"doc-1": [], "doc-2": [], "doc-3": []},
            {"doc-1": ["c1"], "doc-2": ["c2"], "doc-3": ["c3"]}, "run-1", "idx-1"))

    assert ok_ids == ["doc-1", "doc-3"]
    assert record_failure.call_args[0][0:3] == ("run-1", "doc-2", "doc_canonical")


def test_canonicalize_documents_in_batch_respects_the_configured_concurrency_cap():
    """Unbounded fan-out would hold every document's extraction results in
    memory at once, against this module's O(batch) memory contract."""
    peak, live = 0, 0

    async def _track(doc_id, chunk_ids, doc_results, mongo_collections=None):
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1

    doc_ids = [f"doc-{i}" for i in range(10)]
    with patch("core.pipeline._canonicalize_document_with_cache", side_effect=_track), \
         patch("core.pipeline.clear_failure"), \
         patch("core.pipeline.settings") as mock_settings:
        mock_settings.doc_canonicalization_concurrency = 3
        asyncio.run(_canonicalize_documents_in_batch(
            {d: [] for d in doc_ids}, {d: [d] for d in doc_ids}, None, None))

    assert peak == 3


# ---------------------------------------------------------------------------
# Step 4 Neo4j write-flush fault isolation (code review must-fix #13c —
# found on re-review 2026-07-16: _flush_pending had NO try/except at all, so
# one bad row in a batched write raised straight out and aborted the entire
# run, discarding every document processed after that flush point.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# edge_pool_map sharding — one BSON doc with all edge_keys exceeds Mongo's
# 16MB limit at ~100k edges (DocumentTooLarge on the final Step 4 checkpoint).
# ---------------------------------------------------------------------------

def test_save_edge_pool_map_shards_edge_keys_and_keeps_root_thin():
    saved: dict = {}

    def _save(collection, key, data):
        saved[key] = data

    n = _EDGE_POOL_MAP_CHUNK_SIZE * 2 + 7
    records = [{"key": f"k{i}", "tuple": [f"s{i}", "REL", f"t{i}"]} for i in range(n)]
    with patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.delete_checkpoint"):
        _save_edge_pool_map("edge_pool_map", "base", records, claim_count=3, mention_count=4)

    root = saved["base"]
    assert "edge_keys" not in root
    assert root == {
        "edge_key_chunk_count": 3,
        "edge_count": n,
        "claim_count": 3,
        "mention_count": 4,
    }
    assert len(saved[_edge_pool_map_chunk_key("base", 0)]["edge_keys"]) == _EDGE_POOL_MAP_CHUNK_SIZE
    assert len(saved[_edge_pool_map_chunk_key("base", 1)]["edge_keys"]) == _EDGE_POOL_MAP_CHUNK_SIZE
    assert len(saved[_edge_pool_map_chunk_key("base", 2)]["edge_keys"]) == 7
    assert all(len(v.get("edge_keys", [])) <= _EDGE_POOL_MAP_CHUNK_SIZE
               for k, v in saved.items() if k != "base")


def test_save_edge_pool_map_empty_pool_writes_zero_chunks():
    saved: dict = {}

    def _save(collection, key, data):
        saved[key] = data

    with patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.delete_checkpoint"):
        _save_edge_pool_map("edge_pool_map", "base", [], claim_count=0, mention_count=0)

    assert saved == {
        "base": {
            "edge_key_chunk_count": 0,
            "edge_count": 0,
            "claim_count": 0,
            "mention_count": 0,
        },
    }


def test_load_edge_key_records_reads_chunked_map():
    records = [{"key": f"k{i}", "tuple": [f"s{i}", "REL", f"t{i}"]} for i in range(5)]
    store = {
        "base": {"edge_key_chunk_count": 2, "claim_count": 0, "mention_count": 0},
        _edge_pool_map_chunk_key("base", 0): {"edge_keys": records[:3]},
        _edge_pool_map_chunk_key("base", 1): {"edge_keys": records[3:]},
    }

    def _load(collection, key):
        return store.get(key)

    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        assert _load_edge_key_records(store["base"], "edge_pool_map", "base") == records


def test_load_edge_key_records_supports_legacy_inline_edge_keys():
    legacy = {"edge_keys": [{"key": "abc", "tuple": ["A", "REL", "B"]}],
              "claim_count": 1, "mention_count": 0}
    assert _load_edge_key_records(legacy, "edge_pool_map", "base") == legacy["edge_keys"]


def test_load_edge_key_records_returns_none_when_a_chunk_is_missing():
    store = {"base": {"edge_key_chunk_count": 2, "claim_count": 0, "mention_count": 0},
             _edge_pool_map_chunk_key("base", 0): {"edge_keys": [{"key": "a", "tuple": ["A", "R", "B"]}]}}

    def _load(collection, key):
        return store.get(key)

    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        assert _load_edge_key_records(store["base"], "edge_pool_map", "base") is None


def test_save_edge_pool_map_successful_rewrite_replaces_existing_root_and_chunks():
    old_records = [{"key": "old0", "tuple": ["A", "R", "B"]},
                   {"key": "old1", "tuple": ["C", "R", "D"]}]
    new_records = [{"key": "new0", "tuple": ["X", "R", "Y"]}]
    store = {
        "base": {"edge_key_chunk_count": 2, "edge_count": 2, "claim_count": 0, "mention_count": 0},
        _edge_pool_map_chunk_key("base", 0): {"edge_keys": [old_records[0]]},
        _edge_pool_map_chunk_key("base", 1): {"edge_keys": [old_records[1]]},
    }

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        _save_edge_pool_map("edge_pool_map", "base", new_records, claim_count=1, mention_count=2)

    assert store["base"] == {
        "edge_key_chunk_count": 1,
        "edge_count": 1,
        "claim_count": 1,
        "mention_count": 2,
    }
    assert store[_edge_pool_map_chunk_key("base", 0)]["edge_keys"] == new_records
    assert _edge_pool_map_chunk_key("base", 1) not in store  # trailing old chunk cleaned up
    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        assert _load_edge_key_records(store["base"], "edge_pool_map", "base") == new_records


def test_save_edge_pool_map_invalidates_root_before_chunk_rewrite_failure():
    """Rewrite crash mid-chunk must not leave the old root pointing at mixed shards."""
    store = {
        "base": {"edge_key_chunk_count": 2, "edge_count": 2, "claim_count": 0, "mention_count": 0},
        _edge_pool_map_chunk_key("base", 0): {"edge_keys": [{"key": "old0", "tuple": ["A", "R", "B"]}]},
        _edge_pool_map_chunk_key("base", 1): {"edge_keys": [{"key": "old1", "tuple": ["C", "R", "D"]}]},
    }
    # Force two chunks so the first can succeed and the second can fail after overwrite.
    new_records = [{"key": f"n{i}", "tuple": [f"S{i}", "R", f"T{i}"]}
                   for i in range(_EDGE_POOL_MAP_CHUNK_SIZE + 1)]
    writes = {"chunks": 0}

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        if ":chunk:" in key:
            writes["chunks"] += 1
            if writes["chunks"] >= 2:
                raise RuntimeError("chunk write failed")
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        with pytest.raises(RuntimeError, match="chunk write failed"):
            _save_edge_pool_map(
                "edge_pool_map", "base", new_records, claim_count=0, mention_count=0,
            )

    assert "base" not in store  # root invalidated; warm-start must miss
    # First chunk was overwritten with new data — without root invalidation this
    # would mix with the remaining old chunk under the stale root.
    assert store[_edge_pool_map_chunk_key("base", 0)]["edge_keys"][0]["key"] == "n0"
    assert store[_edge_pool_map_chunk_key("base", 1)]["edge_keys"][0]["key"] == "old1"
    # Caller's cache check: missing root => miss (loaders are never consulted).
    assert store.get("base") is None


def test_save_edge_pool_map_first_time_write_failure_leaves_cache_miss():
    store: dict = {}

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        if ":chunk:" in key:
            raise RuntimeError("first-time chunk write failed")
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        with pytest.raises(RuntimeError, match="first-time chunk write failed"):
            _save_edge_pool_map(
                "edge_pool_map", "base",
                [{"key": "k0", "tuple": ["A", "R", "B"]}],
                claim_count=0, mention_count=0,
            )

    assert "base" not in store
    # Caller treats missing root as a miss (same as load_checkpoint returning None).
    assert store.get("base") is None


def test_build_and_write_edges_persists_chunked_map_not_inline_edge_keys():
    repo = MagicMock()
    fake_edges = ({("A", "REL", "B"): _fake_edge()}, [{"claim": 1}], [("c1", "e1", "T")])
    all_saves: list = []
    bulk_saves: list = []

    def _save_all(collection, key, data):
        all_saves.append((collection, key, data))

    def _bulk_save(collection, items, batch_size=None):
        bulk_saves.append((collection, list(items)))
        for key, data in items:
            all_saves.append((collection, key, data))

    with patch("core.pipeline.load_checkpoint", side_effect=_checkpoint_side_effect), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint", side_effect=_save_all), \
         patch("core.pipeline.save_checkpoints_batch", side_effect=_bulk_save), \
         patch("core.pipeline.delete_checkpoint"), \
         patch("core.pipeline.build_edge_pool", return_value=fake_edges), \
         patch("core.pipeline.record_failure"), \
         patch("core.pipeline.clear_failure"), \
         patch("core.pipeline._corpus_key", return_value="corpus"), \
         patch("core.pipeline.DOC_CANON_VERSION", "dv"), \
         patch("core.pipeline.GRAPH_CANON_VERSION", "gv"), \
         patch("core.pipeline.EDGE_POOL_VERSION", "ev"):
        asyncio.run(_build_and_write_edges(
            ["doc-1"], {}, {}, {}, repo, "tenant-1", run_id="run-1", index="idx-1",
        ))

    assert len(bulk_saves) == 3  # edges, claims, mentions in parallel jobs
    bulk_by_coll = {c: items for c, items in bulk_saves}
    assert any("edge_pool_edges" in c for c in bulk_by_coll)
    assert any("edge_pool_claims" in c for c in bulk_by_coll)
    assert any("edge_pool_mentions" in c for c in bulk_by_coll)

    map_saves = [(k, d) for (c, k, d) in all_saves if "edge_pool_map" in c]
    assert map_saves, f"expected edge_pool_map saves, got: {all_saves!r}"
    root_saves = [(k, d) for (k, d) in map_saves if ":chunk:" not in k]
    chunk_saves = [(k, d) for (k, d) in map_saves if ":chunk:" in k]
    assert len(root_saves) == 1
    _root_key, root_data = root_saves[0]
    assert "edge_keys" not in root_data
    assert root_data["edge_key_chunk_count"] == 1
    assert root_data["edge_count"] == 1
    assert len(chunk_saves) == 1
    assert chunk_saves[0][1]["edge_keys"][0]["tuple"] == ["A", "REL", "B"]


def _checkpoint_side_effect(collection, key):
    if collection == "edge_pool_map":
        return None  # skip the whole-corpus cache path
    if collection == "doc_canonical":
        return {"chunk_ids": ["c1"], "merge_map": {}}
    return None


def _fake_edge() -> dict:
    """One edge in the exact shape build_edge_pool returns. All four list fields
    matter — _merge_edges unions them across contributing batches."""
    return {"descriptions": [], "chunk_ids": [], "raw_types": [], "acl": [],
            "src_type": "X", "tgt_type": "Y"}

def test_flush_pending_records_failure_and_continues_when_neo4j_write_raises():
    repo = MagicMock()
    repo.write_relationships.side_effect = RuntimeError("Neo4j 500")
    fake_edges = ({("A", "REL", "B"): _fake_edge()}, [], [])

    with patch("core.pipeline.load_checkpoint", side_effect=_checkpoint_side_effect), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.save_checkpoints_batch"), \
         patch("core.pipeline.delete_checkpoint"), \
         patch("core.pipeline.build_edge_pool", return_value=fake_edges), \
         patch("core.pipeline.record_failure") as record_failure, \
         patch("core.pipeline.clear_failure") as clear_failure:
        # Must not raise — a Neo4j write failure is recorded, not propagated.
        edges, claims, mentions = asyncio.run(_build_and_write_edges(
            ["doc-1", "doc-2"], {}, {}, {}, repo, "tenant-1", run_id="run-1", index="idx-1",
        ))

    repo.write_relationships.assert_called_once()
    repo.write_claims.assert_not_called()  # never reached — write_relationships raised first
    # edge_pool step succeeded for both docs (it's build_edge_pool that's mocked to succeed;
    # only the WRITE fails) — filter to the neo4j_write step specifically, since both docs
    # also get a legitimate clear_failure("edge_pool") call from the unrelated Step 4 build stage.
    neo4j_write_failures = [c for c in record_failure.call_args_list if c.args[2] == "neo4j_write"]
    assert {c.args[1] for c in neo4j_write_failures} == {"doc-1", "doc-2"}
    neo4j_write_clears = [c for c in clear_failure.call_args_list if c.args[2] == "neo4j_write"]
    assert neo4j_write_clears == []
    # The build_edge_pool result is still returned for logging even though the write failed.
    # Both docs produce the same edge key here, so this also proves _merge_edges combines
    # them instead of the second doc's dict.update() silently replacing the first's.
    assert edges == {("A", "REL", "B"): {"src_type": "X", "tgt_type": "Y",
                                         "descriptions": [], "chunk_ids": [], "raw_types": [], "acl": []}}


def test_flush_pending_clears_ledger_entries_on_successful_write():
    repo = MagicMock()
    fake_edges = ({("A", "REL", "B"): _fake_edge()}, [], [])

    with patch("core.pipeline.load_checkpoint", side_effect=_checkpoint_side_effect), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.save_checkpoints_batch"), \
         patch("core.pipeline.delete_checkpoint"), \
         patch("core.pipeline.build_edge_pool", return_value=fake_edges), \
         patch("core.pipeline.record_failure") as record_failure, \
         patch("core.pipeline.clear_failure") as clear_failure:
        asyncio.run(_build_and_write_edges(
            ["doc-1"], {}, {}, {}, repo, "tenant-1", run_id="run-1", index="idx-1",
        ))

    repo.write_relationships.assert_called_once()
    record_failure.assert_not_called()
    neo4j_write_clears = [c for c in clear_failure.call_args_list if c.args[2] == "neo4j_write"]
    assert neo4j_write_clears == [call("run-1", "doc-1", "neo4j_write")]


def test_flush_pending_skips_ledger_entirely_without_a_run_id():
    repo = MagicMock()
    repo.write_relationships.side_effect = RuntimeError("Neo4j 500")
    fake_edges = ({("A", "REL", "B"): _fake_edge()}, [], [])

    with patch("core.pipeline.load_checkpoint", side_effect=_checkpoint_side_effect), \
         patch("core.pipeline.load_checkpoints_batch", return_value={}), \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.save_checkpoints_batch"), \
         patch("core.pipeline.delete_checkpoint"), \
         patch("core.pipeline.build_edge_pool", return_value=fake_edges), \
         patch("core.pipeline.record_failure") as record_failure:
        asyncio.run(_build_and_write_edges(["doc-1"], {}, {}, {}, repo, "tenant-1"))
    record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# --retry-failed (code review must-fix #13)
# ---------------------------------------------------------------------------

def test_run_retry_failed_async_returns_none_when_no_failures():
    with patch("core.pipeline.list_failures", return_value=[]), \
         patch("core.pipeline.run_bulk_async") as run_bulk_async:
        result = asyncio.run(run_retry_failed_async("run-1"))
    assert result is None
    run_bulk_async.assert_not_called()


def test_run_retry_failed_async_returns_none_for_corpus_only_failures():
    """A corpus-level failure (Step 3/5's "__corpus__" sentinel) isn't tied to
    a doc_id, so a targeted retry can't isolate it — must not silently call
    run_bulk_async with an empty doc filter."""
    failures = [{"run_id": "run-1", "item_id": "__corpus__", "step": "corpus_canonicalization",
                "error_class": "transient", "message": "boom", "index": "idx-1"}]
    with patch("core.pipeline.list_failures", return_value=failures), \
         patch("core.pipeline.run_bulk_async") as run_bulk_async:
        result = asyncio.run(run_retry_failed_async("run-1"))
    assert result is None
    run_bulk_async.assert_not_called()


def test_run_retry_failed_async_queries_only_failed_doc_ids():
    failures = [
        {"run_id": "run-1", "item_id": "doc-1", "step": "extraction",
         "error_class": "transient", "message": "boom", "index": "idx-1"},
        {"run_id": "run-1", "item_id": "doc-2", "step": "doc_canonical",
         "error_class": "permanent", "message": "bad shape", "index": "idx-1"},
    ]
    with patch("core.pipeline.list_failures", return_value=failures), \
         patch("core.pipeline.run_bulk_async") as run_bulk_async:
        asyncio.run(run_retry_failed_async("run-1", tenant_id="acme-corp",
                                            spec=get_connector("wikipedia")))

    run_bulk_async.assert_called_once()
    _, kwargs = run_bulk_async.call_args
    assert kwargs["index"] == "idx-1"
    assert kwargs["run_id"] == "run-1"
    assert kwargs["tenant_id"] == "acme-corp"
    assert sorted(kwargs["query"]["pageids"]) == ["doc-1", "doc-2"]


def test_run_retry_failed_async_still_retries_docs_alongside_corpus_level_failures():
    """Mixed ledger: doc-level failures ARE re-driven even when corpus-level
    failures also exist for the same run — only a pure corpus-only ledger
    short-circuits (see test above)."""
    failures = [
        {"run_id": "run-1", "item_id": "doc-1", "step": "extraction",
         "error_class": "transient", "message": "boom", "index": "idx-1"},
        {"run_id": "run-1", "item_id": "__corpus__", "step": "entity_write",
         "error_class": "transient", "message": "neo4j down", "index": "idx-1"},
    ]
    with patch("core.pipeline.list_failures", return_value=failures), \
         patch("core.pipeline.run_bulk_async") as run_bulk_async:
        asyncio.run(run_retry_failed_async("run-1", spec=get_connector("wikipedia")))
    run_bulk_async.assert_called_once()
    _, kwargs = run_bulk_async.call_args
    assert kwargs["query"]["pageids"] == ["doc-1"]


# --- _canonicalize_corpus_with_cache PERSON split/merge (experiment-parity review #8, Phase 2) ---

def test_canonicalize_corpus_with_cache_splits_person_before_clustering_and_merges_after():
    """PERSON entries must never reach canonicalize_corpus (the normal
    cross-document name clustering) -- confirms they're stripped from its
    input, and confirms the separately-resolved person entity/merge-map
    entries land in the final merged output."""
    checkpoints = [
        (f"doc-1:{DOC_CANON_VERSION}",
         {"canonical_entities": {"Globex": {"type": "ORGANIZATION", "description": "A company"},
                                 "Jane Doe": {"type": "PERSON", "description": "an engineer"}},
          "merge_map": {"Globex": "Globex", "Jane Doe": "Jane Doe"}}),
    ]
    person_final_entities = {"ent_person1": {"type": "PERSON", "name": "Jane Doe",
                                             "source_docs": ["doc-1"], "descriptions": [], "aliases": []}}
    person_merge_map = {"doc-1": {"Jane Doe": "ent_person1"}}

    with patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.iter_checkpoints", return_value=checkpoints), \
         patch("core.pipeline.resolve_persons_for_corpus",
              new=AsyncMock(return_value=(
                  {"doc-1": {"Globex": {"type": "ORGANIZATION", "description": "A company"}}},
                  person_final_entities, person_merge_map,
              ))) as resolve_persons, \
         patch("core.pipeline.canonicalize_corpus",
              new=AsyncMock(return_value=(
                  {"Globex": {"type": "ORGANIZATION", "source_docs": ["doc-1"]}},
                  {"doc-1": {"Globex": "Globex"}},
              ))) as canon_corpus, \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.save_checkpoints_batch"), \
         patch("core.pipeline._save_corpus_map"):
        final_entities, final_merge_map_by_doc = _run(
            _canonicalize_corpus_with_cache(["doc-1"], "tenant-1", {})
        )

    # canonicalize_corpus never saw the PERSON entity.
    corpus_input = canon_corpus.call_args.args[0]
    assert "Jane Doe" not in corpus_input["doc-1"][0]
    assert "Globex" in corpus_input["doc-1"][0]
    resolve_persons.assert_called_once()

    # Merged output has both.
    assert "Globex" in final_entities
    assert "ent_person1" in final_entities
    assert final_merge_map_by_doc["doc-1"]["Globex"] == "Globex"
    assert final_merge_map_by_doc["doc-1"]["Jane Doe"] == "ent_person1"


def test_canonicalize_corpus_with_cache_passes_materialized_persons_through():
    """What the corpus stage is handed is MATERIALIZE's already-minted entries,
    not a raw people payload — core/ never holds one now."""
    materialized = {"ent_1": {"type": "PERSON", "name": "Jane Doe", "source_docs": ["doc-people-1"],
                              "descriptions": [], "aliases": [],
                              "person_fields": {"source_key": "user-99", "connector_type": "zeus"}}}
    with patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.iter_checkpoints", return_value=[]), \
         patch("core.pipeline.resolve_persons_for_corpus",
              new=AsyncMock(return_value=({}, {}, {}))) as resolve_persons, \
         patch("core.pipeline.canonicalize_corpus", new=AsyncMock(return_value=({}, {}))), \
         patch("core.pipeline.save_checkpoint"), \
         patch("core.pipeline.save_checkpoints_batch"), \
         patch("core.pipeline._save_corpus_map"):
        _run(_canonicalize_corpus_with_cache([], "tenant-1", materialized))
    resolve_persons.assert_called_once_with({}, materialized, "tenant-1")


# ---------------------------------------------------------------------------
# corpus_map sharding — one BSON doc with all entity_names + merge maps
# exceeds Mongo's 16MB limit at Confluence scale (DocumentTooLarge on the
# final Step 3 checkpoint).
# ---------------------------------------------------------------------------

def test_save_corpus_map_shards_names_and_merge_maps_and_keeps_root_thin():
    saved: dict = {}

    def _save(collection, key, data):
        saved[key] = data

    n_names = _CORPUS_MAP_NAME_CHUNK_SIZE * 2 + 3
    names = [f"Entity{i}" for i in range(n_names)]
    n_docs = _CORPUS_MAP_MERGE_DOC_CHUNK_SIZE + 5
    merge = {f"doc-{i}": {f"raw-{i}": f"Entity{i}"} for i in range(n_docs)}

    with patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.delete_checkpoint"):
        _save_corpus_map("corpus_map", "base", names, merge)

    root = saved["base"]
    assert "entity_names" not in root
    assert "final_merge_map_by_doc" not in root
    assert root == {
        "entity_name_chunk_count": 3,
        "merge_map_chunk_count": 2,
        "entity_count": n_names,
        "doc_count": n_docs,
    }
    assert len(saved[_corpus_map_entity_names_chunk_key("base", 0)]["entity_names"]) == _CORPUS_MAP_NAME_CHUNK_SIZE
    assert len(saved[_corpus_map_entity_names_chunk_key("base", 2)]["entity_names"]) == 3
    assert len(saved[_corpus_map_merge_chunk_key("base", 0)]["final_merge_map_by_doc"]) == _CORPUS_MAP_MERGE_DOC_CHUNK_SIZE
    assert len(saved[_corpus_map_merge_chunk_key("base", 1)]["final_merge_map_by_doc"]) == 5


def test_save_corpus_map_empty_writes_zero_chunks():
    saved: dict = {}

    def _save(collection, key, data):
        saved[key] = data

    with patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.load_checkpoint", return_value=None), \
         patch("core.pipeline.delete_checkpoint"):
        _save_corpus_map("corpus_map", "base", [], {})

    assert saved == {
        "base": {
            "entity_name_chunk_count": 0,
            "merge_map_chunk_count": 0,
            "entity_count": 0,
            "doc_count": 0,
        },
    }


def test_load_corpus_map_payload_reads_chunked_map():
    names = [f"E{i}" for i in range(5)]
    merge = {"doc-1": {"a": "E0"}, "doc-2": {"b": "E1"}}
    store = {
        "base": {
            "entity_name_chunk_count": 2,
            "merge_map_chunk_count": 2,
            "entity_count": 5,
            "doc_count": 2,
        },
        _corpus_map_entity_names_chunk_key("base", 0): {"entity_names": names[:3]},
        _corpus_map_entity_names_chunk_key("base", 1): {"entity_names": names[3:]},
        _corpus_map_merge_chunk_key("base", 0): {"final_merge_map_by_doc": {"doc-1": merge["doc-1"]}},
        _corpus_map_merge_chunk_key("base", 1): {"final_merge_map_by_doc": {"doc-2": merge["doc-2"]}},
    }

    def _load(collection, key):
        return store.get(key)

    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        got_names, got_merge = _load_corpus_map_payload(store["base"], "corpus_map", "base")
    assert got_names == names
    assert got_merge == merge


def test_load_corpus_map_payload_supports_legacy_inline_fields():
    legacy = {
        "entity_names": ["A", "B"],
        "final_merge_map_by_doc": {"doc-1": {"A": "A"}},
    }
    assert _load_corpus_map_payload(legacy, "corpus_map", "base") == (
        ["A", "B"], {"doc-1": {"A": "A"}},
    )


def test_load_corpus_map_payload_returns_none_when_a_chunk_is_missing():
    store = {
        "base": {
            "entity_name_chunk_count": 2,
            "merge_map_chunk_count": 1,
            "entity_count": 2,
            "doc_count": 1,
        },
        _corpus_map_entity_names_chunk_key("base", 0): {"entity_names": ["A"]},
    }

    def _load(collection, key):
        return store.get(key)

    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        assert _load_corpus_map_payload(store["base"], "corpus_map", "base") is None


def test_save_corpus_map_successful_rewrite_replaces_existing_root_and_chunks():
    store = {
        "base": {
            "entity_name_chunk_count": 2,
            "merge_map_chunk_count": 2,
            "entity_count": 2,
            "doc_count": 2,
        },
        _corpus_map_entity_names_chunk_key("base", 0): {"entity_names": ["OldA"]},
        _corpus_map_entity_names_chunk_key("base", 1): {"entity_names": ["OldB"]},
        _corpus_map_merge_chunk_key("base", 0): {"final_merge_map_by_doc": {"doc-1": {"a": "OldA"}}},
        _corpus_map_merge_chunk_key("base", 1): {"final_merge_map_by_doc": {"doc-2": {"b": "OldB"}}},
    }
    new_names = ["NewX"]
    new_merge = {"doc-9": {"x": "NewX"}}

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        _save_corpus_map("corpus_map", "base", new_names, new_merge)

    assert store["base"] == {
        "entity_name_chunk_count": 1,
        "merge_map_chunk_count": 1,
        "entity_count": 1,
        "doc_count": 1,
    }
    assert store[_corpus_map_entity_names_chunk_key("base", 0)]["entity_names"] == new_names
    assert store[_corpus_map_merge_chunk_key("base", 0)]["final_merge_map_by_doc"] == new_merge
    assert _corpus_map_entity_names_chunk_key("base", 1) not in store
    assert _corpus_map_merge_chunk_key("base", 1) not in store
    with patch("core.pipeline.load_checkpoint", side_effect=_load):
        assert _load_corpus_map_payload(store["base"], "corpus_map", "base") == (new_names, new_merge)


def test_save_corpus_map_invalidates_root_before_chunk_rewrite_failure():
    """Rewrite crash mid-chunk must not leave the old root pointing at mixed shards."""
    store = {
        "base": {
            "entity_name_chunk_count": 2,
            "merge_map_chunk_count": 1,
            "entity_count": 2,
            "doc_count": 1,
        },
        _corpus_map_entity_names_chunk_key("base", 0): {"entity_names": ["OldA"]},
        _corpus_map_entity_names_chunk_key("base", 1): {"entity_names": ["OldB"]},
        _corpus_map_merge_chunk_key("base", 0): {"final_merge_map_by_doc": {"doc-1": {"a": "OldA"}}},
    }
    # Enough names for two entity_names chunks so the first can succeed and the
    # second can fail after a partial overwrite.
    new_names = [f"N{i}" for i in range(_CORPUS_MAP_NAME_CHUNK_SIZE + 1)]
    new_merge = {"doc-9": {"x": "N0"}}
    writes = {"name_chunks": 0}

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        if ":entity_names:" in key:
            writes["name_chunks"] += 1
            if writes["name_chunks"] >= 2:
                raise RuntimeError("corpus chunk write failed")
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        with pytest.raises(RuntimeError, match="corpus chunk write failed"):
            _save_corpus_map("corpus_map", "base", new_names, new_merge)

    assert "base" not in store  # root invalidated; warm-start must miss
    assert store[_corpus_map_entity_names_chunk_key("base", 0)]["entity_names"][0] == "N0"
    assert store[_corpus_map_entity_names_chunk_key("base", 1)]["entity_names"] == ["OldB"]
    assert store.get("base") is None


def test_save_corpus_map_first_time_write_failure_leaves_cache_miss():
    store: dict = {}

    def _load(collection, key):
        return store.get(key)

    def _save(collection, key, data):
        if ":entity_names:" in key or ":merge_map:" in key:
            raise RuntimeError("first-time corpus chunk write failed")
        store[key] = data

    def _delete(collection, key):
        store.pop(key, None)

    with patch("core.pipeline.load_checkpoint", side_effect=_load), \
         patch("core.pipeline.save_checkpoint", side_effect=_save), \
         patch("core.pipeline.delete_checkpoint", side_effect=_delete):
        with pytest.raises(RuntimeError, match="first-time corpus chunk write failed"):
            _save_corpus_map("corpus_map", "base", ["A"], {"doc-1": {"a": "A"}})

    assert "base" not in store
    assert store.get("base") is None


# ---------------------------------------------------------------------------
# _assert_upstream_vector_space (the consume_upstream_vector startup gate)
# ---------------------------------------------------------------------------

def test_assert_upstream_vector_space_refuses_the_flag():
    """wikipedia and arxiv do not ship precomputed vectors."""
    with pytest.raises(ValueError, match="consume_upstream_vector"):
        _assert_upstream_vector_space(MagicMock(), "idx", None, _CFG)
