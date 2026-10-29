"""Unit tests for repository/run_registry.py (the Mongo-backed
pointer from a run_id to its per-run Qdrant/Mongo store names)."""
from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import DuplicateKeyError

import repository.run_registry as run_registry
from config.settings import settings
from services.run_context import RunConfig

_CFG = RunConfig(
    embedding_backend="azure", embedding_model="text-embedding-3-large",
    embedding_dimension=1536, consume_upstream_vector=True,
    es_chunk_vector_field="embeddings_openai_large_1536",
    llm_backend="ministral", llm_model="ministral-3b",
    build_authorship_edges=True, contributor_entity_types=("PROJECT",),
    people_org_graph_enabled=True,
)
# The nested sub-document a full (post-RunConfig) doc carries — reused by the
# get_run/get_latest_run fixtures so they exercise the normal (non-legacy) path.
_CFG_DOC = {
    "embedding_backend": "azure", "embedding_model": "text-embedding-3-large",
    "embedding_dimension": 1536, "consume_upstream_vector": True,
    "es_chunk_vector_field": "embeddings_openai_large_1536",
    "llm_backend": "ministral", "llm_model": "ministral-3b",
    "build_authorship_edges": True, "contributor_entity_types": ["PROJECT"],
    "people_org_graph_enabled": True,
}


@pytest.fixture(autouse=True)
def _reset_index_cache():
    # _ensure_index() is cached process-wide (like entity_registry.py's own) —
    # reset it so each test observes create_index being called against its own
    # fake db, instead of only the first test in the run ever calling it.
    run_registry._index_ensured = False
    yield
    run_registry._index_ensured = False


def test_register_run_inserts_a_running_doc_keyed_by_run_id():
    fake_db = MagicMock()
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_id, run_ctx = run_registry.register_run("idx-1", "acme", "confluence", _CFG)

    assert run_id == f"idx-1:acme:confluence:{run_ctx.run_suffix}"
    fake_db["run_registry"].create_index.assert_called_once_with(
        [("tenant_id", 1), ("connector_type", 1), ("run_suffix", 1)], unique=True,
    )
    doc = fake_db["run_registry"].insert_one.call_args[0][0]
    assert doc["_id"] == run_id
    assert doc["status"] == "RUNNING"
    assert doc["tenant_id"] == "acme"
    assert doc["connector_type"] == "confluence"
    assert doc["run_suffix"] == run_ctx.run_suffix
    assert doc["neo4j_database"] == run_ctx.neo4j_database
    assert doc["qdrant_entities_collection"] == run_ctx.qdrant_entities_collection
    assert doc["completed_at"] is None
    # The run's resolved config is persisted so incremental events can reuse it.
    assert doc["run_config"] == _CFG_DOC


def test_register_run_retries_on_suffix_collision():
    fake_db = MagicMock()
    fake_db["run_registry"].insert_one.side_effect = [DuplicateKeyError("dup"), None]
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_id, run_ctx = run_registry.register_run("idx-1", "acme", "confluence", _CFG)
    assert fake_db["run_registry"].insert_one.call_count == 2
    assert run_id.endswith(run_ctx.run_suffix)


def test_register_run_gives_up_after_max_retries():
    fake_db = MagicMock()
    fake_db["run_registry"].insert_one.side_effect = DuplicateKeyError("dup")
    with patch("repository.run_registry._get_db", return_value=fake_db), \
         pytest.raises(RuntimeError):
        run_registry.register_run("idx-1", "acme", "confluence", _CFG)
    assert fake_db["run_registry"].insert_one.call_count == settings.run_registry_max_suffix_retries


def test_get_run_returns_none_when_never_registered():
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = None
    with patch("repository.run_registry._get_db", return_value=fake_db):
        assert run_registry.get_run("nope") is None


def test_get_run_reconstructs_run_context_from_the_stored_doc():
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = {
        "tenant_id": "acme", "connector_type": "confluence", "run_suffix": "ab12",
        "neo4j_database": "g-acme-confluence-ab12",
        "qdrant_entities_collection": "g-acme-confluence-entities-ab12",
        "qdrant_chunks_collection": "g-acme-confluence-chunks-ab12",
        "mongo_collections": {"extraction": "g-acme-confluence-extraction-ab12"},
        "run_config": _CFG_DOC,
    }
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_ctx = run_registry.get_run("idx-1:acme:confluence:ab12")
    fake_db["run_registry"].find_one.assert_called_once_with({"_id": "idx-1:acme:confluence:ab12"})
    assert run_ctx.run_suffix == "ab12"
    assert run_ctx.qdrant_entities_collection == "g-acme-confluence-entities-ab12"
    # The stored config round-trips back into the RunContext.
    assert run_ctx.run_config == _CFG


def test_get_latest_run_queries_by_tenant_and_connector_sorted_newest_first():
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = {
        "tenant_id": "acme", "connector_type": "confluence", "run_suffix": "zz99",
        "neo4j_database": "n", "qdrant_entities_collection": "e", "qdrant_chunks_collection": "c",
        "mongo_collections": {}, "run_config": _CFG_DOC,
    }
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_ctx = run_registry.get_latest_run("acme", "confluence")
    fake_db["run_registry"].find_one.assert_called_once_with(
        {"tenant_id": "acme", "connector_type": "confluence", "status": "COMPLETED"},
        sort=[("created_at", -1)],
    )
    assert run_ctx.run_suffix == "zz99"


def test_legacy_doc_without_run_config_falls_back_to_process_snapshot_and_warns(caplog):
    """A run registered before RunConfig existed has no run_config sub-doc.
    _to_run_context must not KeyError — it fills from a live snapshot of process
    settings (behaviour-preserving) and warns, rather than DLQ'ing the event."""
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = {
        "run_id": "idx-1:acme:confluence:old0", "tenant_id": "acme",
        "connector_type": "confluence", "run_suffix": "old0",
        "neo4j_database": "n", "qdrant_entities_collection": "e", "qdrant_chunks_collection": "c",
        "mongo_collections": {},
        # no "run_config"
    }
    with patch("repository.run_registry._get_db", return_value=fake_db), \
         patch("services.llm_backend.snapshot_run_config", return_value=_CFG) as snap:
        run_ctx = run_registry.get_run("idx-1:acme:confluence:old0")
    snap.assert_called_once()
    assert run_ctx.run_config == _CFG
    assert any("predates RunConfig" in r.message for r in caplog.records)


def test_get_latest_run_returns_none_when_no_completed_run_exists():
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = None
    with patch("repository.run_registry._get_db", return_value=fake_db):
        assert run_registry.get_latest_run("acme", "confluence") is None


def test_get_run_by_suffix_queries_the_pinned_run():
    """runIdSuffix routing: the event names one specific COMPLETED run's
    suffix, resolved via the unique (tenant, connector, run_suffix) index."""
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = {
        "tenant_id": "acme", "connector_type": "zeus", "run_suffix": "ab12",
        "neo4j_database": "n", "qdrant_entities_collection": "e", "qdrant_chunks_collection": "c",
        "mongo_collections": {}, "run_config": _CFG_DOC,
    }
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_ctx = run_registry.get_run_by_suffix("acme", "zeus", "ab12")
    fake_db["run_registry"].find_one.assert_called_once_with(
        {"tenant_id": "acme", "connector_type": "zeus", "run_suffix": "ab12", "status": "COMPLETED"},
    )
    assert run_ctx.run_suffix == "ab12"


def test_get_run_by_suffix_returns_none_when_no_matching_run():
    fake_db = MagicMock()
    fake_db["run_registry"].find_one.return_value = None
    with patch("repository.run_registry._get_db", return_value=fake_db):
        assert run_registry.get_run_by_suffix("acme", "zeus", "nope") is None


def test_mark_run_completed_sets_status_and_completed_at():
    fake_db = MagicMock()
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_registry.mark_run_completed("run-1")
    args, kwargs = fake_db["run_registry"].update_one.call_args
    assert args[0] == {"_id": "run-1"}
    assert args[1]["$set"]["status"] == "COMPLETED"
    assert "completed_at" in args[1]["$set"]


def test_mark_run_failed_sets_status_and_completed_at():
    fake_db = MagicMock()
    with patch("repository.run_registry._get_db", return_value=fake_db):
        run_registry.mark_run_failed("run-1")
    args, kwargs = fake_db["run_registry"].update_one.call_args
    assert args[0] == {"_id": "run-1"}
    assert args[1]["$set"]["status"] == "FAILED"
