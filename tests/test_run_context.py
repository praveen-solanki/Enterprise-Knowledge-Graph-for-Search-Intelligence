"""Unit tests for services/run_context.py (bundling every per-run Neo4j/
Qdrant/Mongo store name derived from services/run_naming.py, plus the run's
pinned RunConfig)."""
from services.run_context import (
    MONGO_CHECKPOINT_KINDS,
    RunConfig,
    active_run_config,
    bind_run_config,
    new_run_context,
)

_CFG = RunConfig(
    embedding_backend="azure", embedding_model="text-embedding-3-large",
    embedding_dimension=1536, consume_upstream_vector=True,
    es_chunk_vector_field="embeddings_openai_large_1536",
    llm_backend="ministral", llm_model="ministral-3b",
    build_authorship_edges=True, contributor_entity_types=("PROJECT",),
    people_org_graph_enabled=True,
)


def _new():
    return new_run_context("acme", "confluence", _CFG)


def test_new_run_context_derives_neo4j_database_from_tenant_and_connector():
    run_ctx = _new()
    assert run_ctx.neo4j_database == f"g-acme-confluence-{run_ctx.run_suffix}"


def test_new_run_context_derives_qdrant_collections_from_tenant_and_connector():
    run_ctx = _new()
    assert run_ctx.qdrant_entities_collection == f"g-acme-confluence-entities-{run_ctx.run_suffix}"
    assert run_ctx.qdrant_chunks_collection == f"g-acme-confluence-chunks-{run_ctx.run_suffix}"


def test_new_run_context_derives_every_mongo_checkpoint_collection():
    run_ctx = _new()
    assert set(run_ctx.mongo_collections) == set(MONGO_CHECKPOINT_KINDS)
    for kind, name in run_ctx.mongo_collections.items():
        assert name == f"g-acme-confluence-{kind.replace('_', '-')}-{run_ctx.run_suffix}"


def test_new_run_context_two_calls_never_share_a_suffix():
    """Different bulk runs of the same (tenant, connector) must never collide
    onto the same store names — see repository/run_registry.py's uniqueness
    guard for the enforced version of this guarantee."""
    ctx1 = _new()
    ctx2 = _new()
    assert ctx1.run_suffix != ctx2.run_suffix
    assert ctx1.mongo_collections != ctx2.mongo_collections


def test_new_run_context_stores_tenant_and_connector_verbatim():
    run_ctx = _new()
    assert run_ctx.tenant_id == "acme"
    assert run_ctx.connector_type == "confluence"


def test_new_run_context_carries_run_config_verbatim():
    assert _new().run_config == _CFG


def test_bind_run_config_sets_active_and_unbinds_to_none():
    # Runs in the test's own context; a fresh context (each event) starts None.
    bind_run_config(_CFG)
    assert active_run_config() == _CFG
    bind_run_config(None)
    assert active_run_config() is None
