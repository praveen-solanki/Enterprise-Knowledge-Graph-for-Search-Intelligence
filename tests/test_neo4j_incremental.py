"""Unit tests for the incremental Neo4j write/read methods (AIS-3634).

Neo4jRepository talks over plain HTTP (requests.post) — these tests patch
_execute directly so nothing here ever touches a real Neo4j server.
"""
import hashlib
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from repository.neo4j_repository import Neo4jRepository
from services.run_context import RunConfig, bind_run_config


def _run_cfg(consume: bool = True, dim: int = 1536) -> RunConfig:
    """A RunConfig pinning the consume flag + dimension write_chunks reads
    through effective_run_config(); the rest is filler the chunk path never
    touches."""
    return RunConfig(
        embedding_backend="azure", embedding_model="text-embedding-3-large",
        embedding_dimension=dim, consume_upstream_vector=consume,
        es_chunk_vector_field="embeddings_openai_large_1536",
        llm_backend="azure", llm_model="gpt", build_authorship_edges=True,
        contributor_entity_types=("X",), people_org_graph_enabled=True,
    )


@contextmanager
def _bound_run_cfg(consume: bool = True, dim: int = 1536):
    """Bind _run_cfg for the whole call, the same way a real run does.

    These tests assert on an upstream 1536-d vector, which is read in two
    places that BOTH resolve the run's dimension/ES field independently
    (connectors/chunking.upstream_chunk_vector when the SourceDocument is
    built, repository/neo4j_repository when it is pushed) — patching one
    module's effective_run_config leaves the other reading the process's own
    embedding space, so the assertion breaks the moment that space is not
    1536-d. Binding the config covers every reader at once."""
    bind_run_config(_run_cfg(consume=consume, dim=dim))
    try:
        yield
    finally:
        bind_run_config(None)


def _sd_list(raws):
    """write_documents/write_chunks take SourceDocuments. These tests are about
    what lands on a Neo4j node, so the document is built here rather than by a
    connector."""
    from connectors.base import Chunk, ConnectorType, PersonRef, SourceDocument
    from connectors.chunking import dedup_chunks, upstream_chunk_vector
    from services.acl.types import AclDecision

    docs = []
    for raw in raws:
        created = raw.get("createdbyid")
        modified = raw.get("lastmodifiedbyid")
        chunks = [
            Chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk.get("text") or "",
                title=chunk.get("chunk_title"),
                # Explicit field, not upstream_chunk_vector's default: the
                # default is the PROCESS's embedding space, which is only the
                # 1536-d azure one these fixtures use when that happens to be
                # the configured space. A real run threads the run's own field
                # the same way (core/pipeline.py).
                vector=upstream_chunk_vector(chunk, _run_cfg().es_chunk_vector_field),
                recursive_chunk_id=chunk.get("recursive_chunk_id") or chunk["chunk_id"],
            )
            for chunk in dedup_chunks(raw)
        ]
        docs.append(SourceDocument(
            doc_id=raw["id"],
            connector_type=ConnectorType.WIKIPEDIA,
            object_type=raw.get("object_type") or "",
            title=(raw.get("content_title") or raw.get("file_title") or raw.get("file_name")
                   or raw.get("user_name") or raw.get("title") or ""),
            url=raw.get("url") or raw.get("file_url") or "",
            published_at=(raw.get("content_publish_on_date") or raw.get("createddate") or "")[:10],
            updated_at=raw.get("lastmodifieddate") or "",
            chunks=chunks,
            author=PersonRef(source_key=created, display_name=raw.get("content_authored_by_name")) if created else None,
            editor=PersonRef(source_key=modified) if modified else None,
            mentions=[],
            acl=AclDecision([], False, "test"),
            raw=raw,
        ))
    return docs


TENANT = "tenant-1"
VALID_ENTITY_EMB = [0.1] * 1536  # matches the azure space's 1536-d dimension


def _repo() -> Neo4jRepository:
    with patch("repository.neo4j_repository.settings") as mock_settings:
        mock_settings.neo4j_http_url = "http://localhost:7474"
        mock_settings.neo4j_username = "neo4j"
        mock_settings.neo4j_password = "pw"
        # Real (non-MagicMock) fallback batch sizes — __init__ captures these once,
        # as instance attributes, so a MagicMock here would poison every batched
        # write call for the lifetime of the repo, even after this patch exits.
        mock_settings.neo4j_default_batch_size = 5000
        mock_settings.neo4j_default_entity_batch_size = 100
        return Neo4jRepository(database="g-test-ab12")


@pytest.fixture(autouse=True)
def _no_real_entity_registry():
    """Safety net: upsert_entity_incremental resolves entity_id via
    repository/entity_registry.py's link_or_mint, which talks to a REAL
    shared MongoDB Atlas instance if not mocked -- autouse so no test in this
    file can accidentally write real documents into the shared entity_registry
    collection just by forgetting to mock this explicitly (confirmed this
    happened once: stray test documents landed in the real collection before
    this fixture existed). Individual tests can still override this with their
    own nested `with patch("repository.entity_registry.link_or_mint", ...)`
    when they need to assert on the exact entity_id resolved."""
    with patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent_default"}):
        yield


def test_find_similar_entities_combines_qdrant_score_with_neo4j_descriptions():
    """The vector search itself runs against Qdrant now (services/vector_store.py)
    -- this method's own Neo4j round trip is only a follow-up lookup for
    descriptions_json, the one piece of entity metadata Qdrant doesn't carry."""
    repo = _repo()
    vector_store = MagicMock()
    vector_store.search.return_value = [{"entity_id": "ent-1", "score": 0.91},
                                        {"entity_id": "ent-2", "score": 0.85}]
    fake_response = {
        "data": {"fields": ["entity_id", "name", "descriptions_json"],
                 "values": [["ent-1", "Globex Inc.", '[{"text": "A company", "doc_id": "d1"}]'],
                            ["ent-2", "Globex Corp", None]]},
    }
    with patch.object(repo, "_execute", return_value=fake_response) as execute:
        results = repo.find_similar_entities(vector_store, [0.1, 0.2], "ORGANIZATION", TENANT,
                                              k=5, min_similarity=0.83)
    vector_store.search.assert_called_once_with([0.1, 0.2], "ORGANIZATION", TENANT, k=5, min_similarity=0.83)
    _, params = execute.call_args[0]
    assert set(params["entity_ids"]) == {"ent-1", "ent-2"}
    assert results == [
        {"name": "Globex Inc.", "descriptions_json": '[{"text": "A company", "doc_id": "d1"}]', "score": 0.91},
        {"name": "Globex Corp", "descriptions_json": None, "score": 0.85},
    ]


def test_find_similar_entities_empty_when_qdrant_has_no_hits_and_skips_neo4j_entirely():
    repo = _repo()
    vector_store = MagicMock()
    vector_store.search.return_value = []
    with patch.object(repo, "_execute") as execute:
        assert repo.find_similar_entities(vector_store, [0.1], "ORGANIZATION", TENANT) == []
    execute.assert_not_called()


def test_find_similar_entities_fail_opens_when_qdrant_raises():
    """Missing/down Qdrant must not abort incremental Neo4j entity writes —
    treat as no ANN candidates (under-merge until vectors are back)."""
    repo = _repo()
    vector_store = MagicMock()
    vector_store.search.side_effect = RuntimeError("Not found: Collection doesn't exist!")
    with patch.object(repo, "_execute") as execute:
        assert repo.find_similar_entities(vector_store, [0.1], "ORGANIZATION", TENANT) == []
    execute.assert_not_called()


def test_find_similar_entities_sorts_by_score_descending_after_the_neo4j_lookup():
    """Neo4j's IN-list MATCH doesn't preserve Qdrant's own ordering -- must
    re-sort by score after combining, not just trust row order."""
    repo = _repo()
    vector_store = MagicMock()
    vector_store.search.return_value = [{"entity_id": "ent-1", "score": 0.80},
                                        {"entity_id": "ent-2", "score": 0.95}]
    fake_response = {
        "data": {"fields": ["entity_id", "name", "descriptions_json"],
                 "values": [["ent-1", "A", None], ["ent-2", "B", None]]},
    }
    with patch.object(repo, "_execute", return_value=fake_response):
        results = repo.find_similar_entities(vector_store, [0.1], "ORGANIZATION", TENANT)
    assert [r["name"] for r in results] == ["B", "A"]


def test_entity_exists_exact_true_when_node_has_a_vector():
    repo = _repo()
    with patch.object(repo, "_execute",
                      return_value={"data": {"fields": ["has_vector"], "values": [[True]]}}) as execute:
        assert repo.entity_exists_exact("Globex", "ORGANIZATION", TENANT) is True
    cypher, params = execute.call_args[0]
    assert "tenant_id: $tenant_id" in cypher
    assert params == {"name": "Globex", "type": "ORGANIZATION", "tenant_id": TENANT}


def test_entity_exists_exact_false_when_no_node_matches():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}):
        assert repo.entity_exists_exact("Nonexistent", "ORGANIZATION", TENANT) is False


def test_entity_exists_exact_false_when_node_exists_without_a_vector():
    """A legacy/edge-case row missing its vector must still go through the
    slow path once, so it gets backfilled -- not fast-pathed past forever."""
    repo = _repo()
    with patch.object(repo, "_execute",
                      return_value={"data": {"fields": ["has_vector"], "values": [[False]]}}):
        assert repo.entity_exists_exact("Globex", "ORGANIZATION", TENANT) is False


def _no_existing_entity_response() -> dict:
    """OPTIONAL MATCH found nothing — a brand-new entity."""
    return {"data": {"fields": ["descriptions_json", "aliases_json", "already_contributed"],
                     "values": [[None, None, False]]}}


def test_upsert_entity_incremental_sends_expected_parameters():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_no_existing_entity_response(), {}]) as execute, \
         patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent_1"}) as link_or_mint:
        repo.upsert_entity_incremental(
            name="Globex Inc.", entity_type="ORGANIZATION", description="A company",
            embedding=[0.1, 0.2], doc_id="doc-1", acl=["public"], tenant_id=TENANT, raw_name="Globex",
        )
    link_or_mint.assert_called_once_with(
        "Globex Inc.", "ORGANIZATION", "globex inc", TENANT, aliases=["globex inc", "globex"],
    )
    assert execute.call_count == 2
    cypher, params = execute.call_args_list[1].args
    assert "ON CREATE SET" in cypher and "ON MATCH SET" in cypher
    assert "entity_id: $entity_id" in cypher
    assert params == {
        "entity_id": "ent_1",
        "name": "Globex Inc.", "type": "ORGANIZATION",
        "has_new_embedding": True, "doc_id": "doc-1", "acl": ["public"],
        "description": "A company",
        "descriptions_json": json.dumps([{"text": "A company", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
        "aliases_json": json.dumps([{"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
        "aliases_flat": ["Globex"],
        "tenant_id": TENANT,
    }


def test_upsert_entity_incremental_defaults_none_acl_to_empty_list():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_no_existing_entity_response(), {}]) as execute:
        repo.upsert_entity_incremental(
            name="X", entity_type="ORG", description="", embedding=None, doc_id="doc-1", acl=None,
            tenant_id=TENANT,
        )
    _, params = execute.call_args_list[1].args
    assert params["acl"] == []


def _n(value: int) -> dict:
    return {"data": {"fields": ["n"], "values": [[value]]}}


def _no_prune_rows() -> dict:
    """The descriptions/aliases prune step's read found no affected entities
    (or none worth pruning) — its conditional write is skipped entirely, so
    this consumes exactly one _execute call, not two."""
    return {"data": {"fields": ["name", "type", "descriptions_json", "aliases_json"], "values": []}}


def _entity_acl_rows(pairs: list) -> dict:
    """Simulates a per-row RETURN entity_id, acl response -- pairs: [(entity_id, [acl_token, ...]), ...].
    Replaces the old aggregate `count(e) AS n` shape now that these steps also
    need each entity_id/acl to push to Qdrant."""
    return {"data": {"fields": ["entity_id", "acl"], "values": [[eid, acl] for eid, acl in pairs]}}


def _entity_id_rows(entity_ids: list) -> dict:
    """Simulates a per-row RETURN entity_id response (the zero-support/orphan
    entity DETACH DELETE steps, which now return the deleted ids for Qdrant
    cleanup instead of just a count)."""
    return {"data": {"fields": ["entity_id"], "values": [[eid] for eid in entity_ids]}}


def _chunk_ids_row(chunk_ids: list) -> dict:
    """Simulates the single-row RETURN collect(chunk_id) AS chunk_ids shape
    the Chunk DETACH DELETE / acl-update steps use to hand back what they
    touched for the parallel Qdrant vec_chunks push."""
    return {"data": {"fields": ["chunk_ids"], "values": [[chunk_ids]]}}


# --- _execute transient-error retry (experiment-parity review #24) ---

def _http_error(status: int):
    from requests.exceptions import HTTPError
    exc = HTTPError(f"{status} error")
    exc.response = type("Resp", (), {"status_code": status})()
    return exc


def test_execute_retries_a_transient_failure_on_a_fresh_session_and_succeeds():
    """Regression test for a real failure found during the 100-doc parity dry
    run: the shared dev Neo4j sits behind a load balancer fronting more than
    one member, and the HTTP Query API has no leader-routing the way Bolt
    does -- a plain POST can land on a non-writer member and get a
    deterministic "no longer accepts writes" 500. Since self._session is one
    persistent, keep-alive connection, retrying on the SAME session hits the
    SAME wrong member every time. _execute must open a fresh session before
    each retry so the load balancer gets another chance to route correctly."""
    repo = _repo()
    ok_response = MagicMock()
    ok_response.json.return_value = {"data": {"fields": [], "values": []}}
    fresh_session = MagicMock()
    fresh_session.post.return_value = ok_response
    with patch.object(repo._session, "post", side_effect=_http_error(500)) as first_post, \
         patch("repository.neo4j_repository.requests.Session", return_value=fresh_session), \
         patch("repository.neo4j_repository.time.sleep"):
        result = repo._execute("MATCH (n) RETURN n", {})
    first_post.assert_called_once()          # first attempt used the original (bad) session
    fresh_session.post.assert_called_once()  # retry used a brand-new session, not the same one
    assert result == {"data": {"fields": [], "values": []}}


def test_execute_gives_up_after_max_retries_on_persistent_transient_failure():
    repo = _repo()
    with patch.object(repo._session, "post", side_effect=_http_error(500)) as original_post, \
         patch("repository.neo4j_repository.requests.Session",
               side_effect=lambda: MagicMock(post=MagicMock(side_effect=_http_error(500)))), \
         patch("repository.neo4j_repository.time.sleep"):
        with pytest.raises(Exception):
            repo._execute("MATCH (n) RETURN n", {})
    original_post.assert_called_once()  # only the first of the 3 attempts used the original session


def test_execute_does_not_retry_a_permanent_failure():
    repo = _repo()
    with patch.object(repo._session, "post", side_effect=_http_error(400)) as post, \
         patch("repository.neo4j_repository.time.sleep"):
        with pytest.raises(Exception):
            repo._execute("MATCH (n) RETURN n", {})
    assert post.call_count == 1


def test_execute_does_not_retry_a_fatal_failure():
    repo = _repo()
    with patch.object(repo._session, "post", side_effect=_http_error(401)) as post, \
         patch("repository.neo4j_repository.time.sleep"):
        with pytest.raises(Exception):
            repo._execute("MATCH (n) RETURN n", {})
    assert post.call_count == 1


def test_count_documents_scopes_to_tenant():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(42)) as execute:
        n = repo.count_documents(TENANT)
    cypher, params = execute.call_args[0]
    assert "tenant_id: $tenant_id" in cypher
    assert params == {"tenant_id": TENANT}
    assert n == 42


def test_retract_document_runs_eight_statements_and_reports_stats():
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.retract_document("doc-1", TENANT)
    assert execute.call_count == 8
    assert stats == {
        "entities_decremented": 2, "entities_deleted": 1, "relationships_pruned": 1,
        "relationships_acl_recomputed": 1, "claims_deleted": 3, "doc_deleted": True,
    }
    for call in execute.call_args_list:
        cypher = call.args[0]
        params = call.args[1]
        if "doc_id" in params:
            assert params["doc_id"] == "doc-1"
        if "tenant_id" in params:
            assert params["tenant_id"] == TENANT


def test_retract_document_deletes_chunk_vectors_when_chunks_vector_store_given():
    """Parity review -- reference's vec_chunks collection: a retracted Chunk
    node's vector must be deleted too, or it's an orphaned point in Qdrant
    forever, invisible from the Neo4j side but still live and searchable."""
    repo = _repo()
    chunks_vector_store = MagicMock()
    responses = [_entity_acl_rows([]), _no_prune_rows(), _entity_id_rows([]), _n(0), _n(0), _n(0),
                _chunk_ids_row(["c1", "c2"]), _n(1)]
    with patch.object(repo, "_execute", side_effect=responses):
        repo.retract_document("doc-1", TENANT, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.delete_entities.assert_called_once_with(["c1", "c2"])


def test_retract_document_skips_chunk_vector_delete_when_nothing_was_deleted():
    repo = _repo()
    chunks_vector_store = MagicMock()
    responses = [_entity_acl_rows([]), _no_prune_rows(), _entity_id_rows([]), _n(0), _n(0), _n(0),
                _chunk_ids_row([]), _n(1)]
    with patch.object(repo, "_execute", side_effect=responses):
        repo.retract_document("doc-1", TENANT, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.delete_entities.assert_not_called()


# --- claim survives retraction if shared with another document (review #33) ---

def test_retract_document_claims_step_only_deletes_the_claim_when_no_supported_by_edges_remain():
    """claim_id is hashed from subject|predicate|object|valid_from|valid_to alone (no
    chunk_id/doc_id) -- two different documents' chunks CAN merge onto the same Claim
    node. Retraction must strip THIS doc's SUPPORTED_BY edges and only delete the claim
    if that leaves zero remaining, never an unconditional delete-by-chunk_id-property."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    claims_call = execute.call_args_list[5]
    cypher = claims_call.args[0]
    assert "<-[sb:SUPPORTED_BY]-(cl:Claim)" in cypher
    assert "FOREACH (r IN this_docs_edges | DELETE r)" in cypher
    assert "OPTIONAL MATCH (cl)-[remaining:SUPPORTED_BY]->(:Chunk)" in cypher
    assert "WHERE remaining_count = 0" in cypher
    assert "{chunk_id: c.chunk_id}" not in cypher


def test_create_constraints_calibrates_by_default():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}), \
         patch.object(repo, "calibrate_batch_sizes") as calibrate:
        repo.create_constraints()
    calibrate.assert_called_once()


def test_calibrate_batch_sizes_caps_entity_batch_at_configured_max():
    """Synthetic calibration can return ~1280; neo4j_entity_batch_max must win."""
    repo = _repo()
    fake_calibrator = MagicMock()
    # First call = entity probe, second = document probe.
    fake_calibrator.calibrate.side_effect = [1280, 200]

    with patch("services.adaptive_controller.Neo4jBatchCalibrator",
               return_value=fake_calibrator), \
         patch.object(repo, "_execute", return_value={}), \
         patch.object(repo, "_cleanup_calibration_nodes"), \
         patch("repository.neo4j_repository.settings") as mock_settings:
        mock_settings.neo4j_entity_batch_max = 50
        repo.calibrate_batch_sizes()

    assert fake_calibrator.calibrate.call_count == 2
    assert repo._entity_batch_size == 50
    assert repo._batch_size == 200


def test_neo4j_entity_batch_max_rejects_non_positive_values():
    from pydantic import ValidationError

    from config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(neo4j_entity_batch_max=0)
    with pytest.raises(ValidationError):
        Settings(neo4j_entity_batch_max=-1)


def test_create_constraints_skips_calibration_when_disabled():
    """AIS-3634: the Kafka worker passes calibrate=False — batch-size tuning
    exists for bulk's multi-thousand-row writes, not the worker's tiny
    per-document ones, and shouldn't pay for a redundant synthetic probe."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}), \
         patch.object(repo, "calibrate_batch_sizes") as calibrate:
        repo.create_constraints(calibrate=False)
    calibrate.assert_not_called()


def test_create_constraints_drops_and_recreates_the_entity_constraint_with_tenant():
    """Code review must-fix #4: the old global (name, type) uniqueness
    constraint must be dropped and replaced with a (name, type, tenant_id)
    one — confirmed safe on this disposable dev graph rather than a live
    migration."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute, \
         patch.object(repo, "calibrate_batch_sizes"):
        repo.create_constraints()
    statements = [call.args[0] for call in execute.call_args_list]
    assert any("DROP CONSTRAINT entity_name_type IF EXISTS" in s for s in statements)
    assert any("entity_name_type_tenant" in s and "e.tenant_id" in s for s in statements)


def test_retract_document_relationship_prune_is_scoped_to_mentioned_entities_not_whole_graph():
    """Regression test for the unscoped `MATCH ()-[r]-()` full-graph scan —
    the relationship-prune step must only ever touch entities reachable from
    this document's own chunks, not every relationship in the database."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    relationship_prune_cypher = execute.call_args_list[3].args[0]
    assert "MENTIONS" in relationship_prune_cypher
    assert "MATCH ()-[r]-()" not in relationship_prune_cypher


def test_retract_document_entity_deletion_check_ignores_support():
    """Regression test: deletion must key off doc_ids emptiness alone, not
    support — an entity with stale support>0 but no real doc_ids left must
    still be eligible for cleanup."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    entity_delete_cypher = execute.call_args_list[2].args[0]
    assert "support" not in entity_delete_cypher
    assert "e.doc_ids IS NOT NULL AND size(e.doc_ids) = 0" in entity_delete_cypher


def test_retract_document_entity_deletion_never_uses_coalesce_on_doc_ids():
    """Regression test for a real incident: an earlier version of this query
    used size(coalesce(e.doc_ids, [])) = 0, which treats "this entity never
    had doc_ids tracked at all" (every entity written before that tracking
    existed) identically to "explicitly emptied out" — that deleted 153
    legitimate entities from the shared dev graph in one shot. Must always
    require the property to actually exist first."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    entity_delete_cypher = execute.call_args_list[2].args[0]
    assert "coalesce(e.doc_ids" not in entity_delete_cypher


def test_retract_document_entity_deletion_is_scoped_to_tenant():
    """Code review must-fix #4: the zero-support sweep must not scan/delete
    entities belonging to other tenants."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    entity_delete_cypher, entity_delete_params = execute.call_args_list[2].args
    assert "e.tenant_id = $tenant_id" in entity_delete_cypher
    assert entity_delete_params["tenant_id"] == TENANT


# --- ACL shrink via live recompute (code review must-fix #3) ---

def test_retract_document_entities_step_recomputes_acl_from_remaining_docs():
    """The entity decrement step must also recompute acl as a fresh union of
    the entity's REMAINING contributing documents, not keep appending forever."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.retract_document("doc-1", TENANT)
    entities_cypher = execute.call_args_list[0].args[0]
    assert "e.allow_access_control = reduce" in entities_cypher
    assert "remaining_doc_ids" in entities_cypher


def test_retract_document_recomputes_relationship_acl_for_survivors():
    """A new step, run after the prune step commits (a deleted relationship
    can't be read again in the same query), recomputes acl for every
    relationship that survived pruning from its remaining chunk_ids'
    documents — not the historical union."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}, _n(1)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.retract_document("doc-1", TENANT)
    relationship_acl_cypher = execute.call_args_list[4].args[0]
    assert "r.allow_access_control = reduce" in relationship_acl_cypher
    assert "size(r.chunk_ids) > 0" in relationship_acl_cypher
    assert stats["relationships_acl_recomputed"] == 1


def test_retract_chunks_prunes_this_docs_own_description_and_alias_entries():
    """Code review must-fix #8 follow-up: an entity's descriptions_json/
    aliases_json must lose THIS doc_id's entry (not just doc_ids/support/acl)
    so a later re-upsert of the same document doesn't append a second,
    duplicate entry alongside the stale one."""
    repo = _repo()
    prune_read_response = {
        "data": {"fields": ["entity_id", "descriptions_json", "aliases_json"],
                 "values": [["ent_globex",
                             json.dumps([
                                 {"text": "from doc-1", "doc_id": "doc-1", "allow_access_control": ["public"]},
                                 {"text": "from doc-2", "doc_id": "doc-2", "allow_access_control": ["public"]},
                             ]),
                             json.dumps([
                                 {"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]},
                                 {"name": "Globex Corp", "doc_id": "doc-2", "allow_access_control": ["public"]},
                             ])]]}
    }
    responses = [_entity_acl_rows([("ent_globex", ["public"])]), prune_read_response, {},
                _entity_id_rows([]), _n(0), _n(0), _n(0), {}]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.replace_document_chunks("doc-1", TENANT)
    prune_write_cypher, prune_write_params = execute.call_args_list[2].args
    # Experiment-parity review #27: this write-back must MATCH by entity_id,
    # the stable identity now -- not by (name, type, tenant_id).
    assert "MATCH (e:Entity {entity_id: row.entity_id})" in prune_write_cypher
    assert "e.descriptions_json = row.descriptions_json" in prune_write_cypher
    batch = prune_write_params["batch"]
    assert len(batch) == 1
    assert batch[0]["entity_id"] == "ent_globex"
    descriptions = json.loads(batch[0]["descriptions_json"])
    aliases = json.loads(batch[0]["aliases_json"])
    assert descriptions == [{"text": "from doc-2", "doc_id": "doc-2", "allow_access_control": ["public"]}]
    assert aliases == [{"name": "Globex Corp", "doc_id": "doc-2", "allow_access_control": ["public"]}]


def test_write_entities_merges_doc_ids_into_existing_rather_than_overwriting():
    """Regression test: bulk must UNION its source_docs into e.doc_ids, never
    overwrite — an overwrite would erase a document incremental added after
    bulk's own corpus scan ran."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "source_docs": ["bulk-doc-1"], "embedding": [0.1],
                        "descriptions": [{"text": "d", "doc_id": "bulk-doc-1", "allow_access_control": ["public"]}],
                        "aliases": [{"name": "Globex", "doc_id": "bulk-doc-1", "allow_access_control": ["public"]}],
                        "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "coalesce(e.doc_ids, [])" in cypher
    assert "e.support = size(merged_doc_ids)" in cypher
    assert params["batch"][0]["source_docs"] == ["bulk-doc-1"]


def test_write_entities_merges_acl_into_existing_rather_than_overwriting():
    """Experiment-parity review #11: core/pipeline.py's _tag_entity_acls only
    covers documents fetched in THIS run (documented limitation) -- a bulk run
    scoped narrower than the full corpus (--limit, a retry run's restricted
    query) must not overwrite an entity's acl with only its own partial view,
    dropping an access grant a previously-processed-but-not-rescanned document
    legitimately contributed. Must union into e.allow_access_control, never
    overwrite -- same pattern as e.doc_ids just above."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "source_docs": ["doc-2"], "embedding": [0.1],
                        "acl": ["group:site:deal-room"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "coalesce(e.allow_access_control, [])" in cypher
    assert "e.allow_access_control = row.acl" not in cypher
    assert params["batch"][0]["acl"] == ["group:site:deal-room"]


def test_write_entities_scopes_merge_key_by_tenant():
    """Experiment-parity review #27, Phase 1: MERGE now keys on entity_id alone
    (the stable identity), not (name, type, tenant_id) -- but tenant_id must
    still be written as a plain property on the node, since filtering/display
    and the tenant index still rely on it."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "source_docs": ["doc-1"], "embedding": [0.1],
                        "descriptions": [{"text": "d", "doc_id": "doc-1", "allow_access_control": ["public"]}],
                        "aliases": [{"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]}],
                        "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "MERGE (e:Entity {entity_id: row.entity_id})" in cypher
    assert "e.tenant_id = row.tenant_id" in cypher
    assert params["batch"][0]["tenant_id"] == TENANT
    assert params["batch"][0]["entity_id"] == "ent_globex"


# --- entity description/alias ACL leak via union (code review must-fix #8) ---

def test_write_entities_writes_descriptions_and_aliases_as_json_not_flat_fields():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {
                "type": "ORGANIZATION", "source_docs": ["doc-1", "doc-2"], "embedding": [0.1],
                "descriptions": [
                    {"text": "public blurb", "doc_id": "doc-1", "allow_access_control": ["public"]},
                    {"text": "private detail", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
                ],
                "aliases": [
                    {"name": "Globex Inc.", "doc_id": "doc-1", "allow_access_control": ["public"]},
                    {"name": "codename-x", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
                ],
                "acl": ["public", "group:site:s1"],
            },
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "e.descriptions_json = row.descriptions_json" in cypher
    assert "e.aliases_json = row.aliases_json" in cypher
    # e.description (singular) IS expected now -- the LLM's one merged description
    # per canon entity (decision 2026-08-05), guarded so a writer that carries none
    # (persons, incremental) never blanks what an earlier writer stored. e.aliases
    # likewise (experiment-parity review #22) -- only e.alias (singular) is banned.
    assert "e.description = CASE WHEN row.description <> ''" in cypher
    assert "e.alias " not in cypher
    row = params["batch"][0]
    descriptions = json.loads(row["descriptions_json"])
    aliases = json.loads(row["aliases_json"])
    assert descriptions == [
        {"text": "public blurb", "doc_id": "doc-1", "allow_access_control": ["public"]},
        {"text": "private detail", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
    ]
    assert aliases == [
        {"name": "Globex Inc.", "doc_id": "doc-1", "allow_access_control": ["public"]},
        {"name": "codename-x", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
    ]
    # Neither entry's own acl was blended into the other's.
    assert descriptions[0]["allow_access_control"] != descriptions[1]["allow_access_control"]


def test_upsert_entity_incremental_appends_new_doc_description_and_alias_as_json():
    repo = _repo()
    existing_response = {
        "data": {"fields": ["descriptions_json", "aliases_json", "already_contributed"],
                 "values": [[
                     json.dumps([{"text": "from doc-1", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
                     json.dumps([{"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
                     False,
                 ]]},
    }
    with patch.object(repo, "_execute", side_effect=[existing_response, {}]) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="from doc-2", embedding=[0.2],
            doc_id="doc-2", acl=["group:site:s1"], tenant_id=TENANT, raw_name="Globex Corp",
        )
    assert execute.call_count == 2
    _, write_params = execute.call_args_list[1].args
    descriptions = json.loads(write_params["descriptions_json"])
    aliases = json.loads(write_params["aliases_json"])
    assert descriptions == [
        {"text": "from doc-1", "doc_id": "doc-1", "allow_access_control": ["public"]},
        {"text": "from doc-2", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
    ]
    assert aliases == [
        {"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]},
        {"name": "Globex Corp", "doc_id": "doc-2", "allow_access_control": ["group:site:s1"]},
    ]


def test_upsert_entity_incremental_does_not_duplicate_entry_on_redelivered_doc_id():
    repo = _repo()
    existing_response = {
        "data": {"fields": ["descriptions_json", "aliases_json", "already_contributed"],
                 "values": [[
                     json.dumps([{"text": "from doc-1", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
                     json.dumps([{"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]}]),
                     True,  # doc-1 already contributed — redelivered message
                 ]]},
    }
    with patch.object(repo, "_execute", side_effect=[existing_response, {}]) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="from doc-1", embedding=[0.2],
            doc_id="doc-1", acl=["public"], tenant_id=TENANT,
        )
    _, write_params = execute.call_args_list[1].args
    assert len(json.loads(write_params["descriptions_json"])) == 1
    assert len(json.loads(write_params["aliases_json"])) == 1


def test_retract_document_reports_doc_deleted_false_when_nothing_matched():
    repo = _repo()
    responses = [_entity_acl_rows([]), _no_prune_rows(), _entity_id_rows([]), _n(0), _n(0), _n(0), {},
                {"data": {"fields": ["n"], "values": [[0]]}}]
    with patch.object(repo, "_execute", side_effect=responses):
        stats = repo.retract_document("missing-doc", TENANT)
    assert stats["doc_deleted"] is False
    assert stats["entities_decremented"] == 0


# --- replace_document_chunks (code review must-fix #2 — upsert hard-replace) ---

def test_replace_document_chunks_runs_seven_statements_and_never_deletes_the_document():
    """Unlike retract_document, this must NOT delete the Document node — the
    caller (process_document) is about to rewrite this exact doc_id's chunks
    right after calling this."""
    repo = _repo()
    responses = [_entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"])]), _no_prune_rows(),
                _entity_id_rows(["ent-c"]), _n(1), _n(1), _n(3), {}]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.replace_document_chunks("doc-1", TENANT)
    assert execute.call_count == 7
    assert stats == {
        "entities_decremented": 2, "entities_deleted": 1, "relationships_pruned": 1,
        "relationships_acl_recomputed": 1, "claims_deleted": 3,
    }
    assert "doc_deleted" not in stats
    for call in execute.call_args_list:
        cypher, params = call.args[0], call.args[1]
        assert "DETACH DELETE d" not in cypher
        if "doc_id" in params:
            assert params["doc_id"] == "doc-1"


def test_replace_document_chunks_is_safe_on_a_brand_new_doc_id():
    """A doc_id that has never been ingested before must produce all-zero
    stats, not an error — every MATCH inside simply finds nothing."""
    repo = _repo()
    responses = [_entity_acl_rows([]), _no_prune_rows(), _entity_id_rows([]), _n(0), _n(0), _n(0), {}]
    with patch.object(repo, "_execute", side_effect=responses):
        stats = repo.replace_document_chunks("brand-new-doc", TENANT)
    assert stats["entities_decremented"] == 0
    assert stats["entities_deleted"] == 0
    assert stats["relationships_pruned"] == 0
    assert stats["relationships_acl_recomputed"] == 0
    assert stats["claims_deleted"] == 0


# --- ACL/tenant contract rename: acl -> allow_access_control (code review must-fix #11) ---

def test_write_documents_sets_allow_access_control_property():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "content_title": "T"}]), TENANT, {"doc-1": ["public"]})
    cypher, params = execute.call_args[0]
    assert "d.allow_access_control = row.acl" in cypher
    assert "d.acl" not in cypher
    assert params["batch"][0]["acl"] == ["public"]


# --- Retrieval schema contract: source_type + documented_on (code review must-fix #12) ---

def test_write_documents_source_type_now_always_mirrors_object_type():
    """source_type used to be "the raw connector_type field, else object_type",
    and the raw field was absent on every intranet document — so object_type is
    the value it has always had in practice. It is now a plain duplicate of
    d.source; d.connector_type carries the connector."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content"}]),
            TENANT, {"doc-1": ["public"]},
        )
    cypher, params = execute.call_args[0]
    assert "d.source_type = row.source_type" in cypher
    assert params["batch"][0]["source_type"] == "content"
    assert params["batch"][0]["source_type"] == params["batch"][0]["source"]


def test_write_documents_also_writes_object_type_under_its_own_name():
    """ai-graph-retrieval gates its people-profile join on `d.object_type = 'people'`
    (graph/cypher.py CHUNK_QUOTES). The value was only ever written as `source` and
    `source_type`, so that join never fired on any graph this pipeline built — a
    quote about a former employee could not be flagged. object_type must land under
    its own name, in addition to (not instead of) the existing two."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "content_title": "T", "object_type": "people"}]),
            TENANT, {"doc-1": ["public"]},
        )
    cypher, params = execute.call_args[0]
    assert "d.object_type = row.object_type" in cypher
    assert params["batch"][0]["object_type"] == "people"
    # The pre-existing names keep the same value — this is additive, not a rename.
    assert params["batch"][0]["source"] == "people"
    assert params["batch"][0]["source_type"] == "people"


def test_write_documents_falls_back_to_object_type_when_no_connector_type():
    """source_type must always be populated, never left empty, so a document
    written by the bulk path is never queryable-but-typeless. It used to reach
    object_type by falling back to it; it now reads object_type outright. The
    assertion is the one that has always mattered and is unchanged."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content"}]),
                             TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["source_type"] == "content"


def test_write_documents_source_type_empty_string_when_no_fields_present():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "content_title": "T"}]), TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["source_type"] == ""


# --- write_documents: per-doc connector_type/url resolution ---

def test_write_documents_takes_connector_type_and_url_from_the_contract():
    """Both used to be guessed from the payload — a raw connector_type field if
    one happened to be there, else "confluence" if a url field was, else
    a legacy spelling. A document now states which connector it came from and what its
    URL is, so there is nothing to infer and no connector's field names appear
    in this method. Built here as a SourceDocument directly rather than through
    a connector, precisely because the write no longer knows or cares which one
    produced it."""
    from connectors.base import ConnectorType, SourceDocument
    from services.acl.types import AclDecision

    repo = _repo()
    foreign = SourceDocument(
        doc_id="p1", connector_type=ConnectorType.WIKIPEDIA, object_type="article",
        title="T", url="http://localhost:8090/wiki/spaces/x/pages/1/Page",
        published_at="2026-01-02", updated_at="2026-01-03", chunks=[],
        author=None, editor=None, mentions=[],
        acl=AclDecision(["group_id:g1"], False, "test"),
    )
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents([foreign], TENANT, {"p1": ["group_id:g1"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["connector_type"] == "wikipedia"
    assert params["batch"][0]["url"] == "http://localhost:8090/wiki/spaces/x/pages/1/Page"
    assert params["batch"][0]["title"] == "T"
    assert params["batch"][0]["published_at"] == "2026-01-02"


def test_write_documents_stamps_zeus_and_an_empty_url_when_no_route_can_be_built():
    """No content_site_id, so no app route -- url is never fabricated when the
    required source field is absent. connector_type is "zeus": the one name for
    the concept, where this used to write a legacy spelling."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content"}]),
                             TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["connector_type"] == "wikipedia"
    assert params["batch"][0]["url"] == ""


def test_write_documents_builds_intranet_content_url_with_site_and_type():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content",
              "content_site_id": "site-1", "content_type": "page"}]),
            TENANT, {"doc-1": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == ""


def test_write_documents_builds_intranet_content_url_falls_back_without_content_type():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content", "content_site_id": "site-1"}]),
            TENANT, {"doc-1": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == ""


def test_write_documents_builds_intranet_file_url_with_site_and_provider():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "object_type": "file", "file_site_id": "site-1", "file_provider": "gdrive"}]),
            TENANT, {"doc-1": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == ""


def test_write_documents_builds_intranet_file_url_defaults_provider_to_intranet():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "object_type": "file", "file_site_id": "site-1"}]),
                             TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == ""


def test_write_documents_builds_intranet_file_url_falls_back_to_file_url_without_site():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "object_type": "file", "file_url": "/content/r/28be4d97"}]),
            TENANT, {"doc-1": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == "/content/r/28be4d97"


def test_write_documents_builds_intranet_people_url_from_doc_id():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "user-1", "object_type": "people"}]), TENANT, {"user-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["url"] == ""


# --- Authorship graph: write_documents stamps created_by/modified_by/author_name ---

def test_write_documents_stores_a_files_real_title_where_it_used_to_store_nothing():
    """Behaviour change. title read `content_title` literally, which only a
    content document has, so every file landed with title "". The connector's
    own fallback chain — content_title -> file_title -> file_name — is the same
    one already used to frame that document's chunk text."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "f1", "object_type": "file", "file_type": "pptx",
                       "file_title": "Q3 Deck", "file_name": "q3.pptx"}]),
            TENANT, {"f1": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["title"] == "Q3 Deck"


def test_write_documents_falls_back_to_a_files_name_when_it_has_no_title():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "f2", "object_type": "file", "file_type": "pptx",
                       "file_name": "q3.pptx"}]),
            TENANT, {"f2": ["public"]},
        )
    _, params = execute.call_args[0]
    assert params["batch"][0]["title"] == "q3.pptx"


def test_write_documents_titles_a_people_document_with_the_persons_name():
    """Reversal of the old "leave it empty" rule. An HR record has no title field,
    so the title used to land "" — but Document.title is MANDATORY for
    ai-graph-retrieval: it is shown on every citation and is the ONLY thing
    DOCUMENT_LOOKUP / DOCUMENT_CONTRIBUTORS match on
    (`toLower(d.title) CONTAINS toLower($title)`). A blank title made all 1,008
    people profiles on the live graph uncitable and unfindable by name, so the
    person's own name is the title now."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "u1", "object_type": "people",
                                        "user_name": "Ada Lovelace"}]),
                             TENANT, {"u1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["title"] == "Ada Lovelace"


def test_write_documents_stamps_authorship_fields_for_content():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(
            _sd_list([{"id": "doc-1", "content_title": "T", "object_type": "content",
              "createdbyid": "u1", "lastmodifiedbyid": "u2", "content_authored_by_name": "Rachel"}]),
            TENANT, {"doc-1": ["public"]},
        )
    cypher, params = execute.call_args[0]
    row = params["batch"][0]
    assert row["created_by_id"] == "u1"
    assert row["modified_by_id"] == "u2"
    assert row["author_name"] == "Rachel"
    assert "d.created_by_id = coalesce(row.created_by_id, d.created_by_id)" in cypher


def test_write_documents_authorship_fields_none_for_people():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_documents(_sd_list([{"id": "doc-1", "object_type": "people"}]), TENANT, {})
    _, params = execute.call_args[0]
    row = params["batch"][0]
    assert row["created_by_id"] is None
    assert row["modified_by_id"] is None
    assert row["author_name"] is None


# --- materialize_authorship_edges ---

def test_materialize_authorship_edges_returns_authored_edited_contributed_counts():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_n(2), _n(1), _n(3)]) as execute:
        result = repo.materialize_authorship_edges(TENANT, ["doc-1", "doc-2"])
    assert result == {"authored": 2, "edited": 1, "contributed_to": 3}
    authored_cypher, authored_params = execute.call_args_list[0].args
    assert "MERGE (p)-[:AUTHORED]->(d)" in authored_cypher
    assert authored_params["doc_ids"] == ["doc-1", "doc-2"]
    edited_cypher, _ = execute.call_args_list[1].args
    assert "MERGE (p)-[:EDITED]->(d)" in edited_cypher
    assert "d.modified_by_id <> coalesce(d.created_by_id, '')" in edited_cypher
    contributed_cypher, contributed_params = execute.call_args_list[2].args
    assert "MERGE (p)-[:CONTRIBUTED_TO]->(e)" in contributed_cypher
    assert contributed_params["types"] == ["PROJECT", "PRODUCT", "PROGRAM", "FEATURE", "INITIATIVE"]


def test_materialize_authorship_edges_with_no_doc_ids_scopes_to_whole_tenant():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(0)) as execute:
        repo.materialize_authorship_edges(TENANT, None)
    _, params = execute.call_args_list[0].args
    assert params["doc_ids"] is None
    cypher, _ = execute.call_args_list[0].args
    assert "AND d.doc_id IN $doc_ids" not in cypher


def test_write_relationships_documented_on_flows_through_descriptions_json():
    """documented_on is produced by services/edge_pool_service.py, not this
    repository method — write_relationships must just pass descriptions
    through to descriptions_json verbatim, whatever fields they carry."""
    repo = _repo()
    final_edges = {
        ("A", "RELATED_TO", "B"): {
            "src_type": "ORG", "tgt_type": "ORG",
            "descriptions": [{"text": "hi", "acl": ["public"], "chunk_id": "c1", "documented_on": "2026-01-15"}],
            "chunk_ids": ["c1"], "raw_types": ["related_to"], "acl": ["public"],
        },
    }
    with patch.object(repo, "_execute", return_value={}), \
         patch.object(repo, "_run_batched") as run_batched:
        repo.write_relationships(final_edges, TENANT)
    _, rows, _ = run_batched.call_args[0]
    descriptions = json.loads(rows[0]["descriptions_json"])
    assert descriptions == [{"text": "hi", "acl": ["public"], "chunk_id": "c1", "documented_on": "2026-01-15"}]


def test_write_chunks_sets_allow_access_control_property():
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi"}]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    cypher, _ = execute.call_args[0]
    assert "c.allow_access_control = row.acl" in cypher
    assert "c.acl" not in cypher


def test_write_chunks_url_inherits_parent_document_url():
    repo = _repo()
    doc = {"id": "doc-1", "object_type": "content", "content_site_id": "site-1", "content_type": "page",
          "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi"}]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    cypher, params = execute.call_args[0]
    assert "c.url = row.url" in cypher
    assert params["batch"][0]["url"] == ""


def test_write_chunks_dedupes_overlapping_recursive_chunks():
    """Experiment-parity review C1: confirmed against real production documents
    that the upstream recursive chunker's overlapping windows share one
    recursive_chunk_id and carry byte-identical text — writing every raw entry
    as its own Chunk node inflates the graph 4-6x for a typical multi-chunk doc."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "recursive_chunk_id": "rid-1", "text": "hello"},
        {"chunk_id": "c2", "recursive_chunk_id": "rid-1", "text": "hello"},
        {"chunk_id": "c3", "recursive_chunk_id": "rid-1", "text": "hello"},
    ]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert [row["chunk_id"] for row in params["batch"]] == ["c1"]


# --- c.recursive_chunk_id + chunk-diff (experiment-parity review #34) ---

def test_write_chunks_persists_recursive_chunk_id():
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "recursive_chunk_id": "rid-1", "text": "hi"}]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    cypher, params = execute.call_args[0]
    assert "c.recursive_chunk_id = row.recursive_chunk_id" in cypher
    assert params["batch"][0]["recursive_chunk_id"] == "rid-1"


def test_write_chunks_falls_back_to_chunk_id_when_recursive_chunk_id_missing():
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi"}]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert params["batch"][0]["recursive_chunk_id"] == "c1"


def test_write_chunks_never_sends_the_vector_to_neo4j():
    """The Cypher never references row.vector -- Chunk nodes carry no vector
    property, only Qdrant does. A chunk's full embedding must not be
    transmitted to Neo4j's query engine just to be ignored there."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi", "vector": [0.1] * 1536}]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    _, params = execute.call_args[0]
    assert "vector" not in params["batch"][0]


# --- vec_chunks push (parity review -- reference's chunk-vector collection, previously missing) ---

def test_write_chunks_pushes_upstream_vector_to_qdrant():
    """The chunk's own pre-computed ES embedding gets copied to Qdrant --
    never re-embedded (the embedding-space rule)."""
    repo = _repo()
    vector = [0.1] * 1536
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": vector}}},
    ]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}), _bound_run_cfg():
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.upsert_entities_batch.assert_called_once_with([{
        "entity_id": "c1", "embedding": vector, "type": "CHUNK",
        "tenant_id": TENANT, "acl": ["public"],
    }])


def test_write_chunks_skips_qdrant_push_for_wrong_dimension_vector():
    """Confirmed live-parity bug: an upstream export can populate the vector
    field with a wrong-dimension vector (data-quality issue, not just a
    missing field). Qdrant's upsert is all-or-nothing per batch, so a
    wrong-dimension vector must be filtered out BEFORE the push rather than
    left to fail server-side -- otherwise it takes every correctly-dimensioned
    sibling chunk in the same call down with it (see c-good below, which must
    still land)."""
    repo = _repo()
    good_vector = [0.1] * 1536
    bad_vector = [0.2] * 128  # wrong dimension under the one recognized field
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c-good", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": good_vector}}},
        {"chunk_id": "c-bad", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": bad_vector}}},
    ]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}), _bound_run_cfg():
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.upsert_entities_batch.assert_called_once_with([{
        "entity_id": "c-good", "embedding": good_vector, "type": "CHUNK",
        "tenant_id": TENANT, "acl": ["public"],
    }])


# --- consume_upstream_vector: copy-or-skip vs embed-everything ---

def test_write_chunks_consume_true_never_calls_the_embedder():
    """consume_upstream_vector=true: a mismatched/missing vector gets no vector
    at all, even when an embedder is given -- the flag itself forbids us
    embedding chunks."""
    repo = _repo()
    bad_vector = [0.2] * 128
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c-bad", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": bad_vector}}},
    ]}}
    chunks_vector_store = MagicMock()
    embedder = MagicMock()
    with patch.object(repo, "_execute", return_value={}), \
         patch("repository.neo4j_repository.effective_run_config", return_value=_run_cfg(consume=True)):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]},
                          chunks_vector_store=chunks_vector_store, embedder=embedder)
    embedder.embed_batch.assert_not_called()
    chunks_vector_store.upsert_entities_batch.assert_not_called()


def test_write_chunks_consume_false_embeds_every_chunk_even_with_a_matching_upstream_vector():
    """consume_upstream_vector=false: the upstream vector is never trusted --
    every chunk, including one with a perfectly good upstream vector, is
    embedded by us in the run's own space."""
    repo = _repo()
    upstream_vector = [0.1] * 1536
    fresh_vector = [0.9] * 1536
    fresh_vector_2 = [0.8] * 1536
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": upstream_vector}}},
        {"chunk_id": "c2", "text": "no vector here"},
    ]}}
    chunks_vector_store = MagicMock()
    embedder = MagicMock()
    embedder.embed_batch.return_value = [fresh_vector, fresh_vector_2]
    with patch.object(repo, "_execute", return_value={}), \
         patch("repository.neo4j_repository.effective_run_config", return_value=_run_cfg(consume=False)):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]},
                          chunks_vector_store=chunks_vector_store, embedder=embedder)
    embedder.embed_batch.assert_called_once_with(["hi", "no vector here"])
    pushed = chunks_vector_store.upsert_entities_batch.call_args.args[0]
    assert {"entity_id": "c1", "embedding": fresh_vector, "type": "CHUNK",
            "tenant_id": TENANT, "acl": ["public"]} in pushed
    assert {"entity_id": "c2", "embedding": fresh_vector_2, "type": "CHUNK",
            "tenant_id": TENANT, "acl": ["public"]} in pushed
    assert len(pushed) == 2


def test_write_chunks_consume_false_skips_without_an_embedder():
    """consume_upstream_vector=false with no embedder given: nothing to embed
    with, so every chunk is skipped (never falls back to the upstream vector --
    that's exactly what this flag says not to trust)."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": [0.1] * 1536}}},
    ]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}), \
         patch("repository.neo4j_repository.effective_run_config", return_value=_run_cfg(consume=False)):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.upsert_entities_batch.assert_not_called()


def test_write_chunks_rejects_a_freshly_embedded_vector_not_equal_to_the_runs_dimension():
    """Defense in depth: even a freshly embedded vector must EQUAL the run's
    own dimension -- an unexpected model/config change must not silently push
    a vector Qdrant would otherwise choke on."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi"}]}}
    chunks_vector_store = MagicMock()
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 999]  # not the run's dimension
    with patch.object(repo, "_execute", return_value={}), \
         patch("repository.neo4j_repository.effective_run_config", return_value=_run_cfg(consume=False)):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]},
                          chunks_vector_store=chunks_vector_store, embedder=embedder)
    chunks_vector_store.upsert_entities_batch.assert_not_called()


def test_write_chunks_rejects_an_upstream_vector_of_another_spaces_dimension():
    """The guard is equality with THIS run's dimension, not membership in a
    global accepted set: a 1536-d vector must not pass into a 256-d-configured
    space just because 1536 is a dimension some other space uses."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": [0.1] * 1536}}},
    ]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}), \
         patch("repository.neo4j_repository.effective_run_config",
               return_value=_run_cfg(consume=True, dim=256)):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.upsert_entities_batch.assert_not_called()


def test_write_chunks_skips_qdrant_push_for_chunks_with_no_upstream_vector():
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [{"chunk_id": "c1", "text": "hi"}]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}):
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    chunks_vector_store.upsert_entities_batch.assert_not_called()


def test_write_chunks_skips_qdrant_entirely_when_no_vector_store_given():
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": "c1", "text": "hi", "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": [0.1]}}},
    ]}}
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]})
    execute.assert_called_once()  # just the Neo4j write, no Qdrant call to assert on either


def test_write_chunks_chunks_qdrant_push_at_256():
    """Same fixed value as the entity-vector push and the reference's own
    storage/vector_store.py B = 256 -- a large chunk batch can exceed Qdrant's
    request-body limit the same way a large entity batch can."""
    repo = _repo()
    doc = {"id": "doc-1", "_chunks": {"v1": [
        {"chunk_id": f"c{i}", "text": "hi",
         "_embeddings": {"embeddings_openai_large_1536": {"text_embedding": [0.1] * 1536}}}
        for i in range(300)
    ]}}
    chunks_vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}), _bound_run_cfg():
        repo.write_chunks(_sd_list([doc]), TENANT, {"doc-1": ["public"]}, chunks_vector_store=chunks_vector_store)
    assert chunks_vector_store.upsert_entities_batch.call_count == 2
    assert len(chunks_vector_store.upsert_entities_batch.call_args_list[0].args[0]) == 256
    assert len(chunks_vector_store.upsert_entities_batch.call_args_list[1].args[0]) == 44


def test_get_document_chunk_rids_returns_rid_to_chunk_id_map():
    repo = _repo()
    response = {"data": {"fields": ["rid", "chunk_id"],
                         "values": [["rid-1", "c1"], ["rid-2", "c2"]]}}
    with patch.object(repo, "_execute", return_value=response) as execute:
        result = repo.get_document_chunk_rids("doc-1", TENANT)
    assert result == {"rid-1": "c1", "rid-2": "c2"}
    _, params = execute.call_args[0]
    assert params == {"doc_id": "doc-1", "tenant_id": TENANT}


def test_get_document_chunk_rids_empty_for_brand_new_doc():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}):
        assert repo.get_document_chunk_rids("brand-new-doc", TENANT) == {}


def test_retract_chunks_short_circuits_on_empty_list_without_any_query():
    repo = _repo()
    with patch.object(repo, "_execute") as execute:
        stats = repo.retract_chunks("doc-1", TENANT, [])
    execute.assert_not_called()
    assert stats == {"relationships_pruned": 0, "claims_deleted": 0, "chunks_deleted": 0, "entities_deleted": 0}


def test_retract_chunks_runs_four_statements_and_never_touches_entity_doc_ids_or_acl():
    """Unlike replace_document_chunks/_retract_chunks_and_dependents, a partial
    chunk-diff retraction must not decrement entity doc_ids/support/acl or run
    the whole-tenant zero-support sweep -- the owning Document isn't being
    removed as a contributor, only these specific stale chunks are. (It DOES
    still run a narrower, live orphan check -- see the dedicated test below --
    but this scenario has no candidate entities, so that step is skipped.)"""
    repo = _repo()
    responses = [_entity_id_rows([]), _n(1), _n(2), _n(3)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.retract_chunks("doc-1", TENANT, ["c-old-1", "c-old-2"])
    assert execute.call_count == 4
    assert stats == {"relationships_pruned": 1, "claims_deleted": 2, "chunks_deleted": 3, "entities_deleted": 0}
    for call in execute.call_args_list:
        cypher, params = call.args[0], call.args[1]
        assert "entities_decremented" not in cypher
        assert "e.doc_ids" not in cypher
        assert params["chunk_ids"] == ["c-old-1", "c-old-2"]


def test_retract_chunks_deletes_chunk_vectors_when_chunks_vector_store_given():
    """Parity review -- reference's vec_chunks collection: this is the
    partial chunk-diff path (an UPDATE that dropped some chunks) -- those
    specific chunk_ids' vectors must go too, same as a full retraction."""
    repo = _repo()
    chunks_vector_store = MagicMock()
    responses = [_entity_id_rows([]), _n(1), _n(2), _n(3)]
    with patch.object(repo, "_execute", side_effect=responses):
        repo.retract_chunks("doc-1", TENANT, ["c-old-1", "c-old-2"],
                            chunks_vector_store=chunks_vector_store)
    chunks_vector_store.delete_entities.assert_called_once_with(["c-old-1", "c-old-2"])


def test_retract_chunks_deletes_the_given_chunk_nodes():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_entity_id_rows([])) as execute:
        repo.retract_chunks("doc-1", TENANT, ["c-old-1"])
    chunk_delete_call = execute.call_args_list[3]
    assert "DETACH DELETE c" in chunk_delete_call.args[0]
    assert chunk_delete_call.args[1]["chunk_ids"] == ["c-old-1"]


def test_retract_chunks_deletes_entities_that_become_truly_orphaned():
    """Mirrors the reference implementation's own scope-safe, live-checked
    orphan sweep on a partial update: an entity mentioned only by a DROPPED
    chunk, with no relationships of any kind left anywhere, must be cleaned
    up in both Neo4j and Qdrant -- not left behind as a stale, unreachable
    node with an orphaned vector."""
    repo = _repo()
    vector_store = MagicMock()
    responses = [_entity_id_rows(["ent-orphan"]), _n(0), _n(0), _n(0), _entity_id_rows(["ent-orphan"])]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.retract_chunks("doc-1", TENANT, ["c-old-1"], vector_store=vector_store)
    assert stats["entities_deleted"] == 1
    orphan_check_cypher, orphan_check_params = execute.call_args_list[4].args
    assert "NOT (e)--()" in orphan_check_cypher
    assert orphan_check_params["candidate_ids"] == ["ent-orphan"]
    vector_store.delete_entities.assert_called_once_with(["ent-orphan"])


def test_retract_chunks_skips_orphan_check_entirely_when_no_candidates():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_entity_id_rows([]), _n(0), _n(0), _n(0)]) as execute:
        stats = repo.retract_chunks("doc-1", TENANT, ["c-old-1"])
    assert execute.call_count == 4
    assert stats["entities_deleted"] == 0


def test_write_entities_sets_allow_access_control_property():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": [0.1], "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "e.allow_access_control = merged_acl" in cypher
    assert "e.acl " not in cypher and not cypher.rstrip().endswith("e.acl")
    assert params["batch"][0]["acl"] == ["public"]


# --- entity vectors moved to Qdrant, out of Neo4j (embeddings architecture change) ---

def test_write_entities_sets_has_vector_true_and_never_writes_e_embedding():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}) as execute:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": [0.1], "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, _ = execute.call_args[0]
    assert "e.has_vector" in cypher
    assert "e.embedding" not in cypher


def test_write_entities_uses_explicit_name_override_when_present():
    """Experiment-parity review #8, Phase 2: PERSON entries are keyed by
    entity_id itself (two different real people sharing a name can't be
    dict keys), with the real display name carried in data["name"]."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}) as execute:
        repo.write_entities({
            "ent_person1": {"type": "PERSON", "name": "Jane Doe", "source_docs": ["doc-1"],
                            "embedding": [0.1], "aliases": [], "acl": ["public"]},
        }, TENANT, canon_to_id={"ent_person1": "ent_person1"})
    _, params = execute.call_args[0]
    assert params["batch"][0]["name"] == "Jane Doe"
    assert params["batch"][0]["entity_id"] == "ent_person1"


def test_write_entities_writes_person_fields_through_without_reading_them():
    """MATERIALIZE's output (services/people/materialize.py) is written with
    `SET e += ...` and never inspected here, so a connector's own person
    properties need no change to write_entities. A source_key is also what
    earns the second :Person label."""
    repo = _repo()
    person_fields = {"source_key": "hr-1", "connector_type": "zeus", "email": "j@x.com",
                     "person_user_id": "hr-1", "person_manager_id": "hr-mgr-1",
                     "person_department": "R&D", "person_whatever_this_source_knows": "yes"}
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}) as execute:
        repo.write_entities({
            "ent_person1": {
                "type": "PERSON", "name": "Jane Doe", "source_docs": ["doc-1"],
                "embedding": [0.1], "aliases": [], "acl": ["public"],
                "person_fields": person_fields,
            },
        }, TENANT, canon_to_id={"ent_person1": "ent_person1"})
    cypher, params = execute.call_args[0]
    assert params["batch"][0]["person_fields"] == person_fields
    assert "SET e += row.person_fields" in cypher
    assert "SET e:Person" in cypher


def test_write_entities_person_fields_are_empty_when_absent():
    """A bare-name PERSON mention (no person_fields key at all) must not get a
    source_key, so it stays an unlabeled :Entity — `SET e += {}` is a no-op and
    never blanks a previously-known value."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}) as execute:
        repo.write_entities({
            "Some Mentioned Person": {"type": "PERSON", "source_docs": ["doc-1"],
                                      "embedding": [], "aliases": [], "acl": ["public"]},
        }, TENANT, canon_to_id={"Some Mentioned Person": "ent_mention1"})
    cypher, params = execute.call_args[0]
    assert params["batch"][0]["person_fields"] == {}
    assert "row.person_fields.source_key IS NULL THEN [] ELSE [1] END" in cypher


def test_write_entities_rejects_a_vector_not_equal_to_the_runs_dimension():
    """Defense in depth: entity embeddings must EQUAL the run's own dimension
    before the Qdrant push — same rule as write_chunks. has_vector is rolled
    back so Neo4j never claims a vector Qdrant doesn't have."""
    repo = _repo()
    vector_store = MagicMock()
    bad_embedding = [0.1] * 999
    entity_response = {
        "data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                 "values": [["ent_globex", "ORGANIZATION", bad_embedding, ["public"]]]},
    }
    rollback_response = {"data": {"fields": ["n"], "values": [[1]]}}
    with patch.object(repo, "_execute", side_effect=[entity_response, rollback_response]) as execute, \
         patch("repository.neo4j_repository.effective_run_config", return_value=_run_cfg()):
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": bad_embedding, "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"}, vector_store=vector_store)
    vector_store.upsert_entities_batch.assert_not_called()
    assert execute.call_count == 2
    rollback_cypher, rollback_params = execute.call_args_list[1].args
    assert "SET e.has_vector = false" in rollback_cypher
    assert rollback_params["entity_ids"] == ["ent_globex"]


def test_write_entities_pushes_embeddings_to_qdrant_with_the_final_merged_acl():
    """The Qdrant push must use the ACTUAL final (unioned) acl the Cypher
    RETURNs, not just this batch's own partial view -- same "acl is a union,
    computed once, reused everywhere" principle the Neo4j side follows."""
    repo = _repo()
    vector_store = MagicMock()
    entity_response = {
        "data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                 "values": [["ent_globex", "ORGANIZATION", VALID_ENTITY_EMB, ["public", "group:site:s1"]]]},
    }
    with patch.object(repo, "_execute", return_value=entity_response), _bound_run_cfg():
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": VALID_ENTITY_EMB, "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"}, vector_store=vector_store)
    vector_store.upsert_entities_batch.assert_called_once_with([{
        "entity_id": "ent_globex", "embedding": VALID_ENTITY_EMB, "type": "ORGANIZATION",
        "tenant_id": TENANT, "acl": ["public", "group:site:s1"],
    }])


def test_write_entities_chunks_qdrant_push_at_256_to_stay_under_its_payload_limit():
    """Regression test for a real failure found during the 100-doc parity dry
    run: pushing a large batch's vectors to Qdrant in one call can exceed
    Qdrant's own request-body limit regardless of the Neo4j batch size
    (confirmed against the real dev server: 1280 entities' vectors serialized
    to ~50MB, over Qdrant's 32MB limit). Must chunk the Qdrant push at 256 --
    same fixed value the reference implementation already uses for this
    exact problem (storage/vector_store.py's own B = 256) -- rather than
    reusing whatever size Neo4j's own write happened to use."""
    repo = _repo()
    repo._entity_batch_size = 1000  # one single outer Neo4j batch -- isolates the Qdrant-side chunking
    vector_store = MagicMock()
    entity_rows = {
        f"Entity{i}": {"type": "ORGANIZATION", "description": "d", "source_docs": [f"doc-{i}"],
                       "embedding": VALID_ENTITY_EMB, "aliases": [f"Entity{i}"], "acl": ["public"]}
        for i in range(300)
    }
    canon_to_id = {name: f"ent_{i}" for i, name in enumerate(entity_rows)}
    response = {
        "data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                 "values": [[f"ent_{i}", "ORGANIZATION", VALID_ENTITY_EMB, ["public"]] for i in range(300)]},
    }
    with patch.object(repo, "_execute", return_value=response), _bound_run_cfg():
        repo.write_entities(entity_rows, TENANT, canon_to_id=canon_to_id, vector_store=vector_store)
    assert vector_store.upsert_entities_batch.call_count == 2  # 256 + 44, not one 300-row call
    first_batch = vector_store.upsert_entities_batch.call_args_list[0].args[0]
    second_batch = vector_store.upsert_entities_batch.call_args_list[1].args[0]
    assert len(first_batch) == 256
    assert len(second_batch) == 44


# --- Dual-write safety (manager requirement, full-traceability project) ---

def test_write_entities_rolls_back_has_vector_when_qdrant_push_exhausts_retries():
    """Neo4j already committed has_vector = true as part of the SAME write
    above -- if the Qdrant push then fails 3 times, that flag must be
    corrected back to false, never left claiming a vector Qdrant doesn't
    actually have."""
    repo = _repo()
    vector_store = MagicMock()
    vector_store.upsert_entities_batch.side_effect = RuntimeError("qdrant down")
    entity_response = {
        "data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                 "values": [["ent_globex", "ORGANIZATION", VALID_ENTITY_EMB, ["public"]]]},
    }
    rollback_response = {"data": {"fields": ["n"], "values": [[1]]}}
    with patch.object(repo, "_execute", side_effect=[entity_response, rollback_response]) as execute, \
         _bound_run_cfg(), \
         patch("services.dual_write_guard.time.sleep"), \
         patch("services.dual_write_guard.record_call_failure") as record_call:
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": VALID_ENTITY_EMB, "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"}, vector_store=vector_store)
    assert vector_store.upsert_entities_batch.call_count == 3  # retried before giving up
    assert execute.call_count == 2  # entity write, then the has_vector rollback
    rollback_cypher, rollback_params = execute.call_args_list[1].args
    assert "SET e.has_vector = false" in rollback_cypher
    assert rollback_params["entity_ids"] == ["ent_globex"]
    record_call.assert_called_once()
    assert record_call.call_args.args[0] == "qdrant_entity_upsert_batch"


def test_write_entities_splits_batch_in_half_and_retries_on_timeout():
    """Regression test for a real failure found during the 100-doc parity dry
    run: calibrate_batch_sizes() times a batch of tiny synthetic rows, so its
    "safe" batch size can still be too heavy once real (larger) entities are
    written -- confirmed against the real dev server, where a calibrated-1280
    batch of real entities hit a 60s read timeout. write_entities must recover
    by halving the batch and retrying, not just fail after _execute's own
    retries are exhausted."""
    repo = _repo()
    entity_rows = {
        f"Entity{i}": {"type": "ORGANIZATION", "description": "d", "source_docs": [f"doc-{i}"],
                       "embedding": [], "aliases": [f"Entity{i}"], "acl": ["public"]}
        for i in range(4)
    }
    canon_to_id = {name: f"ent_{i}" for i, name in enumerate(entity_rows)}

    def _response_for(ids: list[str]) -> dict:
        return {"data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                         "values": [[eid, "ORGANIZATION", [], ["public"]] for eid in ids]}}

    with patch.object(repo, "_execute", side_effect=[
        TimeoutError("read timed out"),                       # full batch of 4 -- too slow
        _response_for(["ent_0", "ent_1"]),                    # first half succeeds
        _response_for(["ent_2", "ent_3"]),                    # second half succeeds
    ]) as execute:
        repo.write_entities(entity_rows, TENANT, canon_to_id=canon_to_id)
    assert execute.call_count == 3
    assert len(execute.call_args_list[0].args[1]["batch"]) == 4
    assert len(execute.call_args_list[1].args[1]["batch"]) == 2
    assert len(execute.call_args_list[2].args[1]["batch"]) == 2


def test_write_entities_splits_batch_on_http_400_for_multi_row_batches():
    """Neo4j Query API can return 400 for an oversized multi-row MERGE that
    succeeds when halved. Treat 400 like a size issue until a single row."""
    from requests import Response
    from requests.exceptions import HTTPError

    repo = _repo()
    entity_rows = {
        f"Entity{i}": {"type": "ORGANIZATION", "description": "d", "source_docs": [f"doc-{i}"],
                       "embedding": [], "aliases": [f"Entity{i}"], "acl": ["public"]}
        for i in range(4)
    }
    canon_to_id = {name: f"ent_{i}" for i, name in enumerate(entity_rows)}

    def _response_for(ids: list[str]) -> dict:
        return {"data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                         "values": [[eid, "ORGANIZATION", [], ["public"]] for eid in ids]}}

    bad = Response()
    bad.status_code = 400
    err = HTTPError("400 Client Error: Bad Request", response=bad)

    with patch.object(repo, "_execute", side_effect=[
        err,
        _response_for(["ent_0", "ent_1"]),
        _response_for(["ent_2", "ent_3"]),
    ]) as execute:
        repo.write_entities(entity_rows, TENANT, canon_to_id=canon_to_id)
    assert execute.call_count == 3
    assert len(execute.call_args_list[0].args[1]["batch"]) == 4
    assert len(execute.call_args_list[1].args[1]["batch"]) == 2
    assert len(execute.call_args_list[2].args[1]["batch"]) == 2


def test_write_entities_gives_up_splitting_at_a_single_row_and_propagates():
    """A single row that still times out is a real problem (a giant
    description, a stuck server) -- not a batch-size issue -- so it must
    propagate up to run_bulk_async's own retry-then-log handling instead of
    splitting forever."""
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=TimeoutError("read timed out")):
        with pytest.raises(TimeoutError):
            repo.write_entities({
                "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                            "embedding": [], "aliases": ["Globex"], "acl": ["public"]},
            }, TENANT, canon_to_id={"Globex": "ent_globex"})


def test_mark_vector_missing_sets_has_vector_false_for_given_entities():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(2)) as execute:
        n = repo.mark_vector_missing(TENANT, ["ent-1", "ent-2"])
    assert n == 2
    cypher, params = execute.call_args[0]
    assert "SET e.has_vector = false" in cypher
    assert params == {"tenant_id": TENANT, "entity_ids": ["ent-1", "ent-2"]}


def test_mark_vector_missing_empty_list_is_a_no_op():
    repo = _repo()
    with patch.object(repo, "_execute") as execute:
        n = repo.mark_vector_missing(TENANT, [])
    assert n == 0
    execute.assert_not_called()


def test_upsert_entity_incremental_rolls_back_has_vector_when_qdrant_push_exhausts_retries():
    repo = _repo()
    vector_store = MagicMock()
    vector_store.upsert_entity.side_effect = RuntimeError("qdrant down")
    entity_response = {"data": {"fields": ["descriptions_json", "aliases_json", "already_contributed", "acl"],
                                "values": [[None, None, False, ["public"]]]}}
    rollback_response = {"data": {"fields": ["n"], "values": [[1]]}}
    with patch.object(repo, "_execute", side_effect=[entity_response, entity_response, rollback_response]), \
         patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent-1"}), \
         patch("services.dual_write_guard.time.sleep"), \
         patch("services.dual_write_guard.record_call_failure") as record_call:
        repo.upsert_entity_incremental(
            "Globex", "ORGANIZATION", "desc", [0.1, 0.2], "doc-1", ["public"], TENANT,
            vector_store=vector_store,
        )
    assert vector_store.upsert_entity.call_count == 3
    record_call.assert_called_once()
    assert record_call.call_args.args[0] == "qdrant_entity_upsert"


def test_write_entities_skips_qdrant_rows_with_no_embedding():
    repo = _repo()
    vector_store = MagicMock()
    response = {
        "data": {"fields": ["entity_id", "type", "embedding", "merged_acl"],
                 "values": [["ent_globex", "ORGANIZATION", [], ["public"]]]},
    }
    with patch.object(repo, "_execute", return_value=response):
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": [], "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"}, vector_store=vector_store)
    vector_store.upsert_entities_batch.assert_not_called()


def test_write_entities_skips_qdrant_entirely_when_no_vector_store_given():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": [], "values": []}}):
        # Must not raise even though vector_store defaults to None.
        repo.write_entities({
            "Globex": {"type": "ORGANIZATION", "description": "d", "source_docs": ["doc-1"],
                        "embedding": [0.1], "aliases": ["Globex"], "acl": ["public"]},
        }, TENANT, canon_to_id={"Globex": "ent_globex"})


def test_write_relationships_sets_allow_access_control_property():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": [],
                                       "acl": ["public"], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    cypher, params = execute.call_args[0]
    assert "r.allow_access_control = merged_acl" in cypher
    assert "r.acl " not in cypher
    assert params["batch"][0]["acl"] == ["public"]


# --- single EXTRACTED_REL relationship type (experiment-parity review #26) ---

def test_write_relationships_uses_a_single_extracted_rel_label():
    """A dynamic per-type Neo4j label needed Cypher-injection sanitization on
    every call and can't be parameterized; a single fixed EXTRACTED_REL label
    with relationship_type as a normal property mirrors the experiment's
    graph_writer.py and lets the type be passed as an ordinary $batch value."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "ADDS_FEATURE", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": [],
                                         "acl": [], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    cypher, params = execute.call_args[0]
    assert "MERGE (src)-[r:EXTRACTED_REL {relationship_type: row.relationship_type}]->(tgt)" in cypher
    assert ":ADDS_FEATURE" not in cypher
    assert params["batch"][0]["relationship_type"] == "ADDS_FEATURE"


def test_write_relationships_keys_the_merge_on_relationship_type_so_distinct_types_dont_collapse():
    """Two different real-world relationship types between the same pair of
    entities must stay two distinct edges, not collapse into one just because
    they now share the same Neo4j label."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "FOUNDED", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": [],
                                    "acl": [], "src_type": "X", "tgt_type": "Y"},
            ("A", "WORKS_AT", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": [],
                                     "acl": [], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    _, params = execute.call_args[0]
    rel_types = {row["relationship_type"] for row in params["batch"]}
    assert rel_types == {"FOUNDED", "WORKS_AT"}
    assert len(params["batch"]) == 2


# --- edge chunk_ids/support overwrite (experiment-parity review #6) ---

def test_write_relationships_merges_chunk_ids_into_existing_rather_than_overwriting():
    """Regression test: on the incremental path write_relationships is called once
    PER DOCUMENT, so a plain SET r.chunk_ids = row.chunk_ids would silently erase
    whatever a previously-ingested document had already contributed to a shared
    edge. Must union into r.chunk_ids, the same coalesce-and-append pattern
    write_entities already uses for e.doc_ids."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [], "chunk_ids": ["c2"], "raw_types": ["related_to"],
                                       "acl": ["public"], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    cypher, params = execute.call_args[0]
    assert "coalesce(r.chunk_ids, [])" in cypher
    assert "r.chunk_ids = row.chunk_ids" not in cypher
    assert params["batch"][0]["chunk_ids"] == ["c2"]


def test_write_relationships_writes_support_as_size_of_merged_chunk_ids():
    """r.support was never written at all before this fix -- a ranking signal
    that must reflect every contributing chunk, not just the most recent call's."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [], "chunk_ids": ["c1"], "raw_types": [],
                                       "acl": [], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    cypher, _ = execute.call_args[0]
    assert "r.support = size(merged_chunk_ids)" in cypher


def test_write_relationships_merges_raw_types_and_acl_into_existing_rather_than_overwriting():
    """Same clobbering bug applies to raw_types and allow_access_control -- both
    are native list properties that must union with what's already on the edge,
    not overwrite it, for the same incremental per-document reason as chunk_ids."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [], "chunk_ids": [], "raw_types": ["related_to"],
                                       "acl": ["public"], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    cypher, _ = execute.call_args[0]
    assert "coalesce(r.raw_types, [])" in cypher
    assert "coalesce(r.allow_access_control, [])" in cypher
    assert "r.raw_types = row.raw_types" not in cypher
    assert "r.allow_access_control = row.acl" not in cypher


# --- descriptions_json merge, not overwrite (edge overwrite bug #24) ---

def test_write_relationships_merges_descriptions_with_what_already_exists():
    """Unlike chunk_ids/raw_types/acl (native lists Cypher can coalesce),
    descriptions_json is a JSON-encoded string -- must be read, merged in
    Python, and written back, not just overwritten with this call's own data."""
    repo = _repo()
    # Column names are exactly what the pre-read RETURNs (src_id,
    # relationship_type, tgt_id, descriptions_json) — _read_existing_edge_
    # descriptions indexes the row by those names.
    existing_response = {
        "data": {"fields": ["src_id", "relationship_type", "tgt_id", "descriptions_json"],
                 "values": [["A", "RELATED_TO", "B", json.dumps([{"text": "old", "chunk_id": "c1"}])]]},
    }
    with patch.object(repo, "_execute", side_effect=[existing_response, {}]) as execute, \
         patch.object(repo, "_run_batched") as run_batched:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [{"text": "new", "chunk_id": "c2"}],
                                       "chunk_ids": ["c2"], "raw_types": ["related_to"],
                                       "acl": ["public"], "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    assert execute.call_count == 1  # the pre-read; the write itself goes through _run_batched
    _, rows, _ = run_batched.call_args[0]
    descriptions = json.loads(rows[0]["descriptions_json"])
    assert descriptions == [{"text": "old", "chunk_id": "c1"}, {"text": "new", "chunk_id": "c2"}]


def test_write_relationships_reads_nothing_extra_when_edge_is_new():
    """A brand-new edge (no prior descriptions_json on Neo4j) must not error
    or invent phantom entries -- the pre-read simply returns no match."""
    repo = _repo()
    no_match_response = {"data": {"fields": [], "values": []}}
    with patch.object(repo, "_execute", return_value=no_match_response), \
         patch.object(repo, "_run_batched") as run_batched:
        repo.write_relationships({
            ("A", "RELATED_TO", "B"): {"descriptions": [{"text": "new", "chunk_id": "c1"}],
                                       "chunk_ids": ["c1"], "raw_types": [], "acl": [],
                                       "src_type": "X", "tgt_type": "Y"},
        }, TENANT)
    _, rows, _ = run_batched.call_args[0]
    descriptions = json.loads(rows[0]["descriptions_json"])
    assert descriptions == [{"text": "new", "chunk_id": "c1"}]


def test_write_relationships_skips_the_pre_read_when_there_are_no_edges():
    repo = _repo()
    with patch.object(repo, "_execute") as execute, patch.object(repo, "_run_batched") as run_batched:
        repo.write_relationships({}, TENANT)
    execute.assert_not_called()
    run_batched.assert_called_once()


# --- claim -> chunk provenance edge (experiment-parity review #12) ---

def test_write_claims_links_supported_by_chunk_edge():
    """A claim's source-chunk link must be a real, traversable graph edge --
    same as (:Claim)-[:ABOUT]->(:Entity) -- not just a chunk_id property that
    only a separate lookup (never a graph query) could follow."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "A", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    cypher, params = execute.call_args[0]
    assert "MATCH (chunk:Chunk {chunk_id: row.chunk_id})" in cypher
    assert "MERGE (cl)-[:SUPPORTED_BY]->(chunk)" in cypher
    assert params["batch"][0]["chunk_id"] == "c1"


def test_write_claims_sets_allow_access_control_property():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "A", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    cypher, params = execute.call_args[0]
    assert "cl.allow_access_control = row.acl" in cypher
    assert "cl.acl " not in cypher
    assert params["batch"][0]["acl"] == ["public"]


# --- documented_on + claim serving indexes (experiment-parity review #36) ---

def test_write_claims_sets_subject_and_documented_on_properties():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"], "documented_on": "2026-01-15",
        }], TENANT)
    cypher, params = execute.call_args[0]
    assert "cl.subject = row.subject" in cypher
    assert "cl.documented_on = row.documented_on" in cypher
    assert params["batch"][0]["subject"] == "ent-1"
    assert params["batch"][0]["documented_on"] == "2026-01-15"


def test_write_claims_writes_documented_on_as_empty_string_not_absent_when_missing():
    """Reverses an earlier decision that wrote null here so the property would be
    entirely absent — Neo4j has no way to store an actual null value, so a `SET
    cl.documented_on = null` just never creates the property, and callers who rely
    on `keys(cl)` containing documented_on always find it missing on the ~7,364
    people-profile claims with no publish date. The field must always exist on
    every Claim now, so this writes the honest-unknown "" instead."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    _, params = execute.call_args[0]
    assert params["batch"][0]["documented_on"] == ""


def test_write_claims_keeps_upstreams_empty_documented_on_as_empty_string():
    """edge_pool_service emits "" for a dateless document; write_claims no longer
    converts that to null (see test above), so it passes through unchanged."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"], "documented_on": "",
        }], TENANT)
    _, params = execute.call_args[0]
    assert params["batch"][0]["documented_on"] == ""


def test_write_claims_writes_valid_from_and_valid_to_as_empty_string_not_absent():
    """Same reasoning as documented_on: the LLM extraction schema returns these as
    None when the source text gives no date, and a null SET would leave the
    property key entirely absent on every claim without one."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "DYNAMIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    _, params = execute.call_args[0]
    assert params["batch"][0]["valid_from"] == ""
    assert params["batch"][0]["valid_to"] == ""


def test_write_claims_keeps_a_real_valid_from_and_valid_to_untouched():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "DYNAMIC", "valid_from": "2024-01-01",
            "valid_to": "2024-06-01", "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    _, params = execute.call_args[0]
    assert params["batch"][0]["valid_from"] == "2024-01-01"
    assert params["batch"][0]["valid_to"] == "2024-06-01"


def test_create_constraints_creates_claim_serving_indexes():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute, \
         patch.object(repo, "calibrate_batch_sizes"):
        repo.create_constraints()
    statements = [call.args[0] for call in execute.call_args_list]
    assert any("claim_subject" in s and "cl.subject" in s for s in statements)
    assert any("claim_valid_from" in s and "cl.valid_from" in s for s in statements)
    assert any("claim_documented_on" in s and "cl.documented_on" in s for s in statements)


def test_upsert_entity_incremental_sets_allow_access_control_property():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="d", embedding=[0.1],
            doc_id="doc-1", acl=["public"], tenant_id=TENANT,
        )
    cypher, params = execute.call_args[0]
    assert "e.allow_access_control = $acl" in cypher
    assert "e.allow_access_control = CASE" in cypher
    assert "e.acl" not in cypher
    assert params["acl"] == ["public"]


def test_create_constraints_creates_allow_access_control_indexes_not_acl():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute, \
         patch.object(repo, "calibrate_batch_sizes"):
        repo.create_constraints()
    statements = [call.args[0] for call in execute.call_args_list]
    assert any("document_allow_access_control" in s and "d.allow_access_control" in s for s in statements)
    assert not any("document_acl" in s for s in statements)


# --- entity embedding refresh on update (experiment-parity review #31),
# now pushed to Qdrant instead of stored on the Entity node ---

def test_upsert_entity_incremental_sets_has_vector_true_when_a_new_embedding_is_given():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="d", embedding=[0.9, 0.9],
            doc_id="doc-2", acl=["public"], tenant_id=TENANT,
        )
    cypher, params = execute.call_args[0]
    assert "e.has_vector" in cypher
    assert "e.embedding" not in cypher
    assert params["has_new_embedding"] is True


def test_upsert_entity_incremental_pushes_the_new_embedding_to_qdrant():
    """A repeat mention of an existing entity with a freshly computed
    embedding must refresh its Qdrant vector, not leave it frozen at
    whatever text first created the node."""
    repo = _repo()
    vector_store = MagicMock()
    with patch.object(repo, "_execute",
                      return_value={"data": {"fields": ["acl"], "values": [[["public"]]]}}):
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="d", embedding=[0.9, 0.9],
            doc_id="doc-2", acl=["public"], tenant_id=TENANT, vector_store=vector_store,
        )
    vector_store.upsert_entity.assert_called_once_with(
        "ent_default", [0.9, 0.9], "ORGANIZATION", TENANT, ["public"],
    )


def test_upsert_entity_incremental_skips_qdrant_push_when_no_new_embedding_computed():
    """The exact-name fast path (review #10) intentionally skips computing a
    new embedding and passes None -- must not push a null/empty vector to
    Qdrant, and must not touch e.has_vector's existing true value either."""
    repo = _repo()
    vector_store = MagicMock()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="d", embedding=None,
            doc_id="doc-2", acl=["public"], tenant_id=TENANT, vector_store=vector_store,
        )
    cypher, params = execute.call_args[0]
    assert "coalesce(e.has_vector, false)" in cypher
    assert params["has_new_embedding"] is False
    vector_store.upsert_entity.assert_not_called()


def test_upsert_entity_incremental_skips_qdrant_entirely_when_no_vector_store_given():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}):
        # Must not raise even though vector_store defaults to None.
        entity_id = repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="d", embedding=[0.1],
            doc_id="doc-2", acl=["public"], tenant_id=TENANT,
        )
    assert entity_id == "ent_default"


# --- native e.aliases + fulltext index (experiment-parity review #22) ---

def test_create_constraints_creates_entity_aliases_fulltext_index():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute, \
         patch.object(repo, "calibrate_batch_sizes"):
        repo.create_constraints()
    statements = [call.args[0] for call in execute.call_args_list]
    assert any("CREATE FULLTEXT INDEX entity_aliases" in s and "[e.aliases]" in s for s in statements)


def test_create_constraints_creates_broadened_entity_search_fulltext_index():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute, \
         patch.object(repo, "calibrate_batch_sizes"):
        repo.create_constraints()
    statements = [call.args[0] for call in execute.call_args_list]
    assert any("CREATE FULLTEXT INDEX entity_search" in s
              and "e.name" in s and "e.aliases" in s and "e.person_search_terms" in s
              for s in statements)


def test_write_entities_writes_a_denormalized_native_aliases_list():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}) as execute:
        repo.write_entities({
            "Globex": {
                "type": "ORGANIZATION", "source_docs": ["doc-1"], "embedding": [0.1],
                "aliases": [
                    {"name": "Globex Inc.", "doc_id": "doc-1", "allow_access_control": ["public"]},
                    {"name": "Globex", "doc_id": "doc-1", "allow_access_control": ["public"]},
                ],
                "acl": ["public"],
            },
        }, TENANT, canon_to_id={"Globex": "ent_globex"})
    cypher, params = execute.call_args[0]
    assert "e.aliases = row.aliases" in cypher
    assert params["batch"][0]["aliases"] == ["Globex", "Globex Inc."]


def test_upsert_entity_incremental_writes_a_denormalized_native_aliases_list():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_no_existing_entity_response(), {}]) as execute:
        repo.upsert_entity_incremental(
            name="Globex", entity_type="ORGANIZATION", description="A company",
            embedding=[0.1], doc_id="doc-1", acl=["public"], tenant_id=TENANT, raw_name="Globex Inc.",
        )
    cypher, params = execute.call_args_list[1].args
    assert "e.aliases = $aliases_flat" in cypher
    assert params["aliases_flat"] == ["Globex Inc."]


def test_retract_document_delegates_shared_steps_to_retract_chunks_and_dependents():
    """retract_document = _retract_chunks_and_dependents (steps 1-6) + delete
    the Document node (step 7) — proves the refactor kept that composition."""
    repo = _repo()
    with patch.object(repo, "_retract_chunks_and_dependents",
                      return_value={"entities_decremented": 1}) as shared_steps, \
         patch.object(repo, "_execute", return_value=_n(1)) as execute:
        stats = repo.retract_document("doc-1", TENANT)
    shared_steps.assert_called_once_with("doc-1", TENANT, None, None)
    execute.assert_called_once()  # only the Document-node delete itself
    assert stats["entities_decremented"] == 1
    assert stats["doc_deleted"] is True


# --- update_document_acl (acl_update event, search-utility-agents confirmed) ---

def test_update_document_acl_runs_five_statements_and_reports_stats():
    repo = _repo()
    responses = [_n(1), _chunk_ids_row(["c1", "c2"]), _n(1),
                _entity_acl_rows([("ent-a", ["public"]), ("ent-b", ["public"]), ("ent-c", ["public"])]),
                _n(2)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        stats = repo.update_document_acl("doc-1", TENANT, ["group:site:site-1"])
    assert execute.call_count == 5
    assert stats == {
        "document_updated": True, "chunks_updated": 2, "claims_updated": 1,
        "entities_acl_recomputed": 3, "relationships_acl_recomputed": 2,
    }
    for call in execute.call_args_list:
        params = call.args[1]
        if "doc_id" in params:
            assert params["doc_id"] == "doc-1"
        if "tenant_id" in params:
            assert params["tenant_id"] == TENANT


def test_update_document_acl_sets_document_and_chunk_acl_exactly():
    repo = _repo()
    responses = [_n(1), _n(2), _n(0), _n(0), _n(0)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.update_document_acl("doc-1", TENANT, ["group:site:site-1"])
    doc_cypher, doc_params = execute.call_args_list[0].args
    assert "SET d.allow_access_control = $acl" in doc_cypher
    assert doc_params["acl"] == ["group:site:site-1"]
    chunk_cypher, chunk_params = execute.call_args_list[1].args
    assert "SET c.allow_access_control = $acl" in chunk_cypher
    assert chunk_params["acl"] == ["group:site:site-1"]


def test_update_document_acl_syncs_chunk_acl_to_qdrant_when_chunks_vector_store_given():
    """Parity review -- reference's vec_chunks collection: every chunk of this
    document gets the SAME new acl in Neo4j (exact, own-document semantics),
    so each one's Qdrant payload must be synced too -- otherwise a permission
    REVOCATION would correct Neo4j but leave Qdrant's vec_chunks holding the
    old, more permissive acl forever, same security gap as the entity-acl
    sync above but at the chunk level."""
    repo = _repo()
    chunks_vector_store = MagicMock()
    responses = [_n(1), _chunk_ids_row(["c1", "c2"]), _n(0), _entity_acl_rows([]), _n(0)]
    with patch.object(repo, "_execute", side_effect=responses):
        repo.update_document_acl("doc-1", TENANT, ["group:site:site-1"],
                                 chunks_vector_store=chunks_vector_store)
    assert chunks_vector_store.set_acl.call_count == 2
    chunks_vector_store.set_acl.assert_any_call("c1", ["group:site:site-1"])
    chunks_vector_store.set_acl.assert_any_call("c2", ["group:site:site-1"])


def test_update_document_acl_recomputes_entity_acl_as_a_live_union():
    """Must recompute (not overwrite with the raw acl param) -- an entity's
    acl is the union across every contributing document's CURRENT acl, not
    just this one document's."""
    repo = _repo()
    responses = [_n(1), _n(0), _n(0), _n(2), _n(0)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.update_document_acl("doc-1", TENANT, ["group:site:site-1"])
    entity_cypher, _ = execute.call_args_list[3].args
    assert "e.allow_access_control = reduce" in entity_cypher
    assert "coalesce(e.doc_ids, [])" in entity_cypher


def test_update_document_acl_recomputes_relationship_acl_as_a_live_union():
    repo = _repo()
    responses = [_n(1), _n(0), _n(0), _n(0), _n(2)]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        repo.update_document_acl("doc-1", TENANT, ["group:site:site-1"])
    rel_cypher, _ = execute.call_args_list[4].args
    assert "r.allow_access_control = reduce" in rel_cypher


def test_update_document_acl_is_a_safe_no_op_for_an_unknown_doc_id():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(0)) as execute:
        stats = repo.update_document_acl("nonexistent-doc", TENANT, ["public"])
    assert execute.call_count == 5
    assert stats["document_updated"] is False
    assert stats["chunks_updated"] == 0


# --- e.degree (experiment-parity review #35) ---

def test_recompute_degree_counts_distinct_neighbour_entities_not_edges():
    """degree is how many DISTINCT entities an entity is connected to, not how many
    EXTRACTED_REL edges touch it — a pair joined by two relationship types is ONE
    connection. Edge counting inflated hubs (live graph: "Globex" 7,284 edges vs
    4,955 real neighbours). ai-graph-retrieval reads this as a ranking prior, so
    the DISTINCT is a contract, not a detail."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(3)) as execute:
        n = repo.recompute_degree(TENANT, ["ent-1", "ent-2"])
    assert n == 3
    cypher, params = execute.call_args[0]
    assert (
        "SET e.degree = COUNT { MATCH (e)-[:EXTRACTED_REL]-(nbr:Entity) "
        "RETURN DISTINCT nbr }"
    ) in cypher
    assert params == {"tenant_id": TENANT, "entity_ids": ["ent-1", "ent-2"]}


def test_recompute_degree_scopes_to_given_entity_ids_not_the_whole_tenant():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(1)) as execute:
        repo.recompute_degree(TENANT, ["ent-1"])
    cypher, params = execute.call_args[0]
    assert "WHERE e.entity_id IN $entity_ids" in cypher
    assert params["entity_ids"] == ["ent-1"]


def test_recompute_degree_with_no_entity_ids_scopes_to_whole_tenant():
    """None means "the whole tenant" (bulk's fresh-run case) — no entity_ids
    filter at all, batched server-side since a from-scratch tenant can be
    150k+ entities."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(50)) as execute:
        n = repo.recompute_degree(TENANT, None)
    assert n == 50
    cypher, params = execute.call_args[0]
    assert "IN TRANSACTIONS OF" in cypher
    assert "entity_id IN" not in cypher
    assert params == {"tenant_id": TENANT}


def test_recompute_degree_batches_large_entity_id_lists():
    """A single autocommit SET across a huge entity_ids list is exactly what
    silently exhausted the server's heap on a Confluence-sized tenant
    (150k+ entities) and left degree unset for the whole run. Splitting into
    self._batch_size-sized calls is the fix — verify it actually happens."""
    repo = _repo()
    repo._batch_size = 2
    with patch.object(repo, "_execute", return_value=_n(1)) as execute:
        n = repo.recompute_degree(TENANT, ["ent-1", "ent-2", "ent-3"])
    assert execute.call_count == 2
    assert execute.call_args_list[0].args[1]["entity_ids"] == ["ent-1", "ent-2"]
    assert execute.call_args_list[1].args[1]["entity_ids"] == ["ent-3"]
    assert n == 2  # one per batch, from the mocked return


# --- cl.support (evidence-count backfill follow-up) ---

def test_recompute_claim_support_counts_supported_by_edges():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(2)) as execute:
        n = repo.recompute_claim_support(TENANT, ["cl-1", "cl-2"])
    assert n == 2
    cypher, params = execute.call_args[0]
    assert "SET cl.support = COUNT { MATCH (cl)-[:SUPPORTED_BY]->(:Chunk) }" in cypher
    assert params == {"tenant_id": TENANT, "claim_ids": ["cl-1", "cl-2"]}


def test_recompute_claim_support_with_no_claim_ids_scopes_to_whole_tenant():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(50)) as execute:
        n = repo.recompute_claim_support(TENANT, None)
    assert n == 50
    cypher, params = execute.call_args[0]
    assert "IN TRANSACTIONS OF" in cypher
    assert "claim_id IN" not in cypher
    assert params == {"tenant_id": TENANT}


def test_recompute_claim_support_batches_large_claim_id_lists():
    repo = _repo()
    repo._batch_size = 2
    with patch.object(repo, "_execute", return_value=_n(1)) as execute:
        repo.recompute_claim_support(TENANT, ["cl-1", "cl-2", "cl-3"])
    assert execute.call_count == 2
    assert execute.call_args_list[1].args[1]["claim_ids"] == ["cl-3"]


def test_write_claims_returns_written_claim_ids():
    """Lets incremental_pipeline.py scope the follow-up recompute_claim_support
    to just this document's claims instead of the whole tenant."""
    repo = _repo()
    with patch.object(repo, "_execute", return_value={}):
        claim_ids = repo.write_claims([{
            "subject": "ent-1", "subject_type": "X", "predicate": "p", "object": "o",
            "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None,
            "valid_to": None, "chunk_id": "c1", "acl": ["public"],
        }], TENANT)
    assert claim_ids == [hashlib.sha1(b"ent-1|p|o||").hexdigest()]


# --- Connection reuse (code review must-fix #6 / B4) ---

def test_execute_reuses_the_repository_session_not_a_fresh_connection():
    repo = _repo()
    repo._session = MagicMock()
    repo._session.post.return_value.json.return_value = {"data": {"fields": [], "values": []}}
    repo._execute("MATCH (n) RETURN n", {})
    repo._session.post.assert_called_once()


def test_two_execute_calls_share_the_same_session_instance():
    repo = _repo()
    with patch.object(repo._session, "post") as post:
        post.return_value.json.return_value = {"data": {"fields": [], "values": []}}
        repo._execute("MATCH (n) RETURN n", {})
        repo._execute("MATCH (m) RETURN m", {})
    assert post.call_count == 2


def test_close_closes_the_session():
    repo = _repo()
    repo._session = MagicMock()
    repo.close()
    repo._session.close.assert_called_once()


# --- People / org graph ---

def test_label_person_records_sets_person_label():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(2)) as execute:
        n = repo.label_person_records(TENANT, ["ent-1", "ent-2"])
    assert n == 2
    cypher, params = execute.call_args[0]
    assert "SET e:Person" in cypher
    assert "e.source_key" in cypher
    assert params == {"tenant_id": TENANT, "entity_ids": ["ent-1", "ent-2"]}


def test_materialize_reporting_edges_merges_reports_to():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(1)) as execute:
        n = repo.materialize_reporting_edges(TENANT, None)
    assert n == 1
    cypher, params = execute.call_args[0]
    assert "MERGE (e)-[:REPORTS_TO]->(m)" in cypher
    # The manager resolves by the key their own source handed out, within that
    # source — not by person_user_id, an HR field name only Zeus populates.
    assert "source_key: e.person_manager_id" in cypher
    assert "connector_type: e.connector_type" in cypher
    assert "person_user_id" not in cypher
    assert params["entity_ids"] is None


def test_retire_person_manages_deletes_person_to_person_manages_only():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(3)) as execute:
        n = repo.retire_person_manages(TENANT, ["ent-1"])
    assert n == 3
    cypher, _ = execute.call_args[0]
    assert "relationship_type: 'MANAGES'" in cypher
    assert "DELETE r" in cypher


def test_materialize_org_membership_returns_department_and_team_counts():
    repo = _repo()
    with patch.object(repo, "_execute", side_effect=[_n(2), _n(1)]) as execute:
        result = repo.materialize_org_membership(TENANT, ["ent-1"])
    assert result == {"department_edges": 2, "team_edges": 1}
    dep_cypher, _ = execute.call_args_list[0].args
    assert "MERGE (d:Department" in dep_cypher
    team_cypher, _ = execute.call_args_list[1].args
    assert "MERGE (t:Team" in team_cypher


def test_compute_org_rollups_with_entity_ids_scopes_via_reports_to_upward_lookup():
    repo = _repo()
    up_rows = {"data": {"fields": ["id"], "values": [["mgr-1"], ["mgr-2"]]}}
    with patch.object(repo, "_execute", side_effect=[up_rows, _n(2)]) as execute:
        n = repo.compute_org_rollups(TENANT, ["ent-1"])
    assert n == 2
    assert execute.call_count == 2
    rollup_cypher, rollup_params = execute.call_args_list[1].args
    assert "m.direct_reports = COUNT" in rollup_cypher
    assert rollup_params["ids"] == ["mgr-1", "mgr-2"]


def test_compute_org_rollups_with_no_entity_ids_scopes_to_whole_tenant():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(5)) as execute:
        n = repo.compute_org_rollups(TENANT, None)
    assert n == 5
    assert execute.call_count == 1
    _, params = execute.call_args[0]
    assert params["ids"] is None


def test_find_person_mention_merges_returns_unique_match_pairs():
    repo = _repo()
    rows = {"data": {"fields": ["m", "c"], "values": [["ent-mention", "ent-canonical"]]}}
    with patch.object(repo, "_execute", return_value=rows) as execute:
        pairs = repo.find_person_mention_merges(TENANT, None)
    assert pairs == [{"m": "ent-mention", "c": "ent-canonical"}]
    cypher, _ = execute.call_args[0]
    assert "size(cids) = 1" in cypher


def test_merge_person_mentions_empty_pairs_is_a_no_op():
    repo = _repo()
    with patch.object(repo, "_execute") as execute:
        result = repo.merge_person_mentions(TENANT, [])
    assert result == {"merged": 0}
    execute.assert_not_called()


def test_merge_person_mentions_rewires_and_deletes_the_mention():
    """One pair, no EXTRACTED_REL edges on either side, exercises the full call
    sequence: read node JSON (must happen before the delete destroys it), MENTIONS
    rewire, ABOUT rewire, EXTRACTED_REL-out read+rewire, EXTRACTED_REL-in
    read+rewire, node-property migration, DETACH DELETE. No merged_edge_json means
    the two conditional per-edge-type descriptions_json SETs are skipped."""
    repo = _repo()
    pairs = [{"m": "ent-mention", "c": "ent-canonical"}]
    node_rows = {
        "data": {"fields": ["m", "c", "m_desc", "m_aliases", "c_desc", "c_aliases"],
                 "values": [["ent-mention", "ent-canonical",
                             '[{"doc_id": "d-m", "text": "about mention"}]',
                             '[{"doc_id": "d-m", "name": "Mention Name"}]',
                             '[{"doc_id": "d-c", "text": "about canonical"}]',
                             '[{"doc_id": "d-c", "name": "Canonical Name"}]']]}}
    no_edges = {"data": {"fields": ["c", "o", "rel_type", "old_desc", "existing_desc"], "values": []}}
    responses = [
        node_rows,   # 1. read node JSON before anything is destroyed
        _n(0),       # 2. MENTIONS rewire
        _n(0),       # 3. ABOUT rewire
        no_edges,    # 4. EXTRACTED_REL-out read (nothing to merge -> no extra SET call)
        _n(0),       # 5. EXTRACTED_REL-out rewire
        no_edges,    # 6. EXTRACTED_REL-in read
        _n(0),       # 7. EXTRACTED_REL-in rewire
        _n(0),       # 8. node-property migration
        _n(0),       # 9. DETACH DELETE
    ]
    with patch.object(repo, "_execute", side_effect=responses) as execute:
        result = repo.merge_person_mentions(TENANT, pairs)
    assert result == {"merged": 1}
    assert execute.call_count == 9
    last_cypher, _ = execute.call_args_list[-1].args
    assert "DETACH DELETE m" in last_cypher

    # The node migration must carry the canonical's own descriptions/aliases
    # PLUS the mention's, union-merged (order-stable: survivor's own entries first) --
    # never just overwritten, which is the whole point of reading them first.
    migrate_cypher, migrate_params = next(
        c.args for c in execute.call_args_list if "c.descriptions_json = row.descriptions_json" in c.args[0])
    row = migrate_params["rows"][0]
    assert json.loads(row["descriptions_json"]) == [
        {"doc_id": "d-c", "text": "about canonical"}, {"doc_id": "d-m", "text": "about mention"},
    ]
    assert json.loads(row["aliases_json"]) == [
        {"doc_id": "d-c", "name": "Canonical Name"}, {"doc_id": "d-m", "name": "Mention Name"},
    ]


def test_mark_person_departed_sets_inactive_status_and_departed_on():
    repo = _repo()
    with patch.object(repo, "_execute", return_value=_n(1)) as execute:
        n = repo.mark_person_departed(TENANT, "people-doc-1", when="2026-07-17T00:00:00+00:00")
    assert n == 1
    cypher, params = execute.call_args[0]
    assert "person_employment_status = 'inactive'" in cypher
    # Found through the document that defined the person, keyed on
    # (connector_type, source_key) — never on an HR field name.
    assert "$doc_id IN e.source_docs" in cypher
    assert "connector_type: d.connector_type" in cypher
    assert "e.source_key IS NOT NULL" in cypher
    assert "person_user_id" not in cypher
    assert params["doc_id"] == "people-doc-1"
    assert params["when"] == "2026-07-17T00:00:00+00:00"


def test_doc_object_type_returns_the_stored_source_value():
    repo = _repo()
    rows = {"data": {"fields": ["ot"], "values": [["people"]]}}
    with patch.object(repo, "_execute", return_value=rows) as execute:
        ot = repo.doc_object_type("doc-1", TENANT)
    assert ot == "people"
    cypher, params = execute.call_args[0]
    assert "RETURN d.source AS ot" in cypher
    assert params == {"doc_id": "doc-1", "tenant_id": TENANT}


def test_doc_object_type_returns_none_for_unknown_doc():
    repo = _repo()
    with patch.object(repo, "_execute", return_value={"data": {"fields": ["ot"], "values": []}}):
        ot = repo.doc_object_type("nonexistent", TENANT)
    assert ot is None
