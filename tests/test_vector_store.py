"""Unit tests for services/vector_store.py (entity embeddings moved to Qdrant,
out of Neo4j). All tests mock the QdrantClient -- nothing here touches a real
Qdrant instance."""
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

from services.vector_store import VectorStore, _point_id


def _store(client=None) -> VectorStore:
    # collection is a required, explicit arg now (PR-41 review point 7) — no
    # settings default — so it can no longer drift with the developer's local
    # .env QDRANT_COLLECTION the way the old settings fallback could.
    return VectorStore(collection="entity_embeddings", client=client or MagicMock())


def test_point_id_is_deterministic_and_a_valid_uuid():
    """Qdrant only accepts an unsigned integer or a real UUID as a point id
    -- our own entity_id format ("ent_<uuid4 hex>") is rejected outright
    (confirmed against a live instance). Must derive the same point id for
    the same entity_id every time, or upserts stop being idempotent."""
    import uuid
    p1 = _point_id("ent_1a2b3c4d5e6f")
    p2 = _point_id("ent_1a2b3c4d5e6f")
    assert p1 == p2
    assert str(uuid.UUID(p1)) == p1  # raises ValueError if not a valid UUID


def test_point_id_differs_for_different_entity_ids():
    assert _point_id("ent_aaa") != _point_id("ent_bbb")


def test_ensure_collection_creates_when_missing():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    store = _store(client)
    store.ensure_collection(dimensions=3072)
    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == "entity_embeddings"
    assert kwargs["vectors_config"] == VectorParams(size=3072, distance=Distance.COSINE)


def test_ensure_collection_waits_for_readiness_after_creating():
    """A freshly-created collection can 404 briefly on a clustered/Cloud Qdrant
    before it is writable; ensure_collection must poll until it stops erroring
    so the run's first upserts don't race the create."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    # count() 404s twice (not ready), then succeeds — ensure_collection must
    # not return until it succeeds.
    client.count.side_effect = [RuntimeError("404 not found"), RuntimeError("404 not found"),
                                MagicMock()]
    store = _store(client)
    with patch("services.vector_store.time.sleep") as sleep:
        store.ensure_collection(dimensions=3072)
    assert client.count.call_count == 3
    assert sleep.call_count == 2  # slept between the two failed polls


def test_ensure_collection_does_not_wait_when_it_already_exists():
    client = MagicMock()
    existing = MagicMock()
    existing.name = "entity_embeddings"
    client.get_collections.return_value = MagicMock(collections=[existing])
    store = _store(client)
    store.ensure_collection(dimensions=3072)
    client.create_collection.assert_not_called()
    client.count.assert_not_called()


def test_ensure_collection_is_a_no_op_when_it_already_exists():
    client = MagicMock()
    existing = MagicMock()
    existing.name = "entity_embeddings"
    client.get_collections.return_value = MagicMock(collections=[existing])
    store = _store(client)
    store.ensure_collection(dimensions=3072)
    client.create_collection.assert_not_called()


def test_upsert_entity_sends_a_valid_point_id_vector_and_payload_with_the_real_entity_id():
    client = MagicMock()
    store = _store(client)
    store.upsert_entity("ent-1", [0.1, 0.2], "ORGANIZATION", "tenant-1", ["public"])
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == "entity_embeddings"
    point = kwargs["points"][0]
    assert point.id == _point_id("ent-1")
    assert point.vector == [0.1, 0.2]
    assert point.payload == {"entity_id": "ent-1", "type": "ORGANIZATION",
                             "tenant_id": "tenant-1", "acl": ["public"]}


def test_upsert_entity_defaults_none_acl_to_empty_list():
    client = MagicMock()
    store = _store(client)
    store.upsert_entity("ent-1", [0.1], "ORGANIZATION", "tenant-1", None)
    _, kwargs = client.upsert.call_args
    assert kwargs["points"][0].payload["acl"] == []


def test_upsert_entities_batch_sends_one_call_for_every_row():
    client = MagicMock()
    store = _store(client)
    store.upsert_entities_batch([
        {"entity_id": "ent-1", "embedding": [0.1], "type": "ORGANIZATION",
         "tenant_id": "tenant-1", "acl": ["public"]},
        {"entity_id": "ent-2", "embedding": [0.2], "type": "PERSON",
         "tenant_id": "tenant-1", "acl": []},
    ])
    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    ids = [p.id for p in kwargs["points"]]
    assert ids == [_point_id("ent-1"), _point_id("ent-2")]
    entity_ids = [p.payload["entity_id"] for p in kwargs["points"]]
    assert entity_ids == ["ent-1", "ent-2"]


def test_upsert_entities_batch_empty_list_is_a_no_op():
    client = MagicMock()
    store = _store(client)
    store.upsert_entities_batch([])
    client.upsert.assert_not_called()


def test_set_acl_updates_payload_only_never_touches_the_vector():
    client = MagicMock()
    store = _store(client)
    store.set_acl("ent-1", ["group:site:s1"])
    _, kwargs = client.set_payload.call_args
    assert kwargs["collection_name"] == "entity_embeddings"
    assert kwargs["payload"] == {"acl": ["group:site:s1"]}
    assert kwargs["points"] == [_point_id("ent-1")]
    client.upsert.assert_not_called()


def test_delete_entity_deletes_by_id():
    client = MagicMock()
    store = _store(client)
    store.delete_entity("ent-1")
    _, kwargs = client.delete.call_args
    assert kwargs["collection_name"] == "entity_embeddings"
    assert kwargs["points_selector"] == [_point_id("ent-1")]


def test_delete_entities_deletes_every_given_id_in_one_call():
    client = MagicMock()
    store = _store(client)
    store.delete_entities(["ent-1", "ent-2"])
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["points_selector"] == [_point_id("ent-1"), _point_id("ent-2")]


def test_delete_entities_empty_list_is_a_no_op():
    client = MagicMock()
    store = _store(client)
    store.delete_entities([])
    client.delete.assert_not_called()


def test_search_scopes_by_type_and_tenant_and_returns_id_plus_score():
    client = MagicMock()
    hit = MagicMock(score=0.91, payload={"entity_id": "ent-2"})
    client.query_points.return_value = MagicMock(points=[hit])
    store = _store(client)
    results = store.search([0.1, 0.2], "ORGANIZATION", "tenant-1", k=5, min_similarity=0.83)
    assert results == [{"entity_id": "ent-2", "score": 0.91}]
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "entity_embeddings"
    assert kwargs["query"] == [0.1, 0.2]
    assert kwargs["score_threshold"] == 0.83
    assert kwargs["limit"] == 5
    conditions = kwargs["query_filter"].must
    assert {c.key for c in conditions} == {"type", "tenant_id"}


def test_search_empty_when_no_hits():
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    store = _store(client)
    assert store.search([0.1], "ORGANIZATION", "tenant-1") == []
