"""
Entity vector store — Qdrant, not Neo4j. Storing embeddings as node
properties directly inside Neo4j was degrading the graph engine at
production scale, so entity vectors live in a dedicated Qdrant collection;
Neo4j keeps everything structural plus a lightweight `has_vector` boolean so
callers can check without a Qdrant round trip.

Division of responsibility: Qdrant owns the raw vector plus enough payload
(entity_type, tenant_id, acl) to scope a similarity search. Neo4j stays the
source of truth for everything else about an entity (name, descriptions,
aliases, ...) — this module never returns entity metadata, only ids+scores;
a caller does a follow-up Neo4j MATCH by entity_id for the rest.

One point per entity, keyed by entity_id, but NOT literally as the Qdrant
point id: Qdrant only accepts an unsigned integer or a real UUID as a point
id, and our entity_id format ("ent_<uuid4 hex>") is rejected with a 400.
Point ids are a deterministic uuid5 derived from entity_id (_point_id) so
upserts/deletes stay idempotent, with the real entity_id carried as its own
payload field so search() can hand back something the caller recognizes.

ACL travels as a payload field for a future retrieval-side Qdrant filter;
this module's own search() does not ACL-filter today since its only caller
(incremental entity linking) is an internal merge decision, not a
user-facing query.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from config.settings import settings

logger = logging.getLogger("graphrag.services.vector_store")

# Arbitrary but MUST stay constant — changing it orphans every existing
# point (re-upsert of the same entity_id would compute a different point
# id and create a duplicate instead of replacing it).
_POINT_ID_NAMESPACE = uuid.UUID("6f9d6f1a-2b3c-4d5e-8f9a-1b2c3d4e5f6a")

# A just-created Qdrant collection is not instantly writable on a
# clustered/Cloud instance — create_collection returns before the collection
# has propagated to the shard node the first upsert lands on, so that upsert
# gets a 404 "collection doesn't exist" for a short window. Bound how long
# ensure_collection waits for readiness before giving up (~6s), matched to
# observed Cloud propagation, not the dual-write guard's own write retries.
_READY_POLL_ATTEMPTS = 20
_READY_POLL_INTERVAL_SECONDS = 0.3

# One client per (url, api_key) — the incremental worker may talk to two Qdrant
# instances (primary and EC2) over the life of the process, so a single global
# client no longer suffices. Reused across messages hitting the same endpoint.
_clients: dict[tuple[str, str], QdrantClient] = {}


def _point_id(entity_id: str) -> str:
    """Deterministically maps an entity_id string to a valid Qdrant point id
    (a UUID) so the same entity_id always maps to the same point."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, entity_id))


def get_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    """A client for the given Qdrant endpoint, cached per (url, api_key) and
    reused for the life of the process. url/api_key default to the process-global
    settings (bulk path, and any caller not routing per-message); the incremental
    Kafka worker passes the endpoint its message's `use_ec2_database` header
    resolved to.

    port=None is deliberate: QdrantClient's own `port` defaults to 6333 and
    silently wins whenever `url` has no explicit port, which breaks a
    reverse-proxied https:// instance (connection reset on 6333). Passing
    port=None lets it fall through to the scheme's own standard port."""
    url = url if url is not None else settings.qdrant_url
    api_key = api_key if api_key is not None else settings.qdrant_api_key
    key = (url, api_key or "")
    client = _clients.get(key)
    if client is None:
        # Explicit timeout: qdrant-client defaults to 5s, too short for a cold TLS
        # handshake to the reverse-proxied dev endpoint (caused "handshake operation
        # timed out" failures). See settings.qdrant_timeout_seconds.
        client = QdrantClient(url=url, api_key=api_key or None, port=None,
                              timeout=settings.qdrant_timeout_seconds)
        _clients[key] = client
    return client


def _get_client() -> QdrantClient:
    """Back-compat shim — the process-global endpoint's client."""
    return get_qdrant_client()


class VectorStore:
    """Thin wrapper around one Qdrant collection. `collection` is required and
    names the run's own isolated collection — its entities or its chunks
    (RunContext.qdrant_entities_collection / qdrant_chunks_collection). One class
    serves both, avoiding a second near-duplicate."""

    def __init__(self, collection: Optional[str] = None, client: Optional[QdrantClient] = None):
        if not collection:
            raise ValueError("VectorStore requires an explicit collection — pass the run's own "
                             "RunContext.qdrant_entities_collection or qdrant_chunks_collection. "
                             "There is no settings default.")
        self._client = client or _get_client()
        self._collection = collection

    def ensure_collection(self, dimensions: int) -> None:
        """Idempotent — safe to call on every process/worker startup.

        After creating a NEW collection, block until Qdrant reports it ready
        for operations. On a clustered/Cloud Qdrant, create_collection returns
        before the collection is writable on the shard node the run's first
        upserts land on, so those batches would otherwise race the create and
        get a 404 "collection doesn't exist". The dual-write guard retries that
        (transient) a few times, but slow propagation can exhaust the retries —
        and the chunk push has no rollback, so those chunks would silently end
        up with no vector. Waiting here closes the race once, at the source."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )
        self._wait_until_ready()
        logger.info("Qdrant collection %r created (dimensions=%d)", self._collection, dimensions)

    def _wait_until_ready(self) -> None:
        """Poll a cheap shard-level op (count) until the just-created collection
        stops 404ing, so the caller never returns into a write that races the
        create. Bounded — if it never confirms, log and fall through rather than
        block the run forever; the dual-write guard's retries remain a backstop."""
        last_exc: Optional[Exception] = None
        for _ in range(_READY_POLL_ATTEMPTS):
            try:
                self._client.count(collection_name=self._collection, exact=False)
                return
            except Exception as exc:  # noqa: BLE001 — any error means "not ready yet", keep polling
                last_exc = exc
                time.sleep(_READY_POLL_INTERVAL_SECONDS)
        logger.warning("Qdrant collection %r not confirmed ready after %d attempts (%.1fs); "
                       "proceeding — writes fall back to the dual-write retry: %s",
                       self._collection, _READY_POLL_ATTEMPTS,
                       _READY_POLL_ATTEMPTS * _READY_POLL_INTERVAL_SECONDS, last_exc)

    def upsert_entity(self, entity_id: str, embedding: list[float], entity_type: str,
                      tenant_id: str, acl: list[str]) -> None:
        """Writes (or overwrites) one entity's vector + scoping payload.
        Idempotent — re-upserting the same entity_id replaces the point,
        never duplicates it."""
        self.upsert_entities_batch([{
            "entity_id": entity_id, "embedding": embedding,
            "type": entity_type, "tenant_id": tenant_id, "acl": acl,
        }])

    def upsert_entities_batch(self, rows: list[dict]) -> None:
        """rows: [{entity_id, embedding, type, tenant_id, acl}, ...] — batches
        a run's entity writes into one Qdrant call instead of one round trip
        per entity."""
        if not rows:
            return
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=_point_id(row["entity_id"]), vector=row["embedding"],
                payload={"entity_id": row["entity_id"], "type": row["type"],
                         "tenant_id": row["tenant_id"], "acl": row.get("acl") or []},
            ) for row in rows],
        )

    def set_acl(self, entity_id: str, acl: list[str]) -> None:
        """Updates ONLY the acl payload field, without touching the vector —
        so an acl_update event never has to re-embed."""
        self._client.set_payload(
            collection_name=self._collection,
            payload={"acl": acl or []},
            points=[_point_id(entity_id)],
        )

    def delete_entity(self, entity_id: str) -> None:
        """Safe to call on an entity_id with no point — Qdrant's delete is a
        no-op on a missing id, not an error."""
        self.delete_entities([entity_id])

    def delete_entities(self, entity_ids: list[str]) -> None:
        """Batched delete-by-id, safe with ids that have no point."""
        if not entity_ids:
            return
        self._client.delete(collection_name=self._collection,
                            points_selector=[_point_id(eid) for eid in entity_ids])

    def search(self, embedding: list[float], entity_type: str, tenant_id: str,
              k: int = 5, min_similarity: float = 0.83) -> list[dict]:
        """Returns [{entity_id, score}, ...] ordered by similarity descending,
        scoped to same-type and same-tenant only. Doesn't return entity
        metadata — see module docstring. entity_id is read from the payload,
        not the point id (which is an opaque derived uuid — see _point_id)."""
        hits = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            query_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value=entity_type)),
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            ]),
            score_threshold=min_similarity,
            limit=k,
            with_payload=["entity_id"],
        ).points
        return [{"entity_id": hit.payload["entity_id"], "score": hit.score} for hit in hits]
