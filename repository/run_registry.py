"""One Mongo document per bulk run: which Neo4j database / Qdrant / Mongo
collections that run's data lives in. Lets the incremental worker resolve a
tenant+connector's latest run, and --retry-failed resume the same stores."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo.errors import DuplicateKeyError

from config.settings import settings
from repository.mongo_repository import _get_db
from services.run_context import RunConfig, RunContext, new_run_context

logger = logging.getLogger("graphrag.repository.run_registry")


class NoCompletedRunError(Exception):
    """No COMPLETED bulk run yet for this (tenant_id, connector_type) —
    classified as permanent, so the event is DLQ'd, not retried forever."""


_COLLECTION = "run_registry"
# settings.run_registry_max_suffix_retries — headroom, not an expected count,
# see _ensure_index.

_index_ensured = False


def _ensure_index() -> None:
    """Guards against two runs of the same (tenant_id, connector_type)
    minting the same run_suffix. Cached process-wide — create_index is a
    network round trip."""
    global _index_ensured
    if _index_ensured:
        return
    _get_db()[_COLLECTION].create_index(
        [("tenant_id", 1), ("connector_type", 1), ("run_suffix", 1)], unique=True,
    )
    _index_ensured = True


def _to_run_config(doc: dict) -> RunConfig:
    """Rehydrate the run's config, or fall back for a legacy doc.

    A run registered before RunConfig existed has no `run_config` sub-doc.
    Rather than KeyError (which would DLQ every legacy tenant's events at
    rollout), fill from a live snapshot of this process's settings — exactly the
    behaviour those events had before this change — and warn. The tenant becomes
    pinned at its next bulk run."""
    raw = doc.get("run_config")
    if raw is None:
        # Imported lazily: run_registry must stay importable without eagerly
        # pulling in the settings/backend stack that llm_backend depends on.
        from services.llm_backend import snapshot_run_config
        logger.warning(
            "run_registry: run_id=%s predates RunConfig — falling back to this process's "
            "current settings; it will be pinned at the next bulk run for this tenant/connector.",
            doc.get("run_id"),
        )
        return snapshot_run_config()
    return RunConfig(
        embedding_backend=raw["embedding_backend"],
        embedding_model=raw["embedding_model"],
        embedding_dimension=raw["embedding_dimension"],
        consume_upstream_vector=raw["consume_upstream_vector"],
        es_chunk_vector_field=raw["es_chunk_vector_field"],
        llm_backend=raw["llm_backend"],
        llm_model=raw["llm_model"],
        build_authorship_edges=raw["build_authorship_edges"],
        contributor_entity_types=tuple(raw["contributor_entity_types"]),
        people_org_graph_enabled=raw["people_org_graph_enabled"],
    )


def _to_run_context(doc: dict) -> RunContext:
    return RunContext(
        tenant_id=doc["tenant_id"],
        connector_type=doc["connector_type"],
        run_suffix=doc["run_suffix"],
        neo4j_database=doc["neo4j_database"],
        qdrant_entities_collection=doc["qdrant_entities_collection"],
        qdrant_chunks_collection=doc["qdrant_chunks_collection"],
        mongo_collections=doc["mongo_collections"],
        run_config=_to_run_config(doc),
    )


def _run_config_doc(run_config: RunConfig) -> dict:
    """RunConfig as a Mongo sub-document (scalars + a list). Mirrored 1:1 by
    _to_run_config above."""
    return {
        "embedding_backend": run_config.embedding_backend,
        "embedding_model": run_config.embedding_model,
        "embedding_dimension": run_config.embedding_dimension,
        "consume_upstream_vector": run_config.consume_upstream_vector,
        "es_chunk_vector_field": run_config.es_chunk_vector_field,
        "llm_backend": run_config.llm_backend,
        "llm_model": run_config.llm_model,
        "build_authorship_edges": run_config.build_authorship_edges,
        "contributor_entity_types": list(run_config.contributor_entity_types),
        "people_org_graph_enabled": run_config.people_org_graph_enabled,
    }


def register_run(index: str, tenant_id: str, connector_type: str,
                 run_config: RunConfig) -> tuple[str, RunContext]:
    """Mints a new isolated RunContext + run_id, persists it as RUNNING.
    Retries with a fresh suffix on a genuine collision.

    run_config is this run's resolved model/space/flags (services/llm_backend.
    snapshot_run_config), captured once by the caller and persisted so
    incremental events reuse it. Never re-resolved on resume."""
    _ensure_index()
    coll = _get_db()[_COLLECTION]
    for attempt in range(1, settings.run_registry_max_suffix_retries + 1):
        run_ctx = new_run_context(tenant_id, connector_type, run_config)
        run_id = f"{index}:{tenant_id}:{connector_type}:{run_ctx.run_suffix}"
        doc = {
            "_id": run_id,
            "run_id": run_id,
            "index": index,
            "tenant_id": tenant_id,
            "connector_type": connector_type,
            "run_suffix": run_ctx.run_suffix,
            "neo4j_database": run_ctx.neo4j_database,
            "qdrant_entities_collection": run_ctx.qdrant_entities_collection,
            "qdrant_chunks_collection": run_ctx.qdrant_chunks_collection,
            "mongo_collections": run_ctx.mongo_collections,
            "run_config": _run_config_doc(run_ctx.run_config),
            "status": "RUNNING",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        try:
            coll.insert_one(doc)
        except DuplicateKeyError:
            logger.warning(
                "run_registry: run_suffix %s collided for (tenant_id=%s, connector_type=%s) — "
                "retrying with a new suffix (attempt %d/%d)",
                run_ctx.run_suffix, tenant_id, connector_type, attempt,
                settings.run_registry_max_suffix_retries,
            )
            continue
        logger.info("run_registry: registered run_id=%s (run_suffix=%s)", run_id, run_ctx.run_suffix)
        return run_id, run_ctx
    raise RuntimeError(
        f"run_registry: could not mint a unique run_suffix for tenant_id={tenant_id!r} "
        f"connector_type={connector_type!r} after {settings.run_registry_max_suffix_retries} attempts"
    )


def get_run(run_id: str) -> Optional[RunContext]:
    """Recovers a run's RunContext by run_id, or None if never registered."""
    doc = _get_db()[_COLLECTION].find_one({"_id": run_id})
    return _to_run_context(doc) if doc else None


def get_latest_run(tenant_id: str, connector_type: str) -> Optional[RunContext]:
    """Most recently COMPLETED run for this (tenant_id, connector_type),
    or None — a RUNNING/FAILED run is never returned."""
    doc = _get_db()[_COLLECTION].find_one(
        {"tenant_id": tenant_id, "connector_type": connector_type, "status": "COMPLETED"},
        sort=[("created_at", -1)],
    )
    return _to_run_context(doc) if doc else None


def get_run_by_suffix(tenant_id: str, connector_type: str, run_suffix: str) -> Optional[RunContext]:
    """The COMPLETED run pinned by an explicit run_suffix (the Kafka
    `runIdSuffix` header), or None. Hits the unique (tenant_id, connector_type,
    run_suffix) index directly, so an incremental event can target one specific
    run's stores instead of whatever get_latest_run would pick. A RUNNING/FAILED
    run is never returned, matching get_latest_run."""
    doc = _get_db()[_COLLECTION].find_one(
        {"tenant_id": tenant_id, "connector_type": connector_type,
         "run_suffix": run_suffix, "status": "COMPLETED"},
    )
    return _to_run_context(doc) if doc else None


def mark_run_completed(run_id: str) -> None:
    _get_db()[_COLLECTION].update_one(
        {"_id": run_id},
        {"$set": {"status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat()}},
    )


def mark_run_failed(run_id: str) -> None:
    _get_db()[_COLLECTION].update_one(
        {"_id": run_id},
        {"$set": {"status": "FAILED", "completed_at": datetime.now(timezone.utc).isoformat()}},
    )
