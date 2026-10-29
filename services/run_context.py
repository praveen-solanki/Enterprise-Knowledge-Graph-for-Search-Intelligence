"""Bundles one bulk run's Neo4j database, Qdrant collections, and Mongo
checkpoint collection names — build once, thread unchanged through the run.

Also carries a RunConfig: the embedding space, LLM model, and graph-shaping
flags that run actually used. An incremental event resolves its run via the
registry and reuses this config, so it lands in the SAME vector space and uses
the SAME model as the bulk run it updates — instead of whatever this worker's
own process-global settings/backend probe happen to be. See
services/llm_backend.py (snapshot_run_config / effective_run_config) for how it
is captured and read.

This module stays a pure name-deriver plus these frozen value types — it imports
neither config.settings nor services.llm_backend, so it never pulls the whole
config stack in behind it. The snapshot lives in llm_backend, which owns those."""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Optional

from services.run_naming import build_resource_name, generate_run_suffix

# Every Mongo checkpoint collection a bulk run writes to (core/pipeline.py).
MONGO_CHECKPOINT_KINDS = (
    "extraction", "doc_canonical", "corpus_map", "corpus_entities",
    "edge_pool_edges", "edge_pool_claims", "edge_pool_mentions", "edge_pool_map",
)


@dataclass(frozen=True)
class RunConfig:
    """The model/space/behaviour a run is pinned to. Frozen and scalar-only so
    RunContext stays hashable and a resumed/incremental run can never drift from
    what the bulk run recorded.

    - Embedding space: ONE per run — every vector the run writes, entity or
      chunk, comes from embedding_model on embedding_backend at
      embedding_dimension (config/settings.py's SUPPORTED_EMBEDDING_SPACES
      holds the two allowed combinations; the dimension identifies the model).
      vec_entities and vec_chunks stay two separate Qdrant collections, both
      created at this dimension. consume_upstream_vector says whether upstream
      ES already carries vectors in this space (copy them; a chunk without one
      is skipped, never embedded by us) or not (we embed every chunk
      ourselves); es_chunk_vector_field is the ES field they're copied from.
      Entities are always embedded by us either way.
    - LLM: extraction/gleaning backend ("ministral"|"azure") and its model.
    - Graph-shaping flags: whether authorship edges / the people-org graph are
      built, and which entity types count as contributors."""
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int
    consume_upstream_vector: bool
    es_chunk_vector_field: str
    llm_backend: str
    llm_model: str
    build_authorship_edges: bool
    contributor_entity_types: tuple[str, ...]
    people_org_graph_enabled: bool


@dataclass(frozen=True)
class RunContext:
    tenant_id: str
    connector_type: str
    run_suffix: str
    neo4j_database: str
    qdrant_entities_collection: str
    qdrant_chunks_collection: str
    mongo_collections: dict[str, str]
    run_config: RunConfig


def new_run_context(tenant_id: str, connector_type: str, run_config: RunConfig) -> RunContext:
    """Mints a brand-new random run_suffix and derives every store name from
    it. Call exactly once per bulk run (repository/run_registry.py's
    register_run does this and persists the result) — never call this again
    mid-run, or a later step would land in a different, empty set of stores.

    run_config is this run's captured model/space/flags (services/llm_backend.
    snapshot_run_config), persisted alongside the store names and reused by
    incremental events."""
    run_suffix = generate_run_suffix()
    return RunContext(
        tenant_id=tenant_id,
        connector_type=connector_type,
        run_suffix=run_suffix,
        neo4j_database=build_resource_name(tenant_id, connector_type, run_suffix),
        qdrant_entities_collection=build_resource_name(tenant_id, connector_type, run_suffix, kind="entities"),
        qdrant_chunks_collection=build_resource_name(tenant_id, connector_type, run_suffix, kind="chunks"),
        mongo_collections={
            kind: build_resource_name(tenant_id, connector_type, run_suffix, kind=kind)
            for kind in MONGO_CHECKPOINT_KINDS
        },
        run_config=run_config,
    )


# The run whose config the current work should use, bound per incremental event
# from its resolved RunContext (core/kafka_worker_entrypoint.py). Consumers read
# it through services/llm_backend.effective_run_config(), which falls back to a
# live snapshot of process settings when nothing is bound (the bulk path — where
# the process settings ARE the run's config, so no bind is needed).
#
# ContextVar, not a module global, for the same reason entity_registry's
# connector_type is one: each event runs in its own asyncio context, so a
# binding never leaks between events handled by the same worker.
_run_config_ctx: contextvars.ContextVar[Optional[RunConfig]] = contextvars.ContextVar(
    "active_run_config", default=None,
)


def bind_run_config(run_config: Optional[RunConfig]) -> None:
    """Bind this run's config for the current execution context, so the deep
    embedding/LLM/graph call sites use the config the bulk run recorded."""
    _run_config_ctx.set(run_config)


def active_run_config() -> Optional[RunConfig]:
    """The bound run's RunConfig, or None to fall back to process settings."""
    return _run_config_ctx.get()
