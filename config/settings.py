"""
Centralized configuration. Every credential and tunable value for the bulk
pipeline lives here — no other file reads an environment variable directly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.kafka_constants import (
    KAFKA_CONSUMER_GROUP_TEMPLATE,
    KAFKA_TOPIC_GRAPH_EXTRACT_DLQ_TEMPLATE,
    KAFKA_TOPIC_GRAPH_EXTRACT_TEMPLATE,
    lower_env,
    title_env,
    upper_env,
)

# backend -> the space's model, dimension, and upstream ES vector field. THE
# definition of the two supported embedding spaces. Vault/env exposes a bool
# (use_in_house_embeddings); settings.embedding_backend is derived from it
# ("inhouse" if true else "azure") and everything else about the space resolves
# through this table (services/llm_backend.snapshot_run_config), so a config
# mixing one backend with the other's model or dimension is unrepresentable.
# Adding a third space means adding a row here, nowhere else.
SUPPORTED_EMBEDDING_SPACES: dict[str, dict] = {
    "azure": {
        "model": "text-embedding-3-large",
        "dimension": 1536,
        "es_vector_field": "embeddings_openai_large_1536",
    },
    # Filled in for real right after `settings = Settings()` below, from
    # whichever of the two inference_provider presets (nvidia/inhouse) is
    # active — see Settings._apply_inference_provider. The dict literal here
    # is just the nvidia-shaped placeholder so this table is never empty if
    # something reads it at import time, before that assignment runs.
    "inhouse": {
        "model": "nvidia/nemotron-3.5-embed-1b",
        "dimension": 2048,
        "es_vector_field": "embeddings_nim_2048",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime environment ---
    # Same semantics as connectors/vault_manager.py's ENV routing: local_env =
    # laptop with plain .env; local_vault = AppRole Vault; anything else = deployed.
    env: str = Field(default="local_env", validation_alias=AliasChoices("ENV"))

    # --- Elasticsearch ---
    es_host: str = Field(default="localhost", description="Elasticsearch host")
    es_port: int = 9200
    es_scheme: str = "http"
    es_username: str = "elastic"
    es_password: str = Field(default="", description="Elasticsearch password")
    verify_certs: bool = True  # keep True in production; disable only for local dev

    use_ec2_databases: bool = Field(
        default=False,
        validation_alias=AliasChoices("USE_EC2_DATABASES", "USE_LOCAL_EC2_DBS"),
        description="When true, overwrite Neo4j/Mongo/Qdrant settings with EC2 docker endpoints",
    )

    ec2_neo4j_http_url: str = Field(
        default="http://localhost:7474",
        validation_alias=AliasChoices("EC2_NEO4J_HTTP_URL"),
    )
    ec2_neo4j_username: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("EC2_NEO4J_USERNAME"),
    )
    ec2_neo4j_password: str = Field(
        default="",
        validation_alias=AliasChoices("EC2_NEO4J_PASSWORD"),
    )
    ec2_mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        validation_alias=AliasChoices("EC2_MONGO_URI"),
    )
    ec2_mongo_db: str = Field(
        default="graph_ingestion_checkpoints_dev",
        validation_alias=AliasChoices("EC2_MONGO_DB"),
    )
    ec2_qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("EC2_QDRANT_URL"),
    )
    ec2_qdrant_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EC2_QDRANT_API_KEY"),
    )

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_http_url: str = Field(default="http://localhost:7474",
                                description="Base URL for the Neo4j Query API")
    neo4j_username: str = "neo4j"
    neo4j_password: str = Field(default="", description="Neo4j password")

    # --- MongoDB (checkpoints) ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection string")
    mongo_db: str = "graphrag_checkpoints"
    # Chunk size for save_checkpoints_batch's bulk_write calls (corpus entity
    # persistence in Step 3, edge-pool persistence in Step 4). Matches the
    # batch-load size already used for corpus entity reads (core/pipeline.py).
    mongo_checkpoint_bulk_size: int = 1000

    # --- Organization-specific values ---
    # Everything that names a particular company, product, or hosted endpoint
    # lives here so a deployment can point the pipeline at its own values.
    # Base URL of the intranet app that document citation links point into.
    citation_base_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("CITATION_BASE_URL"),
    )
    # In-house embeddings server (the "inhouse" embedding_backend). Default is
    # NVIDIA NIM's /v1/embeddings; INFERENCE_PROVIDER=inhouse switches this (and
    # the three fields below) to a local vLLM preset — see
    # _apply_inference_provider. An explicit override here always wins.
    in_house_embeddings_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/embeddings",
        validation_alias=AliasChoices("IN_HOUSE_EMBEDDINGS_URL"),
    )
    # The embedding model/dimension/Qdrant-field triple for the "inhouse" row
    # of SUPPORTED_EMBEDDING_SPACES above — kept as settings (not a hardcoded
    # dict entry) so INFERENCE_PROVIDER=inhouse can switch all three together
    # to match whatever model a self-hosted vLLM is actually serving.
    in_house_embedding_model: str = Field(
        default="nvidia/nemotron-3.5-embed-1b",
        validation_alias=AliasChoices("IN_HOUSE_EMBEDDING_MODEL"),
    )
    in_house_embedding_dimension: int = Field(
        default=2048, gt=0,
        validation_alias=AliasChoices("IN_HOUSE_EMBEDDING_DIMENSION"),
    )
    in_house_embedding_es_field: str = Field(
        default="embeddings_nim_2048",
        validation_alias=AliasChoices("IN_HOUSE_EMBEDDING_ES_FIELD"),
    )

    # --- Qdrant (entity/chunk embeddings) ---
    qdrant_url: str = "http://localhost:6333"
    # Vault's ai-graph-ingestion/<env>/static/qdrant secret holds a bare hostname
    # under QDRANT_HOST (same platform convention ai-graph-retrieval's own Settings
    # uses — see its qdrant_host field), not a full qdrant_url — combined into
    # qdrant_url below (_derive_qdrant_url) when the latter is left at its default.
    qdrant_host: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_HOST"))
    qdrant_api_key: str = Field(default="", description="Qdrant API key — empty for a local/dev instance")
    # --- The embedding space ---
    # ONE embedding space per run: every vector this run writes, entity or chunk,
    # comes from the same model at the same dimension (vec_entities and vec_chunks
    # stay two separate Qdrant collections, both created at this dimension).
    # Vault key (service-config): USE_IN_HOUSE_EMBEDDINGS — bool flips the space;
    # embedding_backend is derived in _derive_embedding_backend (never set in Vault).
    use_in_house_embeddings: bool = True
    embedding_backend: Literal["azure", "inhouse"] = "azure"
    # Whether the upstream ES index already carries vectors in THIS run's
    # embedding space (the space's es_vector_field):
    #   true  -- copy a chunk's upstream vector; a chunk without one is skipped
    #            (searchable by text/graph only), NEVER embedded by us. A bulk
    #            run asserts one sampled document's vector dimension against
    #            the space's dimension at startup and fails the run on mismatch
    #            (core/pipeline.py), so a whole index in the wrong space dies
    #            loudly up front instead of silently skipping every chunk.
    #   false -- upstream vectors are ignored; we embed every chunk ourselves,
    #            in the run's space. This is the mode for running the inhouse
    #            256-d space while source ES still carries azure 1536-d vectors.
    # Entities are ALWAYS embedded by us, either way — which is what makes our
    # config the definition of the space, and upstream the thing that has to
    # match it.
    # Vault key (service-config): CONSUME_UPSTREAM_VECTOR.
    consume_upstream_vector: bool = False

    # --- Azure OpenAI (extraction + embeddings) ---
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_completion_model: str = "gpt-5.4-nano"
    # Vault's llm secret has this under AZURE_OPENAI_EMBEDD_DEPLOYMENT (missing
    # "ING") — accept both so the real secret's key actually gets picked up,
    # instead of silently keeping this field's own default forever.
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-large-global",
        validation_alias=AliasChoices("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_EMBEDD_DEPLOYMENT"))
    # Optional dedicated embedding-only Azure resource — Vault's llm secret
    # provisions these separately (AZURE_OPENAI_EMBEDD_ENDPOINT/_API_KEY, missing
    # "ING" — same Vault typo as azure_openai_embedding_deployment above) from
    # the chat/extraction ones above. embedding_service.py falls back to the
    # shared azure_openai_endpoint/api_key when these are left empty, so a
    # deployment that only sets the shared pair still works unchanged.
    azure_openai_embedding_endpoint: str = Field(
        default="", validation_alias=AliasChoices("AZURE_OPENAI_EMBEDDING_ENDPOINT", "AZURE_OPENAI_EMBEDD_ENDPOINT"),
        description="Dedicated embedding Azure OpenAI endpoint URL — falls back to azure_openai_endpoint when empty")
    azure_openai_embedding_api_key: str = Field(
        default="", validation_alias=AliasChoices("AZURE_OPENAI_EMBEDDING_API_KEY", "AZURE_OPENAI_EMBEDD_API_KEY"),
        description="Dedicated embedding Azure OpenAI API key — falls back to azure_openai_api_key when empty")
    azure_llm_temperature: float = 0.0
    # Toggles extraction to the in-house Ministral model (services/llm_client.py).
    # Vault key: ai-graph-ingestion/<env>/static/service-config -> USE_IN_HOUSE_MODEL.
    # Default True: Ministral is the preferred backend wherever it's actually reachable
    # (see services/llm_backend.py) -- this only expresses intent, not the resolved
    # decision; a plain local_env or an unreachable Ministral both still fall back to
    # Azure regardless of this value.
    use_in_house_model: bool = True
    # The explicit switch between the two non-Azure backends (only matters
    # when use_in_house_model/use_in_house_embeddings is True):
    #   nvidia  -- NVIDIA NIM, hosted, free tier, nothing to run locally (default)
    #   inhouse -- a self-hosted vLLM server anyone can run on their own box/GPU
    # This REPLACES inferring "is this NIM?" from the URL string: flipping this
    # one setting is what picks the chat + embeddings URL/model defaults below
    # (see _apply_inference_provider), not editing MINISTRAL_URL by hand.
    inference_provider: Literal["nvidia", "inhouse"] = Field(
        default="nvidia",
        validation_alias=AliasChoices("INFERENCE_PROVIDER"),
        description="nvidia (NIM, default) or inhouse (self-hosted vLLM) — picks "
                    "chat + embeddings defaults for both at once",
    )
    # Default is NVIDIA NIM (OpenAI-compatible, no local model download).
    # Set INFERENCE_PROVIDER=inhouse to switch every default below to the
    # self-hosted vLLM preset instead (see _apply_inference_provider) — an
    # explicit MINISTRAL_URL/_MODEL override here always still wins over
    # either preset. USE_IN_HOUSE_MODEL=false selects Azure instead of both.
    ministral_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        description="OpenAI-compatible chat completions endpoint. Default: NVIDIA NIM")
    ministral_model: str = "nvidia/nemotron-3.5-bolt-e4b"
    ministral_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MINISTRAL_API_KEY", "NVIDIA_API_KEY"),
        description="Bearer token. For NIM, the key from build.nvidia.com. Not "
                    "needed for a self-hosted vLLM with no auth configured.",
    )
    extraction_gleaning_enabled: bool = True
    people_org_graph_enabled: bool = True
    build_authorship_edges: bool = True
    contributor_entity_types: tuple = ("PROJECT", "PRODUCT", "PROGRAM", "FEATURE", "INITIATIVE")
    # --- Wikipedia connector (connectors/wikipedia/) ---
    # The zero-local-dependency connector: reads live from the public,
    # no-auth MediaWiki Action API — no Elasticsearch, no download/indexing
    # step. --index is the search phrase, or "Category:<name>" to walk a
    # category instead. wikipedia_api_base_url can point at any MediaWiki
    # install (a self-hosted mirror, a different-language Wikipedia, ...).
    wikipedia_api_base_url: str = Field(
        default="https://en.wikipedia.org/w/api.php",
        validation_alias=AliasChoices("WIKIPEDIA_API_BASE_URL"),
    )
    wikipedia_page_size: int = Field(
        default=20, gt=0, le=500,
        description="Pages requested per API call (MediaWiki's own cap is 50 for anonymous callers)",
    )
    wikipedia_request_timeout_seconds: float = 30.0
    # Not Wikipedia-specific despite living in this block: Wikimedia's
    # User-Agent policy (meta.wikimedia.org/wiki/User-Agent_policy) is the
    # strictest of the public APIs here — it returns 403 without a descriptive
    # UA — but arXiv asks the same courtesy, and both connectors send this one
    # value. Set HTTP_USER_AGENT to your own contact info for a production
    # deployment.
    http_user_agent: str = Field(
        default="ai-graph-ingestion/1.0 (open-source; no contact configured)",
        validation_alias=AliasChoices("HTTP_USER_AGENT", "WIKIPEDIA_USER_AGENT"),
    )

    # --- arXiv connector (connectors/arxiv/) ---
    # Public Atom API, no auth, no download. --index is the search phrase.
    # arXiv asks clients to wait 3 seconds between requests; the connector does.
    arxiv_api_base_url: str = Field(
        default="https://export.arxiv.org/api/query",
        validation_alias=AliasChoices("ARXIV_API_BASE_URL"),
    )
    arxiv_page_size: int = Field(default=10, gt=0, le=100)

    # --- In-house embeddings gateway (the "inhouse" embedding_backend) ---
    # Transport plumbing only — WHICH model/dimension to use comes from the
    # embedding-space fields above, never from here.
    # The gateway URL is in_house_embeddings_url (organization block above).
    # Gateway headers are "<prefix>-service-code" and "<prefix>-embedding-priority".
    in_house_embeddings_header_prefix: str = "x"
    in_house_embeddings_service_code: str = "graph"  # <prefix>-service-code header
    in_house_embeddings_quantize: bool = False

    # Hard ceiling after calibration — synthetic probes overestimate safe size
    # for real entities (aliases/descriptions). A large tenant hit calibrated-1280 then
    # 500/timeout/400 storms; keep writes small.
    neo4j_entity_batch_max: int = Field(
        default=50, gt=0,
        description="Upper bound on calibrated Neo4j entity UNWIND batch size",
    )
    # Pause between entity bulk writes to ease Query API pressure. 0 disables.
    neo4j_entity_write_pause_seconds: float = Field(
        default=0.15, ge=0,
        description="Seconds to sleep between successive entity write batches",
    )

    embedding_azure_max_tokens_per_request: int = 250000
    embedding_azure_chars_per_token_estimate: float = 3.0
    # --- Kafka (incremental worker) ---
    # deploy_env: the env slug Helm sets (matches its own helm-values/<slug>-values.yaml
    # filename, e.g. "uat-eu") — the ONLY per-env Kafka input Helm needs to supply; the
    # three names below are derived from it via config/kafka_constants.py's templates
    # unless explicitly overridden (see _default_kafka_names below).
    deploy_env: str = Field(default="dev", description="Env slug (dev/qa/uat-eu/prod-ca/...) "
                            "used to derive the Kafka names below when not set explicitly")
    # Vault's infra/<env>/static/kafka secret holds this under kafka_host (a shared,
    # infra-team-owned secret — its own key naming, not this service's convention).
    kafka_bootstrap_servers: str = Field(
        default="", validation_alias=AliasChoices("KAFKA_BOOTSTRAP_SERVERS", "kafka_host"),
        description="Kafka bootstrap brokers, comma-separated")
    kafka_security_protocol: str = "SSL"
    kafka_topic_graph_extract: str = Field(default="",
                                           description="Topic the incremental worker consumes "
                                           "(GraphExtractRequest) — derived from deploy_env if unset")
    kafka_consumer_group: str = Field(default="", description="derived from deploy_env if unset")
    kafka_poll_timeout_seconds: float = 1.0
    # When True, on partition assign the worker jumps to the tip (high watermark)
    # and only consumes messages produced after this process started. Local/demo
    # use only — production must stay False so a new group still drains backlog
    # from auto.offset.reset=earliest. Env: KAFKA_SEEK_TO_END_ON_START.
    kafka_seek_to_end_on_start: bool = False
    # DLQ: without this, a poison message (malformed JSON, a missing required field,
    # an unreachable S3 claim-check pointer) would be left uncommitted forever,
    # wedging every later message on its partition behind it.
    kafka_topic_graph_extract_dlq: str = Field(default="",
                                               description="Dead-letter topic for permanent "
                                               "failures and transient failures that exhausted "
                                               "kafka_max_delivery_attempts — derived from deploy_env if unset")
    kafka_max_delivery_attempts: int = 5  # transient failures get this many redeliveries before DLQ

    # --- Mass-retraction safety guard ---
    mass_retraction_guard_enabled: bool = True
    mass_retraction_frac: float = 0.5  # abort deletes once this fraction of a tenant's corpus would go
    mass_retraction_window_seconds: float = 3600.0
    mass_retraction_min_corpus: int = 20  # below this, a small tenant clearing its own docs isn't a runaway burst

    # --- AWS / S3 (claim-check payload download) ---
    aws_region: str = "us-west-2"
    s3_connect_timeout_seconds: float = 10.0
    s3_read_timeout_seconds: float = 30.0

    # --- Default ES index ---
    # --- Connector selection ---
    # Which connector the incremental Kafka worker reads. The bulk path takes
    # the same choice as --connector/BULK_CONNECTOR on the command line; the
    # worker has no command line under ai-jobs-manager, so it reads it here.
    incremental_connector: str = Field(
        default="wikipedia", validation_alias=AliasChoices("incremental_connector",
                                                      "INCREMENTAL_CONNECTOR"),
        description="Connector the incremental worker reads (see connectors/registry.py)")
    # TESTING/DEBUGGING ONLY — never set this in a production deployment.
    # Production always uses the latest run for a (tenant, connector); this pins
    # the incremental worker to one specific run's graph instead, and labels its
    # logs/failure-ledger entries. Unset (the default) leaves production behavior
    # unchanged. See core/kafka_worker_entrypoint.py's _TARGET_RUN_ID.
    incremental_run_id: str | None = Field(
        default=None, validation_alias=AliasChoices("incremental_run_id", "INCREMENTAL_RUN_ID"),
        description="TESTING/DEBUGGING ONLY — never set in production. Pins the incremental "
                    "worker to a specific run instead of the latest one.")

    # --- Bulk run inputs (core/entrypoint.py) ---
    # ai-jobs-manager passes these as env vars to the one-shot bulk Job; config/
    # settings.py is the single place that reads the environment, so entrypoint
    # reads them here instead of calling os.environ itself (the same rule the
    # incremental worker already follows for INCREMENTAL_CONNECTOR above). A CLI
    # flag still wins over the env value — see core/entrypoint.py's resolvers.
    bulk_connector: str | None = Field(default=None, validation_alias=AliasChoices("BULK_CONNECTOR"))
    bulk_index: str | None = Field(default=None, validation_alias=AliasChoices("BULK_INDEX"))
    bulk_limit: int | None = Field(default=None, validation_alias=AliasChoices("BULK_LIMIT"))
    bulk_tenant_id: str | None = Field(default=None, validation_alias=AliasChoices("BULK_TENANT_ID"))
    bulk_retry_failed: str | None = Field(default=None, validation_alias=AliasChoices("BULK_RETRY_FAILED"))

    # ==========================================================================
    # Calibration constants, ordered by pipeline stage (Step 0 fetch -> 1
    # extract -> 2 doc-canon -> 3 corpus-canon -> 4 edge pool -> 5 write),
    # incremental worker last. CALIBRATED = measured against real Azure/
    # Ministral calls on real prod chunks, 2026-08-07. Everything else is
    # still a guess. Ministral's own performance isn't tuned here — the
    # deploying team's call, not ours.
    # ==========================================================================

    # --- Step 0: fetch (connectors/elasticsearch_client.py) + chunk-batch sizer ---
    es_request_timeout_seconds: float = 60.0
    es_search_max_retries: int = 3
    es_search_retry_backoff_seconds: float = 1.0
    es_pit_keep_alive: str = "4h"
    es_max_single_page: int = 10000
    es_site_fetch_page_size: int = 100
    es_fetch_default_batch_size: int = 200
    bulk_batch_initial_chunks: int = 50
    bulk_batch_target_seconds: float = 45.0
    bulk_batch_max_change_factor: float = 0.25
    bulk_batch_min_chunks: int = 10
    bulk_batch_max_chunks: int = 5000

    # --- Step 1: extraction (services/extraction_service.py) ---
    llm_probe_timeout_seconds: float = 5.0  # startup backend probe, not per-call

    # Ministral. All call kinds share ONE throttle pool (services/llm_client.py
    # get_throttle_for), so these numbers bound INGESTION's total in-flight to
    # the whole service, not one kind's.
    #
    # Sized against the g6e.xlarge (L40S) serving deployment: each of the 2
    # pods admits at most 30 running sequences (--max-num-seqs 30, fp8 KV
    # cache), so the service-wide admission ceiling is 60. The ceiling below
    # (16) deliberately claims only ~a quarter of that: the L40S goodput knee
    # is estimated (~2.9x the old node's bandwidth), not yet measured, and
    # ingestion is not the pods' only consumer. AIMD does the fine-tuning
    # inside the cap — if latency holds, the pool sits pinned at the ceiling;
    # if the pods degrade, timeouts halve it. Raise after a k6 constant-VU
    # re-measurement on the L40S nodes, not before.
    #
    # (History: the previous 3/6 values were the measured knee of the OLD
    # 8k-context node — 0% errors up to 7 concurrent, collapse past 8.)
    ministral_max_concurrency: int = 8
    # 5, not 10: with the ceiling hard-capped at the pod's knee, probing
    # upward is cheap and bounded — the only cost of a lower value is
    # reaching the cap sooner. At ~2 min per extraction call, 10 successes
    # per +1 step meant a knocked-down pool spent tens of minutes crawling
    # back to capacity; 5 halves that recovery time with no overload risk
    # (the cap, not the ramp rate, is what protects the pod).
    ministral_throttle_increase_after: int = 5
    # Floor equals the starting concurrency, so a dropped connection cannot
    # halve the pool (8 -> 4) and leave the run crawling. The hosted API's
    # own limits are what apply backpressure, not this process.
    ministral_throttle_min_concurrency: int = 8
    ministral_throttle_max_concurrency: int = 16
    # Client-side request timeout for the Ministral gateway, its OWN knob —
    # previously the Azure timeout (120s) was reused, but extraction p90 was
    # measured at 313s on real prod chunks (run x9j2, 2026-08-12): a 120s
    # client deadline abandons calls the pod then finishes for nobody, which
    # is how the load test's goodput collapse happens. Must stay >= the
    # longest legitimate generation (max_tokens / worst-case tok/s ≈ 260s)
    # and <= the model-orchestrator's own vllm_read_timeout (360s).
    ministral_request_timeout: float = 360.0
    ministral_throttle_slow_call_multiplier: float = 3.0
    ministral_throttle_recent_window: int = 20
    # NO client-side token budgeting for Ministral: max_tokens is simply not
    # sent, and vLLM's chat API defaults it to (context window − prompt
    # tokens) using the real tokenizer. The char-based estimate this replaces
    # (chars/3.9 + safety buffer) could only miss in two directions, and one
    # bulk run produced both at scale: 86 under-requests (wasted
    # correct-and-retry round trips) and 67 over-requests (outright 400
    # rejections with no retry path). The window itself (8192,
    # prompt+completion combined) is the serving deployment's
    # --max-model-len; a genuine overflow surfaces as finish_reason='length'
    # and extraction recovers by salvage + follow-up.

    # Azure throttle. CALIBRATED: ramp on 18 real chunks — 10 concurrent =
    # 14.14 docs/min, 20 = 27.9/min, 30 = 27.96/min (no gain past 20).
    azure_throttle_initial_concurrency: int = 20
    azure_throttle_increase_after: int = 20
    azure_throttle_min_concurrency: int = 1
    azure_throttle_max_concurrency: int = 500
    azure_openai_request_timeout: float = 120.0

    # Only feeds Ministral's throttle (Azure ignores it). RECALIBRATED against
    # a real bulk run at full concurrency: extraction p50 ≈ 125s, p90 ≈ 313s
    # (the old 70s came from n=23 lightly-loaded calls). This number does two
    # things in LatencyAdaptiveThrottle and both were broken at 70:
    #   growth:   only calls <= target count toward +1 concurrency. At 70s a
    #             loaded pool (p50 125s) could NEVER earn its way back up
    #             after a knockback — halve-fast, recover-never.
    #   back-off: a SUCCESS slower than target x slow_call_multiplier (3.0)
    #             halves the pool. 70x3 = 210s < p90, so legitimate long
    #             generations were punished as overload. 130x3 = 390s sits
    #             deliberately ABOVE the 360s client timeout: a genuinely
    #             overloaded pod times out first, which records a real
    #             failure (the correct overload signal) — latency-based
    #             halving of successful calls can no longer fire, by design.
    # Overload protection is unchanged: the hard concurrency ceiling (6, the
    # pod's measured knee) plus halving on timeout/connection/5xx/429.
    llm_extraction_target_latency_seconds: float = 130.0
    llm_extraction_max_retries: int = 3  # was 5 -- truncation retries are deterministic, no point burning 5x
    llm_extraction_base_backoff_seconds: float = 1.0
    llm_extraction_max_backoff_seconds: float = 30.0
    llm_extraction_gleaning_max_retries: int = 3
    # Documentation ceilings only — no Ministral request carries max_tokens
    # anymore (see the no-client-side-budgeting note above); the server
    # derives the real per-request budget itself. Kept because PROFILES
    # exposes a per-kind ceiling for any future caller that needs an explicit
    # cap (an explicit max_tokens passed to complete() is still forwarded).
    llm_extraction_max_tokens: int = 7792
    llm_extraction_gleaning_max_tokens: int = 7792

    # Fallback for any call site not yet individually calibrated (Step 2/3's
    # description-merge below still uses this).
    llm_default_target_latency_seconds: float = 15.0
    llm_default_max_retries: int = 5
    llm_default_base_backoff_seconds: float = 1.0
    llm_default_max_backoff_seconds: float = 30.0
    # Above extraction's measured p50 (125s under load) so the warning marks
    # genuinely unusual calls, not every extraction. NOTE: an env override of
    # 0 makes every single call warn — run x9j2's logs are full of
    # "took 0.8s (> 0s)" spam from exactly that; don't set this to 0.
    llm_slow_call_warn_seconds: float = 120.0

    # --- Step 2: doc-level canonicalization (services/doc_canonicalization_service.py,
    # services/clustering_utils.py, services/people/disambiguation.py) ---
    doc_canonicalization_concurrency: int = 20
    doc_cluster_resolve_max_depth: int = 3
    doc_descriptions_per_group_max: int = 20
    # Payload budget per doc-level description-merge/reduce call, in characters.
    # Smaller than corpus_description_merge_char_budget (20_000): doc descriptions
    # are ~250-400 chars vs corpus's ~900-1300 already-merged summaries.
    doc_description_merge_char_budget: int = 8000
    doc_cluster_merge_batch_max: int = 50

    # Only feeds Ministral's throttle (Azure ignores it). CALIBRATED
    # (Ministral, n=6): p50=13.2s/p90=14.3s. Was 8.0 (guess).
    llm_cluster_merge_target_latency_seconds: float = 15.0
    llm_cluster_merge_max_retries: int = 5
    llm_cluster_merge_base_backoff_seconds: float = 1.0
    llm_cluster_merge_max_backoff_seconds: float = 30.0
    # CALIBRATED (Ministral, n=2, 10-20 candidates): completion 404-576.
    llm_cluster_merge_max_tokens: int = 1500

    # Yes/no confirm — also used by incremental merge-confirm and person
    # disambiguation's single-pair path. CALIBRATED (Ministral, n=8):
    # p50=1.75s/p90=1.82s. Was 5.0.
    llm_confirm_target_latency_seconds: float = 2.5
    llm_confirm_max_retries: int = 3
    llm_confirm_base_backoff_seconds: float = 1.0
    llm_confirm_max_backoff_seconds: float = 15.0
    # CALIBRATED (Ministral, n=6): completion is a flat 9 tokens every time
    # (just {"merge": true/false}) — plenty of headroom at 50.
    llm_confirm_max_tokens: int = 50

    # CALIBRATED (Ministral, n=5): p50=4.19s/p90=4.34s. Was 15.0 (fallback).
    llm_doc_description_merge_target_latency_seconds: float = 5.0
    llm_doc_description_merge_max_retries: int = 5
    llm_doc_description_merge_base_backoff_seconds: float = 1.0
    llm_doc_description_merge_max_backoff_seconds: float = 30.0
    # CALIBRATED (Ministral, n=4): completion 107-151 tokens.
    llm_doc_description_merge_max_tokens: int = 500

    # CALIBRATED (Ministral, n=5): p50=0.89s/p90=0.9s. Was 15.0 (fallback).
    llm_disambiguation_target_latency_seconds: float = 1.5
    llm_disambiguation_max_retries: int = 5
    llm_disambiguation_base_backoff_seconds: float = 1.0
    llm_disambiguation_max_backoff_seconds: float = 30.0
    # CALIBRATED (Ministral, n=4): completion a flat 19 tokens every time.
    llm_disambiguation_max_tokens: int = 100

    disambiguation_batch_max_candidates: int = 20
    disambiguation_items_per_call: int = 25
    # Shared with Step 3's cluster_to_fixpoint. 2, not 3: round 1 does the real
    # merging and round 2 confirms the fixpoint; a third round almost never
    # merges anything new (run x9j2: round 2 already regrouped, round 3 idle)
    # but costs a full cluster-merge LLM sweep per document.
    canon_max_iterations: int = 2

    # --- Step 3: corpus-wide canonicalization (services/graph_canonicalization_service.py) ---
    person_resolution_concurrency: int = 8
    # Bounds resolve_entity_ids' concurrent entity_registry.link_or_mint calls —
    # same conservative default as person_resolution_concurrency above, which
    # already runs the identical registry concurrently for PERSON entities.
    corpus_entity_id_resolution_concurrency: int = 8
    corpus_cluster_resolve_max_depth: int = 6
    corpus_description_merge_max_descriptions: int = 30
    corpus_description_merge_char_budget: int = 20_000
    corpus_description_signal_max_chars: int = 1400
    corpus_cluster_merge_batch_max: int = 50
    # Hard cap on the per-entity PERSISTED "descriptions" list (one entry per
    # contributing document, ACL-taggable) — protects that one entity's Mongo
    # document from the 16MB BSON limit when a name is mentioned in an extreme
    # number of documents. Well above any realistic entity's mention count, so
    # this only ever trims the pathological tail; the merge/embedding signal
    # above already samples/reduces separately and is unaffected.
    corpus_entity_max_persisted_descriptions: int = 5000

    # CALIBRATED (Ministral, n=5): p50=7.88s/p90=9.33s — slower than doc-level,
    # bigger packed payload. Was 15.0 (fallback).
    llm_corpus_description_merge_target_latency_seconds: float = 10.0
    llm_corpus_description_merge_max_retries: int = 5
    llm_corpus_description_merge_base_backoff_seconds: float = 1.0
    llm_corpus_description_merge_max_backoff_seconds: float = 30.0
    # CALIBRATED (Ministral, n=3): 2 completed at 311-365, 1 exceeded 5000 —
    # big packed batches are as unpredictable as extraction, same max-safe ceiling.
    llm_corpus_description_merge_max_tokens: int = 7792

    # --- Step 4: edge pool (core/pipeline.py) ---
    bulk_edge_pool_doc_batch: int = 50
    bulk_write_flush_threshold: int = 2000

    # --- Step 5: Neo4j + Qdrant write, including embeddings ---
    # CALIBRATED (real batches): Azure plateaus ~8 workers (42.4 texts/s @8 vs
    # 45.5 @16, +7%); in-house keeps climbing (48.5 @8 vs 66.2 @16, +36%, no
    # ceiling found yet) — the two backends genuinely don't share one number.
    embedding_azure_concurrency: int = 8
    embedding_inhouse_concurrency: int = 16
    embedding_azure_batch_size: int = 500
    # Azure caps each embeddings request by TOTAL tokens across all inputs
    # (rejects ~300k tokens/request), independent of the input-count cap above —
    # so an Azure batch is also closed once adding the next text would exceed
    # this budget (services/embedding_service.py's _split_texts). 290k leaves a
    # margin under the observed ~300k limit. In-house batches stay count-only.
    embedding_azure_max_tokens_per_request: int = 290_000
    # Chars→tokens estimate used only for the token-budget batching above (a
    # cheap heuristic, not a real tokenizer). ~4 chars/token is the standard
    # English estimate for OpenAI embedding models.
    embedding_azure_chars_per_token_estimate: float = 4.0
    # DISCOVERED (a 502 from the gateway names it): the gateway rejects batches over
    # 512 inputs. 500 keeps the same margin Azure's batch size does. Overridable
    # so a server-side limit drop (512 → 256) is an env change, not a code one.
    embedding_inhouse_batch_size: int = 500
    embedding_azure_max_retries: int = 5
    embedding_azure_base_backoff_seconds: float = 1.0
    embedding_azure_max_backoff_seconds: float = 30.0
    embedding_inhouse_max_retries: int = 5
    embedding_inhouse_base_backoff_seconds: float = 1.0
    embedding_inhouse_max_backoff_seconds: float = 30.0
    embedding_inhouse_request_timeout_seconds: float = 60.0
    neo4j_http_timeout_seconds: float = 60.0
    neo4j_max_retries: int = 3
    neo4j_retry_backoff_seconds: float = 0.5
    neo4j_default_batch_size: int = 5000
    neo4j_default_entity_batch_size: int = 100
    # Hard cap applied AFTER batch-size calibration (repository/neo4j_repository.py):
    # synthetic probe rows understate real entity weight, so unconstrained calibration
    # previously landed ~1280 and overloaded the Query API (500/timeout/400 storms).
    # Must be positive — it's a min() cap and a non-positive value would zero out writes.
    neo4j_entity_batch_max: int = Field(default=500, gt=0)
    # Optional pause between entity-write batches to relieve the Query API under load;
    # 0.0 = no pause (default). Only applied between batches, never before the first.
    neo4j_entity_write_pause_seconds: float = 0.0
    neo4j_database_online_max_wait_seconds: float = 30.0
    neo4j_database_online_poll_interval_seconds: float = 0.5
    neo4j_calibration_target_seconds: float = 8.0
    neo4j_calibration_min_batch: int = 10
    neo4j_calibration_max_probe: int = 2000
    qdrant_max_retries: int = 3  # push batch size (256) is a hard limit, kept in dual_write_guard.py
    qdrant_base_backoff_seconds: float = 0.5
    qdrant_max_backoff_seconds: float = 5.0
    # Per-request timeout for the Qdrant client. qdrant-client defaults to 5s, which is too
    # short for a cold TLS handshake to the reverse-proxied dev endpoint
    # — that produced "handshake operation timed out" storms. 30s absorbs a slow handshake while
    # still bounding a genuinely dead endpoint. Applied in services/vector_store.get_qdrant_client.
    qdrant_timeout_seconds: float = 30.0

    # --- Run bookkeeping (repository/run_registry.py) ---
    run_registry_max_suffix_retries: int = 5

    # --- Incremental Kafka worker (separate runtime shape — one event at a
    # time, not the batch sequence above); reuses Step 1-5 constants too ---
    incremental_doc_concurrency: int = 5
    incremental_entity_resolve_concurrency: int = 10
    kafka_heartbeat_interval_seconds: float = 30.0
    kafka_redelivery_backoff_cap_seconds: float = 10.0
    kafka_dlq_flush_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _apply_inference_provider(self) -> "Settings":
        """inference_provider=inhouse switches chat + embeddings from the NVIDIA
        NIM defaults to a self-hosted vLLM preset — but ONLY for fields the
        caller left at this class's own default (model_fields_set), so an
        explicit MINISTRAL_URL/IN_HOUSE_EMBEDDINGS_URL/etc. override always
        wins, same precedence every other derived field in this file uses
        (see _derive_qdrant_url's docstring for why model_fields_set and not a
        value comparison). inference_provider=nvidia (the default) is a no-op:
        the field defaults above already are the NIM preset."""
        if self.inference_provider != "inhouse":
            return self
        if "ministral_url" not in self.model_fields_set:
            self.ministral_url = "http://localhost:8000/v1/chat/completions"
        if "ministral_model" not in self.model_fields_set:
            # Whatever `vllm serve <model>` is actually running locally —
            # override via MINISTRAL_MODEL if it isn't this one.
            self.ministral_model = "mistralai/Ministral-8B-Instruct-2410"
        if "in_house_embeddings_url" not in self.model_fields_set:
            self.in_house_embeddings_url = "http://localhost:8001/v1/embeddings"
        if "in_house_embedding_model" not in self.model_fields_set:
            self.in_house_embedding_model = "Snowflake/snowflake-arctic-embed-l-v2.0"
        if "in_house_embedding_dimension" not in self.model_fields_set:
            self.in_house_embedding_dimension = 256
        if "in_house_embedding_es_field" not in self.model_fields_set:
            self.in_house_embedding_es_field = "embeddings_arctic_256"
        return self

    @model_validator(mode="after")
    def _derive_embedding_backend(self) -> "Settings":
        """Map USE_IN_HOUSE_EMBEDDINGS → embedding_backend for all call sites.

        true → inhouse, false → azure. Keeps Vault as a single boolean (no
        string enum to update) while the rest of the codebase keeps reading
        the existing Literal field / SUPPORTED_EMBEDDING_SPACES table."""
        self.embedding_backend = "inhouse" if self.use_in_house_embeddings else "azure"
        return self

    @model_validator(mode="after")
    def _apply_ec2_databases(self) -> "Settings":
        if not self.use_ec2_databases:
            return self
        if not self.ec2_neo4j_http_url:
            raise ValueError("USE_EC2_DATABASES=true requires EC2_NEO4J_HTTP_URL")
        if not self.ec2_neo4j_password:
            raise ValueError("USE_EC2_DATABASES=true requires EC2_NEO4J_PASSWORD")
        if not self.ec2_mongo_uri:
            raise ValueError("USE_EC2_DATABASES=true requires EC2_MONGO_URI")
        if not self.ec2_qdrant_url:
            raise ValueError("USE_EC2_DATABASES=true requires EC2_QDRANT_URL")

        self.neo4j_http_url = self.ec2_neo4j_http_url
        self.neo4j_username = self.ec2_neo4j_username
        self.neo4j_password = self.ec2_neo4j_password
        self.mongo_uri = self.ec2_mongo_uri
        self.mongo_db = self.ec2_mongo_db
        self.qdrant_url = self.ec2_qdrant_url
        self.qdrant_api_key = self.ec2_qdrant_api_key
        # EC2 docker Qdrant is addressed by full URL — ignore Vault QDRANT_HOST.
        self.qdrant_host = None
        return self

    @model_validator(mode="after")
    def _default_kafka_names(self) -> "Settings":
        """Fill in the three Kafka names from deploy_env + the shared templates,
        but only the ones NOT explicitly set by an env var/`.env` — an explicit
        override always wins, same setdefault-style precedence every other
        Vault-sourced value in this codebase already follows."""
        if not self.kafka_topic_graph_extract:
            self.kafka_topic_graph_extract = KAFKA_TOPIC_GRAPH_EXTRACT_TEMPLATE.format(
                env=title_env(self.deploy_env))
        if not self.kafka_topic_graph_extract_dlq:
            self.kafka_topic_graph_extract_dlq = KAFKA_TOPIC_GRAPH_EXTRACT_DLQ_TEMPLATE.format(
                env=upper_env(self.deploy_env))
        if not self.kafka_consumer_group:
            self.kafka_consumer_group = KAFKA_CONSUMER_GROUP_TEMPLATE.format(env=lower_env(self.deploy_env))
        return self

    @model_validator(mode="after")
    def _derive_qdrant_url(self) -> "Settings":
        """Build qdrant_url from qdrant_host (Vault's actual key) when qdrant_url
        itself was never explicitly set — same explicit-wins precedence as
        _default_kafka_names above. HTTPS, no port: this shared dev server fronts
        Qdrant behind a plain HTTPS reverse proxy on the standard 443 port (see
        qdrant_url's own field-level default/comment).

        Checks model_fields_set rather than comparing qdrant_url's value against
        its own default — a value-comparison check can't distinguish "never set"
        from "explicitly set to the same value as the default" (e.g. a developer
        deliberately pointing at their own local Qdrant with
        QDRANT_URL=http://localhost:6333 while QDRANT_HOST is also present from
        Vault), and would silently overwrite the latter."""
        if self.use_ec2_databases:
            return self
        if self.qdrant_host and "qdrant_url" not in self.model_fields_set:
            # The real secret's QDRANT_HOST is already a full URL (scheme included),
            # despite the field name suggesting a bare hostname — confirmed by testing
            # against the real dev secret. Don't double-prepend a scheme if it has one.
            self.qdrant_url = self.qdrant_host if "://" in self.qdrant_host else f"https://{self.qdrant_host}"
        return self


# Singleton — every other module imports `settings` from here, never re-reads env vars itself.
settings = Settings()

# The real "inhouse" row: settings.in_house_embedding_* already resolved the
# nvidia-vs-inhouse preset above (Settings._apply_inference_provider) — this
# just reflects that choice into the table every other module reads by key
# (services/llm_backend.py, services/chunking.py, ...), so there is exactly
# ONE place (inference_provider) that decides it, not two.
SUPPORTED_EMBEDDING_SPACES["inhouse"] = {
    "model": settings.in_house_embedding_model,
    "dimension": settings.in_house_embedding_dimension,
    "es_vector_field": settings.in_house_embedding_es_field,
}
