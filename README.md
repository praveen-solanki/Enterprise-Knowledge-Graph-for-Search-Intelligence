# GraphRAG Ingestion Service

An ingestion service with two runtime shapes: a **bulk job** (`core/entrypoint.py`)
that does one full pass over an Elasticsearch index, and an **always-on
incremental worker** (`core/kafka_worker_entrypoint.py`) that consumes
per-document update/delete events off Kafka as they happen. Both pull from
Elasticsearch, use an LLM to extract entities, relationships, and factual
claims, deduplicate everything into one unified knowledge graph, and write
that graph to Neo4j — with entity embeddings persisted so a future query
agent can vector-search the graph, not just traverse it. Every document is
tagged with access-control tokens derived from its own Elasticsearch fields
(site, ownership, publish status), so a future retrieval layer has something
real to filter on rather than defaulting every document to public.

This service only **builds** the graph. Querying it (a chat/search agent that
reads from this graph) is a separate, not-yet-built service. Every batching/
concurrency setting mentioned below self-tunes at runtime — there is no number
you need to hand-pick before running this against a real corpus.

---

## 1. What this is

- **Input:** any Elasticsearch index of chunked documents (bulk), plus a
  Kafka topic of per-document change events (incremental).
- **Output:** a Neo4j graph of Documents, Chunks, Entities, Relationships, and
  Claims — plus vector embeddings on every Entity, so a future query agent can
  search the graph semantically, not just by exact name — and an
  `allow_access_control` token list on every node/relationship, so a future
  retrieval layer can filter by who's allowed to see what.
- **Scope:** the bulk job does one full run over an index, start to finish;
  re-running is always safe (every write is an idempotent `MERGE`). The
  incremental worker runs continuously alongside it, keeping the graph in
  sync with individual document changes without waiting for the next bulk
  run — see [§2](#2-how-it-works--the-pipeline) and
  [§8](#8-whats-intentionally-not-built-scope-decisions-not-gaps) for what's
  still genuinely out of scope.

---

## 2. How it works — the pipeline

```
Step 0      Step 0.5    Step 1          Step 2              Step 3                Step 4          Step 5
Fetch   ->  Derive  ->  Extract    ->   Doc-level      ->   Corpus-wide     ->   Edge pool   ->   Neo4j
(ES)        ACL         (LLM)           canonicalize        canonicalize         (dedupe          write
                                        (per doc)           (across ALL          edges/claims,
                                                             docs, once)          resolve mentions)
```

**Steps 0-2 stream in adaptively-sized batches** — nothing about batch size is
fixed; see [§3](#3-how-this-scales) for how that's decided at runtime. A
failing document in Steps 1-4 doesn't abort the run — see "Fault isolation
& failure ledger" below.

0.5. **Derive ACL** ([services/acl/](services/acl/)) —
     before anything else touches a document, `build_document_acl()` turns its
     ES fields (or connector-shipped tokens) into a flat list of access-control
     tokens. Zeus uses the production grammar (`site:mem:`, `content:pub`, …);
     connectors pass their own tokens through. A document that's deleted,
     unpublished, or resolves to zero tokens is dropped entirely — fail-closed,
     never defaulted to public. A "public" site can still be individually
     audience-gated (`connectors/zeus/spec.py`'s `fetch_site_audiences()`,
     fetched once per bulk run / once per incremental event's site set). Every
     downstream node (Chunk, Entity, Relationship, Claim) inherits this
     document's tokens into its own `allow_access_control` property — see
     [§4](#4-what-ends-up-in-the-graph).

1. **Fetch** ([connectors/elasticsearch_client.py](connectors/elasticsearch_client.py)
   + the connector's own [spec](connectors/base.py))
   — `fetch_documents_batched()` pulls documents from Elasticsearch via a
   Point-In-Time (PIT) snapshot + `search_after` cursor, yielding one batch at
   a time instead of loading the whole index into memory. The batch boundary
   is drawn by **chunk count**, not document count, so every batch is a
   predictable amount of LLM work regardless of whether a document has 2
   chunks or 50.

2. **Extract** ([services/extraction_service.py](services/extraction_service.py))
   — every chunk in the batch is sent to Azure OpenAI, concurrently, at
   whatever concurrency the adaptive throttle currently allows (see §3). Each
   chunk comes back with its own entities, relationships, and claims. The
   moment **one chunk** finishes, its result is saved to MongoDB — not after
   the whole batch. If the process crashes mid-batch, nothing already
   finished is lost; re-running only re-does the chunks that never completed
   *or genuinely failed* (a failed call is never cached as if it were an
   empty result — see §3).

3. **Doc-level canonicalize** ([services/doc_canonicalization_service.py](services/doc_canonicalization_service.py))
   — once every chunk of *one* document is extracted, that document's
   entities are deduplicated against **themselves**: "Globex" and
   "Globex Inc." mentioned in three different chunks of the same doc become
   one entity, just for that doc. The result (plus that doc's chunk IDs, so
   later steps can find them again) is saved to MongoDB and immediately
   dropped from memory.

   This repeats, batch after batch, until every document in the run has been
   through Steps 0-2. Memory never grows with corpus size — see §3.

4. **Corpus-wide canonicalize** ([services/graph_canonicalization_service.py](services/graph_canonicalization_service.py))
   — runs **once**, after all batches are done, not per-batch. This is the
   one step that can't stream: "Globex" in document #3 and document #4,997
   can only be recognized as the same entity by comparing them directly. It
   reads every document's already-compressed canonical entities back from
   MongoDB in a streaming cursor (never as one giant in-memory dict), so even
   this step doesn't need the whole corpus loaded at once.

5. **Edge pool** ([services/edge_pool_service.py](services/edge_pool_service.py))
   — re-loads each document's saved extraction results from MongoDB and
   re-points every relationship and claim at its final, corpus-deduped entity
   name. The same relationship found in 40 different chunks collapses into
   **one** edge with 40 supporting `chunk_ids` as evidence. Also builds
   `mentions`: which chunk each final entity actually appeared in. Processed
   and written to Neo4j document-by-document, flushed in batches — see §3.

6. **Neo4j write** ([repository/neo4j_repository.py](repository/neo4j_repository.py))
   — writes Documents, Chunks, Entities (with embeddings + aliases),
   Mentions, Relationships, and Claims. Every write is an idempotent `MERGE`,
   keyed on real identity (`doc_id`, `chunk_id`, `(name, type)`, `claim_id`) —
   so the same entity mentioned across 500 or 50,000 documents always becomes
   exactly **one** node, which is what turns many per-document mini-graphs
   into one connected graph.

### Fault isolation & failure ledger

A single bad document used to abort an entire bulk run. Now Steps 1, 2, and 4
isolate per-document failures: a doc whose extraction/canonicalization/edge
build fails is recorded to MongoDB's `ingestion_failures` collection
(`repository/mongo_repository.py` — `record_failure`/`list_failures`/
`clear_failure`) and skipped, and the rest of the run continues. Steps 3
(corpus-wide clustering) and Step 5 (the single batched entity write) aren't
per-item isolatable — a failure there is ledgered under the `"__corpus__"`
sentinel and still aborts the run, since there's no smaller unit to skip.

Re-drive just the ledgered failures for a run without repeating the whole
corpus:

```bash
python -m core.entrypoint --retry-failed <index>:<tenant_id>
```

(`run_id` defaults to `<index>:<tenant_id>` — the same flags you used for the
original run — so nothing needs to be copied down from the logs.) A
`"__corpus__"`-only ledger prints a warning instead of silently doing
nothing — those need a full re-run, which Mongo/extraction checkpointing
makes cheap since already-succeeded documents are skipped via cache.

### Incremental (Kafka) path

`core/kafka_worker_entrypoint.py` is a separate, always-on process — an
at-least-once Kafka consumer for per-document `GraphExtractRequest` events
(upsert/delete), instead of a full-index scan:

- **Payload resolution** ([services/incremental_event_parser.py](services/incremental_event_parser.py),
  [services/s3_client.py](services/s3_client.py)) — an event carries either
  its documents inline or an S3 "claim-check" pointer for large payloads.
- **ACL** — derived via `build_document_acl()` (same entry point as bulk). Zeus
  derives from a fresh Elasticsearch lookup of the live document; connectors
  pass through Gate-2 tokens from the event. If nothing is derivable, the
  event is dropped — fail-closed, never defaulted to public.
- **Per-document processing** ([services/incremental_pipeline.py](services/incremental_pipeline.py))
  — hard-replaces that one document's chunks/entities/relationships (so a
  re-ingested, changed document never leaves its previous chunks/edges
  orphaned), extracts via the same cached `_extract_with_cache` the bulk path
  uses, and resolves each entity against the **live graph**
  ([services/incremental_canonicalization.py](services/incremental_canonicalization.py))
  instead of a corpus-wide cluster, since there's no "whole corpus" available
  per-event.
- **Watermark** — a per-`(tenant_id, doc_id)` completed_at checkpoint drops
  stale/out-of-order redeliveries instead of reprocessing them backward.
- **DLQ** — an exception is classified (`utils/error_classification.py`)
  as permanent (bad JSON, a missing field, an S3 403/404 — sent straight to a
  dead-letter topic) or transient (network blips, Neo4j 5xx — redelivered up
  to `KAFKA_MAX_DELIVERY_ATTEMPTS` times, tracked in Mongo so the count
  survives a worker restart, before also going to the DLQ).

---

## 3. How this scales

Every number below is discovered at runtime, not hand-picked — the service is
meant to work the same way on a quiet Tuesday and a rate-limited Monday
without anyone touching a config file.

- **O(batch) memory, not O(corpus).** Nothing is held in memory across
  batches. Every intermediate result (a chunk's extraction, a document's
  canonical entities, a corpus-wide entity) is written to MongoDB the moment
  it's produced and released from Python immediately after. Running on
  50,000 documents uses the same memory budget as running on 500.

- **Stateless cursor, not scroll.** Fetching uses a PIT snapshot +
  `search_after` instead of the legacy scroll API. A PIT is a cheap,
  stateless index pin — it doesn't hold heavy server-side state the way a
  scroll context does, and it's proactively refreshed on every batch
  boundary (not just when a new page happens to be fetched), so an
  unusually slow batch can't expire it mid-run.

- **Adaptive batch sizing** ([services/adaptive_controller.py](services/adaptive_controller.py) —
  `AdaptiveBatchSizer`). Batches target ~45 seconds of work each. A batch
  that finishes faster than that grows the next one by up to 25%; a batch
  that runs slower shrinks it by up to 25%. Starts conservative (50 chunks)
  and finds the right size for whatever Azure and the corpus are doing that
  day — no manual tuning.

- **Adaptive concurrency** ([services/adaptive_controller.py](services/adaptive_controller.py) —
  `AdaptiveThrottle`). Uses AIMD (the same congestion-control algorithm
  TCP uses): every 20 successful calls, concurrency goes up by 1; every
  rate-limit error, concurrency is immediately halved. It also reads Azure's
  own rate-limit response headers to jump toward the real quota faster
  instead of probing one slot at a time. **Extraction and cluster-merge
  resolution share one throttle instance** — they call the same Azure
  deployment, so a 429 from either one informs both instead of two
  independent controls fighting each other blind.

- **FAISS for entity deduplication** ([services/clustering_utils.py](services/clustering_utils.py)).
  Comparing every entity to every other entity is O(n²) — instead, each
  entity is compared only to its top-10 nearest neighbors, found
  mathematically via FAISS vector search, turning that into roughly
  O(n log n). Only *after* that narrows things down does a small candidate
  cluster (2-10 look-alike names) go to an LLM to confirm the merge — never
  the whole corpus in one prompt.

- **Calibrated, batched Neo4j writes** ([repository/neo4j_repository.py](repository/neo4j_repository.py)).
  Entity writes carry a 3072-dimension embedding per row, so their safe batch
  size is discovered by probing the real server at startup (doubling probe
  sizes until response time crosses a target) rather than guessed. Edge/
  claim/mention writes are flushed every 2,000 combined rows during Step 4 —
  batched across many documents, not once per document, so a large corpus
  doesn't turn into tens of thousands of individual HTTP round-trips.

- **Versioned MongoDB caching.** Every expensive step's cache key folds in a
  hash of that step's prompt, threshold, algorithm, *and* output schema.
  Change any of those — or fix a bug in how a checkpoint is shaped — and the
  old cached results are automatically treated as a miss, never silently
  served stale. Checkpoints are stored **one MongoDB document per
  entity/edge/claim/mention**, not one blob per run, to stay under MongoDB's
  16MB document limit at real corpus scale.

- **Failure doesn't poison the cache.** A chunk that fails (rate-limit
  exhaustion, timeout, content-filter rejection) is never cached as if it
  were a legitimately empty result — it's retried on the next run instead of
  silently losing that chunk's data forever.

- **Explicit request timeouts.** Every Azure OpenAI call has an explicit
  timeout (`AZURE_OPENAI_REQUEST_TIMEOUT`, default 120s) instead of relying
  on the SDK's own default. A call slower than 30 seconds — even a
  successful one — logs a warning, so a degrading backend shows up in the
  logs instead of just quietly making everything take longer.

---

## 4. What ends up in the graph

```
(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:<REL_TYPE>]->(:Entity)          e.g. MANAGES, DEPENDS_ON, PART_OF
(:Claim)-[:ABOUT]->(:Entity)
```

| Node / Edge | Key fields |
|---|---|
| `Document` | `doc_id`, `title`, `source`, `source_type`, `published_at`, `tenant_id`, `allow_access_control[]` |
| `Chunk` | `chunk_id`, `text`, `tenant_id`, `allow_access_control[]` |
| `Entity` | `name`, `type`, `descriptions_json`, `aliases_json`, `source_docs[]`, `doc_ids[]`, `support`, `embedding[3072]`, `tenant_id`, `allow_access_control[]` |
| `Relationship` (dynamic type) | `descriptions_json`, `chunk_ids[]`, `raw_types[]`, `tenant_id`, `allow_access_control[]` |
| `Claim` | `claim_id`, `predicate`, `object`, `statement_type`, `temporal_type`, `valid_from`, `valid_to`, `chunk_id`, `tenant_id`, `allow_access_control[]` |

- **`allow_access_control`** — the ACL token list derived in Step 0.5 (see
  §2). A Document/Chunk carries exactly its own document's tokens; an
  Entity/Relationship carries the **union** of every contributing document's
  tokens ("visible if any source is visible"); a Claim carries exactly its
  one source document's tokens (a claim is never merged across documents).
  There's no retrieval-layer code in this repo yet to actually enforce this —
  it's tagged for whoever builds that next.
- **`descriptions_json` / `aliases_json`** — a JSON-encoded list of
  `{text/name, doc_id, allow_access_control}`, one entry **per contributing
  document**, never collapsed into a single winning string/list. This is
  deliberate: an entity visible through one public source must never leak a
  description/alias that actually came from a document the viewer can't see.
  A relationship's `descriptions_json` entry also carries `documented_on` —
  that source document's own date, same per-source treatment.
- **`embedding`** — a 3072-dim vector (from `text-embedding-3-large`),
  persisted so a query agent can semantic-search entities via the
  `entity_embedding` vector index (cosine similarity), created automatically
  on the first run.
- **`tenant_id`** — every node/relationship is scoped to one tenant; the
  `entity_name_type_tenant` uniqueness constraint (and its Relationship/Claim
  equivalents) means two tenants' data with the same entity name never
  collides into one node.
- **`mentions`** (the `Chunk -[:MENTIONS]-> Entity` edges) — trace an entity
  back to the exact source text it came from, for citations/grounding and
  auditing.
- Relationships and claims match their entity endpoints on **`(name, type,
  tenant_id)`**, not name alone — the same triple the uniqueness constraint
  uses — so a name that resolves to two different types (or tenants) across
  runs can never bind an edge to the wrong node.

---

## 5. Configuration

All settings live in [config/settings.py](config/settings.py) and are read
from environment variables (or a local `.env` file for convenience — a real
environment variable always wins if both are set, so this is Vault-compatible).

| Variable | Purpose | Default |
|---|---|---|
| `ES_HOST` | Elasticsearch host | `localhost` |
| `ES_PORT` | Elasticsearch port | `9200` |
| `ES_SCHEME` | `http` or `https` | `http` |
| `ES_USERNAME` | Elasticsearch username | `elastic` |
| `ES_PASSWORD` | Elasticsearch password | *(required)* |
| `VERIFY_CERTS` | Verify TLS certs (ES + Neo4j) | `true` — set to `false` for local dev only |
| `NEO4J_HTTP_URL` | Base URL for Neo4j's Query API | `http://localhost:7474` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | *(required to write to Neo4j)* |
| `NEO4J_DATABASE` | Which database on the server to target | `neo4j` |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGO_DB` | Mongo database name for checkpoints | `graphrag_checkpoints` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | *(required)* |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | *(required)* |
| `AZURE_OPENAI_API_VERSION` | API version | `2025-01-01-preview` |
| `AZURE_OPENAI_COMPLETION_MODEL` | Completion model/deployment name | `gpt-5-nano` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name | `text-embedding-3-large-global` |
| `EXTRACTION_CONCURRENCY` | Starting reference point for the adaptive throttle (§3) — not a hard cap | `150` |
| `EMBEDDING_DIMENSIONS` | Must match your embedding deployment's output size | `3072` |
| `AZURE_OPENAI_REQUEST_TIMEOUT` | Seconds before a stalled Azure call fails and retries (§3) | `120.0` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers, comma-separated | *(required to run the incremental worker)* |
| `KAFKA_SECURITY_PROTOCOL` | Kafka security protocol | `SSL` |
| `KAFKA_TOPIC_GRAPH_EXTRACT` | Topic the incremental worker consumes | `AI_Graph_Extract_Req_Dev` |
| `KAFKA_CONSUMER_GROUP` | Consumer group ID | `graphrag-ingestion-service-group-dev` |
| `KAFKA_TOPIC_GRAPH_EXTRACT_DLQ` | Dead-letter topic for permanent/exhausted-retry failures | `AI_Graph_Extract_Req_DLQ_DEV` |
| `KAFKA_MAX_DELIVERY_ATTEMPTS` | Redeliveries a transient failure gets before going to the DLQ | `5` |
| `AWS_REGION` | Region for the S3 claim-check payload download | `us-west-2` |
| `CITATION_BASE_URL` | Organization: base URL of the intranet app citation links point into | `http://localhost:3000` |
| `INFERENCE_PROVIDER` | **The switch**: `nvidia` (NIM, hosted, free) or `inhouse` (self-hosted vLLM) — picks chat + embeddings defaults for both at once | `nvidia` |
| `NVIDIA_API_KEY` | Bearer token for NVIDIA NIM — get a free one at [build.nvidia.com](https://build.nvidia.com) (also accepts `MINISTRAL_API_KEY`) | *(required when `INFERENCE_PROVIDER=nvidia`)* |
| `MINISTRAL_URL` | Chat completions endpoint — override only if your own vLLM serves on a different host/port | `https://integrate.api.nvidia.com/v1/chat/completions` (nvidia) / `http://localhost:8000/v1/chat/completions` (inhouse) |
| `MINISTRAL_MODEL` | Chat model name — set to whatever your vLLM is actually serving | `nvidia/nemotron-3.5-bolt-e4b` (nvidia) / `mistralai/Ministral-8B-Instruct-2410` (inhouse) |
| `IN_HOUSE_EMBEDDINGS_URL` | Embeddings endpoint — override only if your own vLLM serves on a different host/port | `https://integrate.api.nvidia.com/v1/embeddings` (nvidia) / `http://localhost:8001/v1/embeddings` (inhouse) |
| `IN_HOUSE_EMBEDDING_MODEL` / `IN_HOUSE_EMBEDDING_DIMENSION` / `IN_HOUSE_EMBEDDING_ES_FIELD` | The "inhouse" embedding space's model/dimension/Qdrant-field triple | `nvidia/nemotron-3.5-embed-1b` @ 2048-d (nvidia) / `Snowflake/snowflake-arctic-embed-l-v2.0` @ 256-d (inhouse) |
| `IN_HOUSE_EMBEDDINGS_HEADER_PREFIX` | Organization: prefix of the gateway's `-service-code` / `-embedding-priority` headers | `x` |
| `EC2_MONGO_DB` | Mongo database name used when `USE_EC2_DATABASES=true` | `graph_ingestion_checkpoints_dev` |
| `PRIVATE_GIT_ORG` | Docker build arg: GitHub org hosting the private logger/vault packages | *(required for `docker build`)* |

Never commit a real `.env` file — it's git-ignored. Rotate any credential
that's ever been accidentally exposed (e.g. printed in a terminal log).

---

## 6. Running it

### Locally — bulk job

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# create .env with the variables listed in §5

# smoke test — 2 documents (every run writes to Neo4j; there is no compute-only mode)
python -m core.entrypoint --connector wikipedia --index "graph database" --tenant-id demo --limit 2

# full run against a specific index, scoped to one tenant
python -m core.entrypoint --connector arxiv --index "graph database" --tenant-id demo

# re-drive only the documents that failed during a previous run (§2's
# "Fault isolation & failure ledger") — run_id copied from the original run's logs
python -m core.entrypoint --connector zeus --retry-failed some-other-index:acme-corp:zeus:em8g
```

| Flag | Meaning |
|---|---|
| `--connector` | **Required.** The source this run reads AND the label for its isolated stores (its `connector_type` — one concept, one flag). Choices come from `connectors/registry.py`; no default, no silent fallback. |
| `--index` | **Required.** The index to read (no default). `--retry-failed` derives its own from the failure ledger and is exempt. |
| `--limit` | Cap the number of documents fetched — for smoke tests |
| `--tenant-id` | **Required.** Tenant this run's data belongs to, stamped on every node. No default (a guessed `default` would risk a cross-tenant merge). `--retry-failed` is exempt. |
| `--retry-failed <run_id>` | Re-drive only the documents ledgered as failed under that run_id, instead of a normal full run. |

Every run writes its result to Neo4j — there is no compute-only mode. Every
flag also has an env var fallback (`BULK_CONNECTOR`, `BULK_INDEX`, `BULK_LIMIT`,
`BULK_TENANT_ID`, `BULK_RETRY_FAILED`) for deployments (e.g. a Kubernetes Job)
that can only pass parameters as environment variables.

Logs go to **stdout** — there's no log file inside the container or on disk.
That's deliberate: a log file inside a container is discarded the moment it
exits, and stdout is what any real log aggregator (or `docker logs`) actually
captures.

### Locally — incremental worker

```bash
# runs continuously until SIGTERM/SIGINT — a Kubernetes-managed long-running
# pod/deployment, not a Job
python -m core.kafka_worker_entrypoint
```

Needs the `KAFKA_*` and `AWS_REGION` variables from §5 in addition to the
bulk job's ES/Neo4j/Mongo/Azure OpenAI ones. Consumes one `GraphExtractRequest`
event at a time, commits only after every document in that event's batch
succeeds, and never dies on a poison message — see §2's "Incremental (Kafka)
path" for the DLQ behavior.

### Via Docker

```bash
docker build -t graphrag-ingestion .

# uses the image's safe default: --limit 2, no Neo4j write
docker run --rm --env-file .env graphrag-ingestion

# override with real arguments
docker run --rm --env-file .env graphrag-ingestion --connector zeus --index some-index --tenant-id acme-corp

# run the incremental worker instead of the bulk job in the same image
docker run --rm --env-file .env --entrypoint python graphrag-ingestion -m core.kafka_worker_entrypoint
```

The image never bakes in credentials — every value comes from `--env-file`
(or however secrets are injected in a real deployment, e.g. Vault). The
committed `zscaler-root-ca.pem` is a public certificate needed to build this
image on a network that TLS-inspects outbound traffic (e.g. corporate
Zscaler) — safe to keep in the repo.

The bulk job does one run and exits — running it on a recurring schedule is a
separate concern (cron, a Kubernetes CronJob, whatever the company's job
scheduler is), not something this codebase does itself. The incremental
worker is the opposite shape: it's meant to run continuously as a long-lived
deployment, not be scheduled at all.

---

## 7. Troubleshooting

- **`SSL: TLSV1_ALERT_INTERNAL_ERROR` connecting to MongoDB Atlas** — almost
  always an IP allowlist issue, not a code or cert problem. Check
  cloud.mongodb.com → Network Access → make sure your current IP (or the
  network you're running from) is allowed.
- **Neo4j Bolt URIs (`neo4j://`, `bolt://`, etc.) fail with
  `ServiceUnavailable`** — the shared dev Neo4j server only exposes Neo4j over
  its HTTPS Query API, not a reachable Bolt port. Use `NEO4J_HTTP_URL`, not
  `NEO4J_URI`.
- **`pip install` fails inside the Docker build with a certificate error** —
  the network is TLS-inspecting outbound traffic; make sure
  `zscaler-root-ca.pem` (or your own network's equivalent root CA) is present
  and trusted in the Dockerfile.
- **You see `"call took N.Ns (> 30s) — succeeded"` warnings in the logs** —
  not necessarily a problem; a single slow Azure response is logged so it's
  visible, but it still succeeded. If this shows up constantly rather than
  occasionally, that's worth investigating (Azure-side degradation, or a
  chunk generating an unusually large completion) — the adaptive throttle
  will shrink concurrency in response if it starts causing real rate limits.

---

## 8. What's intentionally NOT built (scope decisions, not gaps)

**Already built** (previously listed here as not-built — no longer true):
ACL derivation + ABAC audience gating (§2 Step 0.5, [services/acl/](services/acl/)),
tenant isolation (§4's `tenant_id`), incremental/delta ingestion via the Kafka
worker with hard-replace-on-update and a DLQ (§2's "Incremental (Kafka) path"),
and a bulk-run failure ledger with targeted retry (§2's "Fault isolation &
failure ledger").

Still genuinely out of scope:

- **Retrieval / query layer** — nothing in this repo reads the graph for a
  chat/search agent. ACL tokens are stamped on every node so a future
  retrieval layer has something real to filter on, but enforcing them (who
  actually holds a given `group:site:<id>`/`group:audience:<id>` token — that
  membership data lives in Elasticsearch under `object_type: site_role`, not
  read by this service) is explicitly that future layer's job, not this one's.
- **Community detection / summarization (Leiden)** — stubbed out
  ([services/community_service.py](services/community_service.py)), not
  implemented. Real community detection is a separate effort (an algorithm
  choice, summarization prompts, its own vector index), not a small addition
  to the existing pipeline.
- **Delete-event ACL recompute across independently-scoped runs** — an
  entity/relationship's ACL is the union of every contributing document's
  tokens *from documents this service has actually seen*. A bulk run using
  `--limit` (or any narrower query) won't retroactively pick up a wider
  historical run's document ACLs for a shared entity — see
  `core/pipeline.py`'s `_tag_entity_acls` docstring.
- **`--retry-failed` for corpus-level failures** — only re-drives ledgered
  per-**document** failures (extraction/doc-canonicalization/edge-pool). A
  Step 3 (corpus clustering) or Step 5 (batched entity write) failure isn't
  tied to a specific document, so it prints a warning instead of silently
  doing nothing — re-run the full bulk job for those (cheap: checkpointing
  skips every already-succeeded document).
- **Scheduling** — see §6; the bulk job is a one-shot process by design.
- **Automated integration/load tests against a real Kafka broker or Neo4j
  server** — the test suite is unit-level (mocked ES/Mongo/Neo4j/Kafka/S3
  clients); there's no CI step that spins up real infrastructure and runs
  the worker against it end-to-end.

---

## Project structure

One job per file:

| File | Job |
|---|---|
| `config/settings.py` | Every credential/tunable, read once from env vars via `pydantic.BaseSettings`. Nothing else reads env vars directly. |
| `connectors/base.py` | The connector contract. A source answers four questions — what to fetch, what a document is, who the people are, who can see it — and `SourceDocument` is what every stage after fetch is allowed to see. Types only. |
| `connectors/registry.py` | `get_connector()` — connector lookup by type. Adding a source means adding a directory and one line here, never editing `core/`. |
| `connectors/zeus/` | The intranet behind that contract: its index and filters (`spec.py`), its field whitelists and chunk construction (`documents.py`), its ABAC grammar (`acl.py`), the two people stages a connector owns — DISCOVER and ENRICH (`people.py`), and every read of the HR export behind them (`people_fields.py`). `connectors/zeus/` is the only place in the repo that reads a `user_*` field. |
| `connectors/elasticsearch_client.py` | Step 0 — the only file allowed to talk to Elasticsearch, and it knows no source's field names. PIT + `search_after`, chunk-driven batch boundaries; what a hit means comes from the caller's `normalize`/`count_chunks`. |
| `connectors/chunking.py` | `_chunks.v1` / `_embeddings.*` — the upstream indexer's shape, shared by the bulk and incremental readers. |
| `services/acl/` | Step 0.5 — `build_document_acl()` derives `allow_access_control` tokens (Zeus + connectors). Fail-closed: dropped/ambiguous documents never enter the graph. |
| `services/extraction_service.py` | Step 1 — one chunk in, entities/relationships/claims out. Adaptive throttle, retry, per-chunk error isolation. |
| `services/embedding_service.py` | Turns text into embedding vectors. Nothing else. |
| `services/adaptive_controller.py` | Self-calibrating concurrency (`AdaptiveThrottle`), batch sizing (`AdaptiveBatchSizer`), and Neo4j write batch-size probing (`Neo4jBatchCalibrator`). |
| `services/clustering_utils.py` | Shared FAISS clustering + LLM merge-resolution, used by Steps 2 and 3. |
| `services/doc_canonicalization_service.py` | Step 2 — dedupe entities within one document. |
| `services/graph_canonicalization_service.py` | Step 3 — dedupe entities across the whole corpus; computes embeddings, per-source descriptions/aliases. |
| `services/edge_pool_service.py` | Step 4 — re-point relationships/claims at final entity names, dedupe edges, build mentions; per-source description/date/ACL tagging. |
| `services/community_service.py` | Step 6 — Leiden community detection. **Not implemented** (out of scope, see §8). |
| `utils/error_classification.py` | Shared permanent-vs-transient exception classification, used by both the Kafka DLQ and the bulk failure ledger. |
| `services/people/` | Connector-blind person algorithms: MATERIALIZE (`materialize.py` — identity on `(connector_type, source_key)` and the `:Entity:Person` entry, the shared middle of the people phase), the LLM same-person call, and bare-name resolution against the registry. Three modules, no HR field names in any of them. A library plus one stage — connectors call in, it never calls out. |
| `services/incremental_event_parser.py` | Parses the incremental Kafka event and resolves its payload (inline or S3 claim-check). |
| `services/incremental_pipeline.py` | Per-document incremental processing — the Kafka worker's equivalent of `core/pipeline.py`'s per-batch loop. |
| `services/incremental_canonicalization.py` | Resolves one incremental entity against the **live graph** instead of a corpus-wide cluster. |
| `services/s3_client.py` | Downloads a large Kafka event payload staged to S3 via the claim-check pattern. |
| `repository/mongo_repository.py` | Checkpoint storage (save/load one or many by key) plus the bulk-run failure ledger (`record_failure`/`list_failures`/`clear_failure`) and the Kafka DLQ retry-count ledger. |
| `repository/neo4j_repository.py` | Step 5 — writes the graph via Neo4j's HTTPS Query API. Auto-calibrates its own write batch sizes at startup. |
| `models/extraction.py` | Pydantic shapes for what the LLM returns (entities/relationships/claims). |
| `core/pipeline.py` | The bulk orchestrator — one coroutine, wires Steps 0-5 together, owns streaming + caching logic + per-item fault isolation. |
| `core/entrypoint.py` | The bulk job's CLI. The single `asyncio.run()` call for that process; also handles `--retry-failed`. |
| `core/kafka_worker_entrypoint.py` | The always-on incremental worker's entrypoint — Kafka consume loop, manual commit, DLQ routing. |
| `utils/logging_setup.py` | Configures logging to stdout. |
