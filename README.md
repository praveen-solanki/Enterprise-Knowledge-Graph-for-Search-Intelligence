# GraphRAG Ingestion Service

GraphRAG Ingestion Service turns source documents into a tenant-scoped knowledge graph. It fetches papers from arXiv or articles from Wikipedia, extracts entities, relationships, and claims with an LLM, resolves duplicate entities across the corpus, and writes the resulting graph to Neo4j. MongoDB holds resumable processing state, while Qdrant stores entity and chunk vectors for semantic retrieval by a separate application.

The service supports two execution patterns: a one-shot bulk ingestion run and a long-running Kafka worker that applies document updates and deletions. It is an ingestion system; this repository does not provide a user-facing search or retrieval API.

## Contents

- [Architecture](#architecture)
- [Processing workflow](#processing-workflow)
- [Data and storage](#data-and-storage)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Run a bulk ingestion](#run-a-bulk-ingestion)
- [Run the incremental worker](#run-the-incremental-worker)
- [Repository layout](#repository-layout)
- [Operational notes](#operational-notes)

## Architecture

The connector layer adapts source-specific APIs to a shared document contract. The pipeline then applies the same extraction, canonicalization, and persistence stages regardless of source. Each bulk run receives its own Neo4j database, Qdrant collections, and MongoDB checkpoint collections, named from the tenant, connector, and run suffix.

```text
Bulk ingestion
==============
arXiv Atom API ─────┐
                    ├─> Connector ─> ACL + document normalization ─> LLM extraction
Wikipedia API ──────┘                                                │
                                                                     v
                                                  Document + corpus canonicalization
                                                                     │
                                                                     v
                                                       Edge and mention assembly
                                                          /                \
                                                         v                  v
                                              Neo4j graph per run   Qdrant vectors

Incremental ingestion
=====================
Kafka document event ───────────────┐
                                    ├─> Incremental worker ─> ACL + normalization
S3 payload (when referenced) ──────┘                              │
                                                                   v
                                              Extraction + entity resolution
                                                                   │
                                                                   v
                                                    Neo4j graph + Qdrant vectors

MongoDB stores extraction checkpoints, intermediate results, run metadata,
failures, incremental retry counts, and per-document watermarks.
```

| Component | Responsibility |
|---|---|
| `connectors/` | Fetches and normalizes source data. The registry currently includes `arxiv` and `wikipedia`. |
| `core/pipeline.py` | Orchestrates bulk fetch, extraction, canonicalization, edge assembly, and persistence. |
| `core/entrypoint.py` | CLI for one-shot bulk runs and retries of failed documents. |
| `core/kafka_worker_entrypoint.py` | Long-running Kafka consumer for incremental document events. |
| `repository/` | Neo4j graph persistence, MongoDB checkpoints and run registry, and entity identity registry. |
| `services/` | Extraction, embeddings, ACLs, canonicalization, vector storage, and incremental event handling. |

## Processing workflow

For a bulk run, the selected connector fetches documents in batches and converts each into the common `SourceDocument` model. The connector assigns an ACL decision before extraction; documents that the connector marks as dropped do not enter the graph. The currently registered arXiv and Wikipedia connectors expose public material and assign the `public` ACL token.

Each chunk is sent through the configured extraction backend. Successful chunk results are checkpointed in MongoDB so completed work can be reused after a restart. The pipeline first canonicalizes entities within each document, then performs corpus-wide entity resolution across the run. It assembles relationships, claims, and chunk-to-entity mentions against the resolved entities before writing graph data and vectors.

Bulk processing isolates failures at document boundaries where possible. Failed documents are recorded in the MongoDB failure ledger and the run continues; failures in corpus-wide processing can fail the run because there is no smaller independent unit to skip. Use `--retry-failed` with the run ID printed by the original run to re-drive ledgered failures.

The incremental worker consumes `GraphExtractRequest` events from Kafka. Event payloads may be embedded in the message or referenced from S3. Updates replace the affected document's graph contribution, and deletes remove it. The worker commits Kafka offsets after successful processing, filters stale events using a per-document watermark, retries transient failures, and sends permanent or exhausted failures to a dead-letter topic. It resolves the active run for the event's tenant and connector; a completed bulk run must therefore exist before incremental updates can be applied.

## Data and storage

Neo4j is the graph store. A run writes document, chunk, entity, relationship, claim, and mention data to its own database. Writes use identity-based merges to make retries idempotent. The repository creates a database per run, which requires Neo4j Enterprise (the Community edition does not support this operation).

Qdrant stores vector collections for a run's entities and chunks. The embedding model and vector dimension are captured in the run registry so the incremental worker uses the same embedding space as the bulk run. By default, the service generates chunk vectors itself; the current arXiv and Wikipedia connectors do not provide upstream vectors. Entity vectors are generated by the service.

MongoDB stores extraction and canonicalization checkpoints, run metadata, failure records, and incremental retry and watermark state. These records allow work to resume without conflating one run's intermediate state with another's.

ACL tokens are stored with graph and vector data to support downstream authorization filtering. This repository does not implement a retrieval layer that enforces those filters; consumers must apply the relevant ACL policy when querying the stores.

## Requirements

- Python 3.11 (the Docker images use `python:3.11-slim`).
- Neo4j Enterprise 5.x reachable through the Neo4j HTTP Query API.
- MongoDB reachable by the ingestion process.
- Qdrant reachable by the ingestion process.
- A configured LLM and embedding backend. Supported options are Azure OpenAI and the configured OpenAI-compatible Ministral endpoint (NVIDIA NIM by default, or a self-hosted endpoint).
- For incremental ingestion: Kafka access and, for S3-referenced event payloads, AWS credentials with permission to read the referenced objects.

The public Python dependencies are listed in [`requirements.txt`](requirements.txt). The container build also installs two private organization packages for logging and Vault integration; see [Container images](#container-images).

## Configuration

Settings are defined in [`config/settings.py`](config/settings.py). They are loaded from environment variables or a local `.env` file. Entrypoints load Vault secrets before settings are initialized when running in a Vault-backed environment. For local development, set `ENV=local_env` to use `.env` without contacting Vault.

The following variables are the main settings needed for a local bulk run:

| Variable | Purpose |
|---|---|
| `NEO4J_HTTP_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Neo4j Query API endpoint and credentials. The default URL is `http://localhost:7474`. |
| `MONGO_URI`, `MONGO_DB` | MongoDB connection and checkpoint database. Defaults target local MongoDB and `graphrag_checkpoints`. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant endpoint and optional API key. The default URL is `http://localhost:6333`. |
| `USE_IN_HOUSE_MODEL` | Selects the Ministral-compatible chat endpoint when true; set false to use Azure OpenAI for extraction. |
| `MINISTRAL_URL`, `MINISTRAL_MODEL`, `NVIDIA_API_KEY` | OpenAI-compatible chat endpoint, model, and optional bearer key. The defaults target NVIDIA NIM. `NVIDIA_API_KEY` is also accepted as `MINISTRAL_API_KEY`. |
| `INFERENCE_PROVIDER` | Selects the default endpoint/model preset: `nvidia` or `inhouse` (self-hosted vLLM). |
| `USE_IN_HOUSE_EMBEDDINGS` | Selects the configured in-house embedding space when true; set false for the Azure embedding space. |
| `IN_HOUSE_EMBEDDINGS_URL`, `IN_HOUSE_EMBEDDING_MODEL`, `IN_HOUSE_EMBEDDING_DIMENSION` | Endpoint and model configuration for the in-house/NIM embedding space. Defaults use NVIDIA NIM and 2048 dimensions. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | Azure OpenAI endpoint and credential. Required when Azure is selected for extraction or embeddings. |
| `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_COMPLETION_MODEL`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Azure API version and deployment names. The configured defaults are in `config/settings.py`. |
| `CITATION_BASE_URL` | Base URL used when constructing source citation links. |

The bulk CLI requires a connector, an index value, and a tenant ID. These can be provided as flags or through `BULK_CONNECTOR`, `BULK_INDEX`, and `BULK_TENANT_ID`; the CLI flag takes precedence. `BULK_LIMIT` caps documents for a small run. `BULK_RETRY_FAILED` accepts a run ID to retry failures.

For the incremental worker, set `KAFKA_BOOTSTRAP_SERVERS` and, where needed, `KAFKA_SECURITY_PROTOCOL`, `KAFKA_TOPIC_GRAPH_EXTRACT`, `KAFKA_TOPIC_GRAPH_EXTRACT_DLQ`, `KAFKA_CONSUMER_GROUP`, and `KAFKA_MAX_DELIVERY_ATTEMPTS`. Topic and consumer group names are derived from `DEPLOY_ENV` when not explicitly configured. `INCREMENTAL_CONNECTOR` selects the connector used for events and defaults to `wikipedia`. `AWS_REGION` configures S3 payload access.

Do not commit credentials or a populated `.env` file. The repository's `.gitignore` excludes `.env`.

## Run a bulk ingestion

Create and activate a virtual environment, then install the public dependencies:

```bash
git clone https://github.com/praveen-solanki/Enterprise-Knowledge-Graph-for-Search-Intelligence.git
cd Enterprise-Knowledge-Graph-for-Search-Intelligence
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start or provision Neo4j Enterprise, MongoDB, and Qdrant, then configure their endpoints and credentials along with the selected model credentials in `.env`. A minimal Azure-backed local configuration can look like this:

```dotenv
ENV=local_env
NEO4J_HTTP_URL=http://localhost:7474
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace-me
MONGO_URI=mongodb://localhost:27017
MONGO_DB=graphrag_checkpoints
QDRANT_URL=http://localhost:6333
USE_IN_HOUSE_MODEL=false
USE_IN_HOUSE_EMBEDDINGS=false
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=replace-me
```

The Azure embedding deployment defaults to `text-embedding-3-large-global`; set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` if your Azure resource uses another deployment name. The embedding space must match the selected backend configuration.

Run a bounded ingestion first. The connector is required and comes from `connectors/registry.py` (`arxiv` or `wikipedia`). For Wikipedia, `--index` is a search phrase or a category such as `Category:Knowledge_graphs`. For arXiv, it is a search phrase.

```bash
python -m core.entrypoint \
  --connector wikipedia \
  --index "knowledge graph" \
  --tenant-id demo \
  --limit 2
```

The command writes to Neo4j and Qdrant even when `--limit` is set; it is a small ingestion run, not a dry run. Omit `--limit` to ingest all results returned by the connector. Each new invocation creates a new run and isolated stores.

To reprocess failures from a previous bulk run, pass the run ID reported in that run's logs:

```bash
python -m core.entrypoint \
  --connector wikipedia \
  --retry-failed '<run-id-from-logs>'
```

The same inputs can be supplied through environment variables for job schedulers:

```bash
BULK_CONNECTOR=arxiv \
BULK_INDEX="machine learning" \
BULK_TENANT_ID=demo \
BULK_LIMIT=10 \
python -m core.entrypoint
```

## Run the incremental worker

The incremental worker is a separate, continuously running process. Configure Kafka and S3 access, set `INCREMENTAL_CONNECTOR` to the connector matching the events, and ensure a completed bulk run is registered for the same tenant and connector before sending events.

```bash
python -m core.kafka_worker_entrypoint
```

The worker handles SIGTERM and SIGINT by finishing its in-flight message before shutting down. Its default Kafka topics and consumer group are derived from `DEPLOY_ENV`; explicit `KAFKA_TOPIC_GRAPH_EXTRACT`, `KAFKA_TOPIC_GRAPH_EXTRACT_DLQ`, or `KAFKA_CONSUMER_GROUP` values override those names. `INCREMENTAL_RUN_ID` and `--run-id` are intended for testing/debugging to pin a worker to one registered run.

## Container images

[`Dockerfile`](Dockerfile) packages the Kafka worker. [`Dockerfile.bulk`](Dockerfile.bulk) packages the bulk CLI and defaults to a two-document run. Both Dockerfiles install two private Git dependencies and require the organization's build-time configuration: `PRIVATE_GIT_ORG`, a BuildKit secret named `git_token`, and the certificate file `zscaler-root-ca.pem`. Those private dependencies and certificate are not present in this public repository, so the Docker images cannot be built from this checkout alone. Local Python installation uses the public dependencies in `requirements.txt`.

[`docker-compose.yml`](docker-compose.yml) describes local Neo4j Enterprise, MongoDB, and Qdrant services. It requires `EC2_NEO4J_PASSWORD`, `MONGO_ROOT_PASSWORD`, and `EC2_QDRANT_API_KEY` to be set before starting, and accepts an Enterprise license for Neo4j. The application connection settings must be configured to match the services and MongoDB authentication used by the compose file.

## Repository layout

```text
config/       Runtime settings, database endpoint resolution, Kafka names
connectors/   Shared source contract and arXiv/Wikipedia adapters
core/         Bulk pipeline orchestration and Kafka worker entrypoints
models/       Document and extraction data models
repository/   Neo4j, MongoDB, run registry, and entity identity persistence
services/     Extraction, embeddings, ACL, canonicalization, vectors, incremental flow
scripts/      Deployment configuration validation utility
tests/        Unit and integration-oriented tests
```

## Operational notes

- Bulk runs require an explicit tenant ID to keep data isolated across tenants. The connector is also required; there is no fallback connector.
- Neo4j database creation is part of the run setup and requires Enterprise edition.
- The bulk pipeline checkpoints extraction and intermediate results in MongoDB. Use the reported run ID when retrying failures; a retry resumes the run's existing stores.
- A per-run graph is isolated from other runs. Incremental events are routed to a registered run for their tenant and connector.
- The ACL fields and vector payloads prepare the data for downstream retrieval, but no search endpoint or ACL-enforcing query layer is included here.
- `scripts/validate_deployment.py` validates deployment configuration when supplied with a deployment values directory. Run `python scripts/validate_deployment.py --help` to see its supported options.
