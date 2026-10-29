"""
The real entry point. The pipeline steps live in core/pipeline.py and the
per-source index/filter defaults live in each connector's spec — this file's
only job is to expose run_bulk_async() as something runnable from the command
line, instead of only callable via `python -c "from core.pipeline import ..."`.

asyncio.run() is called EXACTLY ONCE here and nowhere else in the codebase.
Every await inside run_bulk_async() — extraction, canonicalization, embeddings —
shares this single event loop for the entire run, so the semaphore and Azure
clients used throughout are bound to one loop and never encounter a
cross-loop RuntimeError on multi-batch runs.

Env-var fallback: when this image runs as a Kubernetes Job via
ai-jobs-manager, parameters can only arrive as environment variables — there
is no way to pass CLI arguments through their job-creation contract. So each
flag below falls back to an env var when not passed on the command line; an
explicit CLI flag still wins for local/manual use.

Run-status tracking: ai-jobs-manager's own "did it finish" check gives up
watching well before a real bulk run completes, and is irrelevant on its
fire-and-forget on-demand trigger path anyway — so this writes its own
RUNNING/COMPLETED/FAILED record via the existing save_checkpoint (no new
Mongo primitive needed), queryable independently of whatever ai-jobs-manager
itself believes happened.

Every run writes its result to Neo4j — there is no compute-only mode (PR-41
review point 6). Each run mints its own isolated stores from (tenant, connector).

Usage (--connector, --index and --tenant-id are all required):
    python -m core.entrypoint --connector zeus --index some-index --tenant-id acme-corp
    python -m core.entrypoint --connector zeus --index some-index --tenant-id acme-corp \
        --limit 5  # smoke test on 5 documents
    python -m core.entrypoint --connector zeus --retry-failed some-index:acme-corp:zeus:em8g   # re-drive failed docs;
                                                                                    # copy run_id from logs

    # Equivalent, driven entirely by env vars (as ai-jobs-manager would do):
    BULK_CONNECTOR=zeus BULK_INDEX=some-index BULK_LIMIT=5 BULK_TENANT_ID=acme-corp \
        python -m core.entrypoint
    BULK_CONNECTOR=zeus BULK_RETRY_FAILED=some-index:acme-corp:zeus:em8g python -m core.entrypoint

--connector/BULK_CONNECTOR is the one name for the source: it both selects which
source to read AND labels this run's isolated stores (it IS the run's
connector_type — one concept, one flag, one spelling). The choices come from
connectors/registry.py, so they can never drift from what is actually
registered. It is required: there is no default and no silent fallback to zeus.

--index/BULK_INDEX and --tenant-id/BULK_TENANT_ID are both required — there is
no default and no guessing (PR-41 review point 3). --retry-failed is the one
exemption: it derives its index from the failure ledger and resumes an
already-registered run_id, so neither is a fresh input to supply.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# Must run before any import that touches config.settings — Settings is built
# eagerly the moment it's first imported, so Vault's values need to already be
# in os.environ by then. See connectors/vault_manager.py's module docstring.
from connectors.vault_manager import setup_environment_variables

setup_environment_variables()

from config.settings import settings
from connectors.registry import get_connector, known_connector_types
from core.pipeline import run_bulk_async, run_retry_failed_async
from repository.mongo_repository import save_checkpoint
from services.llm_backend import resolve_llm_backends
from utils.log_context import set_log_context

logger = logging.getLogger("graphrag.core.entrypoint")

_STATUS_COLLECTION = "bulk_run_status"


# A run_id is always "<index>:<tenant_id>" (see _resolve_args's run_id default
# below), so it always contains a colon -- these boolean-looking tokens can
# never be a real one. Guards against BULK_RETRY_FAILED=false being set by
# mistake (a common true/false env convention) and silently being treated as an
# enabled retry with run_id "false".
_FALSY_ENV_TOKENS = ("", "false", "0", "no", "off")


def _resolve_connector(args: argparse.Namespace) -> str:
    connector = args.connector if args.connector is not None else settings.bulk_connector
    if not connector:
        sys.exit("--connector (or BULK_CONNECTOR) is required — there is no default. "
                 f"Known connectors: {known_connector_types()}.")
    connector = connector.strip().lower()
    if connector not in known_connector_types():
        sys.exit(f"--connector/BULK_CONNECTOR={connector!r} is not a known connector — "
                 f"known: {known_connector_types()}")
    return connector


def _resolve_args(args: argparse.Namespace) -> tuple[str | None, int | None, str]:
    # index, tenant_id and the connector are all required inputs — no defaults,
    is_retry = _resolve_retry_failed(args) is not None
    index = args.index if args.index is not None else settings.bulk_index
    if not index and not is_retry:
        sys.exit("--index (or BULK_INDEX) is required — there is no default. Pass the "
                 "current real index explicitly, e.g. --index gs-prod-<tenant>.")
    # settings.bulk_limit is already parsed to int | None by pydantic.
    limit = args.limit if args.limit is not None else settings.bulk_limit
    # tenant_id: same CLI-flag-then-settings pattern as --index above.
    tenant_id = args.tenant_id if args.tenant_id is not None else settings.bulk_tenant_id
    if not tenant_id and not is_retry:
        # No default: a run written to the shared graph under a guessed
        # tenant_id="default" is exactly the cross-tenant merge that isolation
        # exists to prevent — two real-tenant runs launched without --tenant-id
        # would both land under "default" and merge into each other's entities.
        sys.exit("--tenant-id (or BULK_TENANT_ID) is required — there is no default. "
                 "Pass an explicit value, even a fixed one for a genuinely "
                 "single-tenant deployment.")
    # A retry resumes an existing run_id's own stores, so its tenant is only used
    # for this process's status logging — "default" is harmless there.
    return index, limit, tenant_id or "default"


def _resolve_retry_failed(args: argparse.Namespace) -> str | None:
    # --retry-failed/BULK_RETRY_FAILED: same CLI-flag-then-settings pattern as every
    # other flag here (settings owns the env read — point 5), so ai-jobs-manager can
    # trigger a targeted retry the same way it triggers a normal run.
    run_id = args.retry_failed if args.retry_failed is not None else settings.bulk_retry_failed
    run_id = (run_id or "").strip()
    return run_id if run_id.lower() not in _FALSY_ENV_TOKENS else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bulk GraphRAG ingestion pass.")
    parser.add_argument("--connector", default=None, choices=known_connector_types(),
                        help="The source this run reads AND the label for its isolated stores "
                        "(the run's connector_type — one concept, one flag). Required; falls back "
                        "to env var BULK_CONNECTOR. Choices come from connectors/registry.py.")
    parser.add_argument("--index", default=None, help="Override the connector's default index; "
                        "falls back to env var BULK_INDEX")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of documents fetched; "
                        "falls back to env var BULK_LIMIT")
    parser.add_argument("--tenant-id", default=None, help="Tenant this bulk run's data belongs to — "
                        "written onto every node; falls back to env var BULK_TENANT_ID. Required.")
    parser.add_argument("--retry-failed", default=None, metavar="RUN_ID",
                        help="Re-drive only the documents ledgered as failed under this run_id, "
                        "instead of a normal full run; falls back to env var BULK_RETRY_FAILED. "
                        "run_id must be copied from the original run's own logs — it's no longer "
                        "reconstructable purely from --index/--tenant-id/--connector.")
    args = parser.parse_args()
    # After argparse so `--help` never fires the Ministral/embeddings reachability
    # probe. Resolved once for this process's whole lifetime — see llm_backend.py.
    resolve_llm_backends()
    connector = _resolve_connector(args)
    spec = get_connector(connector)
    index, limit, tenant_id = _resolve_args(args)
    connector_type = connector
    retry_failed_run_id = _resolve_retry_failed(args)

    # Every log line in this file, not just inside run_bulk_async, must carry
    # tenant/connector -- asyncio.run() below isolates run_bulk_async's own
    # set_log_context() call from this synchronous caller, so entrypoint.py's
    # own log lines (the bulk_run_status checkpoint saves) need it set here too,
    # or they'd show up untagged regardless of what run_bulk_async does.
    set_log_context(tenant_id, connector_type)

    if retry_failed_run_id:
        run_key = f"retry:{retry_failed_run_id}"
        save_checkpoint(_STATUS_COLLECTION, run_key,
                        {"status": "RUNNING", "retry_failed": retry_failed_run_id, "tenant_id": tenant_id})
        try:
            asyncio.run(run_retry_failed_async(retry_failed_run_id, index=index,
                                               tenant_id=tenant_id, spec=spec))
        except Exception:
            save_checkpoint(_STATUS_COLLECTION, run_key,
                            {"status": "FAILED", "retry_failed": retry_failed_run_id, "tenant_id": tenant_id})
            raise
        save_checkpoint(_STATUS_COLLECTION, run_key,
                        {"status": "COMPLETED", "retry_failed": retry_failed_run_id, "tenant_id": tenant_id})
        return 0

    # run_key includes tenant_id/connector_type, not just index, so two separate
    # invocations (different tenant and/or connector) targeting the same index never
    # overwrite each other's status record.
    run_key = f"{index or 'default'}:{tenant_id}:{connector_type}"
    save_checkpoint(_STATUS_COLLECTION, run_key,
                    {"status": "RUNNING", "index": index, "limit": limit, "tenant_id": tenant_id,
                     "connector_type": connector_type})
    try:
        # Single asyncio.run() for the entire application lifetime.
        asyncio.run(run_bulk_async(index=index, limit=limit,
                                   tenant_id=tenant_id, connector_type=connector_type, spec=spec))
    except Exception:
        save_checkpoint(_STATUS_COLLECTION, run_key,
                        {"status": "FAILED", "index": index, "limit": limit, "tenant_id": tenant_id,
                         "connector_type": connector_type})
        raise
    save_checkpoint(_STATUS_COLLECTION, run_key,
                    {"status": "COMPLETED", "index": index, "limit": limit, "tenant_id": tenant_id,
                     "connector_type": connector_type})
    return 0


if __name__ == "__main__":
    sys.exit(main())
