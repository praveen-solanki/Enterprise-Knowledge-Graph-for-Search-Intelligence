"""Ambient tenant_id/connector_type/run_suffix/run_id for every log line, via a
contextvars.ContextVar (asyncio-task-scoped) + a filter that stamps every
LogRecord — no per-call-site changes needed."""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

_UNSET = "-"

_tenant_id: ContextVar[str] = ContextVar("log_ctx_tenant_id", default=_UNSET)
_connector_type: ContextVar[str] = ContextVar("log_ctx_connector_type", default=_UNSET)
_run_suffix: ContextVar[str] = ContextVar("log_ctx_run_suffix", default=_UNSET)
# Distinct from run_suffix: the bulk pipeline's run_suffix is a derived Neo4j
# database/collection suffix, not its run_id. run_id here is the incremental
# Kafka worker's own --run-id/INCREMENTAL_RUN_ID (core/kafka_worker_
# entrypoint.py's _KAFKA_RUN_ID) — a separate field so neither meaning is
# stretched to cover the other.
_run_id: ContextVar[str] = ContextVar("log_ctx_run_id", default=_UNSET)


def set_log_context(tenant_id: str, connector_type: str, run_suffix: Optional[str] = None,
                    run_id: Optional[str] = None) -> None:
    """Binds these for every log line in this asyncio task from here on —
    pair with clear_log_context() in a finally block."""
    _tenant_id.set(tenant_id)
    _connector_type.set(connector_type)
    _run_suffix.set(run_suffix or _UNSET)
    _run_id.set(run_id or _UNSET)


def clear_log_context() -> None:
    """Resets to the unset sentinel ("-")."""
    _tenant_id.set(_UNSET)
    _connector_type.set(_UNSET)
    _run_suffix.set(_UNSET)
    _run_id.set(_UNSET)


class RunContextLogFilter(logging.Filter):
    """Stamps every LogRecord with the current tenant_id/connector_type/run_suffix/run_id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.tenant_id = _tenant_id.get()
        record.connector_type = _connector_type.get()
        record.run_suffix = _run_suffix.get()
        record.run_id = _run_id.get()
        return True
