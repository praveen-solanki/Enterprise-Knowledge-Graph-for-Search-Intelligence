"""Emoji-tagged, single-line step logging for the incremental Kafka flow.

Keeps the operator-facing log for one event readable: a clean line per phase
boundary — 📥 received, 🗄️ databases chosen, ▶️ step started, … progress,
✅ step done, ⏭️ step skipped, 🏁 event complete — each with structured
`key=value` fields.

tenant_id / connector_type / run are already stamped on every line by
utils/log_context.py's filter, so they are never repeated here. Steps are
indented two spaces so the phase hierarchy reads at a glance after the log
prefix. Everything logs at INFO; nothing here raises.
"""
from __future__ import annotations

import logging
from typing import Optional


def _fields(pairs: dict) -> str:
    parts = [f"{k}={v}" for k, v in pairs.items() if v is not None and v != ""]
    return (" | " + " | ".join(parts)) if parts else ""


def mongo_host(uri: str) -> str:
    """Host[:port] of a Mongo URI with any `user:pass@` credentials stripped —
    safe to log."""
    if not uri:
        return "<unset>"
    tail = uri.split("://", 1)[-1].split("@", 1)[-1]
    return tail.split("/", 1)[0].split("?", 1)[0]


def event_received(logger: logging.Logger, **fields) -> None:
    logger.info("📥 Event received%s", _fields(fields))


def databases_selected(
    logger: logging.Logger,
    *,
    env: str,
    mongo_uri: str,
    mongo_db: str,
    neo4j_http_url: str,
    neo4j_database: str,
    qdrant_url: str,
    qdrant_entities: str,
    qdrant_chunks: str,
    run_suffix: Optional[str] = None,
) -> None:
    """One line naming every datastore this event will read/write — the whole
    point being that a glance answers 'which DBs did this event actually hit?'."""
    logger.info(
        "🗄️  Databases selected%s",
        _fields({
            "env": env,
            "run_suffix": run_suffix,
            "mongo": f"{mongo_host(mongo_uri)}/{mongo_db}",
            "neo4j": f"{neo4j_http_url} db={neo4j_database}",
            "qdrant": f"{qdrant_url} entities={qdrant_entities} chunks={qdrant_chunks}",
        }),
    )


def step_start(logger: logging.Logger, name: str, **fields) -> None:
    logger.info("  ▶️  %s — started%s", name, _fields(fields))


def step_progress(logger: logging.Logger, name: str, **fields) -> None:
    """Mid-step heartbeat so long phases (e.g. entity resolution) aren't silent."""
    logger.info("  …  %s — progress%s", name, _fields(fields))


def step_done(logger: logging.Logger, name: str, **fields) -> None:
    logger.info("  ✅ %s — done%s", name, _fields(fields))


def step_skipped(logger: logging.Logger, name: str, *, reason: Optional[str] = None, **fields) -> None:
    logger.info("  ⏭️  %s — skipped%s", name, _fields({"reason": reason, **fields}))


def event_done(logger: logging.Logger, **fields) -> None:
    logger.info("🏁 Event complete%s", _fields(fields))
