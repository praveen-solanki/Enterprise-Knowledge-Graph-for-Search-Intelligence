"""Logging bootstrap built on the org's shared python-common-logger.

When the package is installed (the deploy image), this installs its JSON
formatter + ContextFilter on the root handler so every logger.info()/
.warning() call anywhere in the codebase emits the same structured JSON shape
every other service in the org does. When it's absent (local dev — it's a
private package, not in requirements.txt, see the comment there), this falls
back to utils/logging_setup.py's existing plain-text stdout config unchanged,
so local runs behave exactly as they did before this file existed.

There is no FastAPI/request-context piece here (unlike ai-graph-retrieval,
which is a web service) — this is a batch job (core/entrypoint.py) and a
Kafka worker (core/kafka_worker_entrypoint.py), neither of which has an
inbound HTTP request to bind a correlation id to.

tenant_id/connector_type/run_suffix are stamped by utils/log_context.py's
RunContextLogFilter, installed alongside python-common-logger's own
ContextFilter — unverified whether create_json_formatter's output actually
surfaces them (that package isn't available in this repo to check).
"""
from __future__ import annotations

import logging
import sys

from utils.log_context import RunContextLogFilter


def setup_logging(log_level: str = "INFO") -> None:
    """Install structured JSON logging via python-common-logger when present;
    otherwise fall back to utils/logging_setup.py's plain-text config."""
    try:
        from python_common_logger import Logger
        from python_common_logger.src.logger import ContextFilter
        from python_common_logger.src.utils.logger import create_json_formatter
    except ImportError:
        from utils.logging_setup import setup_logging as _fallback_setup_logging

        _fallback_setup_logging(log_level)
        return

    Logger.initialise_logger("logger", "ai-graph-ingestion", level=log_level.upper())
    root = logging.getLogger()
    root.setLevel(log_level.upper())
    formatter = create_json_formatter(service_name="ai-graph-ingestion")
    context_filter = ContextFilter()
    run_context_filter = RunContextLogFilter()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stdout))
    for handler in root.handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        handler.addFilter(run_context_filter)
        handler.stream = sys.stdout
    # Quiet noisy third-party libraries so the log stream stays readable —
    # same list utils/logging_setup.py already applied.
    for noisy in ("httpx", "httpcore", "openai", "elastic_transport", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
