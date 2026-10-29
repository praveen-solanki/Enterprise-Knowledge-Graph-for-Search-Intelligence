"""
Logging configuration for the pipeline. Writes to stdout so container runtimes
(Docker, Kubernetes) capture logs automatically via their standard log drivers.

Previously this wrote to a timestamped file under logs/ inside the container.
Files written inside a container are ephemeral — `docker run --rm` discards
them entirely, and even without --rm they are not accessible to any log
aggregator (Datadog, CloudWatch, etc.) without a sidecar. Stdout is the
conventional and correct target for containerized workloads (M10).

A StreamHandler on the root logger means every logger.info() / logger.warning()
call anywhere in the codebase goes to stdout automatically.

tenant=/connector=/run= is stamped onto every line via utils/log_context.py's
RunContextLogFilter, not passed by each call site.
"""
from __future__ import annotations

import logging
import sys

from utils.log_context import RunContextLogFilter


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger to write structured lines to stdout.

    Called once at the start of run_bulk_async(). Subsequent calls are
    idempotent — basicConfig with force=True replaces any existing handlers.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RunContextLogFilter())
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | tenant=%(tenant_id)s connector=%(connector_type)s "
               "run=%(run_suffix)s run_id=%(run_id)s | %(name)s | %(message)s",
        handlers=[handler],
        force=True,
    )
    # Quiet noisy third-party libraries so the log stream stays readable.
    for noisy in ("httpx", "httpcore", "openai", "elastic_transport", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
