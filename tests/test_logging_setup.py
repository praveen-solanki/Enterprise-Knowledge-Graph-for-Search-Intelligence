"""The local/fallback logging path must attach RunContextLogFilter so
tenant_id/connector_type/run_suffix are available to its format string."""
import logging

from utils.log_context import RunContextLogFilter
from utils.logging_setup import setup_logging


def test_setup_logging_attaches_run_context_filter_to_its_handler():
    setup_logging()
    try:
        root = logging.getLogger()
        assert any(isinstance(f, RunContextLogFilter) for h in root.handlers for f in h.filters)
    finally:
        logging.basicConfig(force=True)  # restore a plain default config for later tests
