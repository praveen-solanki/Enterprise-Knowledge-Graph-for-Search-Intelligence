"""Shared test fixtures."""
import pytest

from services.run_context import bind_run_config
from utils.log_context import clear_log_context


@pytest.fixture(autouse=True)
def _reset_log_context():
    """Same leak, different ContextVar (utils/log_context). The bulk entrypoints
    set it and never clear it, which is right for a one-shot process that exits
    right after, but in one long test process it would stamp a later test's log
    records — including the worker test that asserts the context was cleared."""
    clear_log_context()
    yield
    clear_log_context()


@pytest.fixture(autouse=True)
def _reset_run_config():
    """The active-run-config ContextVar (services/run_context.bind_run_config)
    lives in the worker's main-thread context, so a value a test binds — or
    _handle_message binds — would otherwise leak into later tests in the same
    process and make effective_run_config() return the wrong run. Reset it
    around every test."""
    bind_run_config(None)
    yield
    bind_run_config(None)
