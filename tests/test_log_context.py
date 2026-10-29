"""Unit tests for utils/log_context.py (ambient tenant_id/
connector_type/run_suffix stamped onto every log line)."""
import logging

import pytest

from utils.log_context import RunContextLogFilter, clear_log_context, set_log_context


@pytest.fixture(autouse=True)
def _reset_context():
    clear_log_context()
    yield
    clear_log_context()


def _record() -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)


def test_filter_stamps_defaults_when_context_never_set():
    record = _record()
    assert RunContextLogFilter().filter(record) is True
    assert record.tenant_id == "-"
    assert record.connector_type == "-"
    assert record.run_suffix == "-"


def test_filter_stamps_the_bound_context():
    set_log_context("acme", "confluence", "ab12")
    record = _record()
    RunContextLogFilter().filter(record)
    assert (record.tenant_id, record.connector_type, record.run_suffix) == ("acme", "confluence", "ab12")


def test_set_log_context_without_run_suffix_defaults_to_unset():
    set_log_context("acme", "confluence")
    record = _record()
    RunContextLogFilter().filter(record)
    assert record.run_suffix == "-"


def test_clear_log_context_resets_to_defaults():
    set_log_context("acme", "confluence", "ab12")
    clear_log_context()
    record = _record()
    RunContextLogFilter().filter(record)
    assert (record.tenant_id, record.connector_type, record.run_suffix) == ("-", "-", "-")
