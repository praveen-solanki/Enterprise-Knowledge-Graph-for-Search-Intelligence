"""Unit tests for services/run_metrics.py (experiment-parity review #21)."""
from services import run_metrics


def setup_function():
    run_metrics.reset()


def test_record_increments_a_counter():
    run_metrics.record("extraction_calls")
    run_metrics.record("extraction_calls")
    assert run_metrics.snapshot()["extraction_calls"] == 2


def test_record_with_n_increments_by_that_amount():
    run_metrics.record("embedding_calls", 5)
    assert run_metrics.snapshot()["embedding_calls"] == 5


def test_snapshot_returns_a_copy_not_the_live_dict():
    run_metrics.record("merge_calls")
    snap = run_metrics.snapshot()
    snap["merge_calls"] = 999
    assert run_metrics.snapshot()["merge_calls"] == 1


def test_reset_clears_all_counters():
    run_metrics.record("extraction_calls")
    run_metrics.reset()
    assert run_metrics.snapshot() == {}


def test_distinct_kinds_tracked_independently():
    run_metrics.record("extraction_calls")
    run_metrics.record("gleaning_calls")
    run_metrics.record("extraction_calls")
    assert run_metrics.snapshot() == {"extraction_calls": 2, "gleaning_calls": 1}
