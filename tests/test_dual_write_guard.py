"""Unit tests for services/dual_write_guard.py (manager requirement, full
-traceability project): retry the Qdrant side of a dual write up to 3 times,
and if it still fails, correct Neo4j's state rather than leaving one store
claiming a write that never happened."""
from unittest.mock import MagicMock, patch

from services.dual_write_guard import (
    call_with_retry_or_rollback, delete_vectors, push_vectors_in_batches, sync_acl_for_each,
)


def test_succeeds_on_the_first_try_no_retry_no_rollback():
    fn = MagicMock()
    rollback = MagicMock()
    with patch("services.dual_write_guard.record_call_failure") as record_call:
        result = call_with_retry_or_rollback(fn, step="qdrant_entity_upsert", context="ent-1",
                                             rollback=rollback)
    assert result is True
    fn.assert_called_once()
    rollback.assert_not_called()
    record_call.assert_not_called()


def test_retries_a_transient_failure_then_succeeds():
    fn = MagicMock(side_effect=[RuntimeError("boom"), None])
    rollback = MagicMock()
    with patch("services.dual_write_guard.record_call_failure") as record_call, \
         patch("services.dual_write_guard.time.sleep"):
        result = call_with_retry_or_rollback(fn, step="qdrant_entity_upsert", context="ent-1",
                                             rollback=rollback)
    assert result is True
    assert fn.call_count == 2
    rollback.assert_not_called()
    record_call.assert_not_called()


def test_exhausts_retries_then_logs_and_rolls_back():
    fn = MagicMock(side_effect=RuntimeError("boom"))
    rollback = MagicMock()
    with patch("services.dual_write_guard.record_call_failure") as record_call, \
         patch("services.dual_write_guard.time.sleep"):
        result = call_with_retry_or_rollback(fn, step="qdrant_entity_upsert",
                                             context="entity_ids=['ent-1']", rollback=rollback)
    assert result is False
    assert fn.call_count == 3  # _MAX_RETRIES
    rollback.assert_called_once()
    record_call.assert_called_once_with("qdrant_entity_upsert", "entity_ids=['ent-1']",
                                        "transient", "boom", attempts=3)


def test_non_retryable_error_skips_straight_to_rollback_no_retry():
    from botocore.exceptions import ClientError
    permanent_exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")
    fn = MagicMock(side_effect=permanent_exc)
    rollback = MagicMock()
    with patch("services.dual_write_guard.record_call_failure") as record_call:
        result = call_with_retry_or_rollback(fn, step="qdrant_entity_upsert", context="ent-1",
                                             rollback=rollback)
    assert result is False
    fn.assert_called_once()  # never retried -- permanent, retrying would fail identically
    rollback.assert_called_once()
    record_call.assert_called_once()
    assert record_call.call_args.kwargs["attempts"] == 1


def test_no_rollback_given_is_fine():
    fn = MagicMock(side_effect=RuntimeError("boom"))
    with patch("services.dual_write_guard.record_call_failure"), \
         patch("services.dual_write_guard.time.sleep"):
        result = call_with_retry_or_rollback(fn, step="qdrant_acl_sync", context="ent-1")
    assert result is False


def test_a_failing_rollback_is_logged_but_never_raises():
    fn = MagicMock(side_effect=RuntimeError("boom"))
    rollback = MagicMock(side_effect=RuntimeError("rollback also broke"))
    with patch("services.dual_write_guard.record_call_failure"), \
         patch("services.dual_write_guard.time.sleep"):
        result = call_with_retry_or_rollback(fn, step="qdrant_entity_upsert", context="ent-1",
                                             rollback=rollback)
    assert result is False  # must not raise -- caller gets a clean False, not an exception
    rollback.assert_called_once()


# --- push_vectors_in_batches (shared by write_entities/write_chunks/upsert_entity_incremental) ---

def test_push_vectors_in_batches_sends_everything_in_one_call_when_under_the_limit():
    store = MagicMock()
    rows = [{"entity_id": f"e{i}", "embedding": [0.1]} for i in range(5)]
    push_vectors_in_batches(store, rows, step="qdrant_entity_upsert_batch")
    store.upsert_entities_batch.assert_called_once_with(rows)


def test_push_vectors_in_batches_splits_at_256():
    store = MagicMock()
    rows = [{"entity_id": f"e{i}", "embedding": [0.1]} for i in range(300)]
    push_vectors_in_batches(store, rows, step="qdrant_entity_upsert_batch")
    assert store.upsert_entities_batch.call_count == 2
    assert len(store.upsert_entities_batch.call_args_list[0].args[0]) == 256
    assert len(store.upsert_entities_batch.call_args_list[1].args[0]) == 44


def test_push_vectors_in_batches_uses_custom_id_field_for_chunks():
    store = MagicMock()
    rows = [{"entity_id": "c1", "embedding": [0.1]}]  # chunk rows still key on "entity_id" (see write_chunks)
    with patch("services.dual_write_guard.record_call_failure") as record_call:
        store.upsert_entities_batch.side_effect = RuntimeError("boom")
        with patch("services.dual_write_guard.time.sleep"):
            push_vectors_in_batches(store, rows, step="qdrant_chunk_upsert_batch", id_field="entity_id")
    assert record_call.call_args.args[0] == "qdrant_chunk_upsert_batch"
    assert "entity_ids=['c1']" in record_call.call_args.args[1]


def test_push_vectors_in_batches_calls_rollback_with_the_failed_batchs_own_ids():
    store = MagicMock(upsert_entities_batch=MagicMock(side_effect=RuntimeError("boom")))
    rollback = MagicMock()
    with patch("services.dual_write_guard.record_call_failure"), \
         patch("services.dual_write_guard.time.sleep"):
        push_vectors_in_batches(store, [{"entity_id": "e1", "embedding": [0.1]}],
                                step="qdrant_entity_upsert_batch", rollback=rollback)
    rollback.assert_called_once_with(["e1"])


def test_push_vectors_in_batches_empty_list_calls_nothing():
    store = MagicMock()
    push_vectors_in_batches(store, [], step="qdrant_entity_upsert_batch")
    store.upsert_entities_batch.assert_not_called()


# --- delete_vectors (shared by every orphan/retraction vector-delete call site) ---

def test_delete_vectors_calls_delete_entities_with_the_given_ids():
    store = MagicMock()
    delete_vectors(store, ["e1", "e2"], step="qdrant_entity_delete")
    store.delete_entities.assert_called_once_with(["e1", "e2"])


def test_delete_vectors_is_a_no_op_for_an_empty_list():
    store = MagicMock()
    delete_vectors(store, [], step="qdrant_entity_delete")
    store.delete_entities.assert_not_called()


def test_delete_vectors_logs_on_exhausted_retries_with_no_rollback():
    store = MagicMock(delete_entities=MagicMock(side_effect=RuntimeError("boom")))
    with patch("services.dual_write_guard.record_call_failure") as record_call, \
         patch("services.dual_write_guard.time.sleep"):
        result = delete_vectors(store, ["e1"], step="qdrant_entity_delete", id_label="entity_ids")
    assert result is None
    record_call.assert_called_once()
    assert record_call.call_args.args[0] == "qdrant_entity_delete"
    assert "entity_ids=['e1']" in record_call.call_args.args[1]


# --- sync_acl_for_each (shared by every entity/chunk ACL-sync loop) ---

def test_sync_acl_for_each_calls_set_acl_per_pair():
    store = MagicMock()
    sync_acl_for_each(store, [("e1", ["public"]), ("e2", ["private"])], step="qdrant_acl_sync")
    assert store.set_acl.call_count == 2
    store.set_acl.assert_any_call("e1", ["public"])
    store.set_acl.assert_any_call("e2", ["private"])


def test_sync_acl_for_each_empty_list_calls_nothing():
    store = MagicMock()
    sync_acl_for_each(store, [], step="qdrant_acl_sync")
    store.set_acl.assert_not_called()


def test_sync_acl_for_each_logs_per_failing_id_with_no_rollback():
    store = MagicMock(set_acl=MagicMock(side_effect=RuntimeError("boom")))
    with patch("services.dual_write_guard.record_call_failure") as record_call, \
         patch("services.dual_write_guard.time.sleep"):
        sync_acl_for_each(store, [("e1", ["public"])], step="qdrant_chunk_acl_sync", id_label="chunk_id")
    assert record_call.call_args.args[0] == "qdrant_chunk_acl_sync"
    assert record_call.call_args.args[1] == "chunk_id='e1'"
