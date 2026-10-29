"""Unit tests for repository/mongo_repository.py's increment_retry_count
(code review must-fix #5 — Kafka DLQ retry-count ledger), its bulk-job
failure ledger (code review must-fix #13), and increment_rolling_window_count
(experiment-parity review #9 — atomic mass-retraction counter)."""
from unittest.mock import MagicMock, patch

from repository.mongo_repository import (
    clear_failure,
    increment_retry_count,
    increment_rolling_window_count,
    list_failures,
    record_call_failure,
    record_failure,
)


def test_increment_retry_count_returns_the_post_increment_value():
    fake_db = MagicMock()
    fake_db["kafka_retry_counts"].find_one_and_update.return_value = {"_key": "evt-1", "count": 3}
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        result = increment_retry_count("evt-1")
    assert result == 3


def test_increment_retry_count_upserts_atomically_with_inc():
    fake_db = MagicMock()
    fake_db["kafka_retry_counts"].find_one_and_update.return_value = {"_key": "evt-1", "count": 1}
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        increment_retry_count("evt-1")
    args, kwargs = fake_db["kafka_retry_counts"].find_one_and_update.call_args
    assert args[0] == {"_key": "evt-1"}
    assert args[1] == {"$inc": {"count": 1}}
    assert kwargs["upsert"] is True


def test_increment_rolling_window_count_returns_the_post_increment_value():
    fake_db = MagicMock()
    fake_db["mass_retraction_window"].find_one_and_update.return_value = {"_key": "tenant-1", "count": 4}
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        result = increment_rolling_window_count("mass_retraction_window", "tenant-1", 3600.0, 1000.0)
    assert result == 4


def test_increment_rolling_window_count_upserts_atomically_via_pipeline_update():
    """Must be a single find_one_and_update call given a PIPELINE (a list of
    stages, not a plain $inc dict) — a separate load-then-save round trip
    would let two concurrent callers for the same key both read the same
    starting count and both write back the same incremented value."""
    fake_db = MagicMock()
    fake_db["mass_retraction_window"].find_one_and_update.return_value = {"_key": "tenant-1", "count": 1}
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        increment_rolling_window_count("mass_retraction_window", "tenant-1", 3600.0, 1000.0)
    args, kwargs = fake_db["mass_retraction_window"].find_one_and_update.call_args
    assert args[0] == {"_key": "tenant-1"}
    assert isinstance(args[1], list)  # aggregation-pipeline update, not a plain update dict
    assert kwargs["upsert"] is True


def test_record_failure_upserts_keyed_by_run_item_step():
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        record_failure("run-1", "doc-1", "extraction", "transient", "boom", index="idx-1")
    args, kwargs = fake_db["ingestion_failures"].update_one.call_args
    assert args[0] == {"_key": "run-1:doc-1:extraction"}
    assert args[1]["$set"] == {
        "_key": "run-1:doc-1:extraction", "run_id": "run-1", "item_id": "doc-1",
        "step": "extraction", "error_class": "transient", "message": "boom", "index": "idx-1",
    }
    assert kwargs["upsert"] is True


def test_record_failure_same_key_overwrites_not_duplicates():
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        record_failure("run-1", "doc-1", "extraction", "transient", "first error")
        record_failure("run-1", "doc-1", "extraction", "permanent", "second error")
    assert fake_db["ingestion_failures"].update_one.call_count == 2
    second_args, _ = fake_db["ingestion_failures"].update_one.call_args
    assert second_args[0] == {"_key": "run-1:doc-1:extraction"}
    assert second_args[1]["$set"]["message"] == "second error"


def test_list_failures_returns_all_for_run_id():
    fake_db = MagicMock()
    fake_db["ingestion_failures"].find.return_value = [
        {"_key": "run-1:doc-1:extraction", "run_id": "run-1", "item_id": "doc-1", "step": "extraction"},
        {"_key": "run-1:doc-2:doc_canonical", "run_id": "run-1", "item_id": "doc-2", "step": "doc_canonical"},
    ]
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        result = list_failures("run-1")
    fake_db["ingestion_failures"].find.assert_called_once_with({"run_id": "run-1"})
    assert len(result) == 2


def test_clear_failure_deletes_by_composite_key():
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        clear_failure("run-1", "doc-1", "extraction")
    fake_db["ingestion_failures"].delete_one.assert_called_once_with(
        {"_key": "run-1:doc-1:extraction"}
    )


def test_save_checkpoints_batch_noops_on_empty_items():
    from repository.mongo_repository import save_checkpoints_batch
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        save_checkpoints_batch("edge_pool_edges", [])
    fake_db["edge_pool_edges"].bulk_write.assert_not_called()


def test_delete_checkpoint_deletes_by_key():
    from repository.mongo_repository import delete_checkpoint
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        delete_checkpoint("edge_pool_map", "base")
    fake_db["edge_pool_map"].delete_one.assert_called_once_with({"_key": "base"})


def test_save_checkpoints_batch_uses_unordered_bulk_write_in_chunks():
    from pymongo import UpdateOne
    from repository.mongo_repository import save_checkpoints_batch

    fake_coll = MagicMock()
    fake_db = MagicMock()
    fake_db.__getitem__.return_value = fake_coll
    items = [(f"k{i}", {"v": i}) for i in range(5)]
    with patch("repository.mongo_repository._get_db", return_value=fake_db), \
         patch("repository.mongo_repository.settings") as mock_settings:
        mock_settings.mongo_checkpoint_bulk_size = 2
        save_checkpoints_batch("edge_pool_edges", items)

    assert fake_coll.bulk_write.call_count == 3  # 2+2+1
    first_ops, first_kwargs = fake_coll.bulk_write.call_args_list[0]
    assert first_kwargs["ordered"] is False
    assert len(first_ops[0]) == 2
    assert isinstance(first_ops[0][0], UpdateOne)
    last_ops, _ = fake_coll.bulk_write.call_args_list[-1]
    assert len(last_ops[0]) == 1


# --- record_call_failure (full-traceability requirement: every retry-exhausted
# operation, not just a whole document/step, gets a durable record) ---

def test_record_call_failure_inserts_into_its_own_collection():
    """A separate append-only collection from ingestion_failures -- a single
    call site has no natural (run_id, item_id) key to upsert against."""
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        record_call_failure("cluster_merge_confirm", "members=[A, B]", "transient", "boom", attempts=3)
    fake_db["ingestion_failures"].update_one.assert_not_called()
    fake_db["call_failures"].insert_one.assert_called_once()
    doc = fake_db["call_failures"].insert_one.call_args[0][0]
    assert doc["step"] == "cluster_merge_confirm"
    assert doc["context"] == "members=[A, B]"
    assert doc["error_class"] == "transient"
    assert doc["message"] == "boom"
    assert doc["attempts"] == 3
    assert "logged_at" in doc


def test_record_call_failure_never_deduplicates_repeat_failures():
    """Unlike record_failure's upsert-by-key ledger, every call gets its own
    row -- so the audit trail shows how many times a given call site failed,
    not just the most recent occurrence."""
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        record_call_failure("cluster_merge_confirm", "members=[A, B]", "transient", "first")
        record_call_failure("cluster_merge_confirm", "members=[A, B]", "transient", "second")
    assert fake_db["call_failures"].insert_one.call_count == 2


def test_record_call_failure_attempts_defaults_to_none():
    fake_db = MagicMock()
    with patch("repository.mongo_repository._get_db", return_value=fake_db):
        record_call_failure("rel_type_classify", "raw=ADDED_A_FEATURE", "permanent", "bad request")
    doc = fake_db["call_failures"].insert_one.call_args[0][0]
    assert doc["attempts"] is None
