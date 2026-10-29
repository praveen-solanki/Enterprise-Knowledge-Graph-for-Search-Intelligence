"""Unit tests for core/entrypoint.py's run-status tracking (AIS-3634), its
--retry-failed wiring (code review must-fix #13), the PR-41 review's collapse of
--connector/--connector-type into one required flag (points 2 + 4), its requirement
that --index and --tenant-id be explicit with no defaults (point 3), that the
BULK_* inputs are read from config.settings rather than os.environ (point 5), and
that every run writes to Neo4j — the --write-to-neo4j flag is gone (point 6).

The BULK_* values now come from `settings` (config/settings.py owns the env read),
so these tests patch `core.entrypoint.settings` with a fake carrying only those
fields — hermetic regardless of the real environment."""
import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.entrypoint import _resolve_args, _resolve_connector, _resolve_retry_failed, main


def _args(index=None, limit=None, tenant_id=None,
          retry_failed=None, connector=None) -> argparse.Namespace:
    return argparse.Namespace(index=index, limit=limit,
                              tenant_id=tenant_id, retry_failed=retry_failed, connector=connector)


def _settings(**kw):
    """A fake `settings` carrying only the BULK_* fields entrypoint reads.
    Unspecified fields default to None, so a test that patches nothing sees no
    env at all."""
    base = dict(bulk_connector=None, bulk_index=None, bulk_limit=None,
                bulk_tenant_id=None, bulk_retry_failed=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _patch_settings(**kw):
    return patch("core.entrypoint.settings", _settings(**kw))


def test_cli_flags_take_precedence_over_settings():
    with _patch_settings(bulk_index="env-index", bulk_limit=99, bulk_tenant_id="env-tenant"):
        index, limit, tenant_id = _resolve_args(
            _args(index="cli-index", limit=5, tenant_id="cli-tenant"))
    assert (index, limit, tenant_id) == ("cli-index", 5, "cli-tenant")


def test_falls_back_to_settings_when_no_cli_flags():
    with _patch_settings(bulk_index="env-index", bulk_limit=99, bulk_tenant_id="env-tenant"):
        index, limit, tenant_id = _resolve_args(_args())
    assert (index, limit, tenant_id) == ("env-index", 99, "env-tenant")


def test_resolves_index_limit_and_tenant():
    with _patch_settings():
        index, limit, tenant_id = _resolve_args(_args(index="my-index", tenant_id="acme-corp"))
    assert (index, limit, tenant_id) == ("my-index", None, "acme-corp")


def test_exits_when_no_index_and_not_retry():
    """index is a required input — there is no default and no local-mode fallback
    anymore (PR-41 review point 3)."""
    with _patch_settings(), pytest.raises(SystemExit):
        _resolve_args(_args(tenant_id="acme-corp"))


def test_exits_when_no_tenant_id_and_not_retry():
    """tenant_id is a required input for every run, not only writes — a run that
    lands under a guessed tenant_id="default" is the cross-tenant merge isolation
    exists to prevent (point 3)."""
    with _patch_settings(), pytest.raises(SystemExit):
        _resolve_args(_args(index="my-index"))


def test_does_not_exit_when_no_index_or_tenant_but_retry_failed_given():
    """--retry-failed derives its index from the failure ledger and resumes an
    already-registered run_id, so it's exempt from both requirements. Its tenant
    falls back to 'default' (used only for this process's status logging)."""
    with _patch_settings():
        index, limit, tenant_id = _resolve_args(_args(retry_failed="my-index:acme-corp"))
    assert (index, limit, tenant_id) == (None, None, "default")


def test_bulk_retry_failed_literal_string_false_is_not_a_retry():
    """Code review catch: bool(settings.bulk_retry_failed) would treat
    BULK_RETRY_FAILED=false as truthy (any non-empty string is truthy in Python),
    wrongly exempting a normal run from the required-index check."""
    with _patch_settings(bulk_retry_failed="false"), pytest.raises(SystemExit):
        _resolve_args(_args())


def test_main_binds_log_context_before_any_checkpoint_save():
    """main()'s own log lines (e.g. the bulk_run_status checkpoint save) must
    carry tenant/connector too -- asyncio.run() isolates run_bulk_async's own
    set_log_context() call from this synchronous caller, so main() has to set
    it itself, before doing anything else. connector_type is the connector."""
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run"), \
         patch("core.entrypoint.run_bulk_async"), \
         patch("core.entrypoint.save_checkpoint"), \
         patch("core.entrypoint.set_log_context") as set_log_context, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index",
                            "--tenant-id", "acme-corp"]):
        main()
    set_log_context.assert_called_once_with("acme-corp", "wikipedia")


def test_main_records_completed_status_on_success():
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run"), \
         patch("core.entrypoint.run_bulk_async"), \
         patch("core.entrypoint.save_checkpoint") as save_checkpoint, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index",
                            "--tenant-id", "acme-corp", "--limit", "1"]):
        main()
    statuses = [call.args[2]["status"] for call in save_checkpoint.call_args_list]
    assert statuses == ["RUNNING", "COMPLETED"]


def test_main_records_failed_status_on_exception():
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run", side_effect=RuntimeError("boom")), \
         patch("core.entrypoint.save_checkpoint") as save_checkpoint, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index",
                            "--tenant-id", "acme-corp"]):
        with pytest.raises(RuntimeError):
            main()
    statuses = [call.args[2]["status"] for call in save_checkpoint.call_args_list]
    assert statuses == ["RUNNING", "FAILED"]


def test_main_passes_tenant_id_and_connector_type_through_to_run_bulk_async():
    """Code review must-fix #4: --tenant-id must actually reach run_bulk_async.
    connector_type is the resolved connector — one concept, one flag."""
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run") as asyncio_run, \
         patch("core.entrypoint.run_bulk_async") as run_bulk_async, \
         patch("core.entrypoint.save_checkpoint"), \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index",
                            "--tenant-id", "acme-corp"]):
        main()
    asyncio_run.assert_called_once()
    run_bulk_async.assert_called_once()
    _, kwargs = run_bulk_async.call_args
    assert kwargs["tenant_id"] == "acme-corp"
    assert kwargs["connector_type"] == "wikipedia"


def test_main_connector_type_is_the_connector_and_shapes_the_run_key():
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run"), \
         patch("core.entrypoint.run_bulk_async") as run_bulk_async, \
         patch("core.entrypoint.save_checkpoint") as save_checkpoint, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "gs-prod-acme-corp",
                            "--tenant-id", "acme-corp"]):
        main()
    _, kwargs = run_bulk_async.call_args
    assert kwargs["connector_type"] == "wikipedia"
    # run_key is "{index}:{tenant_id}:{connector_type}".
    run_keys = {call.args[1] for call in save_checkpoint.call_args_list}
    assert run_keys == {"gs-prod-acme-corp:acme-corp:wikipedia"}


def test_main_exits_when_no_connector_given():
    """--connector is required — no default, no silent fallback to wikipedia (point 4)."""
    with _patch_settings(), \
         patch("sys.argv", ["core.entrypoint", "--index", "my-index", "--tenant-id", "acme-corp"]), \
         pytest.raises(SystemExit):
        main()


def test_main_exits_when_no_tenant_id_given():
    """--tenant-id is required for a normal run (point 3)."""
    with _patch_settings(), \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index"]), \
         pytest.raises(SystemExit):
        main()


def test_resolve_retry_failed_cli_flag_takes_precedence_over_settings():
    with _patch_settings(bulk_retry_failed="env-run-id"):
        assert _resolve_retry_failed(_args(retry_failed="cli-run-id")) == "cli-run-id"


def test_resolve_retry_failed_falls_back_to_settings():
    with _patch_settings(bulk_retry_failed="env-run-id"):
        assert _resolve_retry_failed(_args()) == "env-run-id"


def test_resolve_retry_failed_none_when_neither_set():
    with _patch_settings():
        assert _resolve_retry_failed(_args()) is None


def test_main_retry_failed_calls_run_retry_failed_async_not_run_bulk_async():
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run") as asyncio_run, \
         patch("core.entrypoint.run_retry_failed_async") as run_retry_failed_async, \
         patch("core.entrypoint.run_bulk_async") as run_bulk_async, \
         patch("core.entrypoint.save_checkpoint") as save_checkpoint, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia",
                            "--retry-failed", "my-index:acme-corp"]):
        main()
    asyncio_run.assert_called_once()
    run_retry_failed_async.assert_called_once()
    run_bulk_async.assert_not_called()
    args, kwargs = run_retry_failed_async.call_args
    assert args[0] == "my-index:acme-corp"
    statuses = [call.args[2]["status"] for call in save_checkpoint.call_args_list]
    assert statuses == ["RUNNING", "COMPLETED"]


def test_main_retry_failed_records_failed_status_on_exception():
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run", side_effect=RuntimeError("boom")), \
         patch("core.entrypoint.run_retry_failed_async"), \
         patch("core.entrypoint.save_checkpoint") as save_checkpoint, \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia",
                            "--retry-failed", "my-index:acme-corp"]):
        try:
            main()
        except RuntimeError:
            pass
    statuses = [call.args[2]["status"] for call in save_checkpoint.call_args_list]
    assert statuses == ["RUNNING", "FAILED"]


# --- --connector: one required flag, no silent wikipedia fallback (points 2 + 4) ---

def test_resolve_connector_exits_when_neither_flag_nor_settings_given():
    with _patch_settings(), pytest.raises(SystemExit):
        _resolve_connector(_args())


def test_resolve_connector_falls_back_to_settings():
    with _patch_settings(bulk_connector="wikipedia"):
        assert _resolve_connector(_args()) == "wikipedia"


def test_resolve_connector_cli_flag_beats_settings():
    with _patch_settings(bulk_connector="arxiv"):
        assert _resolve_connector(_args(connector="wikipedia")) == "wikipedia"


def test_resolve_connector_exits_on_an_unregistered_name():
    with _patch_settings(bulk_connector="slack"), pytest.raises(SystemExit):
        _resolve_connector(_args())


def test_main_resolves_llm_backends_once_before_any_run_status_checkpoint():
    """This job's whole lifetime shares one resolved backend (services/llm_backend.py)
    -- must resolve before the first save_checkpoint, not after."""
    call_order = []
    with _patch_settings(), \
         patch("core.entrypoint.asyncio.run"), \
         patch("core.entrypoint.run_bulk_async"), \
         patch("core.entrypoint.resolve_llm_backends",
              side_effect=lambda: call_order.append("resolve_llm_backends")), \
         patch("core.entrypoint.save_checkpoint",
              side_effect=lambda *a, **k: call_order.append("save_checkpoint")), \
         patch("sys.argv", ["core.entrypoint", "--connector", "wikipedia", "--index", "my-index",
                            "--tenant-id", "acme-corp"]):
        main()
    assert call_order[0] == "resolve_llm_backends"
    assert "save_checkpoint" in call_order[1:]


def test_main_help_does_not_resolve_llm_backends():
    with patch("core.entrypoint.resolve_llm_backends") as resolve_backends:
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["core.entrypoint", "--help"]):
                main()
        assert exc.value.code == 0
    resolve_backends.assert_not_called()
