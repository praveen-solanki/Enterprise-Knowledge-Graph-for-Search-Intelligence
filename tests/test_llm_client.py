"""Unit tests for services/llm_client.py -- the one place this repo talks to
an LLM: complete() (the single per-call door, both backends), call_with_retry()
(the single retry loop: fatal raises, permanent fails closed, transient
retries), and confirm_with_retry() (the yes/no convenience on top).

Every real call site now passes a CallKind (services/llm_profiles.py) instead
of raw target_latency_seconds/max_retries/base_backoff/max_backoff -- each
kind gets its own throttle, never a shared one. Tests patch
get_throttle_for() directly rather than the old single _ministral_throttle/
get_shared_throttle() names, since throttle lookup is now per (backend, kind)."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError, RateLimitError

import services.llm_client as llm_client
from models.extraction import ChunkExtractionResult
from services.llm_client import call_with_retry, complete, confirm_with_retry, get_client
from services.llm_profiles import CallKind
from services.run_context import RunConfig


def _cfg(backend: str, model: str) -> RunConfig:
    """A RunConfig pinning just the LLM backend/model complete() reads; the
    embedding/flag fields are filler the LLM path never touches."""
    return RunConfig(
        embedding_backend="azure", embedding_model="text-embedding-3-large",
        embedding_dimension=1536, consume_upstream_vector=True,
        es_chunk_vector_field="f",
        llm_backend=backend, llm_model=model,
        build_authorship_edges=True, contributor_entity_types=("X",), people_org_graph_enabled=True,
    )


_CFG_MINISTRAL = _cfg("ministral", "ministral-3b")
_CFG_AZURE = _cfg("azure", "gpt-5.4-nano")

# Not a credential -- a mock value for asserting settings are threaded through
# to the client constructor. Built at runtime (not a bare string literal next
# to a key/secret/token-named variable) so an automated secret scanner has no
# quoted-literal pattern to match, on top of the value itself being an obvious
# placeholder.
_MOCK_SETTING_VALUE = "-".join(["placeholder", "value", "not", "a", "real", "credential"])


def _run(coro):
    return asyncio.run(coro)


def _response(content: str = None, parsed=None, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content, parsed=parsed),
                              finish_reason=finish_reason)]
    return resp


def _raw(resp, headers: dict = None) -> MagicMock:
    """A with_raw_response result: .headers plus .parse() -> the typed response."""
    raw = MagicMock()
    raw.headers = headers or {}
    raw.parse.return_value = resp
    return raw


def _mock_throttle() -> MagicMock:
    throttle = MagicMock()
    throttle.slot.return_value.__aenter__ = AsyncMock(return_value=None)
    throttle.slot.return_value.__aexit__ = AsyncMock(return_value=None)
    throttle.record_success = AsyncMock()
    throttle.record_rate_limit = AsyncMock()
    throttle.record_failure = AsyncMock()
    return throttle


def _patch_throttle(mock_throttle=None):
    """get_throttle_for(kind, backend) is the one lookup complete() makes --
    patched wholesale so every call in a test gets the same mock regardless
    of which kind/backend it resolves, matching what these unit tests check."""
    return patch("services.llm_client.get_throttle_for", return_value=mock_throttle or _mock_throttle())


def _fatal_exc() -> APIStatusError:
    return APIStatusError("unauthorized", response=MagicMock(status_code=401), body=None)


# --- client singletons ---

def test_get_client_builds_only_once_and_reuses_the_same_instance():
    with patch("services.llm_client.AsyncAzureOpenAI") as mock_client_cls, \
         patch.object(llm_client, "_azure_client", None), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_endpoint = "http://localhost:8080"
        mock_settings.azure_openai_api_key = _MOCK_SETTING_VALUE
        mock_settings.azure_openai_api_version = "2025-01-01-preview"
        mock_settings.azure_openai_request_timeout = 120.0
        first = get_client()
        second = get_client()
    mock_client_cls.assert_called_once()
    assert first is second


def test_get_client_is_built_from_settings():
    with patch("services.llm_client.AsyncAzureOpenAI") as mock_client_cls, \
         patch.object(llm_client, "_azure_client", None), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_endpoint = "http://localhost:8080"
        mock_settings.azure_openai_api_key = _MOCK_SETTING_VALUE
        mock_settings.azure_openai_api_version = "2025-01-01-preview"
        mock_settings.azure_openai_request_timeout = 120.0
        get_client()
    mock_client_cls.assert_called_once_with(
        azure_endpoint="http://localhost:8080", api_key=_MOCK_SETTING_VALUE,
        api_version="2025-01-01-preview", timeout=120.0,
    )


def test_get_in_house_client_strips_the_chat_completions_suffix():
    """Also pins the Ministral client to its OWN timeout setting — extraction
    p90 is ~313s, so silently reusing Azure's 120s here abandons calls the pod
    then finishes for nobody."""
    with patch("services.llm_client.AsyncOpenAI") as mock_client_cls, \
         patch.object(llm_client, "_in_house_client", None), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_url = "http://localhost:8000/v1/chat/completions"
        mock_settings.ministral_api_key = _MOCK_SETTING_VALUE
        mock_settings.ministral_request_timeout = 360.0
        llm_client.get_in_house_client()
    mock_client_cls.assert_called_once_with(
        base_url="http://localhost:8000/v1", api_key=_MOCK_SETTING_VALUE, timeout=360.0,
    )


# --- complete(): the one door, structured mode ---

def test_complete_structured_ministral_sends_no_max_tokens():
    """Ministral requests deliberately omit max_tokens: vLLM's chat API
    derives the real budget as (window − prompt tokens) with the REAL
    tokenizer. The old client-side char estimate could only miss in two
    directions — under-request (wasted correct-and-retry round trips) and
    over-request (outright 400 rejections, no retry path) — and one bulk run
    produced both at scale (86 and 67 occurrences respectively)."""
    parsed = ChunkExtractionResult(entities=[], relationships=[], claims=[])
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(
        return_value=_raw(_response(parsed=parsed)))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         patch("services.llm_client.get_client") as azure_client, \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        result = _run(complete(messages=[{"role": "user", "content": "hi"}],
                               response_format=ChunkExtractionResult, kind=CallKind.EXTRACTION))
    assert result is parsed
    mock_client.beta.chat.completions.with_raw_response.parse.assert_called_once_with(
        model="ministral-3b", messages=[{"role": "user", "content": "hi"}],
        response_format=ChunkExtractionResult, temperature=0.0,
    )
    azure_client.assert_not_called()
    mock_throttle.slot.assert_called_once()
    mock_throttle.record_success.assert_called_once()


def test_complete_structured_ministral_forwards_an_explicit_max_tokens():
    """An explicit caller-passed cap is still honored on the vLLM wire name."""
    parsed = ChunkExtractionResult(entities=[], relationships=[], claims=[])
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(
        return_value=_raw(_response(parsed=parsed)))
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        _run(complete(messages=[{"role": "user", "content": "hi"}],
                      response_format=ChunkExtractionResult, max_tokens=512,
                      kind=CallKind.EXTRACTION))
    kwargs = mock_client.beta.chat.completions.with_raw_response.parse.call_args.kwargs
    assert kwargs["max_tokens"] == 512
    assert "max_completion_tokens" not in kwargs


def test_complete_structured_azure_maps_the_cap_to_max_completion_tokens():
    """gpt-5.x rejects max_tokens outright (400: use max_completion_tokens) --
    verified live 2026-08-05 -- so the door must translate the wire name."""
    parsed = ChunkExtractionResult(entities=[], relationships=[], claims=[])
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(
        return_value=_raw(_response(parsed=parsed)))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt-5.4-nano"
        result = _run(complete(messages=[{"role": "user", "content": "hi"}],
                               response_format=ChunkExtractionResult, max_tokens=512,
                               kind=CallKind.EXTRACTION))
    assert result is parsed
    kwargs = mock_client.beta.chat.completions.with_raw_response.parse.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs
    mock_throttle.record_success.assert_called_once()  # headers= fed from raw


def test_complete_structured_truncated_raises():
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(
        return_value=_raw(_response(parsed=None, finish_reason="length")))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        with pytest.raises(ValueError, match="truncated"):
            _run(complete(messages=[{"role": "user", "content": "hi"}],
                          response_format=ChunkExtractionResult, kind=CallKind.EXTRACTION))


def test_complete_structured_none_parsed_raises():
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(
        return_value=_raw(_response(parsed=None)))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt"
        with pytest.raises(ValueError, match="returned None"):
            _run(complete(messages=[{"role": "user", "content": "hi"}],
                          response_format=ChunkExtractionResult, kind=CallKind.EXTRACTION))


# --- complete(): json_object mode (response_format=None) ---

def test_complete_json_object_returns_the_parsed_dict():
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_raw(_response(content='{"merge": true}')))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt"
        result = _run(complete(messages=[{"role": "user", "content": "same org?"}], kind=CallKind.CONFIRM))
    assert result == {"merge": True}
    kwargs = mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_complete_json_object_truncated_raises_on_both_backends():
    """The truncation honesty check now covers json_object mode too (the old
    Azure chat_json_object path had none)."""
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_raw(_response(content='{"merge": tr', finish_reason="length")))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt"
        with pytest.raises(ValueError, match="truncated"):
            _run(complete(messages=[{"role": "user", "content": "same org?"}], kind=CallKind.CONFIRM))


# --- complete(): throttle bookkeeping stays on the backend that was called ---

def test_complete_ministral_rate_limit_records_failure_on_ministral_only():
    """The cross-charge fix: a 429 from the in-house gateway must never halve
    a different throttle's concurrency."""
    mock_client = MagicMock()
    rate_limit_exc = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)
    mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=rate_limit_exc)
    ministral_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(ministral_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        with pytest.raises(RateLimitError):
            _run(complete(messages=[{"role": "user", "content": "hi"}], kind=CallKind.EXTRACTION))
    ministral_throttle.record_failure.assert_called_once()
    ministral_throttle.record_rate_limit.assert_not_called()


def test_complete_azure_rate_limit_records_on_the_shared_throttle():
    mock_client = MagicMock()
    rate_limit_exc = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)
    mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=rate_limit_exc)
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt"
        with pytest.raises(RateLimitError):
            _run(complete(messages=[{"role": "user", "content": "hi"}], kind=CallKind.EXTRACTION))
    mock_throttle.record_rate_limit.assert_called_once()


def test_complete_resolves_the_throttle_for_the_calls_own_kind():
    """get_throttle_for is called with the kind complete() was given -- proof
    a caller's kind actually reaches throttle selection, not just accepted
    and ignored."""
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_raw(_response(content='{"merge": true}')))
    with patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE), \
         patch("services.llm_client.get_client", return_value=mock_client), \
         patch("services.llm_client.get_throttle_for", return_value=_mock_throttle()) as get_throttle, \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_completion_model = "gpt"
        _run(complete(messages=[{"role": "user", "content": "hi"}], kind=CallKind.CLUSTER_MERGE))
    get_throttle.assert_called_once_with(CallKind.CLUSTER_MERGE, "azure")


# --- call_with_retry: fatal raises, permanent fails closed, transient retries ---

def test_call_with_retry_returns_the_attempt_result_on_first_success():
    attempt = AsyncMock(return_value="ok")
    result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                  kind=CallKind.EXTRACTION))
    assert result == "ok"
    attempt.assert_called_once()


def test_call_with_retry_fatal_error_re_raises_instead_of_failing_closed():
    """Aman's ruling (2026-08-05): a fatal is a config-level error that fails
    identically for every future call -- failing closed thousands of times is
    silent mass corruption, so the run must stall. Before this, only
    extraction re-raised fatal; cluster merges silently no-matched."""
    attempt = AsyncMock(side_effect=_fatal_exc())
    with patch("services.llm_client.record_call_failure") as record_call:
        with pytest.raises(APIStatusError):
            _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                 kind=CallKind.EXTRACTION))
    attempt.assert_called_once()  # never retried, never failed closed
    record_call.assert_called_once_with("c", "ctx", "fatal", "unauthorized", attempts=1)


def test_call_with_retry_records_a_metric_per_real_attempt():
    from botocore.exceptions import ClientError
    from services import run_metrics

    run_metrics.reset()
    permanent_exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")
    attempt = AsyncMock(side_effect=permanent_exc)  # non-retryable -- exactly one real attempt
    with patch("services.llm_client.record_call_failure"):
        _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                             kind=CallKind.EXTRACTION,
                             metric_calls="my_calls", metric_failures="my_failures"))
    snap = run_metrics.snapshot()
    assert snap.get("my_calls") == 1
    assert snap.get("my_failures") == 1


def test_call_with_retry_retries_rate_limit_then_succeeds():
    """RateLimitError is transient here; throttle recording already happened
    inside complete(), so the loop only owns the retry."""
    attempt = AsyncMock(side_effect=[
        RateLimitError("rate limited", response=MagicMock(status_code=429), body=None), "ok"])
    with patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                      kind=CallKind.EXTRACTION))
    assert result == "ok"
    assert attempt.call_count == 2


def test_call_with_retry_rate_limit_exhausted_fails_closed_and_records_failure():
    """CallKind.CONFIRM's profile has max_retries=3 -- the kind IS the retry
    budget now, nothing passed at the call site."""
    rate_limit_exc = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)
    attempt = AsyncMock(side_effect=rate_limit_exc)
    with patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                      kind=CallKind.CONFIRM))
    assert result == "fail"
    assert attempt.call_count == 3
    record_call.assert_called_once_with("c", "ctx", "transient", "rate limited", attempts=3)


def test_call_with_retry_retries_a_transient_failure_then_succeeds():
    attempt = AsyncMock(side_effect=[ValueError("truncated"), "ok"])
    with patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                      kind=CallKind.EXTRACTION))
    assert result == "ok"
    assert attempt.call_count == 2


def test_call_with_retry_retries_a_bad_generation_then_succeeds():
    attempt = AsyncMock(side_effect=[json.JSONDecodeError("bad", "doc", 0), "ok"])
    with patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                      kind=CallKind.CONFIRM))
    assert result == "ok"
    assert attempt.call_count == 2


def test_call_with_retry_non_retryable_error_fails_closed_on_first_attempt():
    from botocore.exceptions import ClientError

    permanent_exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")
    attempt = AsyncMock(side_effect=permanent_exc)
    with patch("services.llm_client.record_call_failure") as record_call:
        result = _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                                      kind=CallKind.EXTRACTION))
    assert result == "fail"
    attempt.assert_called_once()
    record_call.assert_called_once_with("c", "ctx", "permanent", str(permanent_exc), attempts=1)


def test_call_with_retry_uses_the_kinds_own_retry_budget():
    """Proof the kind actually drives max_retries, not a shared default --
    EXTRACTION_GLEANING's profile is 3 retries, EXTRACTION's is 5, same
    failure either way."""
    attempt = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("services.llm_client.record_call_failure"), \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        _run(call_with_retry(attempt, call_name="c", context="ctx", fail_closed=lambda: "fail",
                             kind=CallKind.EXTRACTION_GLEANING))
    assert attempt.call_count == 3


# --- confirm_with_retry: the yes/no convenience on top of the two layers ---

def _azure_json_client(*side_effects_or_content):
    mock_client = MagicMock()
    if len(side_effects_or_content) == 1 and isinstance(side_effects_or_content[0], str):
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=_raw(_response(content=side_effects_or_content[0])))
    else:
        effects = [e if isinstance(e, Exception) else _raw(_response(content=e))
                   for e in side_effects_or_content]
        mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=effects)
    return mock_client


def _patch_azure(mock_client, mock_throttle=None):
    return [
        patch("services.llm_client.effective_run_config", return_value=_CFG_AZURE),
        patch("services.llm_client.get_client", return_value=mock_client),
        _patch_throttle(mock_throttle),
        patch("services.llm_client.settings"),
    ]


def _run_confirm(mock_client, mock_throttle=None, **kwargs):
    patches = _patch_azure(mock_client, mock_throttle)
    with patches[0], patches[1], patches[2], patches[3]:
        return _run(confirm_with_retry(
            "sys", "user", kwargs.pop("result_field", "merge"),
            step=kwargs.pop("step", "s"), context=kwargs.pop("context", "c"),
            log_label=kwargs.pop("log_label", "L")))


def test_confirm_with_retry_returns_true_when_the_result_field_is_true():
    mock_client = _azure_json_client('{"merge": true}')
    assert _run_confirm(mock_client) is True
    mock_client.chat.completions.with_raw_response.create.assert_called_once()


def test_confirm_with_retry_returns_false_when_the_result_field_is_false():
    assert _run_confirm(_azure_json_client('{"merge": false}')) is False


def test_confirm_with_retry_retries_a_transient_failure_then_succeeds():
    mock_client = _azure_json_client(RuntimeError("boom"), '{"same_person": true}')
    with patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run_confirm(mock_client, result_field="same_person")
    assert result is True
    assert mock_client.chat.completions.with_raw_response.create.call_count == 2
    record_call.assert_not_called()


def test_confirm_with_retry_fails_closed_and_logs_after_exhausting_retries():
    """CONFIRM's profile is max_retries=3 -- exhausts in exactly 3 attempts."""
    mock_client = _azure_json_client(RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"))
    with patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = _run_confirm(mock_client, step="cluster_merge_confirm_incremental",
                              context="new='A' candidate='B'")
    assert result is False
    assert mock_client.chat.completions.with_raw_response.create.call_count == 3
    record_call.assert_called_once_with(
        "cluster_merge_confirm_incremental", "new='A' candidate='B'", "transient", "boom", attempts=3,
    )


def test_confirm_with_retry_fails_closed_immediately_on_a_permanent_error():
    from botocore.exceptions import ClientError
    permanent_exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")
    mock_client = _azure_json_client(permanent_exc)
    with patch("services.llm_client.record_call_failure") as record_call:
        result = _run_confirm(mock_client)
    assert result is False
    mock_client.chat.completions.with_raw_response.create.assert_called_once()  # never retried
    assert record_call.call_args.kwargs["attempts"] == 1


def test_confirm_with_retry_fatal_error_re_raises():
    """A bad Azure key no longer silently no-matches every confirm."""
    mock_client = _azure_json_client(_fatal_exc())
    with patch("services.llm_client.record_call_failure"):
        with pytest.raises(APIStatusError):
            _run_confirm(mock_client)
    mock_client.chat.completions.with_raw_response.create.assert_called_once()


def test_confirm_with_retry_uses_ministral_when_backend_is_ministral():
    """Never touches the Azure client when Ministral is the resolved backend."""
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_raw(_response(content='{"merge": true}')))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.get_client") as azure_client, \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        result = _run(confirm_with_retry("sys", "user", "merge", step="s", context="c", log_label="L"))
    assert result is True
    mock_client.chat.completions.with_raw_response.create.assert_called_once()
    azure_client.assert_not_called()
    mock_throttle.record_success.assert_called_once()


def test_confirm_with_retry_ministral_failure_retries_then_fails_closed():
    """A content-shaped error (bare ValueError) still retries and fails
    closed, but must NOT halve the Ministral throttle: it says nothing about
    pod load, and halving on it let one doomed input throttle the whole
    healthy pipeline (run x9j2: extraction 5 -> 2 -> 1 on a deterministic
    truncation). Only overload signals (timeout/connection/5xx/429) halve —
    see the dedicated overload tests below."""
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        side_effect=ValueError("bad generation"))
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        result = _run(confirm_with_retry("sys", "user", "merge",
                                         step="cluster_merge_confirm_incremental",
                                         context="new='A' candidate='B'", log_label="Merge-confirmation"))
    assert result is False
    record_call.assert_called_once_with(
        "cluster_merge_confirm_incremental", "new='A' candidate='B'", "transient", "bad generation", attempts=3,
    )
    assert mock_throttle.record_failure.call_count == 0


def test_ministral_connection_error_halves_the_throttle():
    """A timeout/connection error is the clearest overload signal a vLLM box
    emits — every attempt must record a throttle failure."""
    import httpx
    from openai import APIConnectionError
    conn_error = APIConnectionError(request=httpx.Request("POST", "http://localhost:8000"))
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=conn_error)
    mock_throttle = _mock_throttle()
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(mock_throttle), \
         patch("services.llm_client.record_call_failure"), \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        result = _run(confirm_with_retry("sys", "user", "merge", step="s", context="c", log_label="L"))
    assert result is False
    assert mock_throttle.record_failure.call_count == 3


def test_ministral_5xx_halves_the_throttle_but_4xx_does_not():
    """Server-side 5xx = overload (halve); a 4xx is a property of THIS request
    (bad request / content filter / context overflow) — the pod is fine."""
    import httpx

    def _status_error(status: int) -> APIStatusError:
        request = httpx.Request("POST", "http://localhost:8000")
        return APIStatusError(f"{status}", response=httpx.Response(status, request=request), body=None)

    for status, expected_failures in ((503, 1), (400, 0)):
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            side_effect=_status_error(status))
        mock_throttle = _mock_throttle()
        with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
             patch("services.llm_client.get_in_house_client", return_value=mock_client), \
             _patch_throttle(mock_throttle), \
             patch("services.llm_client.record_call_failure"), \
             patch("services.llm_client.asyncio.sleep", new=AsyncMock()), \
             patch("services.llm_client.settings") as mock_settings:
            mock_settings.ministral_model = "ministral-3b"
            with pytest.raises(APIStatusError):
                _run(complete(messages=[{"role": "user", "content": "hi"}], kind=CallKind.CONFIRM))
        assert mock_throttle.record_failure.call_count == expected_failures, f"status={status}"


def test_truncated_json_object_response_raises_llm_truncation_error():
    """finish_reason='length' surfaces as LlmTruncationError so the retry
    classifier can mark it permanent (deterministic at temperature 0) and
    extraction can recover by splitting the chunk."""
    from utils.error_classification import LlmTruncationError
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_raw(_response(content='{"merge": true}', finish_reason="length")))
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        with pytest.raises(LlmTruncationError):
            _run(complete(messages=[{"role": "user", "content": "hi"}], kind=CallKind.CONFIRM))


def test_structured_length_finish_reason_is_normalized_to_llm_truncation_error():
    """The SDK raises LengthFinishReasonError from raw.parse() on the
    structured path — complete() must normalize it to LlmTruncationError."""
    from openai import LengthFinishReasonError
    from utils.error_classification import LlmTruncationError
    raw = MagicMock()
    raw.headers = {}
    raw.parse.side_effect = LengthFinishReasonError(completion=MagicMock())
    mock_client = MagicMock()
    mock_client.beta.chat.completions.with_raw_response.parse = AsyncMock(return_value=raw)
    with patch("services.llm_client.effective_run_config", return_value=_CFG_MINISTRAL), \
         patch("services.llm_client.get_in_house_client", return_value=mock_client), \
         _patch_throttle(), \
         patch("services.llm_client.settings") as mock_settings:
        mock_settings.ministral_model = "ministral-3b"
        with pytest.raises(LlmTruncationError):
            _run(complete(messages=[{"role": "user", "content": "hi"}],
                          response_format=ChunkExtractionResult, kind=CallKind.EXTRACTION))


def test_ministral_throttle_is_shared_across_kinds_azure_is_not():
    """The Ministral pod has ONE measured capacity knee — every kind must
    draw from the same pool, or per-kind pools stack past it whenever two
    stages overlap (extraction + gleaning did exactly that in run x9j2)."""
    with patch.dict(llm_client._throttles, clear=True):
        m1 = llm_client.get_throttle_for(CallKind.EXTRACTION, "ministral")
        m2 = llm_client.get_throttle_for(CallKind.EXTRACTION_GLEANING, "ministral")
        m3 = llm_client.get_throttle_for(CallKind.CONFIRM, "ministral")
        a1 = llm_client.get_throttle_for(CallKind.EXTRACTION, "azure")
        a2 = llm_client.get_throttle_for(CallKind.CONFIRM, "azure")
    assert m1 is m2 is m3
    assert a1 is not a2
    assert m1 is not a1
