"""Unit tests for services/extraction_service.py's gleaning pass
(experiment-parity review C4) -- a second "what did we miss" call merged into
the base extraction result, with no repeats and no dangling relationships."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from models.extraction import ChunkExtractionResult, ExtractedClaim, ExtractedEntity, ExtractedRelationship
from services.extraction_service import ExtractionService

# Every test below exercises the Azure path -- force get_llm_backend() so it never
# depends on the process-wide resolved backend (which a different test module's
# import order could otherwise have already cached to "ministral").
_azure_backend = patch("services.extraction_service.get_llm_backend", new=lambda: "azure")


def _service() -> ExtractionService:
    """Bypass __init__ (which needs real Azure settings) -- tests set _throttle
    directly and patch parse_structured for LLM calls."""
    service = ExtractionService.__new__(ExtractionService)
    service._throttle = MagicMock()
    service._throttle.slot.return_value.__aenter__ = AsyncMock(return_value=None)
    service._throttle.slot.return_value.__aexit__ = AsyncMock(return_value=None)
    service._throttle.record_success = AsyncMock()
    service._throttle.record_rate_limit = AsyncMock()
    return service


def _run(coro):
    return asyncio.run(coro)


@_azure_backend
def test_gleaning_disabled_returns_base_result_without_a_second_call():
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    parse_mock = AsyncMock(return_value=base_result)
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.entities == base_result.entities
    assert parse_mock.await_count == 1


@_azure_backend
def test_extract_from_chunk_records_call_metrics():
    """Experiment-parity review #21: each real API attempt must be counted,
    so a run's final summary log reflects actual LLM call volume."""
    from services import run_metrics
    run_metrics.reset()
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    with patch("services.extraction_service.llm_client.complete",
              new=AsyncMock(return_value=base_result)), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert run_metrics.snapshot().get("extraction_calls") == 1


@_azure_backend
def test_gleaning_merges_new_entities_relationships_and_claims():
    service = _service()
    base_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")],
        relationships=[], claims=[],
    )
    glean_result = ChunkExtractionResult(
        entities=[
            ExtractedEntity(name="Globex", type="ORGANIZATION"),  # repeat -- must not duplicate
            ExtractedEntity(name="Initech", type="ORGANIZATION"),   # new
        ],
        relationships=[
            ExtractedRelationship(source_entity="Globex", target_entity="Initech", relationship_type="ACQUIRED"),
        ],
        claims=[
            ExtractedClaim(subject="Globex", predicate="acquired", object="Initech",
                          statement_type="FACT", temporal_type="STATIC"),
        ],
    )
    parse_mock = AsyncMock(side_effect=[base_result, glean_result])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))

    assert {e.name for e in result.entities} == {"Globex", "Initech"}
    assert len(result.relationships) == 1
    assert len(result.claims) == 1


@_azure_backend
def test_gleaning_drops_a_relationship_whose_endpoint_was_never_extracted():
    """A gleaned relationship pointing at an entity that doesn't exist in
    either the base or the gleaned entity list must never reach the graph."""
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    glean_result = ChunkExtractionResult(
        entities=[],
        relationships=[
            ExtractedRelationship(source_entity="Globex", target_entity="Nonexistent Corp",
                                  relationship_type="ACQUIRED"),
        ],
    )
    with patch("services.extraction_service.llm_client.complete",
              new=AsyncMock(side_effect=[base_result, glean_result])), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.relationships == []


@_azure_backend
def test_gleaning_skipped_when_base_extraction_failed():
    """A PERMANENT-classified failure (e.g. content-filter rejection) fails
    the base call immediately, same as before experiment-parity review #16's
    transient-retry change -- gleaning must still be skipped in that case."""
    service = _service()
    with patch("services.extraction_service.llm_client.complete",
              new=AsyncMock(side_effect=RuntimeError("boom"))), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.classify_llm_retry_error", return_value="permanent"), \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.failed is True


@_azure_backend
def test_gleaning_skipped_when_base_has_no_entities():
    service = _service()
    empty_result = ChunkExtractionResult()
    parse_mock = AsyncMock(return_value=empty_result)
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.entities == []
    assert parse_mock.await_count == 1


# --- base-extraction error classification (experiment-parity review #16) ---

@_azure_backend
def test_transient_error_is_retried_instead_of_failing_immediately():
    """A 5xx/connection-reset/timeout often clears on its own -- must consume
    a retry attempt, not fail on the first try the way a bare `except
    Exception: return failed=True` used to for every non-rate-limit error."""
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    parse_mock = AsyncMock(side_effect=[RuntimeError("connection reset"), base_result])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.classify_llm_retry_error", return_value="transient"), \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.entities == base_result.entities
    assert parse_mock.await_count == 2


@_azure_backend
def test_transient_error_gives_up_after_max_retries():
    service = _service()
    parse_mock = AsyncMock(side_effect=RuntimeError("connection reset"))
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.classify_llm_retry_error", return_value="transient"), \
         patch("services.llm_client.record_call_failure"), \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.failed is True
    assert parse_mock.await_count == 3


@_azure_backend
def test_permanent_error_fails_on_the_first_attempt_no_retry():
    """A content-filter/bad-request rejection will trigger identically every
    time -- retrying wastes calls and delays marking it failed for nothing."""
    service = _service()
    parse_mock = AsyncMock(side_effect=RuntimeError("content filter"))
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.classify_llm_retry_error", return_value="permanent"), \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.failed is True
    assert parse_mock.await_count == 1


@_azure_backend
def test_bad_generation_is_retried_not_failed_immediately():
    """A bad generation (invalid JSON / schema-validation failure) must be
    retried, not treated as permanent -- unlike classify_error(),
    classify_llm_retry_error() (the real classifier, not mocked here) calls
    this transient because a regenerate can succeed."""
    import json as json_module
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    bad_json_error = json_module.JSONDecodeError("bad json", "doc", 0)
    parse_mock = AsyncMock(side_effect=[bad_json_error, base_result])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.entities == base_result.entities
    assert parse_mock.await_count == 2


@_azure_backend
def test_fatal_error_is_re_raised_not_swallowed():
    """A bad API key/endpoint would fail identically for every future chunk
    too -- must propagate so it surfaces as a real problem, not get silently
    marked failed=True and retried the same losing way forever."""
    service = _service()
    parse_mock = AsyncMock(side_effect=RuntimeError("unauthorized"))
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.classify_llm_retry_error", return_value="fatal"), \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        try:
            _run(service.extract_from_chunk("some text", chunk_id="c1"))
            raised = False
        except RuntimeError:
            raised = True
    assert raised is True
    assert parse_mock.await_count == 1


@_azure_backend
def test_gleaning_failure_falls_back_to_base_result():
    """Gleaning is best-effort -- a broken gleaning call must never lose an
    otherwise-good base extraction."""
    service = _service()
    base_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Globex", type="ORGANIZATION")])
    with patch("services.extraction_service.llm_client.complete",
              new=AsyncMock(side_effect=[base_result, RuntimeError("gleaning broke")])), \
         patch("services.extraction_service.settings") as mock_settings:
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("some text", chunk_id="c1"))
    assert result.entities == base_result.entities



# --- truncation recovery (salvage + continuation): a chunk whose output
# overflows the context window fails deterministically at temperature 0, so
# retries and --retry-failed can never fix it. Recovery: repair-parse the
# partial output's complete objects, then one glean-shaped follow-up call
# collects what was missed (full chunk + salvaged names — compressed state,
# never a verbatim replay that wouldn't fit the window). ---

_TRUNCATED_PARTIAL = (
    '{"entities": ['
    '{"name": "Time Off Request tile", "type": "FEATURE", "description": "Submits vacation requests."},'
    '{"name": "BambooHR", "type": "TECHNOLOGY", "description": "HRIS platform."},'
    '{"name": "Ask HR portal", "type": "PRODUCT", "description": "Internal HR portal."},'
    '{"name": "Policy Center", "type": "PRODUCT", "description": "A Globex product for managing pol'
)


def _truncation_error(partial=_TRUNCATED_PARTIAL):
    from utils.error_classification import LlmTruncationError
    return LlmTruncationError("output truncated at the context ceiling", partial_content=partial)


def test_repair_truncated_json_salvages_complete_elements_only():
    from services.extraction_service import _repair_truncated_json
    data = _repair_truncated_json(_TRUNCATED_PARTIAL)
    assert [e["name"] for e in data["entities"]] == [
        "Time Off Request tile", "BambooHR", "Ask HR portal"]  # mid-flight "Policy Center" dropped

    # Cut inside a string VALUE containing brackets/braces — the string-aware
    # scanner must not miscount them.
    tricky = '{"entities": [{"name": "A {weird] name", "type": "CONCEPT"}, {"name": "B", "ty'
    data = _repair_truncated_json(tricky)
    assert [e["name"] for e in data["entities"]] == ["A {weird] name"]

    # Cut inside the very FIRST element -> nothing complete to salvage.
    assert _repair_truncated_json('{"entities": [{"name": "Simp') is None
    assert _repair_truncated_json("") is None


def test_salvage_extraction_validates_items_individually_and_drops_dangling():
    from services.extraction_service import _salvage_extraction
    partial = (
        '{"entities": ['
        '{"name": "Globex", "type": "ORGANIZATION"},'
        '{"type": "MISSING_NAME_SO_INVALID"}'
        '], "relationships": ['
        '{"source_entity": "Globex", "target_entity": "Ghost Corp", "relationship_type": "ACQUIRED"}'
        '], "claims": [{"subject": "Globex", "predicate": "is", "object": "an intranet", '
        '"statement_type": "FACT", "temporal_type": "ATEMPORAL"}], "clai'
    )
    result = _salvage_extraction(partial)
    assert [e.name for e in result.entities] == ["Globex"]   # invalid sibling dropped, not fatal
    assert result.relationships == []                          # dangling endpoint dropped
    assert len(result.claims) == 1
    assert _salvage_extraction(None).entities == []
    assert _salvage_extraction("not json at all").entities == []


@_azure_backend
def test_truncation_salvages_partial_and_normal_gleaning_continues_it():
    """Truncated base call: ONE attempt (permanent — no blind retries), the
    partial's complete entities are salvaged, and the regular gleaning pass
    acts as the continuation — it sees the full chunk plus the salvaged names
    and returns only what was missed, merged with the usual dedupe."""
    service = _service()
    glean_result = ChunkExtractionResult(
        entities=[ExtractedEntity(name="Policy Center", type="PRODUCT"),   # the entity the cut destroyed
                  ExtractedEntity(name="BambooHR", type="TECHNOLOGY")],    # repeat -- must not duplicate
        relationships=[ExtractedRelationship(source_entity="Time Off Request tile",
                                             target_entity="BambooHR",
                                             relationship_type="INTEGRATES_WITH")],
        claims=[ExtractedClaim(subject="Policy Center", predicate="reached_status", object="GA",
                               statement_type="FACT", temporal_type="STATIC")],
    )
    parse_mock = AsyncMock(side_effect=[_truncation_error(), glean_result])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("full chunk text", chunk_id="c1"))
    assert result.failed is False
    assert {e.name for e in result.entities} == {
        "Time Off Request tile", "BambooHR", "Ask HR portal", "Policy Center"}
    assert len(result.relationships) == 1
    assert len(result.claims) == 1
    assert parse_mock.await_count == 2  # 1 truncated attempt + 1 continuation, no blind retries
    # The continuation prompt carries the salvaged names as compressed state.
    continuation_user_msg = parse_mock.await_args_list[1].kwargs["messages"][1]["content"]
    assert "Already extracted entities:" in continuation_user_msg
    assert "Ask HR portal" in continuation_user_msg


@_azure_backend
def test_truncation_runs_an_explicit_topup_when_gleaning_is_disabled():
    """For a truncated chunk the second pass is RECOVERY, not a bonus — it
    must run even with the gleaning feature flag off."""
    service = _service()
    glean_result = ChunkExtractionResult(entities=[ExtractedEntity(name="Policy Center", type="PRODUCT")])
    parse_mock = AsyncMock(side_effect=[_truncation_error(), glean_result])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = False
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("full chunk text", chunk_id="c1"))
    assert result.failed is False
    assert "Policy Center" in {e.name for e in result.entities}
    assert parse_mock.await_count == 2


@_azure_backend
def test_truncation_with_unsalvageable_partial_fails_the_chunk():
    """Cut inside the first object -> nothing to salvage -> failed=True after
    exactly one attempt; the chunk retries next run via the ledger."""
    service = _service()
    parse_mock = AsyncMock(side_effect=_truncation_error(partial='{"entities": [{"name": "Si'))
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("full chunk text", chunk_id="c1"))
    assert result.failed is True
    assert parse_mock.await_count == 1


@_azure_backend
def test_truncation_topup_failure_keeps_the_salvaged_result():
    """If the follow-up pass itself dies, the salvaged entities still beat a
    failed chunk — degraded, never lost."""
    service = _service()
    parse_mock = AsyncMock(side_effect=[_truncation_error(), RuntimeError("followup broke")])
    with patch("services.extraction_service.llm_client.complete", parse_mock), \
         patch("services.extraction_service.settings") as mock_settings, \
         patch("services.llm_client.record_call_failure"):
        mock_settings.extraction_gleaning_enabled = True
        mock_settings.azure_llm_temperature = 0.0
        result = _run(service.extract_from_chunk("full chunk text", chunk_id="c1"))
    assert result.failed is False
    assert {e.name for e in result.entities} == {"Time Off Request tile", "BambooHR", "Ask HR portal"}
