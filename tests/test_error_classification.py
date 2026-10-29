"""Unit tests for utils/error_classification.py."""
import json

import elastic_transport
import elasticsearch
import httpx
from botocore.exceptions import ClientError
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel
from requests.exceptions import HTTPError

from repository.run_registry import NoCompletedRunError
from utils.error_classification import classify_error, classify_llm_retry_error


def _openai_status_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", "http://localhost:8080")
    response = httpx.Response(status, request=request)
    return APIStatusError(f"{status} error", response=response, body=None)


def _es_api_error(status: int) -> elasticsearch.ApiError:
    meta = elastic_transport.ApiResponseMeta(
        status=status, http_version="1.1", headers=elastic_transport.HttpHeaders({}), duration=0.1, node=None,
    )
    return elasticsearch.ApiError(f"{status} error", meta, None)


def test_no_completed_run_error_is_permanent():
    """An incremental event for a (tenant, connector) with no
    COMPLETED bulk run yet has nothing to write into -- retrying the exact
    same event won't change that, so it must go straight to the DLQ rather
    than redeliver forever."""
    assert classify_error(NoCompletedRunError("no run found")) == "permanent"


def test_json_decode_error_is_permanent():
    try:
        json.loads("not json")
    except json.JSONDecodeError as exc:
        assert classify_error(exc) == "permanent"


def test_key_error_is_permanent():
    assert classify_error(KeyError("tenant_id")) == "permanent"


def test_generic_value_error_is_transient():
    """Only json.JSONDecodeError/pydantic.ValidationError (ValueError subclasses
    checked explicitly) count as permanent — a bare, unrelated ValueError from
    some other, possibly-retryable path must not be silently written off as
    unfixable without ever retrying."""
    assert classify_error(ValueError("bad shape")) == "transient"


class _Strict(BaseModel):
    x: int


def test_pydantic_validation_error_is_permanent():
    """A model response that fails schema validation (e.g.
    ChunkExtractionResult.model_validate_json on a malformed Ministral reply)
    will fail identically on every retry -- must not be silently retried 5x
    as if it were a network blip (ValidationError IS a ValueError subclass,
    so this must be checked ahead of the generic-ValueError default above)."""
    try:
        _Strict.model_validate_json("not json at all")
    except Exception as exc:
        assert classify_error(exc) == "permanent"


def test_index_error_is_permanent():
    """An empty `choices` array in a model response is a structural problem
    with that response, not a transient blip."""
    assert classify_error(IndexError("list index out of range")) == "permanent"


def _http_error(status: int) -> HTTPError:
    exc = HTTPError(f"{status} error")
    exc.response = type("Resp", (), {"status_code": status})()
    return exc


def test_neo4j_401_is_fatal():
    """A global auth problem would fail identically for every future message
    too — this must crash the worker, not drain the whole topic into the DLQ
    one message at a time."""
    assert classify_error(_http_error(401)) == "fatal"


def test_neo4j_403_is_fatal():
    assert classify_error(_http_error(403)) == "fatal"


def test_neo4j_400_is_permanent():
    assert classify_error(_http_error(400)) == "permanent"


def test_neo4j_404_is_permanent():
    assert classify_error(_http_error(404)) == "permanent"


def test_neo4j_500_is_transient():
    assert classify_error(_http_error(500)) == "transient"


def test_http_error_with_no_response_is_transient():
    exc = HTTPError("connection reset before a response was received")
    exc.response = None
    assert classify_error(exc) == "transient"


def test_s3_404_is_permanent():
    exc = ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")
    assert classify_error(exc) == "permanent"


def test_s3_403_is_permanent():
    exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")
    assert classify_error(exc) == "permanent"


def test_s3_500_is_transient():
    exc = ClientError({"Error": {"Code": "InternalError"}, "ResponseMetadata": {"HTTPStatusCode": 500}}, "GetObject")
    assert classify_error(exc) == "transient"


def test_generic_exception_is_transient():
    assert classify_error(RuntimeError("connection reset")) == "transient"


def test_timeout_is_transient():
    assert classify_error(TimeoutError("timed out")) == "transient"


# --- Azure OpenAI (experiment-parity review #16) ---

def test_azure_openai_content_filter_rejection_is_permanent():
    """A content-filter/bad-request rejection (400) will trigger identically
    on retry -- that chunk's text is the problem, not a passing blip."""
    assert classify_error(_openai_status_error(400)) == "permanent"


def test_azure_openai_401_is_fatal():
    """A bad API key/endpoint would fail identically for every future chunk
    too -- extraction_service.py must re-raise this, not mark one chunk
    failed=True and keep retrying every future one the same losing way."""
    assert classify_error(_openai_status_error(401)) == "fatal"


def test_azure_openai_500_is_transient():
    assert classify_error(_openai_status_error(500)) == "transient"


def test_azure_openai_connection_error_is_transient():
    request = httpx.Request("POST", "http://localhost:8080")
    assert classify_error(APIConnectionError(request=request)) == "transient"


def test_azure_openai_timeout_error_is_transient():
    """APITimeoutError subclasses APIConnectionError -- must be caught by the
    same branch, not fall through as an unrecognized generic exception."""
    request = httpx.Request("POST", "http://localhost:8080")
    assert classify_error(APITimeoutError(request=request)) == "transient"


# --- Elasticsearch (experiment-parity review #19) ---

def test_elasticsearch_400_is_permanent():
    assert classify_error(_es_api_error(400)) == "permanent"


def test_elasticsearch_404_is_permanent():
    assert classify_error(_es_api_error(404)) == "permanent"


def test_elasticsearch_401_is_fatal():
    """A bad ES credential would fail identically for every future page too
    -- must not be treated as a per-page transient blip."""
    assert classify_error(_es_api_error(401)) == "fatal"


def test_elasticsearch_500_is_transient():
    assert classify_error(_es_api_error(500)) == "transient"


def test_elasticsearch_connection_error_is_transient():
    assert classify_error(elastic_transport.ConnectionError("boom")) == "transient"


def test_elasticsearch_connection_timeout_is_transient():
    assert classify_error(elastic_transport.ConnectionTimeout("timed out")) == "transient"


def test_qdrant_collection_404_is_permanent():
    """qdrant_client.UnexpectedResponse is not requests.HTTPError — without an
    explicit branch, missing-collection 404s were retried as transient."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    exc = UnexpectedResponse(404, "Not Found", b'{"status":{"error":"Not found"}}', None)
    assert classify_error(exc) == "permanent"


def test_qdrant_5xx_is_transient():
    from qdrant_client.http.exceptions import UnexpectedResponse

    exc = UnexpectedResponse(503, "Service Unavailable", b"busy", None)
    assert classify_error(exc) == "transient"


def test_classify_llm_retry_error_treats_bad_generation_as_transient():
    """LLM in-call retries regenerate on invalid output; Kafka/DLQ does not."""
    try:
        json.loads("not json")
    except json.JSONDecodeError as exc:
        assert classify_error(exc) == "permanent"
        assert classify_llm_retry_error(exc) == "transient"

    try:
        _Strict.model_validate_json("not json at all")
    except Exception as exc:
        assert classify_error(exc) == "permanent"
        assert classify_llm_retry_error(exc) == "transient"

    assert classify_llm_retry_error(IndexError("list index out of range")) == "transient"
    assert classify_llm_retry_error(KeyError("merge")) == "transient"


def test_classify_llm_retry_error_still_fails_closed_on_fatal_and_permanent():
    assert classify_llm_retry_error(_http_error(401)) == "fatal"
    assert classify_llm_retry_error(_http_error(400)) == "permanent"
    exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")
    assert classify_llm_retry_error(exc) == "permanent"


def test_truncation_is_permanent_for_llm_retry():
    """finish_reason='length' is deterministic at temperature 0 — a retry
    regenerates the identical truncation (run x9j2 burned 3 identical
    ~2-minute attempts on one chunk). Both the normalized LlmTruncationError
    and openai's own LengthFinishReasonError must be permanent, and must win
    over the bad-generation-is-transient rule (LlmTruncationError IS a
    ValueError)."""
    from unittest.mock import MagicMock

    from openai import LengthFinishReasonError

    from utils.error_classification import LlmTruncationError

    assert classify_llm_retry_error(LlmTruncationError("output truncated")) == "permanent"
    assert classify_llm_retry_error(LengthFinishReasonError(completion=MagicMock())) == "permanent"
    # The generic bad-generation rule is unchanged.
    assert classify_llm_retry_error(ValueError("some other value error")) == "transient"
