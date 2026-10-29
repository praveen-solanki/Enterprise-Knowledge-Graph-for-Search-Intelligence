"""Unit tests for services/embedding_service.py -- the one embed_batch door:
backend routing off the run's embedding space, internal sub-batching,
empty-string sanitization, and retry/backoff on transient failures."""
from unittest.mock import MagicMock, patch

import pytest
from openai import BadRequestError, RateLimitError

from config.settings import settings
from services.embedding_service import EmbeddingService, _split_into_batches
from services.run_context import RunConfig


def _cfg(backend: str = "azure", model: str = "text-embedding-3-large",
         dimension: int = 1536) -> RunConfig:
    return RunConfig(
        embedding_backend=backend, embedding_model=model, embedding_dimension=dimension,
        consume_upstream_vector=True, es_chunk_vector_field="f",
        llm_backend="azure", llm_model="gpt",
        build_authorship_edges=True, contributor_entity_types=("X",), people_org_graph_enabled=True,
    )


# The dimension/backend a call embeds at comes from the running run's config
# (services/llm_backend.effective_run_config). Pin it to the azure 1536-d space
# so these tests are deterministic regardless of process env or any run bound
# by another test module.
_azure_backend = patch("services.embedding_service.effective_run_config", new=_cfg)


def _service() -> EmbeddingService:
    """The Azure client is lazy -- set it directly so no real settings are needed."""
    service = EmbeddingService()
    service._azure_client = MagicMock()
    return service


def _response(vectors: list) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vectors]
    return resp


def _bad_request_too_large() -> BadRequestError:
    return BadRequestError(
        "Error code: 400 - Invalid 'input': maximum request size is 300000 tokens per request.",
        response=MagicMock(status_code=400),
        body={"error": {"message": "maximum request size is 300000 tokens per request."}},
    )


def test_split_into_batches_count_only_when_no_token_budget():
    assert _split_into_batches(["a", "b", "c", "d"], max_items=2) == [["a", "b"], ["c", "d"]]


def test_split_into_batches_respects_token_budget():
    # chars_per_token=1 → each char is ~1 token estimate; "aaaa" ≈ 5 tokens.
    texts = ["aaaa", "bbbb", "cccc"]
    batches = _split_into_batches(
        texts, max_items=10, max_tokens=10, chars_per_token=1.0,
    )
    assert batches == [["aaaa", "bbbb"], ["cccc"]]


def test_split_into_batches_also_respects_max_items_under_token_budget():
    texts = ["a", "b", "c"]
    batches = _split_into_batches(
        texts, max_items=2, max_tokens=10_000, chars_per_token=3.0,
    )
    assert batches == [["a", "b"], ["c"]]


@_azure_backend
def test_embed_batch_empty_list_returns_empty_without_calling_the_api():
    service = _service()
    assert service.embed_batch([]) == []
    service._azure_client.embeddings.create.assert_not_called()


@_azure_backend
def test_embed_batch_sub_batches_at_the_batch_size():
    """A caller must never have to hand-batch itself -- the service does it
    internally, so every caller benefits uniformly."""
    service = _service()
    with patch("services.embedding_service._EMBED_BATCH_SIZE", 2), \
         patch.object(settings, "embedding_azure_max_tokens_per_request", 10_000_000):
        service._azure_client.embeddings.create.side_effect = [
            _response([[0.1], [0.2]]), _response([[0.3]]),
        ]
        result = service.embed_batch(["a", "b", "c"])
    assert result == [[0.1], [0.2], [0.3]]
    assert service._azure_client.embeddings.create.call_count == 2


@_azure_backend
def test_embed_batch_sub_batches_when_token_budget_is_hit():
    """Azure's ~300k-token request cap must split even when item count is under
    embedding_azure_batch_size (the Neo4j chunk re-embed failure mode)."""
    service = _service()
    # Three ~4-char texts with chars_per_token=1 and max_tokens=10 → 2+1 packs.
    with patch("services.embedding_service._EMBED_BATCH_SIZE", 500), \
         patch.object(settings, "embedding_azure_max_tokens_per_request", 10), \
         patch.object(settings, "embedding_azure_chars_per_token_estimate", 1.0):
        service._azure_client.embeddings.create.side_effect = [
            _response([[0.1], [0.2]]), _response([[0.3]]),
        ]
        result = service.embed_batch(["aaaa", "bbbb", "cccc"])
    assert result == [[0.1], [0.2], [0.3]]
    assert service._azure_client.embeddings.create.call_count == 2


@_azure_backend
def test_azure_embed_batch_splits_on_request_too_large_error():
    """If packing underestimates real tokens, halve and retry instead of failing."""
    service = _service()
    service._azure_client.embeddings.create.side_effect = [
        _bad_request_too_large(),
        _response([[0.1]]),
        _response([[0.2]]),
    ]
    with patch.object(settings, "embedding_azure_max_tokens_per_request", 10_000_000):
        result = service.embed_batch(["a", "b"])
    assert result == [[0.1], [0.2]]
    assert service._azure_client.embeddings.create.call_count == 3


@_azure_backend
def test_embed_batch_records_call_metrics():
    """Each real API attempt must be counted."""
    from services import run_metrics
    run_metrics.reset()
    service = _service()
    service._azure_client.embeddings.create.return_value = _response([[0.1]])
    service.embed_batch(["a"])
    assert run_metrics.snapshot().get("embedding_calls") == 1


@_azure_backend
def test_embed_batch_sanitizes_blank_and_whitespace_only_strings():
    """Azure rejects a bare empty string outright -- must not fail the whole
    batch over one bad entry."""
    service = _service()
    service._azure_client.embeddings.create.return_value = _response([[0.1], [0.2], [0.3]])
    service.embed_batch(["real text", "", "   "])
    _, kwargs = service._azure_client.embeddings.create.call_args
    assert kwargs["input"] == ["real text", " ", " "]


@_azure_backend
def test_embed_batch_passes_the_run_spaces_dimension_to_azure():
    service = _service()
    service._azure_client.embeddings.create.return_value = _response([[0.1] * 1536])
    service.embed_batch(["a"])
    _, kwargs = service._azure_client.embeddings.create.call_args
    assert kwargs["dimensions"] == 1536


@_azure_backend
def test_embed_batch_retries_a_rate_limit_and_succeeds():
    service = _service()
    service._azure_client.embeddings.create.side_effect = [
        RateLimitError("429", response=MagicMock(), body=None), _response([[0.1]]),
    ]
    with patch("services.embedding_service.time.sleep"):
        result = service.embed_batch(["a"])
    assert result == [[0.1]]
    assert service._azure_client.embeddings.create.call_count == 2


@_azure_backend
def test_embed_batch_retries_a_transient_failure_and_succeeds():
    service = _service()
    service._azure_client.embeddings.create.side_effect = [RuntimeError("connection reset"), _response([[0.1]])]
    with patch("services.embedding_service.classify_error", return_value="transient"), \
         patch("services.embedding_service.time.sleep"):
        result = service.embed_batch(["a"])
    assert result == [[0.1]]
    assert service._azure_client.embeddings.create.call_count == 2


@_azure_backend
def test_embed_batch_does_not_retry_a_permanent_failure():
    service = _service()
    service._azure_client.embeddings.create.side_effect = RuntimeError("bad request")
    with patch("services.embedding_service.classify_error", return_value="permanent"):
        with pytest.raises(RuntimeError):
            service.embed_batch(["a"])
    assert service._azure_client.embeddings.create.call_count == 1


@_azure_backend
def test_embed_batch_gives_up_after_max_retries():
    service = _service()
    service._azure_client.embeddings.create.side_effect = RuntimeError("connection reset")
    with patch("services.embedding_service.classify_error", return_value="transient"), \
         patch("services.embedding_service.time.sleep"):
        with pytest.raises(RuntimeError):
            service.embed_batch(["a"])
    assert service._azure_client.embeddings.create.call_count == 5


# --- the inhouse backend route ---

def test_embed_batch_routes_to_the_inhouse_gateway_with_the_spaces_model_and_dimension():
    """The run's space decides the backend; the gateway gets the space's model
    and dimension, and Azure is never touched."""
    service = _service()
    inhouse_cfg = _cfg(backend="inhouse", model="arctic", dimension=256)
    with patch("services.embedding_service.effective_run_config", new=lambda: inhouse_cfg), \
         patch("services.embedding_service.inhouse_embed_batch",
               return_value=[[0.1] * 256]) as gateway:
        result = service.embed_batch(["a"])
    assert result == [[0.1] * 256]
    gateway.assert_called_once_with(["a"], model="arctic", dimensions=256)
    service._azure_client.embeddings.create.assert_not_called()


def test_embed_batch_inhouse_splits_at_the_gateway_batch_size():
    """The gateway 502s on batches over 512 inputs, so embed_batch must never
    POST more than embedding_inhouse_batch_size at once."""
    service = _service()
    inhouse_cfg = _cfg(backend="inhouse", model="arctic", dimension=256)
    with patch("services.embedding_service.effective_run_config", new=lambda: inhouse_cfg), \
         patch.object(settings, "embedding_inhouse_batch_size", 2), \
         patch("services.embedding_service.inhouse_embed_batch",
               side_effect=[[[0.1], [0.2]], [[0.3]]]) as gateway:
        result = service.embed_batch(["a", "b", "c"])
    assert result == [[0.1], [0.2], [0.3]]
    assert gateway.call_count == 2
    assert all(len(call.args[0]) <= 2 for call in gateway.call_args_list)


def test_inhouse_client_posts_the_openai_embeddings_payload_to_a_vllm_url():
    """A /v1/embeddings URL is vLLM's OpenAI API, not the old gateway's
    {inputs, dim} body. Vectors come back in input order even if `data` is not."""
    from services import inhouse_embedding_client as client

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [
        {"index": 1, "embedding": [0.2]},
        {"index": 0, "embedding": [0.1]},
    ]}
    with patch.object(client.settings, "in_house_embeddings_url", "http://localhost:8001/v1/embeddings"), \
         patch.object(client.settings, "inference_provider", "inhouse"), \
         patch.object(client.settings, "ministral_api_key", "local-dev"), \
         patch.object(client.requests, "post", return_value=response) as post:
        vectors = client.embed_batch(["a", "b"], model="arctic", dimensions=256)
    assert vectors == [[0.1], [0.2]]
    body = post.call_args.kwargs["json"]
    assert body == {"input": ["a", "b"], "model": "arctic", "dimensions": 256}
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer local-dev"
