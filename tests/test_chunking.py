"""Unit tests for connectors/chunking.py::upstream_chunk_vector -- the ES field
key now comes from the run's embedding space (passed in by the caller, or the
process's own space by default), and the old 128-d fallback is gone entirely
rather than migrated: vec_chunks is created fixed at the space's dimension, so
a 128-d vector could never be validly written there."""
from connectors.chunking import upstream_chunk_vector


def test_upstream_chunk_vector_reads_the_passed_field():
    vector = [0.1] * 1536
    chunk = {"_embeddings": {"embeddings_openai_large_1536": {"text_embedding": vector}}}
    assert upstream_chunk_vector(chunk, "embeddings_openai_large_1536") == vector


def test_upstream_chunk_vector_reads_whatever_field_the_caller_says():
    """Not hardcoded -- a differently-configured field name is honored."""
    vector = [0.2] * 256
    chunk = {"_embeddings": {"some_other_field": {"text_embedding": vector}}}
    assert upstream_chunk_vector(chunk, "some_other_field") == vector


def test_upstream_chunk_vector_defaults_to_the_process_spaces_field():
    """No explicit field: the default is the active embedding space's field."""
    from config.settings import SUPPORTED_EMBEDDING_SPACES, settings
    space = SUPPORTED_EMBEDDING_SPACES[settings.embedding_backend]
    vector = [0.3] * space["dimension"]
    chunk = {"_embeddings": {space["es_vector_field"]: {"text_embedding": vector}}}
    assert upstream_chunk_vector(chunk) == vector


def test_upstream_chunk_vector_no_128d_fallback_anymore():
    """The old embeddings_openai_128 fallback is gone, not just deprioritized --
    a chunk with only that field now returns None, same as having no vector
    at all, since vec_chunks could never accept a 128-d vector anyway."""
    chunk = {"_embeddings": {"embeddings_openai_128": {"text_embedding": [0.1] * 128}}}
    assert upstream_chunk_vector(chunk, "embeddings_openai_large_1536") is None


def test_upstream_chunk_vector_returns_none_when_field_missing():
    assert upstream_chunk_vector({"_embeddings": {}}, "embeddings_openai_large_1536") is None
    assert upstream_chunk_vector({}, "embeddings_openai_large_1536") is None


def test_upstream_chunk_vector_returns_none_for_empty_vector():
    chunk = {"_embeddings": {"embeddings_openai_large_1536": {"text_embedding": []}}}
    assert upstream_chunk_vector(chunk, "embeddings_openai_large_1536") is None
