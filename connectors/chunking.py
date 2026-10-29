"""Chunk conventions shared by every reader of this Elasticsearch index.

`_chunks.v1` and `_embeddings.*.text_embedding` are the upstream indexer's
shape, not any one connector's: the bulk path reads them through
`connectors/zeus/documents.py` and the incremental path reads them straight off
a Kafka payload. They live here so neither has to import the other — before
this, `services/incremental_pipeline.py` reached into the Zeus connector for
them, which is the wrong direction across the boundary.
"""
from __future__ import annotations

from typing import Optional

from config.settings import SUPPORTED_EMBEDDING_SPACES, settings


def upstream_chunk_vector(chunk: dict, es_field: Optional[str] = None) -> Optional[list[float]]:
    """The chunk's own pre-computed embedding, straight off the ES source
    (`_embeddings.<es_field>.text_embedding`). Chunk vectors are COPIED from this
    field, never re-embedded by this pipeline — whatever indexed the content into
    ES already computed it once. Returns None if the field isn't present; callers
    must skip the Qdrant push for that chunk rather than inventing a vector.

    es_field defaults to the process's own embedding space's field (the bulk
    path, where settings ARE the run's config). The incremental path passes the
    field its bulk run recorded (services/run_context.RunConfig.
    es_chunk_vector_field), so both copy the upstream vector from the same field
    and share one vec_chunks space. Threaded as a param rather than read from
    run config here to keep this connector-layer helper free of a services
    dependency.

    No fallback field: vec_chunks is created fixed at the space's dimension, so
    a vector at any other dimension (e.g. an older 128-d export shape) could
    never be validly written there anyway."""
    field = es_field or SUPPORTED_EMBEDDING_SPACES[settings.embedding_backend]["es_vector_field"]
    emb = chunk.get("_embeddings") or {}
    sub = emb.get(field) or {}
    if isinstance(sub, dict):
        vector = sub.get("text_embedding")
        if isinstance(vector, list) and vector:
            return vector
    return None


def dedup_chunks(doc: dict) -> list[dict]:
    """The upstream recursive chunker emits nested/overlapping windows that
    share a recursive_chunk_id with byte-identical text — keeping every raw
    _chunks.v1 entry would re-extract and re-write the same text several
    times over per document, inflating LLM cost and graph nodes accordingly.

    Keeps the first entry per recursive_chunk_id (falls back to chunk_id if
    absent), drops chunks with no usable id or empty text. Used everywhere
    _chunks.v1 is read so the same deduped set is counted, extracted, and
    written consistently."""
    seen: set = set()
    out: list[dict] = []
    for chunk in (doc.get("_chunks") or {}).get("v1", []) or []:
        rid = chunk.get("recursive_chunk_id") or chunk.get("chunk_id")
        if not rid or rid in seen or not (chunk.get("text") or "").strip():
            continue
        seen.add(rid)
        out.append(chunk)
    return out
