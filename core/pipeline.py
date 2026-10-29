"""
Orchestrator: Step 0 (fetch) -> Step 1 (extraction) -> Step 2 (doc-level
canonicalization) -> Step 3 (corpus-wide canonicalization) -> Step 4 (edge
pool) -> Step 5 (Neo4j write, optional).

Memory stays O(batch), not O(corpus): every intermediate result (chunk ids,
doc_canonical entities/merge maps, edges/claims/mentions) is persisted to
Mongo/Neo4j per-doc and released immediately, so peak memory doesn't grow
with corpus size.

Single event loop: asyncio.run() is called exactly once, in
core/entrypoint.py. All awaits here share that loop — the semaphore and
Azure clients are bound to it and must not be reused across a different loop.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import time

from config.settings import settings
from connectors.base import ConnectorSpec, PeopleContext, run_document_preprocess, run_preprocessing_drain
from models.extraction import ChunkExtractionResult
from repository import entity_registry
from repository.mongo_repository import (
    clear_failure,
    delete_checkpoint,
    iter_checkpoints,
    list_failures,
    load_checkpoint,
    load_checkpoints_batch,
    record_failure,
    save_checkpoint,
    save_checkpoints_batch,
)
from repository.run_registry import get_run, mark_run_completed, mark_run_failed, register_run
from services.adaptive_controller import AdaptiveBatchSizer
from services.doc_canonicalization_service import DOC_CANON_VERSION, canonicalize_document
from services.edge_pool_service import EDGE_POOL_VERSION, build_edge_pool, merge_edge_pools
from services.embedding_service import get_embedding_service
from utils.error_classification import classify_error
from services.extraction_service import EXTRACTION_VERSION, ExtractionService
from services.graph_canonicalization_service import GRAPH_CANON_VERSION, canonicalize_corpus, resolve_entity_ids
from services.people import materialize_persons
from services.person_corpus_resolver import resolve_persons_for_corpus
from services.run_metrics import reset as reset_run_metrics
from services.run_metrics import snapshot as run_metrics_snapshot
from services.vector_store import VectorStore
from connectors.logger_setup import setup_logging
from utils.log_context import set_log_context

logger = logging.getLogger("graphrag.core.pipeline")

# Starting chunk budget per batch (settings.bulk_batch_initial_chunks).
# AdaptiveBatchSizer will grow this automatically based on how long each batch
# actually takes, targeting ~45 seconds per batch.


def _retire_merged_person_mentions(pairs: list[dict], vector_store: Optional[VectorStore]) -> None:
    """Cleans up what repo.merge_person_mentions leaves behind: it only
    rewires/deletes the Neo4j node, not the Mongo entity_registry row or the
    Qdrant vector for the merged entity_id. A stale registry row isn't
    cosmetic — a later bare-name mention would resolve through it to a dead
    entity_id and silently undo the merge, so it must be deleted here.

    Must run AFTER repo.merge_person_mentions has migrated the graph edges —
    deleting the registry row first would let a concurrent caller re-mint a
    fresh entity_id for this alias mid-migration."""
    mention_ids = [p["m"] for p in pairs if p.get("m")]
    if not mention_ids:
        return
    for entity_id in mention_ids:
        entity_registry.delete_merged(entity_id)
    if vector_store is not None:
        try:
            vector_store.delete_entities(mention_ids)
        except Exception:  # noqa: BLE001 — best-effort: the Neo4j merge already succeeded
            logger.exception("Vector cleanup for %d merged person mention(s) failed — non-fatal",
                             len(mention_ids))


def _corpus_key(doc_ids: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(doc_ids)).encode("utf-8")).hexdigest()


def _assert_upstream_vector_space(transport, index: str, query: Optional[dict],
                                  run_config) -> None:
    """The old check sampled an Elasticsearch index for precomputed chunk
    vectors. No connector in this build ships those, so the flag is a
    misconfiguration rather than a mode we can satisfy."""
    raise ValueError(
        "consume_upstream_vector=true, but wikipedia and arxiv do not ship "
        "precomputed chunk vectors — set consume_upstream_vector=false so this "
        f"run embeds chunks itself ({run_config.embedding_backend}/"
        f"{run_config.embedding_dimension}-d)"
    )


def _resolve_run_context(run_id: Optional[str], index: str, tenant_id: str,
                         connector_type: str):
    """Resumes an existing run's stores when run_id is given (must already be
    registered), else mints a brand-new isolated run_id + RunContext."""
    if run_id:
        run_ctx = get_run(run_id)
        if run_ctx is None:
            raise ValueError(f"run_bulk_async: no run_registry entry for run_id={run_id!r} — "
                             "cannot resume an unknown run; omit run_id to start a new one")
        return run_id, run_ctx
    # New run: capture the model/space/flags this run resolves to, once, so
    # incremental events reuse the exact same config. Resume (above) never
    # reaches here, so a resumed run keeps its originally-recorded config.
    from services.llm_backend import snapshot_run_config
    return register_run(index, tenant_id, connector_type, snapshot_run_config())


def _coll(mongo_collections: Optional[dict], kind: str) -> str:
    """Resolves a checkpoint kind ('extraction', 'doc_canonical', ...) to this
    run's actual per-run Mongo collection name (services/run_context.py).
    Falls back to the kind itself when no RunContext is given — the direct
    unit-level calls in tests/test_pipeline.py exercise these helpers without
    one, and today's literal collection names are exactly that fallback."""
    return (mongo_collections or {}).get(kind, kind)


def _tag_entity_acls(final_entities: dict, doc_acl_by_doc: dict) -> None:
    """Step 5's ACL pass, mutating final_entities in place.

    Entity-level acl is the union of every supporting document's acl
    ("visible if any source is visible", same as edge_pool_service.py's
    relationship union). Note: doc_acl_by_doc only covers docs fetched in
    THIS run, so an entity whose source_docs include a doc from an older,
    wider run won't have that doc's acl unioned in.

    Per-source (description/alias) tagging uses each entry's OWN document's
    acl, not the entity-level union — a description from a private document
    must never become readable just because the entity also has a public
    source."""
    for entity in final_entities.values():
        entity["acl"] = sorted(set().union(*(
            set(doc_acl_by_doc.get(d, [])) for d in (entity.get("source_docs") or [])
        )))
        entity["descriptions"] = [
            {**d, "allow_access_control": doc_acl_by_doc.get(d["doc_id"], [])}
            for d in (entity.get("descriptions") or [])
        ]
        entity["aliases"] = [
            {**a, "allow_access_control": doc_acl_by_doc.get(a["doc_id"], [])}
            for a in (entity.get("aliases") or [])
        ]


# ---------------------------------------------------------------------------
# Step 1 — extraction
# ---------------------------------------------------------------------------

async def _extract_with_cache(extractor: ExtractionService, prepared_texts: list[str],
                               chunk_ids: list[str], doc_ids: Optional[list[str]] = None,
                               run_id: Optional[str] = None, index: Optional[str] = None,
                               mongo_collections: Optional[dict] = None
                               ) -> list[ChunkExtractionResult]:
    """Batch checkpoint load, then extract only the misses. Failed results are
    NOT cached — they retry next run.

    Failure ledger is attributed at doc_id granularity even though extraction
    fails per-chunk, since a retry re-fetches/re-processes at the document
    level — that's the unit --retry-failed needs. A doc with >=1 failed chunk
    gets one ledger entry for step "extraction"; a doc whose chunks all
    succeed has any stale entry cleared."""
    extraction_coll = _coll(mongo_collections, "extraction")
    keys = [f"{cid}:{EXTRACTION_VERSION}" for cid in chunk_ids]
    saved_map = load_checkpoints_batch(extraction_coll, keys)

    cached: dict = {}
    to_extract_texts, to_extract_ids = [], []
    for text, cid in zip(prepared_texts, chunk_ids):
        key = f"{cid}:{EXTRACTION_VERSION}"
        if key in saved_map:
            cached[cid] = ChunkExtractionResult(**saved_map[key])
        else:
            to_extract_texts.append(text)
            to_extract_ids.append(cid)

    logger.info("Step 1: %d chunk(s) from cache, %d to extract", len(cached), len(to_extract_ids))
    failed_chunks: list[str] = []

    if to_extract_ids:
        async def _checkpoint_as_completed(cid: str, result: ChunkExtractionResult) -> None:
            if result.failed:
                failed_chunks.append(cid)
                logger.warning("chunk %s: failed — skipping cache, will retry next run", cid)
                cached[cid] = ChunkExtractionResult()
                return
            await asyncio.to_thread(
                save_checkpoint, extraction_coll, f"{cid}:{EXTRACTION_VERSION}", result.model_dump()
            )
            cached[cid] = result

        await extractor.extract_chunks_batch(to_extract_texts, to_extract_ids,
                                             on_result=_checkpoint_as_completed)
        if failed_chunks:
            logger.warning("Step 1: %d chunk(s) failed — will retry next run: %s",
                           len(failed_chunks), failed_chunks[:10])

    if run_id and doc_ids:
        chunk_to_doc = dict(zip(chunk_ids, doc_ids))
        failed_docs = {chunk_to_doc[cid] for cid in failed_chunks if cid in chunk_to_doc}
        all_docs_in_batch = set(doc_ids)
        for doc_id in failed_docs:
            record_failure(run_id, doc_id, "extraction", "transient",
                           "one or more chunks failed extraction (see logs for chunk-level detail)",
                           index=index)
        for doc_id in all_docs_in_batch - failed_docs:
            clear_failure(run_id, doc_id, "extraction")

    return [cached.get(cid, ChunkExtractionResult()) for cid in chunk_ids]


# ---------------------------------------------------------------------------
# Step 2 — doc-level canonicalization
# ---------------------------------------------------------------------------

async def _canonicalize_document_with_cache(doc_id: str, chunk_ids: list[str],
                                            doc_results: list[ChunkExtractionResult],
                                            mongo_collections: Optional[dict] = None) -> None:
    """Canonicalize one document and persist both its canonical entities and the
    chunk_id list to Mongo (not separately) so Step 4 can recover both from a
    single read. Neither stays in memory after this call returns."""
    doc_canonical_coll = _coll(mongo_collections, "doc_canonical")
    key = f"{doc_id}:{DOC_CANON_VERSION}"
    # to_thread: pymongo is a blocking driver, and this now runs inside a gather
    # over the whole batch — a synchronous find_one here would serialize the very
    # documents _canonicalize_documents_in_batch is running concurrently. Same
    # treatment the save_checkpoint below already had.
    if await asyncio.to_thread(load_checkpoint, doc_canonical_coll, key) is not None:
        logger.info("Step 2 — doc %s: loaded from cache", doc_id)
        return  # already saved; Step 3/4 will load from Mongo when needed

    canonical_entities, merge_map = await canonicalize_document(doc_results)
    await asyncio.to_thread(
        save_checkpoint, doc_canonical_coll, key,
        {"canonical_entities": canonical_entities, "merge_map": merge_map, "chunk_ids": chunk_ids},
    )


async def _canonicalize_document_isolated(doc_id: str, chunk_ids: list[str],
                                          doc_results: list[ChunkExtractionResult],
                                          run_id: Optional[str], index: Optional[str],
                                          mongo_collections: Optional[dict] = None) -> bool:
    """Step 2 with per-document fault isolation: a single doc's canonicalization
    failure is recorded to the failure ledger and skipped rather than aborting
    the batch. Returns True if the doc succeeded (added to all_doc_ids for
    Step 3/4), False if it failed (excluded from this run's corpus;
    --retry-failed re-drives it later)."""
    try:
        await _canonicalize_document_with_cache(doc_id, chunk_ids, doc_results, mongo_collections)
    except Exception as exc:
        logger.exception("Step 2 — doc %s: canonicalization failed, skipping (isolated)", doc_id)
        if run_id:
            record_failure(run_id, doc_id, "doc_canonical", classify_error(exc), str(exc), index=index)
        return False
    if run_id:
        clear_failure(run_id, doc_id, "doc_canonical")
    return True


async def _canonicalize_documents_in_batch(results_by_doc: dict, chunk_ids_by_doc: dict,
                                           run_id: Optional[str], index: Optional[str],
                                           mongo_collections: Optional[dict] = None) -> list[str]:
    """Step 2 for every document in one batch, concurrently.

    Documents are independent at this stage — nothing one document's
    canonicalization does is visible to another's (cross-document merging is
    Step 3's job) — but they used to run in a plain sequential loop, so the
    batch's LLM calls went out roughly one at a time and left the shared
    AdaptiveThrottle almost entirely idle.

    Concurrency is bounded by settings.doc_canonicalization_concurrency rather
    than unbounded: the LLM calls underneath are already capped by the shared
    throttle, so a wider fan-out here would only queue work at that cap while
    holding every document's extraction results in memory at once, against
    this module's O(batch) memory contract.

    Returns the doc_ids that succeeded, in input order. Failures are already
    isolated and ledgered by _canonicalize_document_isolated, which never
    raises — so one bad document still can't take the batch down."""
    semaphore = asyncio.Semaphore(settings.doc_canonicalization_concurrency)
    doc_ids_in_batch = list(results_by_doc.keys())

    async def _one(doc_id: str) -> bool:
        async with semaphore:
            return await _canonicalize_document_isolated(
                doc_id, chunk_ids_by_doc[doc_id], results_by_doc[doc_id], run_id, index,
                mongo_collections=mongo_collections)

    outcomes = await asyncio.gather(*[_one(doc_id) for doc_id in doc_ids_in_batch])
    return [doc_id for doc_id, ok in zip(doc_ids_in_batch, outcomes) if ok]


# ---------------------------------------------------------------------------
# Step 3 — corpus-wide canonicalization (loads from Mongo, never holds all docs)
# ---------------------------------------------------------------------------

# Max names / docs per corpus_map shard. A single checkpoint packing every
# entity_name plus final_merge_map_by_doc exceeds Mongo's 16MB BSON limit at
# Confluence scale (~170k entities / ~11k docs) — DocumentTooLarge on the
# final Step 3 write. Layout mirrors edge_pool_map sharding.
_CORPUS_MAP_NAME_CHUNK_SIZE = 5000
_CORPUS_MAP_MERGE_DOC_CHUNK_SIZE = 250


def _entity_embed_text(name: str, meta: dict) -> str:
    """Same name[+description] text used when clustering produced the vector."""
    desc = meta.get("description")
    return f"{name}: {desc}" if desc else name


async def _ensure_entity_embeddings(final_entities: dict) -> None:
    """Fill any missing embeddings after a Mongo cache hit.

    Entity checkpoints normally persist vectors; this only backfills gaps
    (e.g. an older run that omitted them) so Step 5 never sees empty embeddings.
    """
    missing = [name for name, meta in final_entities.items() if not meta.get("embedding")]
    if not missing:
        return
    embedder = get_embedding_service()
    batch_size = (
        settings.embedding_inhouse_batch_size
        if settings.embedding_backend == "inhouse"
        else settings.embedding_azure_batch_size
    )
    logger.info(
        "Step 3: re-embedding %d cached entities missing vectors (batch=%d)",
        len(missing), batch_size,
    )
    for start in range(0, len(missing), batch_size):
        names = missing[start:start + batch_size]
        texts = [_entity_embed_text(n, final_entities[n]) for n in names]
        vectors = await asyncio.to_thread(embedder.embed_batch, texts)
        for name, vector in zip(names, vectors):
            final_entities[name]["embedding"] = vector
        done = min(start + batch_size, len(missing))
        if len(missing) >= batch_size:
            logger.info(
                "Step 3: re-embed progress %d/%d (%.1f%%)",
                done, len(missing), 100.0 * done / len(missing),
            )

def _corpus_map_entity_names_chunk_key(base_key: str, chunk_i: int) -> str:
    return f"{base_key}:entity_names:{chunk_i}"


def _corpus_map_merge_chunk_key(base_key: str, chunk_i: int) -> str:
    return f"{base_key}:merge_map:{chunk_i}"


def _load_corpus_map_payload(
    saved_map: dict, corpus_map_coll: str, base_key: str,
) -> Optional[tuple[list, dict]]:
    """Load entity_names + final_merge_map_by_doc from a legacy or chunked map.

    Legacy (pre-shard): root has both fields inline.
    Current: root has ``entity_name_chunk_count`` / ``merge_map_chunk_count``;
    slices live at ``{base_key}:entity_names:{i}`` and ``{base_key}:merge_map:{i}``.

    Returns None if a required chunk is missing so the caller rebuilds rather
    than warm-starting a half-written map.
    """
    if "entity_names" in saved_map and "final_merge_map_by_doc" in saved_map:
        return list(saved_map["entity_names"]), dict(saved_map["final_merge_map_by_doc"])

    name_chunk_count = int(saved_map.get("entity_name_chunk_count") or 0)
    merge_chunk_count = int(saved_map.get("merge_map_chunk_count") or 0)
    # A successful empty corpus still writes chunk counts of 0 with entity_count=0.
    if "entity_count" not in saved_map and name_chunk_count == 0 and merge_chunk_count == 0:
        return None

    entity_names: list = []
    for i in range(name_chunk_count):
        chunk = load_checkpoint(corpus_map_coll, _corpus_map_entity_names_chunk_key(base_key, i))
        if chunk is None:
            logger.warning(
                "Step 3: missing corpus_map entity_names chunk %d/%d for %s — cache miss",
                i, name_chunk_count, base_key,
            )
            return None
        entity_names.extend(chunk.get("entity_names") or [])

    final_merge_map_by_doc: dict = {}
    for i in range(merge_chunk_count):
        chunk = load_checkpoint(corpus_map_coll, _corpus_map_merge_chunk_key(base_key, i))
        if chunk is None:
            logger.warning(
                "Step 3: missing corpus_map merge_map chunk %d/%d for %s — cache miss",
                i, merge_chunk_count, base_key,
            )
            return None
        final_merge_map_by_doc.update(chunk.get("final_merge_map_by_doc") or {})

    return entity_names, final_merge_map_by_doc


def _save_corpus_map(
    corpus_map_coll: str,
    base_key: str,
    entity_names: list,
    final_merge_map_by_doc: dict,
) -> None:
    """Persist corpus_map without packing all names + merge maps into one BSON doc.

    Invalidate the root before rewriting shards so a crash mid-write cannot leave
    an old entity_name_chunk_count / merge_map_chunk_count pointing at a mix of
    old and new chunks. Warm-start then misses and rebuilds.
    """
    name_chunk_size = _CORPUS_MAP_NAME_CHUNK_SIZE
    merge_chunk_size = _CORPUS_MAP_MERGE_DOC_CHUNK_SIZE
    n_names = len(entity_names)
    name_chunk_count = (n_names + name_chunk_size - 1) // name_chunk_size if n_names else 0
    doc_items = list(final_merge_map_by_doc.items())
    n_docs = len(doc_items)
    merge_chunk_count = (n_docs + merge_chunk_size - 1) // merge_chunk_size 
    
    logger.info(
        "Step 3: saving corpus_map shards — %d entity names (%d chunks), "
        "%d merge-map docs (%d chunks)",
        n_names, name_chunk_count, n_docs, merge_chunk_count,
    )

    # Capture prior shard counts for optional trailing cleanup, then drop the
    # root BEFORE any chunk writes.
    old_map = load_checkpoint(corpus_map_coll, base_key)
    old_name_chunk_count = 0
    old_merge_chunk_count = 0
    if old_map is not None and not (
        "entity_names" in old_map and "final_merge_map_by_doc" in old_map
    ):
        old_name_chunk_count = int(old_map.get("entity_name_chunk_count") or 0)
        old_merge_chunk_count = int(old_map.get("merge_map_chunk_count") or 0)
    delete_checkpoint(corpus_map_coll, base_key)

    for i in range(name_chunk_count):
        start = i * name_chunk_size
        save_checkpoint(
            corpus_map_coll,
            _corpus_map_entity_names_chunk_key(base_key, i),
            {"entity_names": entity_names[start:start + name_chunk_size]},
        )
        if name_chunk_count > 1 and ((i + 1) % 5 == 0 or i + 1 == name_chunk_count):
            logger.info(
                "Step 3: corpus_map entity_names shards %d/%d",
                i + 1, name_chunk_count,
            )

    for i in range(merge_chunk_count):
        start = i * merge_chunk_size
        slice_items = doc_items[start:start + merge_chunk_size]
        save_checkpoint(
            corpus_map_coll,
            _corpus_map_merge_chunk_key(base_key, i),
            {"final_merge_map_by_doc": dict(slice_items)},
        )
        if merge_chunk_count > 1 and ((i + 1) % 5 == 0 or i + 1 == merge_chunk_count):
            logger.info(
                "Step 3: corpus_map merge_map shards %d/%d",
                i + 1, merge_chunk_count,
            )

    save_checkpoint(
        corpus_map_coll,
        base_key,
        {
            "entity_name_chunk_count": name_chunk_count,
            "merge_map_chunk_count": merge_chunk_count,
            "entity_count": n_names,
            "doc_count": n_docs,
        },
    )
    logger.info("Step 3: corpus_map root saved for %s", base_key)


    # Best-effort storage cleanup only — correctness does not depend on this.
    for i in range(name_chunk_count, old_name_chunk_count):
        delete_checkpoint(corpus_map_coll, _corpus_map_entity_names_chunk_key(base_key, i))
    for i in range(merge_chunk_count, old_merge_chunk_count):
        delete_checkpoint(corpus_map_coll, _corpus_map_merge_chunk_key(base_key, i))


async def _canonicalize_corpus_with_cache(all_doc_ids: list[str], tenant_id: str,
                                          materialized_persons: dict,
                                          mongo_collections: Optional[dict] = None) -> tuple[dict, dict]:
    """Step 3. Reads doc_canonical checkpoints from Mongo via a streaming
    cursor rather than an in-memory dict; each entity is stored as its own
    Mongo document to avoid the 16MB document-size limit at real corpus scale.

    PERSON entities skip graph_canonicalization_service.py's normal
    cross-document name clustering — its exact-name pass would otherwise
    collapse two different real people sharing a name. They're resolved by
    identity instead (services/person_corpus_resolver.py) and merged back
    into final_entities/final_merge_map_by_doc below; from Step 4 onward a
    person is just another entity, keyed by entity_id."""
    corpus_map_coll = _coll(mongo_collections, "corpus_map")
    corpus_entities_coll = _coll(mongo_collections, "corpus_entities")
    doc_canonical_coll = _coll(mongo_collections, "doc_canonical")
    base_key = f"{_corpus_key(all_doc_ids)}:{DOC_CANON_VERSION}:{GRAPH_CANON_VERSION}"
    saved_map = load_checkpoint(corpus_map_coll, base_key)
    if saved_map is not None:
        payload = _load_corpus_map_payload(saved_map, corpus_map_coll, base_key)
        if payload is not None:
            entity_names, final_merge_map_by_doc = payload
            final_entities: dict = {}
            # Batch-load entity docs — one find_one per name is multi-hour at
            # Confluence scale (~170k entities).
            batch_size = 1000
            n_names = len(entity_names)
            logger.info(
                "Step 3: cache hit — loading %d entity checkpoints from Mongo (batch=%d)",
                n_names, batch_size,
            )
            for start in range(0, len(entity_names), batch_size):
                batch_names = entity_names[start:start + batch_size]
                keys = [f"{base_key}:{name}" for name in batch_names]
                loaded = await asyncio.to_thread(load_checkpoints_batch, corpus_entities_coll, keys)
                for name, key in zip(batch_names, keys):
                    entity = loaded.get(key)
                    if entity is not None:
                        final_entities[name] = entity
                done = min(start + batch_size, n_names)
                if n_names >= batch_size and (done % (batch_size * 5) == 0 or done == n_names):
                    logger.info(
                        "Step 3: entity cache load progress %d/%d (%.1f%%, %d hits)",
                        done, n_names, 100.0 * done / n_names, len(final_entities),
                    )
            await _ensure_entity_embeddings(final_entities)
            logger.info("Step 3: loaded from cache (%d entities)", len(final_entities))
            return final_entities, final_merge_map_by_doc
        logger.warning("Step 3: incomplete corpus_map shards for %s — rebuilding", base_key)

    # Build canonical_by_doc by streaming from Mongo — no giant in-memory dict.
    canonical_entities_by_doc: dict = {}
    doc_merge_maps: dict = {}
    prefix = f":{DOC_CANON_VERSION}"
    logger.info(
        "Step 3: streaming doc_canonical checkpoints from Mongo for %d run doc(s)",
        len(all_doc_ids),
    )
    scanned = 0
    matched = 0
    for _key, data in iter_checkpoints(doc_canonical_coll, ""):
        scanned += 1
        if scanned % 2000 == 0:
            logger.info(
                "Step 3: doc_canonical scan progress — scanned %d checkpoint(s), matched %d/%d docs",
                scanned, matched, len(all_doc_ids),
            )
        if not _key.endswith(prefix):
            continue
        doc_id = _key[: -len(prefix)]
        if doc_id not in all_doc_ids:
            continue  # skip docs from a previous run that aren't in this run's set
        canonical_entities_by_doc[doc_id] = data["canonical_entities"]
        doc_merge_maps[doc_id] = data["merge_map"]
        matched += 1
    logger.info(
        "Step 3: doc_canonical load done — scanned %d checkpoint(s), matched %d/%d docs",
        scanned, matched, len(all_doc_ids),
    )

    non_person_entities_by_doc, person_final_entities, person_final_merge_map_by_doc = \
        await resolve_persons_for_corpus(canonical_entities_by_doc, materialized_persons, tenant_id)
    canonical_by_doc = {
        doc_id: (non_person_entities_by_doc.get(doc_id, {}), doc_merge_maps[doc_id])
        for doc_id in canonical_entities_by_doc
    }
    # Superseded by canonical_by_doc above and not read again — drop them before
    # the corpus-wide clustering pass below, which is this stage's own memory peak,
    # so it isn't stacked on top of these now-dead full-corpus copies too.
    del canonical_entities_by_doc, doc_merge_maps, non_person_entities_by_doc
    gc.collect()

    final_entities, final_merge_map_by_doc = await canonicalize_corpus(
        canonical_by_doc, cluster_checkpoint_key=base_key,
    )
    del canonical_by_doc

    # Merge PERSON's separately-resolved entities/mappings back in. Person
    # entries are keyed by entity_id, not display name, so they can never
    # collide with a name-keyed non-person entry.
    final_entities.update(person_final_entities)
    for doc_id, person_map in person_final_merge_map_by_doc.items():
        final_merge_map_by_doc.setdefault(doc_id, {}).update(person_map)
    del person_final_entities, person_final_merge_map_by_doc
    gc.collect()

    # Persist each entity as its own Mongo document (not one giant blob).
    # On resume after a DocumentTooLarge map failure, nearly all entity docs
    # are already present — skip rewrites so we don't re-pay ~6h of upserts.
    prefix_key = f"{base_key}:"
    already: set[str] = set()
    logger.info("Step 3: scanning existing entity checkpoints under %s*", prefix_key)
    scanned_existing = 0
    for _key, _data in iter_checkpoints(corpus_entities_coll, prefix_key):
        already.add(_key[len(prefix_key):])
        scanned_existing += 1
        if scanned_existing % 10000 == 0:
            logger.info(
                "Step 3: existing entity checkpoint scan — %d so far",
                scanned_existing,
            )
    logger.info(
        "Step 3: existing entity checkpoint scan done — %d already present",
        len(already),
    )
    to_write = [
        (f"{base_key}:{name}", data)
        for name, data in final_entities.items()
        if name not in already
    ]
    if to_write:
        logger.info(
            "Step 3: persisting %d/%d new entity checkpoints (%d already present; embeddings included)",
            len(to_write), len(final_entities), len(already),
        )
        await asyncio.to_thread(save_checkpoints_batch, corpus_entities_coll, to_write)
        logger.info("Step 3: entity checkpoint persist finished (%d docs)", len(to_write))
    else:
        logger.info(
            "Step 3: all %d entity checkpoints already present — skipping rewrite",
            len(final_entities),
        )
    logger.info("Step 3: writing corpus_map for %d entities / %d docs",
                len(final_entities), len(final_merge_map_by_doc))
    await asyncio.to_thread(
        _save_corpus_map,
        corpus_map_coll,
        base_key,
        list(final_entities.keys()),
        final_merge_map_by_doc,
    )
    logger.info("Step 3: corpus_map persist finished")
    return final_entities, final_merge_map_by_doc


# ---------------------------------------------------------------------------
# Step 4 — edge pool, streamed per-doc and written to Neo4j immediately
# ---------------------------------------------------------------------------

# How many documents one build_edge_pool call covers (settings.bulk_edge_pool_doc_batch).
# Larger = fewer calls and better within-batch edge merging; smaller = less peak
# memory holding extraction results. merge_edge_pools afterwards folds same keys
# across batches.

# Max edge_key records per edge_pool_map chunk document. A single Mongo checkpoint
# holding the full edge_keys list exceeds the 16MB BSON limit at real corpus
# scale (~100k edges). Chunking keeps each document small; the root map only
# stores counts. Same records as before — storage layout only.
_EDGE_POOL_MAP_CHUNK_SIZE = 2500


def _edge_pool_map_chunk_key(base_key: str, chunk_i: int) -> str:
    return f"{base_key}:chunk:{chunk_i}"


def _load_edge_key_records(saved_map: dict, edge_pool_map_coll: str, base_key: str) -> Optional[list]:
    """Flatten edge_keys from a legacy inline map or from chunked map docs.

    Legacy (pre-shard): root doc has ``edge_keys``.
    Current: root has ``edge_key_chunk_count``; each chunk is
    ``{base_key}:chunk:{i}`` with its own ``edge_keys`` slice.

    Returns None if a required chunk is missing so the caller can treat the
    whole-corpus cache as a miss and rebuild rather than warm-start half a pool.
    """
    if "edge_keys" in saved_map:
        return saved_map["edge_keys"]
    chunk_count = int(saved_map.get("edge_key_chunk_count") or 0)
    records: list = []
    for i in range(chunk_count):
        chunk = load_checkpoint(edge_pool_map_coll, _edge_pool_map_chunk_key(base_key, i))
        if chunk is None:
            logger.warning(
                "Step 4: missing edge_pool_map chunk %d/%d for %s — cache miss",
                i, chunk_count, base_key,
            )
            return None
        records.extend(chunk.get("edge_keys") or [])
    return records


def _save_edge_pool_map(edge_pool_map_coll: str, base_key: str,
                        edge_key_records: list, claim_count: int, mention_count: int) -> None:
    """Persist the edge-pool index without packing all edge_keys into one BSON doc.

    Invalidate the root before rewriting shards so a crash mid-write cannot leave
    an old edge_key_chunk_count pointing at a mix of old and new chunks.
    Warm-start then misses and rebuilds.
    """
    chunk_size = _EDGE_POOL_MAP_CHUNK_SIZE
    n = len(edge_key_records)
    chunk_count = (n + chunk_size - 1) // chunk_size if n else 0

    # Capture prior shard count for optional trailing cleanup, then drop the
    # root BEFORE any chunk writes.
    old_map = load_checkpoint(edge_pool_map_coll, base_key)
    old_chunk_count = 0
    if old_map is not None and "edge_keys" not in old_map:
        old_chunk_count = int(old_map.get("edge_key_chunk_count") or 0)
    delete_checkpoint(edge_pool_map_coll, base_key)

    for i in range(chunk_count):
        start = i * chunk_size
        save_checkpoint(
            edge_pool_map_coll,
            _edge_pool_map_chunk_key(base_key, i),
            {"edge_keys": edge_key_records[start:start + chunk_size]},
        )
    save_checkpoint(
        edge_pool_map_coll,
        base_key,
        {
            "edge_key_chunk_count": chunk_count,
            "edge_count": n,
            "claim_count": claim_count,
            "mention_count": mention_count,
        },
    )

    # Best-effort storage cleanup only — correctness does not depend on this.
    for i in range(chunk_count, old_chunk_count):
        delete_checkpoint(edge_pool_map_coll, _edge_pool_map_chunk_key(base_key, i))


async def _build_and_write_edges(
    all_doc_ids: list[str],
    final_merge_map_by_doc: dict,
    canon_to_id: dict,
    doc_acl_by_doc: dict,
    repo,
    tenant_id: str,
    run_id: Optional[str] = None,
    index: Optional[str] = None,
    doc_date_by_doc: Optional[dict] = None,
    mongo_collections: Optional[dict] = None,
) -> tuple[dict, list, list]:
    """Build the edge pool in batches of settings.bulk_edge_pool_doc_batch documents, writing
    each batch's edges/claims/mentions to Neo4j instead of accumulating the
    whole corpus in memory.

    canon_to_id: {final_canonical_name: entity_id}, computed once for the
    whole run before this is called and threaded through to build_edge_pool
    so every edge/claim/mention is keyed by entity_id, not name.

    On a cold run, reads each doc's extraction results and merge map from the
    Mongo doc_canonical checkpoint. Returns aggregated (final_edges,
    final_claims, mentions) for logging; the data has already been written to
    Neo4j by the time this returns.

    Fault isolation, under step "edge_pool": per-document while each doc's
    checkpoints are loaded, per-batch for the build itself (nothing in a batched
    build is attributable to one document). Either way the rest of the corpus's
    edge writes continue and --retry-failed re-drives what was ledgered."""
    edge_pool_edges_coll = _coll(mongo_collections, "edge_pool_edges")
    edge_pool_claims_coll = _coll(mongo_collections, "edge_pool_claims")
    edge_pool_mentions_coll = _coll(mongo_collections, "edge_pool_mentions")
    edge_pool_map_coll = _coll(mongo_collections, "edge_pool_map")
    doc_canonical_coll = _coll(mongo_collections, "doc_canonical")
    extraction_coll = _coll(mongo_collections, "extraction")
    base_key = (f"{_corpus_key(all_doc_ids)}:{DOC_CANON_VERSION}"
                f":{GRAPH_CANON_VERSION}:{EDGE_POOL_VERSION}")

    # Check whole-corpus cache first.
    saved_map = load_checkpoint(edge_pool_map_coll, base_key)
    if saved_map is not None:
        edge_key_records = _load_edge_key_records(saved_map, edge_pool_map_coll, base_key)
        if edge_key_records is not None:
            edges: dict = {}
            for edge_rec in edge_key_records:
                # Support both legacy "|"-joined string keys and the current dict format.
                if isinstance(edge_rec, dict):
                    edge_cache_key = edge_rec["key"]
                    edge_tuple = tuple(edge_rec["tuple"])
                else:
                    edge_cache_key = edge_rec
                    edge_tuple = tuple(edge_rec.split("|", 2))
                data = load_checkpoint(edge_pool_edges_coll, f"{base_key}:{edge_cache_key}")
                if data is not None:
                    edges[edge_tuple] = data
            claims = [load_checkpoint(edge_pool_claims_coll, f"{base_key}:{i}")
                      for i in range(saved_map["claim_count"])]
            claims = [c for c in claims if c is not None]
            mentions_raw = [load_checkpoint(edge_pool_mentions_coll, f"{base_key}:{i}")
                            for i in range(saved_map["mention_count"])]
            mentions = [tuple(m["mention"]) for m in mentions_raw if m is not None]
            logger.info("Step 4: loaded from cache (%d edges, %d claims, %d mentions)",
                        len(edges), len(claims), len(mentions))
            return edges, claims, mentions

    all_edges: dict = {}
    all_claims: list = []
    all_mentions: list = []

    # Neo4j writes flush every settings.bulk_write_flush_threshold rows
    # (edges+claims+mentions combined) rather than once per doc, to keep HTTP
    # round trips to Neo4j bounded at real corpus scale. all_edges/all_claims/
    # all_mentions are still accumulated for the whole corpus (needed for the
    # end-of-run checkpoint and return value).
    pending_edges: dict = {}
    pending_claims: list = []
    pending_mentions: list = []
    pending_doc_ids: set = set()  # which docs' data is in the pending batch below

    async def _flush_pending() -> None:
        """Catches failures here so one bad row in a batched UNWIND write
        doesn't abort the whole run. Isolation is at flush-batch granularity,
        not per-row — Neo4j doesn't report which row in a batch failed, so
        every doc_id contributing to this batch is recorded as failed
        together (coarser than Step 1/2's per-doc isolation, but every other
        batch keeps going, and --retry-failed re-drives these doc_ids)."""
        if not repo or not (pending_edges or pending_claims or pending_mentions):
            return
        doc_ids_in_batch = list(pending_doc_ids)
        try:
            await asyncio.to_thread(repo.write_relationships, pending_edges, tenant_id)
            await asyncio.to_thread(repo.write_claims, pending_claims, tenant_id)
            await asyncio.to_thread(repo.write_mentions, pending_mentions, tenant_id)
        except Exception as exc:
            logger.exception("Step 4: Neo4j write batch failed (%d edge(s), %d claim(s), "
                             "%d mention(s), %d contributing doc(s)) — recording and continuing, "
                             "not aborting the run", len(pending_edges), len(pending_claims),
                             len(pending_mentions), len(doc_ids_in_batch))
            if run_id:
                for doc_id in doc_ids_in_batch:
                    record_failure(run_id, doc_id, "neo4j_write", classify_error(exc), str(exc), index=index)
        else:
            if run_id:
                for doc_id in doc_ids_in_batch:
                    clear_failure(run_id, doc_id, "neo4j_write")
        pending_edges.clear()
        pending_claims.clear()
        pending_mentions.clear()
        pending_doc_ids.clear()

    for start in range(0, len(all_doc_ids), settings.bulk_edge_pool_doc_batch):
        doc_id_batch = all_doc_ids[start:start + settings.bulk_edge_pool_doc_batch]

        # Load phase, isolated per document: a missing/corrupt checkpoint or an
        # invalid ChunkExtractionResult is ledgered and dropped here, so the build
        # below only sees documents that loaded cleanly.
        results_by_doc: dict = {}
        chunk_ids_by_doc: dict = {}
        doc_merge_maps: dict = {}
        final_maps: dict = {}
        acls_by_doc: dict = {}
        dates_by_doc: dict = {}
        for doc_id in doc_id_batch:
            canon_data = load_checkpoint(doc_canonical_coll, f"{doc_id}:{DOC_CANON_VERSION}")
            if canon_data is None:
                logger.warning("Step 4: no doc_canonical checkpoint for %s — skipping", doc_id)
                continue
            try:
                chunk_ids = canon_data.get("chunk_ids", [])
                # Batch-load this doc's extraction results in one Mongo query.
                ext_keys = [f"{cid}:{EXTRACTION_VERSION}" for cid in chunk_ids]
                ext_saved = load_checkpoints_batch(extraction_coll, ext_keys)
                results_by_doc[doc_id] = [
                    ChunkExtractionResult(**ext_saved[f"{cid}:{EXTRACTION_VERSION}"])
                    if f"{cid}:{EXTRACTION_VERSION}" in ext_saved else ChunkExtractionResult()
                    for cid in chunk_ids
                ]
            except Exception as exc:
                logger.exception("Step 4 — doc %s: loading extraction results failed, skipping "
                                 "(isolated)", doc_id)
                if run_id:
                    record_failure(run_id, doc_id, "edge_pool", classify_error(exc), str(exc), index=index)
                continue
            chunk_ids_by_doc[doc_id] = chunk_ids
            doc_merge_maps[doc_id] = canon_data.get("merge_map", {})
            final_maps[doc_id] = final_merge_map_by_doc.get(doc_id, {})
            acls_by_doc[doc_id] = doc_acl_by_doc.get(doc_id, [])
            dates_by_doc[doc_id] = (doc_date_by_doc or {}).get(doc_id, "")

        if not results_by_doc:
            continue

        # ONE call for the whole batch: an edge key is corpus-wide, so two documents
        # routinely produce the same one and build_edge_pool folds their
        # descriptions/chunk_ids/raw_types/acl together itself. Isolation is
        # batch-granular here — same reasoning as _flush_pending's writes.
        try:
            batch_edges, batch_claims, batch_mentions = build_edge_pool(
                results_by_doc, chunk_ids_by_doc, doc_merge_maps, final_maps,
                canon_to_id, acls_by_doc, dates_by_doc,
            )
        except Exception as exc:
            logger.exception("Step 4: edge pool build failed for a batch of %d doc(s) — recording "
                             "and continuing, not aborting the run", len(results_by_doc))
            if run_id:
                for doc_id in results_by_doc:
                    record_failure(run_id, doc_id, "edge_pool", classify_error(exc), str(exc), index=index)
            continue
        if run_id:
            for doc_id in results_by_doc:
                clear_failure(run_id, doc_id, "edge_pool")

        # merge_edge_pools, not dict.update: the same edge key can come back from a
        # later batch, and update() would keep only that last batch's data. It also
        # copies the list fields on first insert, so merging the same batch_edges
        # into both pending_edges and all_edges can't leave them sharing one object
        # (the PR #31 double-count bug).
        merge_edge_pools(pending_edges, batch_edges)
        pending_claims.extend(batch_claims)
        pending_mentions.extend(batch_mentions)
        pending_doc_ids.update(results_by_doc)
        merge_edge_pools(all_edges, batch_edges)
        all_claims.extend(batch_claims)
        all_mentions.extend(batch_mentions)

        if len(pending_edges) + len(pending_claims) + len(pending_mentions) >= settings.bulk_write_flush_threshold:
            await _flush_pending()

    await _flush_pending()  # final partial batch

    # Persist edge pool for future warm starts. Edge keys are stored as
    # [src, rel_type, tgt] lists, not "|"-joined strings — a joined key would
    # corrupt on round-trip when entity names contain "|". The index itself is
    # chunked across edge_pool_map docs (see _save_edge_pool_map) so a corpus
    # with ~100k edges never hits Mongo's 16MB BSON limit on one document.
    #
    # Payload docs are bulk_written (and the three collections flushed in
    # parallel): serial save_checkpoint was ~15–22 upserts/s and took ~6h for
    # a production-scale pool. Map is written only after all payloads succeed.
    edge_key_records = []
    edge_items: list[tuple[str, dict]] = []
    for (src, rel_type, tgt), data in all_edges.items():
        edge_cache_key = hashlib.sha1(f"{src}\x00{rel_type}\x00{tgt}".encode()).hexdigest()
        edge_key_records.append({"key": edge_cache_key, "tuple": [src, rel_type, tgt]})
        edge_items.append((f"{base_key}:{edge_cache_key}", data))
    claim_items = [(f"{base_key}:{i}", claim) for i, claim in enumerate(all_claims)]
    mention_items = [
        (f"{base_key}:{i}", {"mention": list(mention)})
        for i, mention in enumerate(all_mentions)
    ]

    persist_jobs = (
        (edge_pool_edges_coll, edge_items, "edges"),
        (edge_pool_claims_coll, claim_items, "claims"),
        (edge_pool_mentions_coll, mention_items, "mentions"),
    )
    logger.info(
        "Step 4: bulk-persisting edge pool (%d edges, %d claims, %d mentions)",
        len(edge_items), len(claim_items), len(mention_items),
    )
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="edge-pool-ckpt") as pool:
        futures = {
            pool.submit(save_checkpoints_batch, coll, items): label
            for coll, items, label in persist_jobs
            if items
        }
        for fut in as_completed(futures):
            label = futures[fut]
            fut.result()  # raise on first failure
            logger.info("Step 4: finished bulk-persist of %s", label)

    _save_edge_pool_map(
        edge_pool_map_coll, base_key, edge_key_records,
        claim_count=len(all_claims), mention_count=len(all_mentions),
    )

    return all_edges, all_claims, all_mentions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_bulk_async(index: Optional[str] = None, query: Optional[dict] = None,
                         limit: Optional[int] = None,
                         batch_size: int = settings.bulk_batch_initial_chunks,
                         tenant_id: str = "default", connector_type: str = "default",
                         run_id: Optional[str] = None,
                         spec: Optional[ConnectorSpec] = None) -> tuple:
    """Single top-level coroutine — called via asyncio.run() in entrypoint.py only.
    Memory stays O(batch_size chunks) throughout (see module docstring).

    spec: the connector this run reads. Everything source-specific — the index
    default, the query, what a document is, who can see it — comes from here,
    and nothing in this module knows which connector it got. Required: there is
    no default, because a run that cannot name its source must stop rather than
    silently read as Zeus (core/entrypoint.py resolves it from
    --connector/BULK_CONNECTOR and passes it in).

    run_id: pass an existing one to resume that run's own stores (must
    already be registered); omit to mint a brand-new isolated run_id + store
    set. --retry-failed needs the run_id copied from the logs — it's no
    longer reconstructable from --index/--tenant-id alone."""
    setup_logging()
    if spec is None:
        raise ValueError(
            "run_bulk_async requires an explicit connector spec — there is no "
            "default. Resolve it via connectors.registry.get_connector and pass "
            "it in (core/entrypoint.py does this from --connector/BULK_CONNECTOR)."
        )
    if not index:
        # No silent default anymore — see core/entrypoint.py's --index/BULK_INDEX
        # requirement. Reaching this with no index means a caller bypassed that
        # check (e.g. a new script calling run_bulk_async directly).
        raise ValueError("run_bulk_async requires an explicit index — there is no default")
    query = query or spec.build_query(None)
    run_id, run_ctx = _resolve_run_context(run_id, index, tenant_id, connector_type)
    set_log_context(run_ctx.tenant_id, run_ctx.connector_type, run_ctx.run_suffix)

    # Reset LLM-call counters so this run's summary reflects only its own calls,
    # not a prior run_bulk_async invocation in the same process (e.g. a
    # run_retry_failed_async call).
    reset_run_metrics()

    logger.info("=== Bulk run starting: run_id=%s index=%s connector_type=%s run_suffix=%s "
                "neo4j_database=%s initial_batch_size=%d chunks ===",
                run_id, index, run_ctx.connector_type, run_ctx.run_suffix, run_ctx.neo4j_database, batch_size)

    # Opaque to this module: an Elasticsearch client for Zeus/Confluence, an
    # HTTP session for an API-backed connector, whatever this source needs.
    # Nothing here imports or knows about any specific transport.
    transport = spec.open_transport()
    # One startup assertion instead of a per-chunk mode ladder: when this run
    # consumes upstream vectors, the index must already be in the run's space.
    # Only meaningful for a connector that embeds documents upstream of this
    # pipeline in the first place; the default is off.
    if run_ctx.run_config.consume_upstream_vector:
        _assert_upstream_vector_space(transport, index, query, run_ctx.run_config)
    extractor = ExtractionService()

    # Whatever this connector needs loaded once per run before the first hit —
    # for the intranet, the ABAC audience map. Deliberately NOT wrapped in
    # try/except: a connector that cannot prepare must abort the run rather than
    # fall back to a permissive default.
    spec.bind_run(transport, index, tenant_id=tenant_id)

    batch_sizer = AdaptiveBatchSizer(initial=batch_size)

    # Every run writes — there is no compute-only mode anymore (PR-41 review
    # point 6). The stores are this run's own isolated ones (run_ctx), never the
    # shared settings defaults.
    from repository.neo4j_repository import Neo4jRepository, ensure_database
    ensure_database(run_ctx.neo4j_database)
    repo = Neo4jRepository(database=run_ctx.neo4j_database)
    repo.create_constraints()
    # ONE embedding space per run: both collections are created at the run's
    # recorded dimension (run_ctx.run_config, not live settings, so a RESUMED
    # run keeps the space it was registered with). Entities and chunks stay two
    # separate collections — same space, different payloads.
    vector_store = VectorStore(collection=run_ctx.qdrant_entities_collection)
    vector_store.ensure_collection(dimensions=run_ctx.run_config.embedding_dimension)
    chunks_vector_store = VectorStore(collection=run_ctx.qdrant_chunks_collection)
    chunks_vector_store.ensure_collection(dimensions=run_ctx.run_config.embedding_dimension)

    all_doc_ids: list[str] = []  # doc IDs only — lightweight, held for Step 3's cache key
    doc_acl_by_doc: dict = {}  # {doc_id: [acl_token, ...]} — lightweight, held for the whole run
    doc_date_by_doc: dict = {}  # {doc_id: "YYYY-MM-DD"} — source for Step 4's edge documented_on
    total_docs = total_entities = total_rels = total_claims = total_dropped = 0

    # Re-reads batch_sizer.current_size before each page fetch, so later pages
    # pick up size adjustments made after earlier batches.
    # --limit is carried into the fetch request rather than applied after the
    # fact. The old shape fetched a fixed page, normalised all of it, and threw
    # away everything past the limit — so a smoke test on 5 documents still
    # built chunks for a hundred, and a malformed document nobody asked for
    # could abort the run. Asking for what we can use is both cheaper and narrower.
    def _docs_wanted() -> Optional[int]:
        return None if limit is None else max(0, limit - total_docs)

    doc_gen = spec.fetch(transport, index, query=query, batch_size=batch_sizer.current_size,
                         batch_sizer=batch_sizer, docs_wanted=_docs_wanted)

    for batch_num, (doc_batch, batch_chunk_count) in enumerate(doc_gen, 1):
        if limit is not None:
            remaining = limit - total_docs
            if remaining <= 0:
                break
            doc_batch = doc_batch[:remaining]

        # ACL: decided once per document by its connector, before anything else
        # touches it. A dropped document (deleted, site deleted, not published,
        # or no derivable token) never enters the graph — excluded before
        # extraction/write.
        kept_batch = []
        for doc in doc_batch:
            if doc.acl.dropped:
                total_dropped += 1
                logger.info("Batch %d: dropping doc %s (%s)", batch_num, doc.doc_id, doc.acl.reason)
                continue
            doc_acl_by_doc[doc.doc_id] = doc.acl.tokens
            doc_date_by_doc[doc.doc_id] = doc.published_at
            kept_batch.append(doc)
        doc_batch = kept_batch

        if doc_batch:
            doc_batch = await run_document_preprocess(spec, doc_batch)

        spec.people.observe(doc_batch)

        total_docs += len(doc_batch)
        batch_start = time.monotonic()

        if repo:
            await asyncio.to_thread(repo.write_documents, doc_batch, tenant_id, doc_acl_by_doc)
            await asyncio.to_thread(repo.write_chunks, doc_batch, tenant_id, doc_acl_by_doc,
                                    chunks_vector_store, get_embedding_service())

        prepared_texts, chunk_ids, doc_ids = [], [], []
        for doc in doc_batch:
            # doc.chunks is already deduped by recursive_chunk_id — without that,
            # the upstream chunker's overlapping windows cause a multi-chunk
            # document to get extracted several times over.
            for chunk in doc.chunks:
                prepared_texts.append(spec.prepare_chunk_text(doc, chunk))
                chunk_ids.append(chunk.chunk_id)
                doc_ids.append(doc.doc_id)

        # Step 1 — extraction.
        results = await _extract_with_cache(extractor, prepared_texts, chunk_ids, doc_ids,
                                            run_id=run_id, index=index,
                                            mongo_collections=run_ctx.mongo_collections)
        total_entities += sum(len(r.entities) for r in results)
        total_rels += sum(len(r.relationships) for r in results)
        total_claims += sum(len(r.claims) for r in results)

        # Step 2 — doc canonicalization. Per-doc fault isolation: a doc whose
        # canonicalization fails is recorded to the failure ledger and excluded
        # from all_doc_ids (so Steps 3/4 never see it) rather than aborting the
        # whole batch — --retry-failed re-drives it later.
        batch_results_by_doc: dict = {}
        batch_chunk_ids_by_doc: dict = {}
        for doc_id, cid, result in zip(doc_ids, chunk_ids, results):
            batch_results_by_doc.setdefault(doc_id, []).append(result)
            batch_chunk_ids_by_doc.setdefault(doc_id, []).append(cid)

        for doc_id in await _canonicalize_documents_in_batch(
                batch_results_by_doc, batch_chunk_ids_by_doc, run_id, index,
                mongo_collections=run_ctx.mongo_collections):
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)

        # Report wall-clock time so the sizer can adjust the next batch's chunk budget.
        batch_elapsed = time.monotonic() - batch_start
        batch_sizer.record_batch(batch_elapsed, len(prepared_texts))

        logger.info("Batch %d done: %d doc(s), %d chunk(s), %.1fs | next budget=%d chunks",
                    batch_num, len(doc_batch), len(prepared_texts),
                    batch_elapsed, batch_sizer.current_size)
        if limit is not None and total_docs >= limit:
            break

    logger.info("=== Steps 0-2 complete: %d docs, %d raw entities, %d rels, %d claims (%d dropped by ACL) ===",
                total_docs, total_entities, total_rels, total_claims, total_dropped)

    await run_preprocessing_drain(spec)

    person_records = await spec.people.discover(PeopleContext(tenant_id=tenant_id))
    materialized_persons = await asyncio.to_thread(
        materialize_persons, person_records, spec.connector_type.value, tenant_id,
    )

    # Step 3 — corpus canonicalization: a single clustering pass over every doc at
    # once, so failures here are NOT per-doc isolatable. Record under the
    # "__corpus__" sentinel and re-raise — this is a genuinely global error and
    # deliberately aborts the whole run, unlike Steps 1/2/4's per-doc isolation.
    try:
        final_entities, final_merge_map_by_doc = await _canonicalize_corpus_with_cache(
            all_doc_ids, tenant_id, materialized_persons, mongo_collections=run_ctx.mongo_collections,
        )
    except Exception as exc:
        logger.exception("Step 3: corpus canonicalization failed — aborting run")
        record_failure(run_id, "__corpus__", "corpus_canonicalization", classify_error(exc), str(exc),
                       index=index)
        mark_run_failed(run_id)
        raise
    clear_failure(run_id, "__corpus__", "corpus_canonicalization")

    # Resolve/mint each final canonical name's stable entity_id via the Mongo-backed
    # registry (repository/entity_registry.py) BEFORE any Neo4j write — a pure Mongo
    # lookup/mint that doesn't depend on Neo4j state, so both this bulk path and the
    # incremental path mint through the exact same registry.
    canon_to_id = await resolve_entity_ids(final_entities, tenant_id) if repo else {}
    # PERSON entries were already resolved by identity in Step 3
    # (services/person_corpus_resolver.py) and keyed by their own entity_id;
    # resolve_entity_ids skips them, so this is a trivial identity mapping.
    for key, meta in final_entities.items():
        if meta.get("type") == "PERSON":
            canon_to_id[key] = key

    # Step 5 — entities must be written BEFORE edges/claims/mentions: Step 4's writes
    # MATCH entities by entity_id, so those Entity nodes must already exist in Neo4j.
    # Writing edges first silently produces zero edges/claims/mentions on a genuinely
    # first-ever run (no prior entities in the graph to match against).
    if repo:
        try:
            _tag_entity_acls(final_entities, doc_acl_by_doc)
            await asyncio.to_thread(repo.write_entities, final_entities, tenant_id, canon_to_id, vector_store)
        except Exception as exc:
            logger.exception("Step 5: entity write failed — aborting run")
            record_failure(run_id, "__corpus__", "entity_write", classify_error(exc), str(exc), index=index)
            mark_run_failed(run_id)
            repo.close()
            raise
        clear_failure(run_id, "__corpus__", "entity_write")

    # Step 4 — edge pool. Processes a batch of docs per call, writes to Neo4j as it goes.
    final_edges, final_claims, mentions = await _build_and_write_edges(
        all_doc_ids, final_merge_map_by_doc, canon_to_id, doc_acl_by_doc, repo, tenant_id, run_id, index,
        doc_date_by_doc, mongo_collections=run_ctx.mongo_collections,
    )
    if repo:
        # ENRICH — the third people stage, and whatever this connector does
        # about the humans in its graph beyond the shared shape MATERIALIZE
        # already gave them. The intranet materializes the HR org chart it
        # ships; a source with no org chart implements nothing here and its
        # graph is simply thinner. Which of those happens is not core's business
        # — it hands over the graph handles and this run's entity scope, and the
        # provider owns the rest, including its own feature flags and its own
        # best-effort boundaries. _retire_merged_person_mentions is passed in
        # rather than imported so a provider never reaches back into core/.
        people_ctx = PeopleContext(
            repo=repo, tenant_id=tenant_id, entity_ids=list(canon_to_id.values()),
            vector_store=vector_store, retire_merged_mentions=_retire_merged_person_mentions,
            final_entities=final_entities, canon_to_id=canon_to_id,
        )
        try:
            await spec.people.enrich(person_records, people_ctx)
        except Exception:
            logger.exception("Connector people ENRICH failed — non-fatal")
        # Degree only needs recomputing for entities this run's edges could have
        # touched, not the whole tenant's graph.
        #
        # Runs AFTER the connector's people phase: a provider may delete
        # EXTRACTED_REL edges (Zeus's merge_person_mentions and
        # retire_person_manages both do), so an earlier degree would already be
        # stale. Nothing below here touches EXTRACTED_REL.
        try:
            await asyncio.to_thread(repo.recompute_degree, tenant_id, list(canon_to_id.values()))
        except Exception:
            logger.exception("Degree recompute failed — non-fatal, graph write already succeeded")
        # Claim.support: whole-tenant, not scoped to this run's claims — write_claims
        # doesn't surface claim_id for the bulk path (final_claims are pre-hash), and a
        # bulk run is exactly recompute_claim_support's "fresh full tenant" case anyway.
        try:
            await asyncio.to_thread(repo.recompute_claim_support, tenant_id, None)
        except Exception:
            logger.exception("Claim support recompute failed — non-fatal, graph write already succeeded")
        if settings.build_authorship_edges:
            # Deterministic (no LLM), scoped to this run's own documents; depends on
            # the :Person nodes Step 5's write created from MATERIALIZE's
            # output, and resolves them by (connector_type, source_key).
            try:
                await asyncio.to_thread(
                    repo.materialize_authorship_edges, tenant_id, all_doc_ids,
                    settings.contributor_entity_types,
                )
            except Exception:
                logger.exception("Authorship-edge materialization failed — non-fatal")
        repo.close()

    logger.info("=== Run complete: %d docs -> %d final entities, %d edges, %d claims, %d mentions ===",
                total_docs, len(final_entities), len(final_edges), len(final_claims), len(mentions))

    logger.info("Done — %d docs -> %d final entities, %d edges, %d claims (written to Neo4j)",
                total_docs, len(final_entities), len(final_edges), len(final_claims))
    failure_count = len(list_failures(run_id))
    logger.info("=== Run metrics: run_id=%s LLM calls=%s | %d unresolved failure(s) in the ledger ===",
                run_id, run_metrics_snapshot(), failure_count)
    mark_run_completed(run_id)
    return final_entities, final_edges, final_claims


async def run_retry_failed_async(run_id: str, index: Optional[str] = None,
                                 tenant_id: str = "default",
                                 spec: Optional[ConnectorSpec] = None) -> Optional[tuple]:
    """--retry-failed <run_id>: re-drive only the documents this run_id's ledger
    says failed, instead of re-running the whole corpus. Reuses run_bulk_async
    unchanged — same run_id means each step's success clears the matching
    ledger entry, and a repeat failure just overwrites the existing record.
    run_id must already be registered in run_registry (an earlier
    run_bulk_async call produced it) so it resumes the same stores.

    Corpus-level failures (the "__corpus__" sentinel — Step 3's clustering or
    Step 5's batched entity write) aren't tied to specific doc_ids, so a
    targeted query can't isolate them; if that's ALL that's ledgered, we say
    so and stop. A full re-run with the same run_id is cheap in that case
    since checkpointing skips already-succeeded docs."""
    failures = list_failures(run_id)
    if not failures:
        logger.info("--retry-failed %s: no ledgered failures found — nothing to do", run_id)
        return None

    doc_ids = sorted({f["item_id"] for f in failures if f["item_id"] != "__corpus__"})
    corpus_level = [f for f in failures if f["item_id"] == "__corpus__"]

    if not doc_ids:
        logger.warning(
            "--retry-failed %s: %d corpus-level failure(s) only (step(s): %s) — not attributable "
            "to specific documents, so this targeted retry can't isolate them. Call "
            "run_bulk_async(run_id=%r, ...) directly to resume this run's corpus-wide steps "
            "(checkpoint cache still applies). A plain re-invocation without this run_id mints a "
            "brand-new isolated run and recomputes everything.",
            run_id, len(corpus_level), sorted({f["step"] for f in corpus_level}), run_id,
        )
        return None

    if corpus_level:
        logger.warning(
            "--retry-failed %s: re-driving %d document-level failure(s); %d corpus-level "
            "failure(s) (step(s): %s) are NOT addressed by this targeted retry — re-run the "
            "full bulk job for those.",
            run_id, len(doc_ids), len(corpus_level), sorted({f["step"] for f in corpus_level}),
        )
    else:
        logger.info("--retry-failed %s: re-driving %d document-level failure(s): %s",
                    run_id, len(doc_ids), doc_ids[:10])

    if spec is None:
        raise ValueError(
            "run_retry_failed_async requires an explicit connector spec — there is "
            "no default. core/entrypoint.py passes the one it resolved from "
            "--connector/BULK_CONNECTOR."
        )
    retry_index = index or next((f.get("index") for f in failures if f.get("index")), None)
    if not retry_index:
        raise ValueError(
            f"--retry-failed {run_id!r}: no index to re-drive against — none was passed "
            "and none of the ledgered failures recorded one. Pass --index explicitly."
        )
    retry_query = spec.build_query(doc_ids)
    return await run_bulk_async(index=retry_index, query=retry_query,
                                tenant_id=tenant_id, run_id=run_id, spec=spec)
