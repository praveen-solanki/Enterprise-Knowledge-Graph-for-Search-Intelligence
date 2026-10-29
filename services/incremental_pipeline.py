"""Per-document incremental processing for the Kafka worker — the equivalent
of core/pipeline.py's per-batch loop, but for one document at a time. Adds
entity resolution against the live graph and a watermark to drop
stale/duplicate events.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from connectors.base import Chunk, ConnectorType, PeopleContext, SourceDocument, resolve_first_present, run_document_preprocess
from connectors.chunking import dedup_chunks, upstream_chunk_vector
from connectors.registry import get_connector
from core.pipeline import _extract_with_cache, _retire_merged_person_mentions, _tag_entity_acls
from repository.mongo_repository import (
    increment_rolling_window_count,
    load_checkpoint,
    load_checkpoints_batch,
    save_checkpoint,
)
from repository.neo4j_repository import Neo4jRepository
from services.acl import AclContext, AclDecision, build_document_acl
from services.edge_pool_service import build_edge_pool
from services.embedding_service import EmbeddingService
from services.extraction_service import EXTRACTION_VERSION, ExtractionService
from services.incremental_canonicalization import resolve_canonical_name
from services.incremental_event_parser import GraphExtractEvent
from services.llm_backend import effective_run_config
from services.people import materialize_persons
from services.run_metrics import snapshot as run_metrics_snapshot
from services.vector_store import VectorStore
from utils import flow_log

logger = logging.getLogger("graphrag.services.incremental_pipeline")

_WATERMARK_COLLECTION = "incremental_watermark"
_MASS_RETRACTION_COLLECTION = "mass_retraction_window"

# Bounds concurrent entity-resolution calls (settings.incremental_entity_resolve_concurrency)
# — resolve_canonical_name isn't rate-limited by an outer throttle, so an unbounded gather
# could fire hundreds of concurrent Azure calls.


def _is_stale(tenant_id: str, doc_id: str, completed_at: int) -> bool:
    saved = load_checkpoint(_WATERMARK_COLLECTION, f"{tenant_id}:{doc_id}")
    return saved is not None and completed_at <= saved.get("completed_at", 0)


def _advance_watermark(tenant_id: str, doc_id: str, completed_at: int) -> None:
    save_checkpoint(_WATERMARK_COLLECTION, f"{tenant_id}:{doc_id}", {"completed_at": completed_at})


def _iso_to_epoch_ms(raw: str) -> Optional[int]:
    """Parses to epoch MILLISECONDS (matching completed_at's convention) — datetime.timestamp()
    returns seconds, so it must be multiplied by 1000, or every comparison against a saved
    watermark looks falsely older and silently stops updating an already-touched document.
    Returns None (not 0) on an unparseable string, so the caller can fall through to its
    next source instead of trusting a bad value."""
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, AttributeError):
        return None


def resolve_watermark(event: GraphExtractEvent, doc: Optional[dict], es_doc: Optional[dict]) -> int:
    """The value _is_stale/_advance_watermark compare on. event.completed_at is stamped at
    Kafka publish time and gets RE-STAMPED on retry, so it can make an older edit look newer
    than one that published cleanly the first time — prefer the document's own real edit
    time instead: doc["doc_updated_at"] (from the payload, no ES round-trip needed), then
    es_doc["lastmodifieddate"] (live ES lookup), falling back to event.completed_at only
    when neither is available (e.g. deletes have no es_doc)."""
    if doc:
        parsed = _iso_to_epoch_ms(doc.get("doc_updated_at") or "")
        if parsed is not None:
            return parsed
    if es_doc:
        raw = es_doc.get("lastmodifieddate")
        if raw:
            parsed = _iso_to_epoch_ms(raw)
            if parsed is not None:
                return parsed
            logger.warning("doc_id=%s: lastmodifieddate=%r is not a valid ISO date — "
                           "falling back to the event's own completed_at",
                           es_doc.get("id"), raw)
    return event.completed_at


def _extract_chunks(doc: dict) -> list[dict]:
    """Dedups doc["_chunks"]["v1"] the same way the bulk path does (dedup_chunks) — the
    upstream chunker's overlapping windows share a recursive_chunk_id, so without this a
    document can get re-extracted several times over."""
    return dedup_chunks(doc)


def _connector_type(connector_type: str) -> ConnectorType:
    """The event's own connector_type when it is one this codebase names. The
    producer's value is free-form and predates the enum — legacy product
    spellings appear on live events — and anything unrecognised is the intranet, which is
    what this path has always served. What lands on the Document node is
    unchanged by this: an event naming a connector still records that connector.
    """
    try:
        return ConnectorType(connector_type)
    except ValueError:
        return ConnectorType.WIKIPEDIA


def _resolve_spec(connector_type: str):
    """Whose field names to read the live ES document under, and whose framing
    to give a chunk. A type named in the enum but with no spec registered here
    (see connectors/registry.py) has nothing to ask, so the intranet's field map
    is used — which is what this path used for every event before the boundary
    existed."""
    try:
        return get_connector(_connector_type(connector_type))
    except ValueError:
        return get_connector(ConnectorType.WIKIPEDIA)


def _to_source_document(doc: dict, chunks: list[dict], connector_type: str,
                        acl: AclDecision, es_doc: Optional[dict] = None) -> SourceDocument:
    """One incremental payload document as the contract sees it.

    This used to be `_adapt_for_bulk_write`, and it did exactly the thing the
    connector boundary exists to stop: it fabricated a Zeus-shaped dict —
    `content_title`, `content_publish_on_date`, `createddate` — so the bulk
    write methods could be reused. Now those methods take SourceDocuments, so
    the payload is read under its own names and typed once, here.

    Dates/author/editor/object_type/url read from `es_doc` when there is one
    (the legacy thin-payload path, unchanged); when there is none, they fall
    back to reading the SAME field names directly off `doc` instead, since a
    fully-embedded payload already carries them as real _source scalar fields.
    Still left empty rather than invented when neither has them.

    author/editor are the live document's, via the connector
    (`spec.document_people`). Hard-coded None before, so a document only this
    path ever saw had a null created_by_id and could never grow an AUTHORED edge.
    """
    es_doc = es_doc or {}
    source = es_doc or doc
    spec = _resolve_spec(connector_type)
    field_map = spec.field_map
    # Same one-to-many chains the bulk path resolves (connectors/base.py's
    # FieldMap). published_fields is (content_publish_on_date, createddate), so a
    # live content document now dates from content_publish_on_date when present —
    # matching the bulk path — rather than createddate only as before.
    created = resolve_first_present(source, field_map.published_fields) or ""
    updated = resolve_first_present(source, field_map.updated_fields) or ""
    author, editor = spec.document_people(source)
    return SourceDocument(
        doc_id=doc["doc_id"],
        connector_type=_connector_type(connector_type),
        # Which raw field carries the object type is per-connector data, not a
        # name this module gets to know — see connectors/base.py's FieldMap.
        # Behaviour change: the adapter this replaces wrote the CONNECTOR's name
        # here (a legacy spelling), so Document.source read that on every
        # incrementally-written node. The live document's own object_type is what
        # the bulk path writes and what a query means by `source`.
        object_type=source.get(field_map.object_type_field) or connector_type or "incremental",
        # doc_title: the fully-embedded payload's own field. title: legacy fallback.
        title=doc.get("doc_title") or doc.get("title", "") or "",
        # Whatever url the source carries, or none. The intranet's ES documents
        # carry no url field, so an intranet document written by this path has
        # always landed with url "" while the same document written by the bulk
        # path gets its real app route. That asymmetry predates the boundary and
        # is not introduced here; closing it means calling the connector's own
        # document_url on the live document, which is a behaviour change worth
        # making on its own.
        url=source.get("url") or None,
        published_at=created[:10],
        updated_at=updated,
        chunks=[
            Chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk.get("text", "") or "",
                title=chunk.get("chunk_title"),
                vector=upstream_chunk_vector(chunk, effective_run_config().es_chunk_vector_field),
                recursive_chunk_id=chunk.get("recursive_chunk_id") or chunk["chunk_id"],
            )
            for chunk in chunks
        ],
        author=author,
        editor=editor,
        mentions=[],
        acl=acl,
        # The live document (or, with no ES doc, the payload doc itself — same
        # `source` fallback as above), plus the payload's chunks under the name a
        # connector looks for them by. Still connector-internal; what changed is
        # that DISCOVER now has something real to read — the {doc_id, _chunks}
        # dict this replaces had nothing, so a Kafka person was skipped as unnamed.
        raw={**source, "doc_id": doc["doc_id"], "_chunks": {"v1": chunks}},
    )


def resolve_incremental_acl(doc: dict, es_doc: Optional[dict],
                           site_audiences_by_id: Optional[dict],
                           connector_type: str = "wikipedia",
                           tenant_id: Optional[str] = None) -> Optional[list[str]]:
    """Resolve ACL via the shared builder; None means fail-closed drop."""
    decision = build_document_acl(doc, AclContext(
        connector_type=connector_type,
        site_audiences=site_audiences_by_id,
        es_document=es_doc,
        tenant_id=tenant_id,
    ))
    return None if decision.dropped else decision.tokens


def _align_source_identity(doc: dict, es_doc: Optional[dict]) -> None:
    """Fill missing identity/date fields from the live source doc so preprocess +
    Neo4j write see the same id/object_type/timestamps the bulk path uses.
    Mutates ``doc`` in place.

    Confluence ES docs carry ``type: page`` (no object_type); after ``_slim_hit`` they
    already have ``object_type: page``. Also accept a raw ``type: page`` fallback so
    Neo4j writes stamp Document.source=\"page\" without inventing a Document.type.
    """
    if not doc.get("id"):
        doc["id"] = doc.get("doc_id") or (es_doc or {}).get("id") or ""
    if es_doc:
        if not doc.get("created_at"):
            doc["created_at"] = es_doc.get("created_at") or es_doc.get("createddate") or ""
        if not doc.get("updated_at"):
            doc["updated_at"] = (
                es_doc.get("updated_at") or es_doc.get("lastmodifieddate") or ""
            )
        if not doc.get("createddate"):
            doc["createddate"] = es_doc.get("createddate") or es_doc.get("created_at") or ""
        if not doc.get("lastmodifieddate"):
            doc["lastmodifieddate"] = (
                es_doc.get("lastmodifieddate") or es_doc.get("updated_at") or ""
            )
    if doc.get("object_type"):
        return
    if not es_doc:
        return
    if es_doc.get("object_type"):
        doc["object_type"] = es_doc["object_type"]
    elif es_doc.get("type") == "page":
        doc["object_type"] = "page"


def _kept_chunk_entity_seed(kept_chunk_ids: list[str], tenant_id: str) -> tuple[dict, dict]:
    """(merge_map_seed, canon_to_id_seed) for entities that appear ONLY in this
    document's unchanged chunks.

    build_edge_pool resolves every claim subject through doc_merge_maps, and a claim's
    subject can be an entity first extracted in an EARLIER chunk of the same document
    (relationships can't — their endpoints are always in their own chunk; see
    _build_doc_type_registry in services/edge_pool_service.py). The bulk path satisfies
    that because it hands the service every chunk of the document at once. Chunk-diff
    hands it only the NEW chunks, so those earlier names are missing from merge_map,
    _resolve_endpoint returns None, and the claim is dropped at no log level.

    Cheap by construction: those entities are already in the graph carrying this doc_id,
    so this needs no re-extraction, no embedding/ANN/LLM resolve, and no upsert — just
    the cached extraction results for the kept chunks plus the entity registry's alias
    index (populated by upsert_entity_incremental's raw_name). An alias resolving to
    more than one registry record is genuinely ambiguous and is left unbound rather
    than guessed at.

    Deliberately NOT folded into the caller's canon_to_id: that drives the people/org
    and recompute_degree scope below, and widening it would re-run reembed_person_profiles
    (an LLM call) over entities this event never touched.

    ponytail: a cache miss — kept chunks extracted under an older EXTRACTION_VERSION —
    leaves those names unseeded, i.e. exactly today's behavior, never worse. Read the raw
    names from the graph's aliases_json instead if that miss rate ever matters.
    """
    if not kept_chunk_ids:
        return {}, {}
    from services.clustering_utils import normalize_name  # local: keeps FAISS off the worker's import path

    keys = [f"{cid}:{EXTRACTION_VERSION}" for cid in kept_chunk_ids]
    saved = load_checkpoints_batch("extraction", keys)
    merge_seed: dict[str, str] = {}
    canon_seed: dict[str, str] = {}
    for key in keys:
        if key not in saved:
            continue
        for entity in ChunkExtractionResult(**saved[key]).entities:
            if entity.name in merge_seed:
                continue
            alias = normalize_name(entity.name) or entity.name.strip().lower()
            records = entity_registry.find_all_by_alias(alias, entity.type, tenant_id)
            if len(records) != 1:
                continue
            canonical = records[0]["canonical_name"]
            merge_seed[entity.name] = canonical
            canon_seed[canonical] = records[0]["_id"]
    return merge_seed, canon_seed


async def process_document(
    event: GraphExtractEvent,
    doc: dict,
    repo: Neo4jRepository,
    extractor: ExtractionService,
    embedder: EmbeddingService,
    vector_store: VectorStore,
    es_doc: Optional[dict] = None,
    site_audiences_by_id: Optional[dict] = None,
    chunks_vector_store: Optional[VectorStore] = None,
) -> None:
    """Processes one document from the payload's docs[] array, start to finish: staleness
    check, ACL resolution (see resolve_incremental_acl — fail-closed, never defaults to
    public), optional preprocess, chunk-diff, extraction, entity resolution, and
    edge/claim writes."""
    doc_id = doc["doc_id"]
    watermark = resolve_watermark(event, doc, es_doc)
    flow_log.step_start(logger, "Upsert document", doc_id=doc_id)

    # No staleness check here: handle_event already runs _is_stale per doc_id
    # before it ever calls this (core/kafka_worker_entrypoint.py), so a second
    # one would just re-read the same watermark it is about to advance below.

    # Step 1/6 — ACL resolution (fail-closed; never defaults to public).
    flow_log.step_start(logger, "1/6 ACL resolution", doc_id=doc_id)
    acl = resolve_incremental_acl(doc, es_doc, site_audiences_by_id,
                                  connector_type=event.connector_type, tenant_id=event.tenant_id)
    if acl is None:
        flow_log.step_skipped(logger, "Upsert document", doc_id=doc_id,
                              reason=f"no ACL derivable for connector_type={event.connector_type} "
                                     "(fail-closed, never defaulted to public)")
        return
    flow_log.step_done(logger, "1/6 ACL resolution", doc_id=doc_id, tokens=len(acl))

    # Normalize id/object_type from live ES before preprocess + Neo4j write
    # (Confluence pages: ES type=page → in-doc object_type=page → Document.source).
    _align_source_identity(doc, es_doc)

    # Step 2/6 — chunk diff + Neo4j document/chunk write.
    flow_log.step_start(logger, "2/6 Chunk diff & graph write", doc_id=doc_id)
    chunks = _extract_chunks(doc)
    if not chunks:
        flow_log.step_skipped(logger, "Upsert document", doc_id=doc_id, reason="no chunks in payload")
        return

    spec = _resolve_spec(event.connector_type)
    source_doc = _to_source_document(doc, chunks, event.connector_type,
                                     AclDecision(acl, False, "incremental"), es_doc)
    enriched = await run_document_preprocess(spec, [source_doc])
    source_doc = enriched[0] if enriched else source_doc
    # Best-effort, like DISCOVER/MATERIALIZE below: a connector's observe()
    # raising must not fail (and redeliver) an already-processable message.
    try:
        spec.people.observe([source_doc])
    except Exception:
        logger.exception("Connector people observe failed for doc_id=%s — non-fatal", doc_id)

    # Chunk-diff: retract only chunks whose recursive_chunk_id disappeared, extract only
    # chunks whose recursive_chunk_id is new; an unchanged chunk is left untouched.
    # recursive_chunk_id (not chunk_id) is the diff key since chunk_id regenerates on every
    # extraction pass. New doc_id: get_document_chunk_rids returns {}, so everything is new.
    new_rids = {(c.get("recursive_chunk_id") or c["chunk_id"]): c for c in chunks}
    old_rid_to_cid = repo.get_document_chunk_rids(doc_id, event.tenant_id)
    dropped_cids = [cid for rid, cid in old_rid_to_cid.items() if rid not in new_rids]
    if dropped_cids:
        repo.retract_chunks(doc_id, event.tenant_id, dropped_cids, vector_store, chunks_vector_store)
    add_chunks = [c for rid, c in new_rids.items() if rid not in old_rid_to_cid]

    repo.write_documents([source_doc], event.tenant_id, {doc_id: acl})
    # write_chunks covers every current chunk (kept + new) since MERGE-refreshing an
    # unchanged chunk is cheap and idempotent; only extraction is worth skipping.
    repo.write_chunks([source_doc], event.tenant_id, {doc_id: acl}, chunks_vector_store, embedder)
    flow_log.step_done(logger, "2/6 Chunk diff & graph write", doc_id=doc_id,
                       new=len(add_chunks), retracted=len(dropped_cids),
                       kept=len(new_rids) - len(add_chunks), total=len(chunks))

    # People, stages one and two — the DISCOVER + MATERIALIZE bulk runs
    # (core/pipeline.py), on one document. `observe` is what decides whether THIS
    # document is a person, and it is the connector's decision. Ungated, as in
    # bulk: MATERIALIZE mints the (connector_type, source_key) identity that
    # authorship joins on, so gating it takes authorship down too. Before the
    # add_chunks return since an identity doesn't depend on new chunks; a fresh
    # spec per document keeps observe's state from outliving the event.
    person_records: list = []
    person_ids: list[str] = []
    # Best-effort DISCOVER/MATERIALIZE: the document is already written, and a
    # connector whose people lookup is a network call must not redeliver it.
    try:
        person_records = await spec.people.discover(PeopleContext(tenant_id=event.tenant_id))
        person_entries = await asyncio.to_thread(
            materialize_persons, person_records, spec.connector_type.value, event.tenant_id,
        )
        if person_entries:
            # Entity-level acl from this document's tokens, bulk's Step 5 pass.
            # Entries are keyed by entity_id already, so canon_to_id is identity.
            _tag_entity_acls(person_entries, {doc_id: acl})
            repo.write_entities(person_entries, event.tenant_id,
                                {eid: eid for eid in person_entries}, vector_store)
            person_ids = list(person_entries)
    except Exception:
        logger.exception("People DISCOVER/MATERIALIZE failed for doc_id=%s — non-fatal", doc_id)
    if person_ids:
        flow_log.step_done(logger, "People discover/materialize", doc_id=doc_id, persons=len(person_ids))

    if not add_chunks:
        try:
            await spec.people.enrich(person_records, PeopleContext(
                repo=repo, tenant_id=event.tenant_id, entity_ids=person_ids,
                vector_store=vector_store, page_doc_ids=[doc_id],
                retire_merged_mentions=_retire_merged_person_mentions,
            ))
        except Exception:
            logger.exception("Connector people ENRICH failed for doc_id=%s — non-fatal", doc_id)
        _advance_watermark(event.tenant_id, doc_id, watermark)
        flow_log.step_skipped(logger, "3/6–6/6 Extraction & graph enrichment", doc_id=doc_id,
                              reason="no new chunks")
        flow_log.step_done(logger, "Upsert document", doc_id=doc_id,
                           result="acl/metadata refreshed (extraction skipped)")
        return

    chunk_ids = [c["chunk_id"] for c in add_chunks]
    # The connector frames its own chunk text, same as the bulk path — resolved
    # from the event's connector rather than hard-wired to the intranet's (`spec`
    # above; the people phase and this share one instance deliberately).
    by_chunk_id = {c.chunk_id: c for c in source_doc.chunks}
    prepared_texts = [spec.prepare_chunk_text(source_doc, by_chunk_id[c["chunk_id"]])
                      for c in add_chunks]
    # Step 4/6 — extraction. Same Mongo-backed cache bulk uses: a redelivered Kafka
    # message skips already-paid-for LLM extraction instead of re-extracting from scratch.
    flow_log.step_start(logger, "4/6 Entity extraction (LLM)", doc_id=doc_id, chunks=len(add_chunks))
    results = await _extract_with_cache(extractor, prepared_texts, chunk_ids)
    flow_log.step_done(logger, "4/6 Entity extraction (LLM)", doc_id=doc_id,
                       entities=sum(len(r.entities) for r in results),
                       relationships=sum(len(r.relationships) for r in results),
                       claims=sum(len(r.claims) for r in results))

    # Step 5/6 — resolve each raw entity name against the live graph (not a corpus-wide cluster);
    # dedup names up front so a repeated name is only resolved once, then resolve
    # concurrently (bounded — see settings.incremental_entity_resolve_concurrency).
    unique_entities: dict[str, tuple[str, str]] = {}
    for result in results:
        for entity in result.entities:
            unique_entities.setdefault(entity.name, (entity.type, entity.description or ""))
    resolve_total = len(unique_entities)
    flow_log.step_start(logger, "5/6 Entity resolution", doc_id=doc_id, unique=resolve_total)

    resolve_sem = asyncio.Semaphore(settings.incremental_entity_resolve_concurrency)
    # ~10 heartbeats across the batch (min every 10) so a 150-name resolve isn't silent.
    resolve_log_every = max(10, resolve_total // 10) if resolve_total else 1
    resolve_done = 0
    resolve_lock = asyncio.Lock()
    resolve_t0 = time.monotonic()

    async def _resolve_one(name: str, entity_type: str, description: str) -> tuple[str, str, str, str, list]:
        nonlocal resolve_done
        async with resolve_sem:
            canonical, embedding = await resolve_canonical_name(
                repo, embedder, vector_store, name, entity_type, description, event.tenant_id,
            )
        async with resolve_lock:
            resolve_done += 1
            if resolve_done == resolve_total or resolve_done % resolve_log_every == 0:
                flow_log.step_progress(
                    logger, "5/6 Entity resolution",
                    phase="resolve", done=resolve_done, total=resolve_total,
                    elapsed_s=int(time.monotonic() - resolve_t0),
                )
        return name, entity_type, description, canonical, embedding

    resolved = await asyncio.gather(
        *[_resolve_one(name, etype, desc) for name, (etype, desc) in unique_entities.items()]
    )

    # Applied sequentially (not concurrently) even though resolution above was concurrent:
    # two raw names can resolve to the same canonical entity, and upsert_entity_incremental
    # is a read-modify-write, so interleaved awaits here could race and lose an update.
    merge_map: dict[str, str] = {}
    canon_to_id: dict[str, str] = {}
    upsert_total = len(resolved)
    upsert_log_every = max(10, upsert_total // 10) if upsert_total else 1
    upsert_t0 = time.monotonic()
    flow_log.step_progress(
        logger, "5/6 Entity resolution",
        phase="upsert", done=0, total=upsert_total,
        elapsed_s=0,
    )
    for upsert_i, (name, entity_type, description, canonical, embedding) in enumerate(resolved, start=1):
        merge_map[name] = canonical
        entity_id = repo.upsert_entity_incremental(
            name=canonical, entity_type=entity_type, description=description,
            embedding=embedding, doc_id=doc_id, acl=acl, tenant_id=event.tenant_id,
            raw_name=name, vector_store=vector_store,
        )
        canon_to_id[canonical] = entity_id
        if upsert_i == upsert_total or upsert_i % upsert_log_every == 0:
            flow_log.step_progress(
                logger, "5/6 Entity resolution",
                phase="upsert", done=upsert_i, total=upsert_total,
                elapsed_s=int(time.monotonic() - upsert_t0),
            )
    flow_log.step_done(logger, "5/6 Entity resolution", doc_id=doc_id,
                       resolved=len(merge_map), nodes=len(canon_to_id),
                       elapsed_s=int(time.monotonic() - resolve_t0))

    # Entities living only in this document's UNCHANGED chunks, so a claim in a new
    # chunk can still bind a subject named back there (see _kept_chunk_entity_seed).
    # merge_map wins on any overlap — this event's resolution is the fresher one.
    kept_cids = [cid for rid, cid in old_rid_to_cid.items() if rid in new_rids]
    merge_seed, canon_seed = _kept_chunk_entity_seed(kept_cids, event.tenant_id)
    edge_merge_map = {**merge_seed, **merge_map}
    edge_canon_to_id = {**canon_seed, **canon_to_id}

    flow_log.step_start(logger, "6/6 Relationship/claim write", doc_id=doc_id)
    final_edges, final_claims, mentions = build_edge_pool(
        results_by_doc={doc_id: results},
        chunk_ids_by_doc={doc_id: chunk_ids},
        doc_merge_maps={doc_id: edge_merge_map},
        final_merge_map_by_doc={doc_id: {v: v for v in edge_merge_map.values()}},
        canon_to_id=edge_canon_to_id,
        doc_acl_by_doc={doc_id: acl},
        # A thin incremental payload carries no document date; published_at is ""
        # (honest "no date") unless the live ES document supplied one.
        doc_dates_by_doc={doc_id: source_doc.published_at},
    )
    if mentions:
        repo.write_mentions(mentions, event.tenant_id)
    if final_edges:
        repo.write_relationships(final_edges, event.tenant_id)
    if final_claims:
        claim_ids = repo.write_claims(final_claims, event.tenant_id)
        # Scoped to just this doc's claims — never touches support outside this set.
        repo.recompute_claim_support(event.tenant_id, claim_ids)
    if canon_to_id:
        # Scoped to just this doc's entities — never changes degree outside this set.
        repo.recompute_degree(event.tenant_id, list(canon_to_id.values()))
    flow_log.step_done(logger, "6/6 Relationship/claim write", doc_id=doc_id,
                       edges=len(final_edges), claims=len(final_claims), mentions=len(mentions))

    # People, stage three — ENRICH, scoped to this document's persons plus its
    # resolved entities (bare-name mentions live in the second set, and folding
    # them onto the first is ENRICH's own first step). Its own feature flags and
    # best-effort boundaries, which the six calls this replaces made core's.
    person_scope = person_ids + [e for e in canon_to_id.values() if e not in person_ids]
    incr_final_entities = {
        canonical: {"type": etype, "name": canonical}
        for name, (etype, _desc) in unique_entities.items()
        if (canonical := merge_map.get(name))
    }
    if person_scope:
        try:
            await spec.people.enrich(person_records, PeopleContext(
                repo=repo, tenant_id=event.tenant_id, entity_ids=person_scope,
                vector_store=vector_store, retire_merged_mentions=_retire_merged_person_mentions,
                final_entities=incr_final_entities, canon_to_id=canon_to_id,
                page_doc_ids=[doc_id],
            ))
        except Exception:
            logger.exception("Connector people ENRICH failed for doc_id=%s — non-fatal", doc_id)

    # Authorship flags come from the bulk run's recorded config, so an update
    # builds (or skips) the same edges its bulk run did, regardless of this
    # worker's current settings.
    run_config = effective_run_config()
    if run_config.build_authorship_edges:
        # Doc-scoped, unconditional of canon_to_id (a document has an author regardless of
        # resolved entities). Depends on the :Person nodes MATERIALIZE wrote above,
        # and resolves them by (connector_type, source_key) — the same join bulk uses.
        try:
            repo.materialize_authorship_edges(event.tenant_id, [doc_id], run_config.contributor_entity_types)
        except Exception:
            logger.exception("Authorship-edge materialization failed for doc_id=%s — non-fatal", doc_id)

    _advance_watermark(event.tenant_id, doc_id, watermark)
    flow_log.step_done(logger, "Upsert document", doc_id=doc_id,
                       entities=len(merge_map), seeded=len(merge_seed),
                       edges=len(final_edges), claims=len(final_claims))
    # This worker is long-lived with no single "end of run", so log the running snapshot
    # per document instead.
    logger.debug("LLM call metrics so far (process lifetime): %s", run_metrics_snapshot())


def _mass_retraction_guard_trips(repo: Neo4jRepository, tenant_id: str) -> Optional[dict]:
    """Guards against a burst of bad delete events (producer bug, misconfigured reindex, bad
    backfill) retracting a tenant's documents one Kafka message at a time with no ceiling —
    each individual retract_document call looks legitimate on its own, so nothing else would
    notice or stop it. Tracks a rolling count of delete attempts per tenant against the
    tenant's current total document count; returns None if the delete may proceed, or a
    stats dict if it must be blocked.

    total is re-read fresh each call (not snapshotted), so the guard trips somewhat sooner
    than mass_retraction_frac's literal percentage as earlier deletes in the burst land —
    conservative by design. The window resets after mass_retraction_window_seconds of
    inactivity, not lifetime-cumulative, so normal long-term churn never permanently trips
    it — only a burst within one window does. increment_rolling_window_count is a single
    atomic Mongo op (not load-then-save), so two concurrent deletes can't both read the same
    starting count and undercount. mass_retraction_min_corpus floors out small/dev tenants
    where a couple of real deletes would trip the fraction trivially."""
    if not settings.mass_retraction_guard_enabled:
        return None
    total = repo.count_documents(tenant_id)
    if total < settings.mass_retraction_min_corpus:
        return None

    attempted_count = increment_rolling_window_count(
        _MASS_RETRACTION_COLLECTION, tenant_id, settings.mass_retraction_window_seconds, time.time(),
    )

    if attempted_count / total > settings.mass_retraction_frac:
        logger.error(
            "MASS-RETRACTION GUARD TRIPPED: tenant_id=%s would have retracted "
            "%d/%d documents (> %.0f%%) within %.0fs — delete BLOCKED. If this "
            "is a genuine bulk offboarding, raise mass_retraction_frac or set "
            "mass_retraction_guard_enabled=false for this run.",
            tenant_id, attempted_count, total, settings.mass_retraction_frac * 100,
            settings.mass_retraction_window_seconds,
        )
        return {"guarded": True, "attempted_count": attempted_count, "total_docs": total}

    return None


def delete_document(doc_id: str, repo: Neo4jRepository, tenant_id: str,
                    vector_store: Optional[VectorStore] = None,
                    chunks_vector_store: Optional[VectorStore] = None) -> dict:
    """Handles "delete" events. Guarded by _mass_retraction_guard_trips before anything is
    deleted — a guarded delete is logged and skipped, not raised, so it doesn't retry-storm
    the consumer or route to the DLQ (a blocked delete is a policy decision, not a transient
    failure). People/org graph: a departed employee is never retracted — their :Person node
    and REPORTS_TO/MEMBER_OF edges are kept for org history, just flipped to inactive. Every
    other object_type falls through to the normal retract_document."""
    guard_stats = _mass_retraction_guard_trips(repo, tenant_id)
    if guard_stats is not None:
        return guard_stats
    if effective_run_config().people_org_graph_enabled and repo.doc_object_type(doc_id, tenant_id) == "people":
        n = repo.mark_person_departed(tenant_id, doc_id, when=datetime.now(timezone.utc).isoformat())
        stats = {"departed": n}
        logger.info("doc_id=%s: person marked departed (%s), not retracted", doc_id, stats)
        return stats
    stats = repo.retract_document(doc_id, tenant_id, vector_store, chunks_vector_store)
    logger.info("doc_id=%s: retracted (%s)", doc_id, stats)
    return stats


def update_acl_only(doc_id: str, repo: Neo4jRepository, tenant_id: str, acl: list[str],
                    vector_store: Optional[VectorStore] = None,
                    chunks_vector_store: Optional[VectorStore] = None) -> dict:
    """Handles "acl_update" events (content/body unchanged, only permissions changed) —
    refreshes permissions on this document's existing nodes without any re-extraction.
    acl is expected to already be translated to this repo's grammar by the caller."""
    stats = repo.update_document_acl(doc_id, tenant_id, acl, vector_store, chunks_vector_store)
    logger.info("doc_id=%s: acl refreshed (%s)", doc_id, stats)
    return stats
