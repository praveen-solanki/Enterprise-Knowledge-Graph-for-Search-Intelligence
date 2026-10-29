"""What a Wikipedia article is — API query building, HTTP pagination, and the
shape of one SourceDocument, straight off the public MediaWiki Action API.

No local index, no download step: `action=query&generator=search` (or
`generator=categorymembers` for a "Category:<name>" topic) streams real pages
live, paginated via the API's own opaque `continue` token
(https://www.mediawiki.org/wiki/API:Continue). Every article on Wikipedia is
public, so ACL here is a constant, not a derivation.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Iterator, Mapping, Optional

import requests

from config.settings import settings
from connectors.base import Chunk, ConnectorType, FieldMap, SourceDocument
from services.acl.builder import DocumentAclBuilder
from services.acl.types import AclDecision
from utils.error_classification import classify_error

logger = logging.getLogger("graphrag.connectors.wikipedia")

OBJECT_TYPE = "article"

WIKIPEDIA_FIELD_MAP = FieldMap(
    object_type_field="object_type",
    title_fields=("title",),
    # The API exposes only the last-edit timestamp, not the page's creation
    # date, so published_at and updated_at read the same source field.
    published_fields=("touched",),
    updated_fields=("touched",),
    author_id_fields=(),
)

# Everything on Wikipedia is public — no site/audience system to derive from.
_ACL_PUBLIC = AclDecision(["public"], False, "wikipedia:public")

# MediaWiki caps pageids/titles lookups per request (50 for anonymous callers,
# higher with a bot flag this connector does not use).
_MAX_IDS_PER_REQUEST = 50
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.0


class WikipediaAclStrategy:
    """Every article is public — nothing to derive, nothing to drop."""

    def build(self, document: dict, context) -> AclDecision:
        return _ACL_PUBLIC


WIKIPEDIA_ACL_STRATEGY = WikipediaAclStrategy()

# Registered under the connector's own name, same as connectors/zeus/acl.py —
# so anything that dispatches by connector_type string (the incremental
# worker's path; this connector has no incremental feed today, but the door
# stays consistent with Zeus/Confluence's) reaches the real constant instead
# of silently falling through to document-token pass-through.
DocumentAclBuilder.register((ConnectorType.WIKIPEDIA.value,), WIKIPEDIA_ACL_STRATEGY)


def search_query(doc_ids: Optional[list[str]] = None) -> dict:
    """What to fetch — a plain dict this connector reads itself (there is no
    Elasticsearch query grammar here). doc_ids narrows to specific pageids,
    e.g. for --retry-failed."""
    return {"pageids": [str(d) for d in doc_ids]} if doc_ids else {}


def _request_params(topic: str, query: dict, page_size: int) -> dict:
    base = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
    }
    pageids = query.get("pageids")
    if pageids:
        base["pageids"] = "|".join(pageids)
        return base
    if not topic:
        raise ValueError(
            "WikipediaSpec.fetch requires a non-empty topic (--index) — a plain "
            "phrase searches article text, 'Category:<name>' walks a category"
        )
    if topic.strip().lower().startswith("category:"):
        base.update({"generator": "categorymembers", "gcmtitle": topic,
                     "gcmlimit": page_size, "gcmnamespace": 0})
    else:
        base.update({"generator": "search", "gsrsearch": topic, "gsrlimit": page_size})
    return base


def _fetch_with_retry(session: requests.Session, params: dict) -> dict:
    """Bounded retry on transient failures, same verdicts as every other
    transport in this repo (utils/error_classification.classify_error) —
    network blips and 5xx are worth retrying, nothing else is."""
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        if attempt:
            delay = _RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            time.sleep(delay)
        try:
            response = session.get(
                settings.wikipedia_api_base_url, params=params,
                timeout=settings.wikipedia_request_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 — classified below, re-raised if not transient
            if classify_error(exc) != "transient":
                raise
            last_exc = exc
            logger.warning("Wikipedia API call failed (attempt %d/%d): %s",
                           attempt + 1, _MAX_RETRIES, exc)
    raise last_exc  # type: ignore[misc] — loop always sets it before exhausting retries


def _pages_from(body: dict) -> list[dict]:
    pages = (body.get("query") or {}).get("pages") or []
    return [p for p in sorted(pages, key=lambda p: p.get("index", 0)) if not p.get("missing")]


def _complete_pages(session: requests.Session, params: dict) -> Iterator[dict]:
    """Walks one continuation chain, yielding each page exactly once and only
    once it is whole.

    A response cannot be read in isolation, because MediaWiki splits one page
    set across several of them:

    * `prop=extracts` with `explaintext` is capped at ONE whole-article extract
      per request — ask for more and it answers `"exlimit" was too large for a
      whole article extracts request, lowered to 1`. So a request for N pages
      returns N-1 of them with no `extract` key at all, plus an `excontinue`
      cursor for the next one.
    * Those continuation responses then drop `prop=info`, so `fullurl` is
      present only in the FIRST response of the chain.

    Reading each response on its own therefore loses a document per request (no
    extract means no chunks means `to_source_document` returns None) and strips
    the URL off every document whose extract arrives second or later — which is
    every document but the first.

    Merging by pageid fixes both at once: `info` from the first response and
    `extract` from whichever later one carried it end up on the same dict. A
    page that never gets an extract (a disambiguation stub, say) is simply
    never yielded, which is what the caller does with it anyway.

    Forwarding the whole `continue` dict back verbatim is also what lets one
    loop serve both generators' differently-shaped cursors (`gsroffset` vs.
    `gcmcontinue`) without this module knowing which is in play.
    """
    pending: dict[int, dict] = {}
    yielded: set[int] = set()
    while True:
        body = _fetch_with_retry(session, params)
        for page in _pages_from(body):
            page_id = page.get("pageid")
            if page_id is not None and page_id not in yielded:
                pending.setdefault(page_id, {}).update(page)
        for page_id, page in list(pending.items()):
            if page.get("extract"):
                del pending[page_id]
                yielded.add(page_id)
                yield page
        cont = body.get("continue")
        if not cont:
            return
        params = {**params, **cont}


def iter_pages(session: requests.Session, topic: str, query: dict, page_size: int) -> Iterator[dict]:
    """Yields raw page dicts, in relevance/category order. A pageids lookup
    (the --retry-failed path) is chunked to the API's own per-request id cap;
    a topic search/category walk pages through the API's `continue` token.
    Either way each chain is drained by _complete_pages, which is where the
    API's per-response quirks are dealt with."""
    pageids = query.get("pageids")
    if pageids:
        for start in range(0, len(pageids), _MAX_IDS_PER_REQUEST):
            chunk = pageids[start:start + _MAX_IDS_PER_REQUEST]
            yield from _complete_pages(
                session, _request_params(topic, {"pageids": chunk}, page_size))
        return
    yield from _complete_pages(session, _request_params(topic, query, page_size))


def _section_heading(paragraph: str) -> Optional[str]:
    """The heading text if this paragraph is a wikitext section heading, else
    None. `explaintext` renders them as "== Early history ==" (level 2) through
    "===== ... =====", never with body text on the same line."""
    stripped = paragraph.strip()
    if len(stripped) < 5 or not stripped.startswith("==") or not stripped.endswith("=="):
        return None
    return stripped.strip("=").strip() or None


def _chunks_from_extract(doc_id: str, extract: str) -> list[Chunk]:
    """One chunk per paragraph of PROSE — plain-text extracts separate them
    with newlines; empty lines (consecutive or otherwise) are not chunks.

    A section heading is not a chunk of its own: on its own it extracts to
    nothing (a real run spent one extraction + one gleaning call per heading to
    be told "0 entities" every time), and the prose under it reads as if it had
    no context ("It has been abandoned since 2020" — what has?). Each heading is
    instead carried as the `title` of the chunks beneath it, which
    prepare_chunk_text puts in front of the text the model actually sees."""
    chunks: list[Chunk] = []
    section: Optional[str] = None
    for paragraph in (p.strip() for p in (extract or "").split("\n")):
        if not paragraph:
            continue
        heading = _section_heading(paragraph)
        if heading is not None:
            section = heading
            continue
        chunk_id = f"{doc_id}::p{len(chunks)}"
        chunks.append(Chunk(chunk_id=chunk_id, text=paragraph, title=section,
                            vector=None, recursive_chunk_id=chunk_id))
    return chunks


def _preview_text(chunks: list[Chunk]) -> Optional[str]:
    if not chunks:
        return None
    return (chunks[0].text or "").strip()[:500] or None


def to_source_document(hit: Mapping) -> Optional[SourceDocument]:
    """One MediaWiki API page object -> SourceDocument. None for a page with
    no extractable text (e.g. a disambiguation stub) — nothing to chunk means
    nothing to extract from."""
    page_id = hit.get("pageid")
    if page_id is None:
        return None
    doc_id = str(page_id)
    chunks = _chunks_from_extract(doc_id, hit.get("extract") or "")
    if not chunks:
        return None
    updated = (hit.get("touched") or "")[:10]
    return SourceDocument(
        doc_id=doc_id,
        connector_type=ConnectorType.WIKIPEDIA,
        object_type=OBJECT_TYPE,
        title=hit.get("title") or "",
        url=hit.get("fullurl") or None,
        # No creation date in this API response — only the last-edit watermark.
        published_at="",
        updated_at=updated,
        chunks=chunks,
        # Collaboratively edited, no single stable author/editor identity, and
        # the public API exposes no per-user directory for this connector's
        # people phase to resolve against — an absence, not a failure.
        author=None,
        editor=None,
        mentions=[],
        acl=_ACL_PUBLIC,
        raw=dict(hit),
        preview_text=_preview_text(chunks),
    )


def document_people(hit: Mapping) -> tuple[None, None]:
    """No per-user directory behind this connector; see to_source_document."""
    return None, None


def chunk_budget_count(doc: SourceDocument) -> int:
    return len(doc.chunks)


def prepare_chunk_text(doc: SourceDocument, chunk: Chunk) -> str:
    """Same framing as Zeus/Confluence, minus the publish-date head this
    connector never has. chunk.title is the article section this paragraph sits
    under (_chunks_from_extract) — without it a mid-article paragraph arrives
    with no subject ("It was acquired in 2018" names nothing on its own)."""
    head = f"Document: {doc.title or ''}"
    if chunk.title:
        head = f"{head}\nSection: {chunk.title}"
    return f"{head}\n\nText:\n{(chunk.text or '').strip()}".strip()


def fetch_batches(
    session: requests.Session, topic: str, query: Optional[dict] = None, *,
    batch_size: int, batch_sizer=None, docs_wanted=None,
) -> Iterator[tuple[list[SourceDocument], int]]:
    """Paginate the API into (doc_batch, batch_chunk_count) pairs, flushing on
    the same chunk-budget rule connectors/elasticsearch_client.py's
    fetch_documents_batched uses — a batch always holds at least one document,
    even one that alone exceeds the budget."""
    query = query or {}
    batch: list[SourceDocument] = []
    batch_chunk_count = 0
    for raw_page in iter_pages(session, topic, query, settings.wikipedia_page_size):
        wanted = docs_wanted() if docs_wanted is not None else None
        if wanted is not None and wanted <= 0:
            break
        doc = to_source_document(raw_page)
        if doc is None:
            continue
        doc_chunk_count = len(doc.chunks)
        max_chunks = batch_sizer.current_size if batch_sizer is not None else batch_size
        if batch and batch_chunk_count + doc_chunk_count > max_chunks:
            yield batch, batch_chunk_count
            batch, batch_chunk_count = [], 0
        batch.append(doc)
        batch_chunk_count += doc_chunk_count
    if batch:
        yield batch, batch_chunk_count
