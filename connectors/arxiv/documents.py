"""arXiv Atom API -> SourceDocument. Abstracts are the text. No download of
the PDF. https://info.arxiv.org/help/api/user-manual.html
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import Iterator, Mapping, Optional

import requests

from config.settings import settings
from connectors.base import Chunk, ConnectorType, FieldMap, SourceDocument
from services.acl.builder import DocumentAclBuilder
from services.acl.types import AclDecision
from utils.error_classification import classify_error

logger = logging.getLogger("graphrag.connectors.arxiv")

OBJECT_TYPE = "paper"
ATOM = "{http://www.w3.org/2005/Atom}"
_ACL_PUBLIC = AclDecision(["public"], False, "arxiv:public")
# arXiv asks for at least 3 seconds between requests.
_PAGE_GAP_SECONDS = 3.0
_MAX_RETRIES = 3


class ArxivAclStrategy:
    def build(self, document: dict, context) -> AclDecision:
        return _ACL_PUBLIC


ARXIV_ACL_STRATEGY = ArxivAclStrategy()
DocumentAclBuilder.register((ConnectorType.ARXIV.value,), ARXIV_ACL_STRATEGY)

ARXIV_FIELD_MAP = FieldMap(
    object_type_field="object_type",
    title_fields=("title",),
    published_fields=("published",),
    updated_fields=("updated",),
    author_id_fields=(),
)


def search_query(doc_ids: Optional[list[str]] = None) -> dict:
    return {"ids": [str(d) for d in doc_ids]} if doc_ids else {}


def _fetch(session: requests.Session, params: dict) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        if attempt:
            time.sleep(1.0 * attempt)
        try:
            response = session.get(
                settings.arxiv_api_base_url, params=params, timeout=30,
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            if classify_error(exc) != "transient":
                raise
            last_exc = exc
            logger.warning("arXiv API call failed (attempt %d/%d): %s",
                           attempt + 1, _MAX_RETRIES, exc)
    raise last_exc  # type: ignore[misc]


def _text(entry: ET.Element, tag: str) -> str:
    value = entry.findtext(f"{ATOM}{tag}") or ""
    return " ".join(value.split())


def _entry_dict(entry: ET.Element) -> dict:
    raw_id = _text(entry, "id")
    arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
    link = ""
    for node in entry.findall(f"{ATOM}link"):
        if node.get("rel") == "alternate" and node.get("href"):
            link = node.get("href") or ""
            break
    return {
        "id": arxiv_id,
        "title": _text(entry, "title"),
        "summary": _text(entry, "summary"),
        "published": _text(entry, "published"),
        "updated": _text(entry, "updated"),
        "url": link or raw_id or None,
    }


def iter_entries(session: requests.Session, topic: str, query: dict, page_size: int) -> Iterator[dict]:
    ids = query.get("ids")
    start = 0
    while True:
        params: dict = {"start": start, "max_results": page_size}
        if ids:
            params["id_list"] = ",".join(ids)
        else:
            if not topic:
                raise ValueError(
                    "ArxivSpec.fetch requires a non-empty search phrase (--index)"
                )
            params["search_query"] = f"all:{topic}"
        root = ET.fromstring(_fetch(session, params))
        entries = root.findall(f"{ATOM}entry")
        if len(entries) == 1 and _text(entries[0], "title") == "Error":
            return
        for entry in entries:
            yield _entry_dict(entry)
        if ids or len(entries) < page_size:
            return
        start += len(entries)
        time.sleep(_PAGE_GAP_SECONDS)


def to_source_document(hit: Mapping) -> Optional[SourceDocument]:
    doc_id = str(hit.get("id") or "")
    summary = (hit.get("summary") or "").strip()
    if not doc_id or not summary:
        return None
    chunk = Chunk(chunk_id=f"{doc_id}::abstract", text=summary, title=None,
                  vector=None, recursive_chunk_id=f"{doc_id}::abstract")
    published = (hit.get("published") or "")[:10]
    return SourceDocument(
        doc_id=doc_id,
        connector_type=ConnectorType.ARXIV,
        object_type=OBJECT_TYPE,
        title=hit.get("title") or "",
        url=hit.get("url") or None,
        published_at=published,
        updated_at=hit.get("updated") or "",
        chunks=[chunk],
        author=None,
        editor=None,
        mentions=[],
        acl=_ACL_PUBLIC,
        raw=dict(hit),
        preview_text=summary[:500],
    )


def document_people(hit: Mapping) -> tuple[None, None]:
    return None, None


def chunk_budget_count(doc: SourceDocument) -> int:
    return len(doc.chunks)


def prepare_chunk_text(doc: SourceDocument, chunk: Chunk) -> str:
    title = doc.title or ""
    head = f"Document published: {doc.published_at}\n" if doc.published_at else ""
    return f"{head}Document: {title}\n\nText:\n{(chunk.text or '').strip()}".strip()


def fetch_batches(
    session: requests.Session, topic: str, query: Optional[dict] = None, *,
    batch_size: int, batch_sizer=None, docs_wanted=None,
) -> Iterator[tuple[list[SourceDocument], int]]:
    query = query or {}
    batch: list[SourceDocument] = []
    batch_chunk_count = 0
    for raw in iter_entries(session, topic, query, settings.arxiv_page_size):
        wanted = docs_wanted() if docs_wanted is not None else None
        if wanted is not None and wanted <= 0:
            break
        doc = to_source_document(raw)
        if doc is None:
            continue
        max_chunks = batch_sizer.current_size if batch_sizer is not None else batch_size
        if batch and batch_chunk_count + len(doc.chunks) > max_chunks:
            yield batch, batch_chunk_count
            batch, batch_chunk_count = [], 0
        batch.append(doc)
        batch_chunk_count += len(doc.chunks)
    if batch:
        yield batch, batch_chunk_count
