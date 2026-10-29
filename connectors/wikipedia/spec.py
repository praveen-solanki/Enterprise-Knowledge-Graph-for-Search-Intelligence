"""Wikipedia's answers to the connector contract.

Transport is a plain `requests.Session`, not an Elasticsearch client — `fetch`
paginates the public MediaWiki Action API directly instead of delegating to
connectors/elasticsearch_client.py. Everything else about the contract (what a
document is, who can see it, who the people are) is exactly as narrow as
Zeus/Confluence's, just answered trivially: every article is public, and
nobody is a person here.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Optional

import requests

from config.settings import settings
from connectors.base import (
    Chunk,
    ConnectorSpec,
    ConnectorType,
    PeopleProvider,
    SourceDocument,
)
from connectors.wikipedia.documents import (
    WIKIPEDIA_ACL_STRATEGY,
    WIKIPEDIA_FIELD_MAP,
    chunk_budget_count,
    document_people,
    fetch_batches,
    prepare_chunk_text,
    search_query,
    to_source_document,
)
from connectors.wikipedia.people import NullPeopleProvider
from services.acl import AclStrategy


class WikipediaSpec:
    """The zero-local-dependency connector: a public, no-auth API read live,
    with no local database or indexing step ahead of it.

    --index doubles as this connector's "what to fetch": a search phrase, or
    "Category:<name>" to walk a category instead (connectors/wikipedia/
    documents.py's _request_params). Holds only the tenant_id bind_run is
    handed — there is no per-run side-lookup to cache, unlike Zeus's site
    audiences.
    """

    connector_type = ConnectorType.WIKIPEDIA
    object_types = frozenset({"article"})
    field_map = WIKIPEDIA_FIELD_MAP

    def __init__(self) -> None:
        self._tenant_id: Optional[str] = None
        self._people: PeopleProvider = NullPeopleProvider()

    # --- what to fetch, and how ---------------------------------------------

    def build_query(self, doc_ids: Optional[list[str]] = None) -> dict:
        return search_query(doc_ids)

    def open_transport(self) -> Any:
        session = requests.Session()
        session.headers["User-Agent"] = settings.http_user_agent
        return session

    def bind_run(self, es, index: str, tenant_id: Optional[str] = None) -> None:
        self._tenant_id = tenant_id

    def fetch(
        self, transport: Any, index: str, query: Optional[dict] = None, *,
        batch_size: int, batch_sizer: Any = None,
        docs_wanted: Optional[Callable[[], Optional[int]]] = None,
    ) -> Iterator[tuple[list[SourceDocument], int]]:
        return fetch_batches(
            transport, index, query, batch_size=batch_size,
            batch_sizer=batch_sizer, docs_wanted=docs_wanted,
        )

    # --- what a document is -------------------------------------------------

    def to_source_document(self, hit: Mapping) -> Optional[SourceDocument]:
        return to_source_document(hit)

    def document_people(self, hit: Mapping) -> tuple:
        return document_people(hit)

    def chunk_budget_count(self, doc: SourceDocument) -> int:
        return chunk_budget_count(doc)

    def prepare_chunk_text(self, doc: SourceDocument, chunk: Chunk) -> str:
        return prepare_chunk_text(doc, chunk)

    # --- incremental ACL lookup ----------------------------------------------
    # No incremental path for this connector — there is no Kafka event source
    # for Wikipedia edits feeding this pipeline. Trivial stubs satisfy the
    # contract; nothing calls them.

    def acl_lookup_document(self, hit: Mapping) -> Optional[dict]:
        page_id = hit.get("pageid", hit.get("id"))
        if page_id is None:
            return None
        out = dict(hit)
        out["id"] = str(page_id)
        return out

    def acl_scope_ids(self, lookup_docs: list[dict]) -> list[str]:
        return []

    def acl_scope_audiences(self, es, index: str, scope_ids: list[str]) -> dict[str, list[str]]:
        return {}

    # --- who the people are, and who can see it -----------------------------

    @property
    def people(self) -> PeopleProvider:
        return self._people

    @property
    def acl(self) -> AclStrategy:
        return WIKIPEDIA_ACL_STRATEGY


# Structural conformance to the contract. There is no type checker in CI, so
# tests/test_connector_contract.py asserts it at runtime instead of trusting
# this annotation.
_SPEC: type[ConnectorSpec] = WikipediaSpec
