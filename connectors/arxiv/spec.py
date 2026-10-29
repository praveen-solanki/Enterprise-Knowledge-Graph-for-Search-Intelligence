"""arXiv's answers to the connector contract. Transport is a requests.Session
against the public Atom API.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Optional

import requests

from config.settings import settings
from connectors.arxiv.documents import (
    ARXIV_ACL_STRATEGY,
    ARXIV_FIELD_MAP,
    chunk_budget_count,
    document_people,
    fetch_batches,
    prepare_chunk_text,
    search_query,
    to_source_document,
)
from connectors.arxiv.people import NullPeopleProvider
from connectors.base import (
    Chunk,
    ConnectorSpec,
    ConnectorType,
    PeopleProvider,
    SourceDocument,
)
from services.acl import AclStrategy


class ArxivSpec:
    connector_type = ConnectorType.ARXIV
    object_types = frozenset({"paper"})
    field_map = ARXIV_FIELD_MAP

    def __init__(self) -> None:
        self._tenant_id: Optional[str] = None
        self._people: PeopleProvider = NullPeopleProvider()

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

    def to_source_document(self, hit: Mapping) -> Optional[SourceDocument]:
        return to_source_document(hit)

    def document_people(self, hit: Mapping) -> tuple:
        return document_people(hit)

    def chunk_budget_count(self, doc: SourceDocument) -> int:
        return chunk_budget_count(doc)

    def prepare_chunk_text(self, doc: SourceDocument, chunk: Chunk) -> str:
        return prepare_chunk_text(doc, chunk)

    def acl_lookup_document(self, hit: Mapping) -> Optional[dict]:
        if not hit.get("id"):
            return None
        out = dict(hit)
        out["id"] = str(hit["id"])
        return out

    def acl_scope_ids(self, lookup_docs: list[dict]) -> list[str]:
        return []

    def acl_scope_audiences(self, es, index: str, scope_ids: list[str]) -> dict[str, list[str]]:
        return {}

    @property
    def people(self) -> PeopleProvider:
        return self._people

    @property
    def acl(self) -> AclStrategy:
        return ARXIV_ACL_STRATEGY


_SPEC: type[ConnectorSpec] = ArxivSpec
