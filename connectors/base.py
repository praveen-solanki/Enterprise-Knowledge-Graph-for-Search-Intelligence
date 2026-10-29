"""The connector contract — the boundary every source enters the pipeline through.

The pipeline was built for one source (the intranet), so every stage
after fetch read that source's field names directly. There
was no shared idea of what a document is, which left a second source only two
ways in: teach the shared pipeline its field names, or rename its documents to
look like intranet documents. This module is the third way.

A connector answers exactly four questions, and nothing else about a source
crosses this line:

    1. What to fetch, and how. `ConnectorSpec.build_query` / `open_transport` / `fetch`
    2. What a document is.     `ConnectorSpec.to_source_document` -> `SourceDocument`
    3. Who the people are.     `ConnectorSpec.people` -> `PeopleProvider`
    4. Who can see it.         `ConnectorSpec.acl` -> `AclStrategy` (services/acl/)

A connector owns its own transport end to end: `open_transport` opens
whatever client it needs (an Elasticsearch client for Zeus/Confluence, a bare
`requests.Session` for an HTTP API, nothing at all for one that needs no
client), and `fetch` paginates it into `SourceDocument` batches. `core/` holds
no import of, and no opinion about, any specific transport — which is what
lets a connector with zero local database/indexing dependency (connectors/
wikipedia/) sit next to Zeus and Confluence without `core/` changing at all.

Everything else stays in `core/`. `SourceDocument` is what every stage after
fetch is allowed to see; `raw` is connector-internal and no stage reaches past
the typed fields into it.

Types only, with one exception: `resolve_first_present`, the pure first-present
rule a `FieldMap` is declared against. No I/O or connector logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, Sequence, runtime_checkable

from services.acl.builder import AclStrategy
from services.acl.types import AclDecision


class ConnectorType(str, Enum):
    """The one name for the concept, with the one set of values.

    Not `source`, not the legacy product spellings — those all meant
    this and disagreed with each other.

    CONFLUENCE is named here without a spec behind it, and that is deliberate:
    it reserves the name so `connectors/registry.py` can say it is unregistered
    rather than guess a connector, and `--connector` only offers what is
    registered.

    One index belongs to exactly one connector — there is no "mixed index". Each
    connector derives ACL through its own door (`ConnectorSpec.acl`), never by
    dispatching on a hit's own connector_type field (see
    `connectors/zeus/documents.py`'s to_source_document).
    """

    # Public, no-auth APIs read live. No download and no local index.
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"


@dataclass(frozen=True)
class Chunk:
    """One unit of text handed to extraction.

    `vector` is upstream-supplied — whatever indexed the content already
    computed it once. It is never invented here; None means the caller skips
    the vector push for this chunk rather than fabricating one.

    `recursive_chunk_id` is a stable structural id, distinct from `chunk_id`
    (which regenerates on every extraction pass). It is what lets a re-ingest
    diff a document's old chunks against its new ones.
    """

    chunk_id: str
    text: str
    title: Optional[str]
    vector: Optional[list[float]]
    recursive_chunk_id: Optional[str] = None


@dataclass(frozen=True)
class PersonRef:
    """A pointer to a human as some source names them. Not an identity.

    Two PersonRefs from two connectors may well be the same human; deciding
    that is `services/identity/`'s job, not this type's.
    """

    source_key: str                 # HR user id, Atlassian account_id, whatever the source uses
    display_name: Optional[str] = None
    email: Optional[str] = None


@dataclass(frozen=True)
class PersonRecord:
    """One person as their source knows them — what DISCOVER hands over.

    Guarantees four fields and nothing else: `source_key`, `display_name`,
    `email`, `url`. MATERIALIZE reads only those, and that is precisely what
    makes it identical for every connector.

    `attributes` is one open map the connector fills and MATERIALIZE carries
    through opaquely — job title, manager, department, whatever this source
    happens to know. Nothing outside the connector interprets it. Two
    conventions, and only two: keys are unprefixed (`job_title`, `department`,
    `manager`) and the node properties they become are `person_<key>`; and a
    key ending in `_id` is an opaque identifier, kept on the node for Cypher
    joins but never rendered into the disambiguation prompt, where an id tells
    an LLM nothing (services/people/disambiguation.py's `attr_text`). Every
    other key is rendered as-is, so a source with its own vocabulary needs no
    change to shared code.

    `source_doc_ids` is provenance, not policy: which of this run's documents
    this record came from, so the person node's doc_ids/support are right.
    Empty when a source's people come from a directory API rather than from
    documents.
    """

    source_key: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    # The person's own profile link, in whatever route scheme their source
    # uses — connector policy, same as email, not part of the opaque
    # `attributes` map (which would prefix it to `person_url`).
    url: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_doc_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceDocument:
    """What every stage after fetch is allowed to see.

    No connector adds a field here without changing this class, and no stage
    reaches past it into `raw`.
    """

    doc_id: str
    connector_type: ConnectorType
    object_type: str                # connector-local, already normalised by the connector
    title: str
    url: Optional[str]
    published_at: str               # YYYY-MM-DD
    updated_at: str                 # incremental watermark
    chunks: list[Chunk]
    author: Optional[PersonRef]
    # Who last changed it, when the source distinguishes that from who wrote it.
    # Not decoration: repository/neo4j_repository.py's EDITED edge is built from
    # this, and folding it into `author` would silently drop that edge.
    editor: Optional[PersonRef]
    mentions: list[PersonRef]
    acl: AclDecision
    raw: Mapping[str, Any] = field(default_factory=dict)   # connector-internal only
    # Short opening excerpt for citation/preview UI — derived from the first
    # chunk's own text (already fetched for extraction), never a second fetch.
    # None when there are no chunks to derive it from.
    preview_text: Optional[str] = None


@dataclass(frozen=True)
class FieldMap:
    """Lookup table mapping each concept (title, published date, ...) to the raw
    field name(s) that hold it. Different doc kinds name a concept differently
    (title = `content_title` / `file_title` / `site_name`), so each concept lists
    the field names in priority order and `resolve_first_present` picks the first
    with a value. Bulk and incremental paths both read this one table.
    """

    object_type_field: str
    title_fields: tuple[str, ...]
    published_fields: tuple[str, ...]
    updated_fields: tuple[str, ...]
    author_id_fields: tuple[str, ...] = ()


def resolve_first_present(raw: Mapping[str, Any], fields: Sequence[str]) -> Optional[Any]:
    """The first field in `fields` present-and-truthy on `raw`, else None.

    The one-to-many resolution every FieldMap concept uses. This is the sole
    piece of behaviour in this module, kept beside the type it reads so the
    declaration and the rule that resolves it stay together; no I/O or
    connector-specific logic lives here.
    """
    for name in fields:
        value = raw.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class PeopleContext:
    """What a connector's people phase is given.

    The graph handles and the run's scope, nothing source-specific — `core/`
    assembles this without knowing what any connector will do with it.
    `retire_merged_mentions` is the orchestrator's own registry/vector cleanup,
    passed in rather than imported, so a provider never reaches back into
    core/.
    """

    repo: Any = None
    tenant_id: str = ""
    entity_ids: list = field(default_factory=list)
    vector_store: Any = None
    retire_merged_mentions: Any = None
    final_entities: Any = None
    canon_to_id: Any = None
    page_doc_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PeopleResult:
    """What a connector's people phase produced.

    `core/` does not read it today — the inlined block this replaces returned
    nothing either — but a phase that reports what it did is the difference
    between a hook and a contract, and the entity scope is the thing a caller
    would want.
    """

    entity_ids: list[str] = field(default_factory=list)


@runtime_checkable
class PeopleProvider(Protocol):
    """Who the people are, per connector — three stages, of which the connector
    owns the first and the third.

        DISCOVER     `observe` + `discover`. The only source-specific part.
                     Zeus reads the People documents it already fetched,
                     Confluence set-dedupes the mentions it walked, Drive calls
                     the users API and ignores `observe` entirely.
        MATERIALIZE  `services/people/materialize.py`. Shared and
                     connector-blind: it reads a PersonRecord's three
                     guaranteed fields and nothing else, and it is what mints
                     the identity and the `:Entity:Person` node.
        ENRICH       `enrich`. Connector-owned, and legitimately nothing at
                     all. Zeus materializes the org chart it ships; a source
                     with no org chart has a thinner graph, which is an absence
                     and not a failure.

    A person is an `Entity` in whatever graph they live in, so
    `(:Chunk)-[:MENTIONS]->(:Entity)` reaches them. Beyond that the connector
    owns its own person properties and edges.

    The shared, connector-blind algorithms these implementations call live in
    `services/people/`. They are called by connectors, never the reverse.
    """

    def observe(self, docs: list[SourceDocument]) -> None: ...

    async def discover(self, ctx: PeopleContext) -> list[PersonRecord]: ...

    async def enrich(self, records: list[PersonRecord], ctx: PeopleContext) -> PeopleResult: ...


@runtime_checkable
class ConnectorSpec(Protocol):
    """One source, behind the boundary. Adding a connector means adding a
    directory that implements this, not editing a file in `core/`.

    The four questions are `build_query`, `to_source_document`, `people` and
    `acl`. The rest of this surface is the plumbing `core/` needs to ask them
    without knowing which connector answered: where to look by default, what a
    document costs against a batch's chunk budget, how its text is framed for
    extraction, and how the incremental path finds the side documents its ACL
    depends on. Every one of these is called from
    `core/`, so every one of them belongs here — a protocol narrower than the
    real call surface would just be a comment.
    """

    connector_type: ConnectorType
    object_types: frozenset[str]
    field_map: FieldMap

    # --- what to fetch, and how ---
    def build_query(self, doc_ids: Optional[list[str]]) -> dict: ...

    # Opens this connector's own transport (an Elasticsearch client, an HTTP
    # session, or nothing at all) — `core/` holds it opaquely and only ever
    # passes it back into `bind_run`/`fetch`/`acl_scope_audiences`.
    def open_transport(self) -> Any: ...

    def bind_run(self, es, index: str, tenant_id: Optional[str] = None) -> None: ...

    # Paginates `transport` into (doc_batch, batch_chunk_count) pairs, honoring
    # a chunk budget (`batch_size`, refreshed from `batch_sizer` between pages
    # when given) and an optional remaining-doc-count callback (`docs_wanted`,
    # for --limit). Elasticsearch-backed connectors delegate straight to
    # connectors/elasticsearch_client.fetch_documents_batched; an API-backed
    # connector paginates its own API instead — `core/` calls this one method
    # either way.
    def fetch(
        self, transport: Any, index: str, query: Optional[dict] = None, *,
        batch_size: int, batch_sizer: Any = None,
        docs_wanted: Optional[Callable[[], Optional[int]]] = None,
    ) -> Iterator[tuple[list[SourceDocument], int]]: ...

    # --- what a document is ---
    def to_source_document(self, hit: Mapping) -> Optional[SourceDocument]: ...   # None = skip this hit

    # (author, editor) for one raw hit, for the incremental path — its Kafka
    # payload carries no author, the live hit it looks up for ACL does. Which
    # fields those are is per-connector logic, not a FieldMap entry.
    def document_people(self, hit: Mapping) -> tuple[Optional[PersonRef], Optional[PersonRef]]: ...

    def chunk_budget_count(self, doc: SourceDocument) -> int: ...

    def prepare_chunk_text(self, doc: SourceDocument, chunk: Chunk) -> str: ...

    # --- incremental ACL lookup ---
    def acl_lookup_document(self, hit: Mapping) -> Optional[dict]: ...

    def acl_scope_ids(self, lookup_docs: list[dict]) -> list[str]: ...

    def acl_scope_audiences(self, es, index: str, scope_ids: list[str]) -> dict[str, list[str]]: ...

    # --- who the people are, and who can see it ---
    @property
    def people(self) -> PeopleProvider: ...

    @property
    def acl(self) -> AclStrategy: ...


async def run_document_preprocess(
    spec: ConnectorSpec, docs: list[SourceDocument],
) -> list[SourceDocument]:
    """Optional per-batch hook — only connectors that need it implement ``preprocess_documents``."""
    preprocess = getattr(spec, "preprocess_documents", None)
    if preprocess is None:
        return docs
    return await preprocess(docs)


async def run_preprocessing_drain(spec: ConnectorSpec) -> None:
    """Optional end-of-bulk hook before corpus-wide steps."""
    drain = getattr(spec, "ensure_preprocessing_complete", None)
    if drain is None:
        return
    await drain()
