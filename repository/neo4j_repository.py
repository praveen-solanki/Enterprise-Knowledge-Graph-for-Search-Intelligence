"""
Step 5 — Neo4j write. Writes the graph produced by Steps 1-4:

    (:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)
    (:Entity)-[:EXTRACTED_REL]->(:Entity)
    (:Claim)-[:ABOUT]->(:Entity)

Every write is an idempotent MERGE, batched via UNWIND (5000 rows/batch).

Uses Neo4j's HTTPS Query API (POST /db/<database>/query/v2), not the Bolt
driver — this shared dev server only exposes Neo4j over HTTPS.

Requires an explicit NEO4J_HTTP_URL; the repository refuses to construct
without it so a run can never accidentally target the wrong graph.

Entity embeddings live in Qdrant (services/vector_store.py), not on the
node — only a lightweight has_vector flag is stored here. No Community/
Leiden support (see services/community_service.py).

Every node/relationship carries `allow_access_control` (see
services/acl/) and `tenant_id`, with Entity identity keyed on
entity_id (see repository/entity_registry.py). Document/Chunk/Claim keep
global doc_id/chunk_id/claim_id identity and carry tenant_id as a plain
property. Document/Chunk/Claim get their one source document's ACL exactly;
Entity and EXTRACTED_REL get the union of every contributing document's ACL,
computed upstream (core/pipeline.py / services/edge_pool_service.py). No
query-time ACL enforcement exists yet — this only makes ingestion ACL-aware.

One requests.Session is reused per repository instance; close() releases it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from typing import Optional

import requests

from config.settings import settings
from services.llm_backend import effective_run_config
from services.dual_write_guard import (
    call_with_retry_or_rollback, delete_vectors, push_vectors_in_batches, sync_acl_for_each,
)
from services.embedding_service import EmbeddingService
from utils.error_classification import classify_error
from services.vector_store import VectorStore

logger = logging.getLogger("graphrag.repository.neo4j")


def _rows(response: dict) -> list[dict]:
    """Flatten the HTTP Query API v2's {"data": {"fields", "values"}} shape
    into a list of dicts — shared by every read method below."""
    fields = response.get("data", {}).get("fields", [])
    return [dict(zip(fields, row)) for row in response.get("data", {}).get("values", [])]


def _scalar(response: dict, field: str):
    rows = _rows(response)
    return rows[0].get(field) if rows else None


def _prune_doc_entries(json_str: Optional[str], doc_id: str) -> list:
    """Drops this doc_id's entry from a descriptions_json/aliases_json array —
    used by _retract_chunks_and_dependents so a re-ingested document doesn't
    end up with a duplicate entry alongside its stale one."""
    try:
        entries = json.loads(json_str or "[]")
    except (TypeError, ValueError):
        return []
    return [e for e in entries if e.get("doc_id") != doc_id]


def _distinct_canonical_batches(pairs: list[dict]) -> list[list[dict]]:
    """Splits merge pairs into rounds holding at most ONE pair per canonical entity_id
    ("c"), preserving input order within each round.

    A Cypher list union inside an UNWIND cannot accumulate across rows writing the SAME
    node or relationship: every row reads the property as it was before the statement ran,
    so the last row silently overwrites the earlier rows instead of adding to them. Two
    bare mentions merging onto one canonical person do exactly that, costing chunk_ids,
    doc_ids, aliases, support and ACL tokens (verified against a real Neo4j).

    Rather than move every union into Python, this makes the collision impossible: one pair
    per canonical per statement means no two rows target the same node or edge. Each round
    is a separate statement reading what the previous one committed, so several mentions of
    one person still accumulate. Every-canonical-distinct is a single round — today's
    behaviour and cost; only a repeat adds one."""
    batches: list[list[dict]] = []
    rounds_used: dict = {}
    for pair in pairs:
        index = rounds_used.get(pair["c"], 0)
        rounds_used[pair["c"]] = index + 1
        if index == len(batches):
            batches.append([])
        batches[index].append(pair)
    return batches


def _union_doc_entries(*json_strs: Optional[str]) -> list:
    """Union of several descriptions_json/aliases_json arrays, deduped on (doc_id, chunk_id,
    text-or-name) and order-stable: first occurrence wins, so the array passed first keeps
    its positions and later ones are appended.

    Needed because these are lists of maps, which a Neo4j property can't hold — they live
    as JSON strings, so Cypher can't union them. Sibling of _prune_doc_entries.

    The key spans all three id fields because the two shapes identify an entry differently:
    an entity's description/alias entry is keyed by source doc_id, a relationship's by
    source chunk_id with no doc_id. Covering both dedupes each shape on its own identity,
    and keeps two chunks producing identical text as two correctly-attributed entries.

    Used by merge_person_mentions (which would destroy these arrays) and
    write_relationships (which would overwrite them)."""
    merged: list = []
    seen: set = set()
    for json_str in json_strs:
        try:
            entries = json.loads(json_str or "[]")
        except (TypeError, ValueError):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = (entry.get("doc_id"), entry.get("chunk_id"),
                   entry.get("text") or entry.get("name"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


# Qdrant push batch size lives in services/dual_write_guard.py, not here.
# Batch sizes (settings.neo4j_default_batch_size / neo4j_default_entity_batch_size):
# starting points for docs/chunks/mentions and entities respectively — calibrated
# at runtime (see calibrate_batch_sizes).

# Retry only transient errors (deadlock/5xx/blip) — a couple of quick retries beat
# wasting a whole redelivery cycle. Kept short: this runs synchronously in a
# thread-pool thread, so a long retry budget would tie it up rather than just wait.
# (settings.neo4j_max_retries / neo4j_retry_backoff_seconds)

# CREATE DATABASE is async on a multi-member Enterprise cluster — bounds how
# long ensure_database() polls for currentStatus="online" before giving up
# (settings.neo4j_database_online_max_wait_seconds / neo4j_database_online_poll_interval_seconds).


def _system_execute(statement: str) -> dict:
    """POSTs to the `system` database — the only one CREATE/SHOW DATABASE can
    run against. Standalone since a database must exist before a
    Neo4jRepository can be constructed to target it."""
    if not settings.neo4j_http_url:
        raise ValueError("NEO4J_HTTP_URL must be set in .env before talking to Neo4j — "
                         "this guard exists specifically so a run never accidentally "
                         "targets a graph nobody meant to write to.")
    response = requests.post(
        f"{settings.neo4j_http_url}/db/system/query/v2",
        auth=(settings.neo4j_username, settings.neo4j_password),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json={"statement": statement, "parameters": {}},
        verify=settings.verify_certs, timeout=settings.neo4j_http_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def ensure_database(name: str) -> None:
    """Creates this run's own Neo4j database if it doesn't exist, and waits
    for it to come online before returning. Enterprise-only. `name` is
    backtick-quoted since it contains dashes, invalid in a bare identifier.

    Cluster members sometimes report ``currentStatus=unknown`` / "Server is
    unavailable" even while the HTTP Query API can serve the database (the
    browser and ``MATCH (n) RETURN count(n)`` succeed). Treat a successful
    probe query as online so resume is not blocked by a stale topology view.
    """
    _system_execute(f"CREATE DATABASE `{name}` IF NOT EXISTS")

    deadline = time.monotonic() + settings.neo4j_database_online_max_wait_seconds
    last_statuses: set = set()
    while True:
        statuses = {row["currentStatus"] for row in _rows(_system_execute(f"SHOW DATABASE `{name}`"))}
        last_statuses = statuses
        if statuses and statuses == {"online"}:
            logger.info("Neo4j database %r is online", name)
            return
        if _database_accepts_queries(name):
            logger.info(
                "Neo4j database %r accepts queries (SHOW DATABASE status=%s) — treating as online",
                name, statuses or "none",
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Neo4j database {name!r} did not come online within "
                               f"{settings.neo4j_database_online_max_wait_seconds}s "
                               f"(last seen status: {last_statuses or 'none'})")
        time.sleep(settings.neo4j_database_online_poll_interval_seconds)


def _database_accepts_queries(name: str) -> bool:
    """True when the HTTP Query API can run a trivial read against ``name``."""
    if not settings.neo4j_http_url:
        return False
    try:
        response = requests.post(
            f"{settings.neo4j_http_url}/db/{name}/query/v2",
            auth=(settings.neo4j_username, settings.neo4j_password),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={"statement": "RETURN 1 AS ok", "parameters": {}},
            verify=settings.verify_certs,
            timeout=min(15, settings.neo4j_http_timeout_seconds),
        )
        return response.status_code in (200, 202)
    except requests.RequestException:
        return False


class Neo4jRepository:
    def __init__(self, database: Optional[str] = None, *, http_url: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None) -> None:
        if not database:
            raise ValueError("Neo4jRepository requires an explicit database — pass the run's "
                             "own RunContext.neo4j_database. There is no settings default.")
        # Endpoint defaults to the process-global settings (bulk path, and any caller that
        # doesn't route per-message); the incremental Kafka worker passes the endpoint its
        # message's `use_ec2_database` header resolved to, so EC2 and primary graphs are
        # reachable from the same process.
        http_url = http_url if http_url is not None else settings.neo4j_http_url
        username = username if username is not None else settings.neo4j_username
        password = password if password is not None else settings.neo4j_password
        if not http_url:
            raise ValueError("NEO4J_HTTP_URL must be set in .env before writing to Neo4j — "
                              "this guard exists specifically so a run never accidentally "
                              "targets a graph nobody meant to write to.")
        self._url = f"{http_url}/db/{database}/query/v2"
        self._auth = (username, password)
        # Overwritten by calibrate_batch_sizes(); settings above are the fallback.
        self._batch_size = settings.neo4j_default_batch_size
        self._entity_batch_size = settings.neo4j_default_entity_batch_size
        # Reused across calls so the HTTP connection is pooled, not re-opened per call.
        self._session = requests.Session()

    def calibrate_batch_sizes(self) -> None:
        """Probes the real Neo4j server to pick safe batch sizes (doubling
        probe: grow until latency exceeds target or an error occurs, use the
        last safe size). Called once from create_constraints() before any
        real data is written."""
        from services.adaptive_controller import Neo4jBatchCalibrator
        calibrator = Neo4jBatchCalibrator()

        # Entity rows carry JSON descriptions/aliases — heavier per-row than doc rows.
        sample_entity = {
            "name": "CalibrationEntity",
            "type": "CALIBRATION",
            "description": "Temporary calibration row — safe to ignore",
            "source_docs": [],
            "aliases": ["CalibrationEntity"],
        }
        entity_cypher = """
            UNWIND $batch AS row
            MERGE (e:Entity {name: row.name, type: row.type})
            SET e.description = row.description, e.source_docs = row.source_docs,
                e.aliases = row.aliases
        """

        def _entity_probe(rows):
            self._execute(entity_cypher, {"batch": rows})

        calibrated_entity_size = calibrator.calibrate(
            _entity_probe, [sample_entity], label="entity"
        )
        # Cap after probe: synthetic rows understate real entity weight;
        # unconstrained calibration previously landed at ~1280 and overloaded
        # the Query API (500/timeout/400 storms on a large tenant).
        capped = min(calibrated_entity_size, settings.neo4j_entity_batch_max)
        if capped < calibrated_entity_size:
            logger.info(
                "Neo4j entity batch size calibrated to %d — capping at %d",
                calibrated_entity_size, capped,
            )
        self._entity_batch_size = max(1, capped)
        logger.info("Neo4j entity batch size calibrated to %d", self._entity_batch_size)

        # General batch size (docs, chunks, mentions — no embeddings)
        sample_doc = {
            "doc_id": "calibration_doc",
            "title": "Calibration",
            "source": "calibration",
            "published_at": "2024-01-01",
        }
        doc_cypher = """
            UNWIND $batch AS row
            MERGE (d:Document {doc_id: row.doc_id})
            SET d.title = row.title
        """

        def _doc_probe(rows):
            self._execute(doc_cypher, {"batch": rows})

        calibrated_batch_size = calibrator.calibrate(
            _doc_probe, [sample_doc], label="document"
        )
        self._batch_size = calibrated_batch_size
        logger.info("Neo4j general batch size calibrated to %d", self._batch_size)

        # Clean up the synthetic probe nodes the calibration above just wrote.
        self._cleanup_calibration_nodes()

    def _cleanup_calibration_nodes(self) -> None:
        """Remove the synthetic nodes created by calibrate_batch_sizes()'s probes."""
        try:
            self._execute("MATCH (e:Entity {type: 'CALIBRATION'}) DETACH DELETE e", {})
            self._execute("MATCH (d:Document {doc_id: 'calibration_doc'}) DETACH DELETE d", {})
            logger.info("Neo4j calibration probe nodes cleaned up")
        except Exception:  # noqa: BLE001 — don't fail the whole run over cleanup
            logger.exception("Could not clean up Neo4j calibration probe nodes — "
                             "a CalibrationEntity node may remain in the graph")

    def close(self) -> None:
        self._session.close()

    def _execute(self, cypher: str, parameters: dict) -> dict:
        """POSTs one statement to the Query API. Transient failures (deadlock,
        5xx, connection blips — classified by utils/error_classification.py)
        get a short retry before propagating; permanent failures raise on the
        first attempt.

        Reopens self._session before each retry: the shared server sits behind
        a load balancer with no leader-routing on the HTTP API, so the
        persistent session can get pinned to a non-writer member — retrying on
        the same session would hit the same wrong member every time. A fresh
        session gives the load balancer another chance to route to a writer."""
        delay = 0.0
        for attempt in range(settings.neo4j_max_retries):
            if delay > 0:
                time.sleep(delay)
            try:
                response = self._session.post(
                    self._url, auth=self._auth,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json={"statement": cypher, "parameters": parameters},
                    verify=settings.verify_certs,
                    timeout=settings.neo4j_http_timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 — classified below, not swallowed
                if classify_error(exc) != "transient" or attempt == settings.neo4j_max_retries - 1:
                    raise
                delay = settings.neo4j_retry_backoff_seconds * (2 ** attempt) + random.uniform(0, 0.25)
                logger.warning("Neo4j call failed (transient, attempt %d/%d), retrying in %.1fs "
                               "on a fresh connection: %s",
                               attempt + 1, settings.neo4j_max_retries, delay, exc)
                self._session.close()
                self._session = requests.Session()

    def _run_batched(self, cypher: str, rows: list, label: str) -> None:
        for i in range(0, len(rows), self._batch_size):
            batch = rows[i:i + self._batch_size]
            self._execute(cypher, {"batch": batch})
            logger.info("%s: wrote %d/%d", label, min(i + self._batch_size, len(rows)), len(rows))

    def create_constraints(self, calibrate: bool = True) -> None:
        """Uniqueness constraints, so every write below can safely MERGE
        instead of risking duplicate nodes.

        calibrate=False skips calibrate_batch_sizes() — used by the
        incremental worker, which doesn't need bulk's calibrated batch sizes
        and shouldn't pay for a redundant calibration probe on every startup."""
        # Entity identity is keyed on entity_id, not (name, type, tenant_id) — two
        # entities sharing a name no longer collide. Constraints dropped/recreated
        # (disposable dev data); name/type/tenant_id remain plain, non-unique properties.
        self._execute("DROP CONSTRAINT entity_name_type IF EXISTS", {})
        self._execute("DROP CONSTRAINT entity_name_type_tenant IF EXISTS", {})
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.claim_id IS UNIQUE",
        ]
        for stmt in statements:
            self._execute(stmt, {})
        logger.info("Constraints ensured")

        # Non-unique index for filtering/display lookups by name/type/tenant.
        try:
            self._execute(
                "CREATE INDEX entity_name_type_tenant IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name, e.type, e.tenant_id)", {},
            )
        except Exception:  # noqa: BLE001
            logger.exception("Could not create entity_name_type_tenant index — non-fatal")

        # ACL filtering indexes — not enforced yet (no retrieval layer), ready for one.
        for stmt in [
            "CREATE INDEX document_allow_access_control IF NOT EXISTS "
            "FOR (d:Document) ON (d.allow_access_control)",
            "CREATE INDEX chunk_allow_access_control IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.allow_access_control)",
            "CREATE INDEX entity_allow_access_control IF NOT EXISTS "
            "FOR (e:Entity) ON (e.allow_access_control)",
            "CREATE INDEX claim_allow_access_control IF NOT EXISTS "
            "FOR (cl:Claim) ON (cl.allow_access_control)",
        ]:
            try:
                self._execute(stmt, {})
            except Exception:  # noqa: BLE001
                logger.exception("Could not create ACL index (%s) — non-fatal", stmt)
        logger.info("ACL indexes ensured")

        # Tenant filtering indexes — Document/Chunk/Claim identity stays global;
        # tenant_id is just a filterable property, not part of a composite constraint.
        for stmt in [
            "CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id)",
            "CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id)",
            "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
            "CREATE INDEX claim_tenant IF NOT EXISTS FOR (cl:Claim) ON (cl.tenant_id)",
        ]:
            try:
                self._execute(stmt, {})
            except Exception:  # noqa: BLE001
                logger.exception("Could not create tenant index (%s) — non-fatal", stmt)
        logger.info("Tenant indexes ensured")

        # Claim serving indexes (mirrors graph_writer.py) — ready for a retrieval
        # layer, not consumed yet.
        for stmt in [
            "CREATE INDEX claim_subject IF NOT EXISTS FOR (cl:Claim) ON (cl.subject)",
            "CREATE INDEX claim_valid_from IF NOT EXISTS FOR (cl:Claim) ON (cl.valid_from)",
            "CREATE INDEX claim_documented_on IF NOT EXISTS FOR (cl:Claim) ON (cl.documented_on)",
        ]:
            try:
                self._execute(stmt, {})
            except Exception:  # noqa: BLE001
                logger.exception("Could not create claim serving index (%s) — non-fatal", stmt)
        logger.info("Claim serving indexes ensured")

        # Fulltext index for typo-tolerant/lexical entity lookup. Needs the native
        # list property e.aliases, not the JSON-encoded aliases_json — Neo4j
        # fulltext can't index into a JSON string.
        try:
            self._execute(
                "CREATE FULLTEXT INDEX entity_aliases IF NOT EXISTS FOR (e:Entity) ON EACH [e.aliases]", {},
            )
            logger.info("Fulltext index 'entity_aliases' ensured")
        except Exception:  # noqa: BLE001
            logger.exception("Could not create fulltext index 'entity_aliases' — non-fatal")

        # People/org graph indexes (mirrors graph_writer.py). Team/Department names
        # are unique within a tenant, not globally.
        for stmt in [
            "CREATE INDEX person_user_id IF NOT EXISTS FOR (e:Entity) ON (e.person_user_id)",
            "CREATE INDEX person_manager_id IF NOT EXISTS FOR (e:Entity) ON (e.person_manager_id)",
            "CREATE INDEX person_label_uid IF NOT EXISTS FOR (p:Person) ON (p.person_user_id)",
            "CREATE INDEX person_account_id IF NOT EXISTS FOR (p:Person) ON (p.tenant_id, p.account_id)",
            # Resolves materialize_authorship_edges' (connector_type, source_key)
            # join, and label_person_records'/find_person_mention_merges' keyed
            # -vs-bare split. A new name, not a redefinition of person_label_uid:
            # CREATE INDEX ... IF NOT EXISTS fails on a same-name index with a
            # different schema, and this key replaced that one.
            "CREATE INDEX person_source_key IF NOT EXISTS FOR (p:Person) "
            "ON (p.connector_type, p.source_key)",
            # Authorship graph: resolves materialize_authorship_edges' created_by_id join.
            "CREATE INDEX doc_created_by IF NOT EXISTS FOR (d:Document) ON (d.created_by_id)",
            # Confluence PERSON identity (account_id / email / configured user_id).
            "CREATE INDEX entity_account_id IF NOT EXISTS FOR (e:Entity) ON (e.account_id)",
            "CREATE INDEX entity_user_email IF NOT EXISTS FOR (e:Entity) ON (e.user_email)",
            "CREATE INDEX entity_user_id IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
        ]:
            try:
                self._execute(stmt, {})
            except Exception:  # noqa: BLE001
                logger.exception("Could not create People/org graph index (%s) — non-fatal", stmt)
        try:
            self._execute(
                "CREATE CONSTRAINT team_name_tenant IF NOT EXISTS FOR (t:Team) "
                "REQUIRE (t.tenant_id, t.name) IS UNIQUE", {},
            )
            self._execute(
                "CREATE CONSTRAINT department_name_tenant IF NOT EXISTS FOR (dp:Department) "
                "REQUIRE (dp.tenant_id, dp.name) IS UNIQUE", {},
            )
            logger.info("People/org graph constraints ensured")
        except Exception:  # noqa: BLE001
            logger.exception("Could not create Team/Department constraints — non-fatal")

        # Lexical arm for ai-graph-retrieval (its `entity_fulltext_index`).
        # e.person_search_terms is vestigial: the rich-profile re-embedding that
        # wrote it is gone, so the property is never set and contributes nothing.
        # Kept in the schema deliberately — CREATE FULLTEXT INDEX IF NOT EXISTS
        # will not redefine an existing index, so dropping the property here is a
        # silent no-op on any graph that already has this index, and an unwritten
        # property costs nothing. Remove it in a migration that drops and
        # recreates, together with retrieval's reference to it.
        try:
            self._execute(
                "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) "
                "ON EACH [e.name, e.aliases, e.person_search_terms]", {},
            )
            logger.info("Fulltext index 'entity_search' ensured")
        except Exception:  # noqa: BLE001
            logger.exception("Could not create fulltext index 'entity_search' — non-fatal")

        # Calibrate batch sizes against the real server before any data is written.
        if calibrate:
            self.calibrate_batch_sizes()

    def write_documents(self, docs: list, tenant_id: str, doc_acl_by_doc: Optional[dict] = None) -> None:
        """docs: SourceDocuments (connectors/base.py). doc_acl_by_doc:
        {doc_id: [acl_token, ...]} — dropped/gated documents never reach here
        (see core/pipeline.py).

        Reads only the contract. This method used to read source field names
        directly — content_title, object_type, content_publish_on_date, plus a
        connector_type/url pair guessed from whether a "url" field happened to
        be present — which is exactly the coupling the connector boundary
        exists to remove. Each of those now arrives already named.

        tenant_id is stamped on every Document; doc_id remains the global
        uniqueness key.

        source_type prefers doc["connector_type"]; falls back to object_type
        for bulk (ES) docs, which don't carry connector_type. `source`
        (object_type) is kept as-is for back-compat.

        Document nodes never receive a ``type`` property. Callers must normalize
        source identity first (e.g. Confluence ES ``type: page`` → in-doc
        ``object_type: page``); that value is written only as ``d.source``.

        created_at / updated_at prefer Confluence-style names, then fall back to
        createddate / lastmodifieddate (content/file + page slim aliases). Empty
        values become null and are coalesced so a thin incremental write does
        not blank timestamps already on the node.

        created_by_id/modified_by_id/author_name are coalesced, never blanked
        by a later write that doesn't carry them — materialize_authorship_edges
        below resolves them to the canonical :Person node."""
        doc_acl_by_doc = doc_acl_by_doc or {}
        rows = []
        for doc in docs:
            rows.append({
                "doc_id": doc.doc_id,
                "tenant_id": tenant_id,
                "title": doc.title,
                "source": doc.object_type,
                "source_type": doc.object_type,
                # ai-graph-retrieval gates its people-profile join on
                # `d.object_type = 'people'` (graph/cypher.py CHUNK_QUOTES), so the
                # value has to land under that exact name too — `source`/`source_type`
                # carried it but no reader looks there for this check.
                "object_type": doc.object_type,
                "connector_type": doc.connector_type.value,
                "url": doc.url or "",
                "published_at": doc.published_at,
                "created_at": doc.published_at or None,
                "updated_at": doc.updated_at or None,
                "acl": doc_acl_by_doc.get(doc.doc_id) or [],
                "created_by_id": doc.author.source_key if doc.author else None,
                "modified_by_id": doc.editor.source_key if doc.editor else None,
                "author_name": doc.author.display_name if doc.author else None,
                "doc_preview_text": doc.preview_text or None,
            })
        self._run_batched(
            """
            UNWIND $batch AS row
            MERGE (d:Document {doc_id: row.doc_id})
            SET d.title = row.title, d.source = row.source, d.published_at = row.published_at,
                d.source_type = row.source_type, d.object_type = row.object_type,
                d.created_at = coalesce(row.created_at, d.created_at),
                d.updated_at = coalesce(row.updated_at, d.updated_at),
                d.connector_type = row.connector_type, d.url = row.url,
                d.allow_access_control = row.acl, d.tenant_id = row.tenant_id,
                d.created_by_id = coalesce(row.created_by_id, d.created_by_id),
                d.modified_by_id = coalesce(row.modified_by_id, d.modified_by_id),
                d.author_name = coalesce(row.author_name, d.author_name),
                d.doc_preview_text = coalesce(row.doc_preview_text, d.doc_preview_text)
            """,
            rows, "Documents",
        )

    def write_chunks(self, docs: list, tenant_id: str, doc_acl_by_doc: Optional[dict] = None,
                     chunks_vector_store: Optional[VectorStore] = None,
                     embedder: Optional[EmbeddingService] = None) -> None:
        """One row per chunk, with its parent doc_id, so the HAS_CHUNK edge
        and MERGE happen in the same batched pass. A chunk's acl and url are
        always exactly its parent document's. tenant_id matches the parent
        document's.

        recursive_chunk_id is a stable structural id used by
        replace_document_chunks to diff a document's old vs. new chunks
        (chunk_id itself regenerates on every extraction pass, so it can't be
        the diff key).

        chunks_vector_store pushes each chunk's vector to Qdrant after the Neo4j
        write commits, per the run's consume_upstream_vector:
          true  -- copy the chunk's upstream ES vector; a chunk without one (or
                   with one at the wrong dimension — a per-chunk anomaly, since
                   the bulk run asserted the index's space at startup) is
                   skipped, NEVER embedded by us.
          false -- upstream vectors are ignored entirely; embedder embeds every
                   chunk in the run's own space.
        Any resulting vector whose dimension doesn't EQUAL the run's
        embedding_dimension is dropped and logged, never pushed — even a freshly
        re-embedded one, in case of an unexpected model/config change. Qdrant's
        upsert is all-or-nothing per batch, so one bad-dimension vector would
        otherwise fail the whole batch, including good sibling vectors.

        Without an embedder, a chunk that would otherwise be embedded is skipped
        instead — still searchable by text/graph, just without a vector."""
        doc_acl_by_doc = doc_acl_by_doc or {}
        # doc.chunks is already deduped by the connector — the upstream recursive
        # chunker's overlapping windows share a recursive_chunk_id and would otherwise
        # each get written as a separate Chunk node with identical text.
        #
        # No "vector" key here -- Chunk nodes carry no vector property (only Qdrant
        # does, chunks_vector_store below), so it never belongs in the batch sent to
        # Neo4j. vectors_by_chunk_id below is where the vector-push logic reads from
        # instead.
        rows = [{
            "doc_id": doc.doc_id,
            "chunk_id": chunk.chunk_id,
            "recursive_chunk_id": chunk.recursive_chunk_id or chunk.chunk_id,
            "tenant_id": tenant_id,
            "text": chunk.text or "",
            "acl": doc_acl_by_doc.get(doc.doc_id) or [],
            "url": doc.url or "",
        } for doc in docs for chunk in doc.chunks]
        self._run_batched(
            """
            UNWIND $batch AS row
            MATCH (d:Document {doc_id: row.doc_id, tenant_id: row.tenant_id})
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c.text = row.text, c.allow_access_control = row.acl, c.tenant_id = row.tenant_id,
                c.recursive_chunk_id = row.recursive_chunk_id, c.url = row.url,
                c.doc_id = row.doc_id
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            rows, "Chunks",
        )
        if chunks_vector_store is not None:
            vectors_by_chunk_id = {chunk.chunk_id: chunk.vector for doc in docs for chunk in doc.chunks}
            # The running run's recorded flag + dimension (settings on the bulk
            # path; the bulk run's own values on an incremental event), so an
            # incrementally-written chunk is copied/embedded exactly as its bulk
            # run's vec_chunks space demands, not by whatever this worker's current
            # settings happen to be.
            run_config = effective_run_config()
            expected_chunk_dim = run_config.embedding_dimension

            def _dimension_ok(chunk_id: str) -> bool:
                vector = vectors_by_chunk_id[chunk_id]
                return bool(vector) and len(vector) == expected_chunk_dim

            skip_reasons: dict[str, str] = {}
            if run_config.consume_upstream_vector:
                good_rows = [r for r in rows if _dimension_ok(r["chunk_id"])]
                needs_reembed: list = []
                for r in rows:
                    if not _dimension_ok(r["chunk_id"]):
                        skip_reasons[r["chunk_id"]] = "consume_upstream_vector=true never embeds"
            else:
                good_rows = []
                needs_reembed = rows  # upstream vector ignored entirely, even a matching one

            reembedded: dict[str, list] = {}
            if needs_reembed:
                if embedder is not None:
                    fresh_vectors = embedder.embed_batch([r["text"] for r in needs_reembed])
                    reembedded = {r["chunk_id"]: v for r, v in zip(needs_reembed, fresh_vectors)}
                else:
                    for r in needs_reembed:
                        skip_reasons[r["chunk_id"]] = "no embedder given to re-embed"

            for chunk_id, reason in skip_reasons.items():
                vector = vectors_by_chunk_id[chunk_id]
                logger.warning(
                    "chunk_id=%s: no vector pushed (%s; upstream dimension=%s, expected=%d) — "
                    "chunk remains searchable by text/graph only",
                    chunk_id, reason, len(vector) if vector else "none", expected_chunk_dim,
                )

            vector_rows = [
                {"entity_id": row["chunk_id"], "embedding": vectors_by_chunk_id[row["chunk_id"]],
                 "type": "CHUNK", "tenant_id": row["tenant_id"], "acl": row["acl"]}
                for row in good_rows
            ] + [
                {"entity_id": row["chunk_id"], "embedding": reembedded[row["chunk_id"]],
                 "type": "CHUNK", "tenant_id": row["tenant_id"], "acl": row["acl"]}
                for row in needs_reembed if row["chunk_id"] in reembedded
            ]

            # Reject any vector whose dimension doesn't EQUAL the run's space --
            # even a freshly re-embedded one, in case of an unexpected model/config
            # change. Equality, not a global accepted set: a 1536-d vector must not
            # pass into a 256-d-configured space just because 1536 is a dimension
            # some OTHER space uses.
            good_vector_rows = []
            for vr in vector_rows:
                if len(vr["embedding"]) == expected_chunk_dim:
                    good_vector_rows.append(vr)
                else:
                    logger.warning(
                        "chunk_id=%s: embedding dimension %d != the run's %d — dropping, "
                        "chunk remains searchable by text/graph only",
                        vr["entity_id"], len(vr["embedding"]), expected_chunk_dim,
                    )
            push_vectors_in_batches(chunks_vector_store, good_vector_rows, step="qdrant_chunk_upsert_batch")

    _ENTITY_WRITE_CYPHER = """
        UNWIND $batch AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        SET e.name = row.name, e.type = row.type, e.tenant_id = row.tenant_id,
            e.description = CASE WHEN row.description <> '' THEN row.description
                                 ELSE e.description END,
            e.descriptions_json = row.descriptions_json, e.source_docs = row.source_docs,
            e.aliases_json = row.aliases_json, e.aliases = row.aliases,
            e.has_vector = CASE WHEN size(row.embedding) > 0 THEN true
                                ELSE coalesce(e.has_vector, false) END
        SET e += row.person_fields
        FOREACH (ignored IN CASE WHEN row.person_fields.source_key IS NULL THEN [] ELSE [1] END |
                 SET e:Person)
        WITH e, row,
             coalesce(e.doc_ids, []) + [d IN row.source_docs WHERE NOT d IN coalesce(e.doc_ids, [])]
             AS merged_doc_ids,
             coalesce(e.allow_access_control, []) + [a IN row.acl WHERE NOT a IN coalesce(e.allow_access_control, [])]
             AS merged_acl
        SET e.doc_ids = merged_doc_ids, e.support = size(merged_doc_ids),
            e.allow_access_control = merged_acl
        RETURN row.entity_id AS entity_id, row.type AS type, row.embedding AS embedding,
               merged_acl
    """

    _PERSON_STAMP_CYPHER = """
        UNWIND $batch AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        SET e += row.person_fields
        FOREACH (ignored IN CASE WHEN row.person_fields.source_key IS NULL THEN [] ELSE [1] END |
                 SET e:Person)
    """

    def stamp_person_fields(self, rows: list[dict], tenant_id: str) -> None:
        """Narrow identity stamp onto EXISTING entities: merges person_fields
        (and the :Person label) only. Never goes through write_entities — that
        path unconditionally SETs name/descriptions_json/aliases/source_docs,
        which would clobber canonicalized state when stamping after Step 5.
        rows: [{entity_id, person_fields}, ...]."""
        batch = [{"entity_id": r["entity_id"], "person_fields": r["person_fields"]}
                 for r in rows if r.get("entity_id") and r.get("person_fields")]
        if not batch:
            return
        self._execute(self._PERSON_STAMP_CYPHER, {"batch": batch})

    @staticmethod
    def _entity_batch_should_split(exc: Exception, batch_len: int) -> bool:
        """True when a failed multi-row entity write should be halved and retried.

        Transient (5xx/timeout) always splits. HTTP 400 on a multi-row batch also
        splits: Neo4j Query API sometimes returns 400 for oversized/heavy
        payloads that succeed as smaller chunks, and splitting isolates a single
        permanently-bad row so the rest of the corpus can still land. A 400 on
        a single row still propagates (truly permanent for that entity)."""
        if batch_len <= 1:
            return False
        if classify_error(exc) == "transient":
            return True
        try:
            from requests.exceptions import HTTPError
        except ImportError:
            return False
        if isinstance(exc, HTTPError) and exc.response is not None:
            return exc.response.status_code == 400
        return False

    def _write_entity_batch(self, batch: list[dict]) -> list[dict]:
        """Writes one batch of entity rows, halving and retrying on transient
        failure (and on HTTP 400 for multi-row batches). self._entity_batch_size
        is calibrated against tiny synthetic rows, but a real entity that
        accumulated aliases/descriptions across many documents is much heavier
        per row and can still time out — this self-corrects instead of guessing
        a smaller constant. Bottoms out at a single row: a failure there is a
        real problem, not a batch-size one, and propagates up."""
        try:
            response = self._execute(self._ENTITY_WRITE_CYPHER, {"batch": batch})
            return _rows(response)
        except Exception as exc:  # noqa: BLE001 — classified below, not swallowed
            if not self._entity_batch_should_split(exc, len(batch)):
                raise
            mid = len(batch) // 2
            logger.warning("Entities: batch of %d timed out/failed (%s) — splitting into "
                           "%d + %d and retrying", len(batch), exc, mid, len(batch) - mid)
            return self._write_entity_batch(batch[:mid]) + self._write_entity_batch(batch[mid:])

    def write_entities(self, final_entities: dict, tenant_id: str, canon_to_id: dict,
                       vector_store: Optional[VectorStore] = None) -> None:
        """final_entities: {name: {type, source_docs, embedding, descriptions, aliases, acl}}
        — Step 3's output, acl added by core/pipeline.py (union of every
        supporting document's acl).

        Embeddings live in Qdrant (services/vector_store.py); only a
        lightweight e.has_vector boolean is stamped here. Each batch RETURNs
        entity_id/merged_acl so the Qdrant push uses the actual final
        (unioned) acl, not just this batch's partial view. vector_store=None
        skips the Qdrant push; the graph write still happens.

        Any embedding whose dimension doesn't EQUAL the run's embedding_dimension
        is dropped before the Qdrant push (same rule as write_chunks), and
        e.has_vector is corrected back to false so Neo4j never claims a vector
        Qdrant doesn't have.

        canon_to_id: {name: entity_id} (repository/entity_registry.py) — MERGE
        keys on entity_id alone, not (name, type, tenant_id), so two
        real-world entities sharing a name stay separate nodes.

        doc_ids/support are merged with whatever's already on the node, never
        overwritten — bulk's source_docs is unioned in. Overwriting instead
        would let a bulk run erase what incremental added after bulk's scan
        started, and would leave freshly-created entities with no real
        doc_ids baseline for incremental's own merge logic (and
        retract_document's deletion check) to build on.

        descriptions_json/aliases_json are JSON-encoded (a Neo4j property
        can't be a list of maps) — one entry per contributing document, each
        tagged with its own acl. Deliberately not collapsed into a single
        description/alias: there's no retrieval layer yet to pick a safe entry
        per user, so nothing here risks blending a private document's text
        into what's shown through a public one.

        e.aliases is a denormalized plain list of alias strings only (no
        doc_id/acl), kept alongside aliases_json because Neo4j's fulltext
        index needs a native list property, not a JSON string.

        e.allow_access_control is unioned with whatever's already on the node,
        never overwritten from row.acl alone — row.acl only reflects
        documents fetched in THIS run, and a narrower-scoped run (--limit, a
        retry) would otherwise wipe out an access grant a
        previously-processed document had contributed. Narrowing only ever
        happens via retract_document's live recompute.

        data["person_fields"] (MATERIALIZE's output, services/people/
        materialize.py — present only on a person their connector named with a
        key): written through with `SET e += ...` and never read here, so a
        connector's own person properties need no change to this method. It
        drops falsy values before it hands them over, which is what keeps a
        later write from blanking a property an earlier one knew — the
        per-field coalesce() this replaces did the same thing, for six
        hard-coded HR field names only.

        `source_key` in that map is also what earns the second :Person label:
        a person some source named with a key, as opposed to a bare name
        extracted from prose that was never confirmed against a record."""
        rows = []
        skipped = 0
        for key, data in final_entities.items():
            entity_id = canon_to_id.get(key)
            if not entity_id:
                skipped += 1
                continue
            # PERSON entries are keyed by entity_id, not name (two people can share a
            # name); data["name"] carries the real display name in that case.
            name = data.get("name") or key
            rows.append({
                "entity_id": entity_id,
                "name": name, "type": data["type"], "tenant_id": tenant_id,
                # The LLM's one merged description (canonicalize_corpus). Empty on
                # writers that don't carry it (persons, incremental) — the CASE in
                # the Cypher keeps the existing value rather than blanking it.
                "description": (data.get("description") or "").strip(),
                "descriptions_json": json.dumps(data.get("descriptions") or []),
                "aliases_json": json.dumps(data.get("aliases") or [{"name": name, "doc_id": None}]),
                "aliases": sorted({
                    (a.get("name") if isinstance(a, dict) else a)
                    for a in (data.get("aliases") or [{"name": name}])
                    if (a.get("name") if isinstance(a, dict) else a)
                }),
                "source_docs": data.get("source_docs") or [], "embedding": data.get("embedding") or [],
                "acl": data.get("acl") or [],
                # MATERIALIZE's output, present only on a person the connector
                # named with a key. Written through as-is, never inspected.
                "person_fields": data.get("person_fields") or {},
            })
        if skipped:
            logger.warning("Entities: %d entit(y/ies) had no canon_to_id entry — skipped, "
                           "not written with a missing entity_id", skipped)
        pause = settings.neo4j_entity_write_pause_seconds
        for i in range(0, len(rows), self._entity_batch_size):
            if i > 0 and pause > 0:
                time.sleep(pause)
            batch = rows[i:i + self._entity_batch_size]
            result_rows = self._write_entity_batch(batch)
            logger.info("Entities: wrote %d/%d", min(i + self._entity_batch_size, len(rows)), len(rows))
            if vector_store is not None:
                # Equality against the run's own space, not a global accepted set —
                # a 1536-d vector must not pass into a 256-d-configured space just
                # because 1536 is a dimension some OTHER space uses.
                expected_dim = effective_run_config().embedding_dimension
                vector_rows = []
                rejected_entity_ids: list[str] = []
                for r in result_rows:
                    embedding = r.get("embedding")
                    if not embedding:
                        continue
                    if len(embedding) != expected_dim:
                        rejected_entity_ids.append(r["entity_id"])
                        logger.warning(
                            "entity_id=%s: embedding dimension %d != the run's %d — dropping, "
                            "entity remains in graph without a vector",
                            r["entity_id"], len(embedding), expected_dim,
                        )
                        continue
                    vector_rows.append({
                        "entity_id": r["entity_id"], "embedding": embedding, "type": r["type"],
                        "tenant_id": tenant_id, "acl": r["merged_acl"],
                    })
                if rejected_entity_ids:
                    self.mark_vector_missing(tenant_id, rejected_entity_ids)
                # Dual-write safety: has_vector was already set true above; roll it
                # back to false if the Qdrant push exhausts its retries.
                push_vectors_in_batches(
                    vector_store, vector_rows, step="qdrant_entity_upsert_batch",
                    rollback=lambda ids: self.mark_vector_missing(tenant_id, ids),
                )

    def write_mentions(self, mentions: list[tuple], tenant_id: str) -> None:
        """mentions: [(chunk_id, entity_id, entity_type), ...] — Step 4's
        output, keyed by entity_id. tenant_id scopes the Chunk match so a
        chunk can only mention an entity from its own tenant."""
        rows = [{"chunk_id": cid, "entity_id": entity_id, "tenant_id": tenant_id}
                for cid, entity_id, _etype in mentions]
        self._run_batched(
            """
            UNWIND $batch AS row
            MATCH (c:Chunk {chunk_id: row.chunk_id, tenant_id: row.tenant_id})
            MATCH (e:Entity {entity_id: row.entity_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            rows, "Mentions",
        )

    def _read_existing_descriptions(self, keys: list[tuple[str, str, str]]) -> dict:
        """Batched pre-read of each edge's current descriptions_json, decoded.
        Neo4j can't append to a JSON-encoded string property natively, so
        write_relationships reads the old value here, merges in Python, and
        writes the combined result back — same batching as _run_batched."""
        existing: dict = {}
        for i in range(0, len(keys), self._batch_size):
            batch = [{"src_id": s, "relationship_type": r, "tgt_id": t}
                     for s, r, t in keys[i:i + self._batch_size]]
            response = self._execute(
                """
                UNWIND $batch AS row
                MATCH (src:Entity {entity_id: row.src_id})
                      -[rel:EXTRACTED_REL {relationship_type: row.relationship_type}]->
                      (tgt:Entity {entity_id: row.tgt_id})
                RETURN row.src_id AS src_id, row.relationship_type AS relationship_type,
                       row.tgt_id AS tgt_id, rel.descriptions_json AS descriptions_json
                """,
                {"batch": batch},
            )
            for row in _rows(response):
                key = (row["src_id"], row["relationship_type"], row["tgt_id"])
                try:
                    existing[key] = json.loads(row.get("descriptions_json") or "[]")
                except (TypeError, ValueError):
                    existing[key] = []
        return existing

    def write_relationships(self, final_edges: dict, tenant_id: str) -> None:
        """final_edges: {(src_id, rel_type, tgt_id): {description, descriptions,
        chunk_ids, raw_types, src_type, tgt_type, acl}} — Step 4's output, keyed
        by entity_id.

        MATCHes on entity_id alone. r.allow_access_control is the union of
        every contributing document's acl (computed in
        services/edge_pool_service.py). r.description is the singular
        LLM-merged text (same split as Entity.description vs
        descriptions_json, edge_pool_service.py's build_edge_pool/
        merge_edge_pools); r.descriptions_json is JSON-encoded (a Neo4j
        property can't be a list of maps) — one entry per contributing chunk,
        each tagged with its own document's acl, deliberately not collapsed
        into one description, for the same ACL-leak reason as write_entities.

        chunk_ids/raw_types/acl/descriptions are all merged with whatever's
        already on the relationship, never overwritten — this method can run
        many times against the same edge (later batches, the incremental
        path), and a plain SET would erase what an earlier write contributed.
        descriptions_json is pre-read via _read_existing_descriptions and
        merged in Python (see that method's docstring for why); the others
        merge natively in Cypher. r.support = size(merged chunk_ids).
        r.description keeps whatever's already there when this batch's value
        is empty, same CASE guard write_entities uses for e.description.

        A single EXTRACTED_REL relationship type is used, with the semantic
        type (e.g. "ADDS_FEATURE") stored as the relationship_type property —
        avoids a dynamic per-type Neo4j label, which can't be parameterized
        and would need Cypher-injection sanitization. relationship_type is
        part of the MERGE key alongside src/tgt, so two different
        relationship types between the same pair of entities get two
        distinct edges."""
        existing_descriptions = self._read_existing_descriptions(list(final_edges.keys()))
        rows = []
        for (src_id, rel_type, tgt_id), data in final_edges.items():
            merged_descriptions = (existing_descriptions.get((src_id, rel_type, tgt_id), [])
                                   + (data.get("descriptions") or []))
            rows.append({
                "src_id": src_id, "relationship_type": rel_type,
                "src_type": data.get("src_type") or "",
                "tgt_id": tgt_id, "tgt_type": data.get("tgt_type") or "",
                "tenant_id": tenant_id,
                "description": (data.get("description") or "").strip(),
                "descriptions_json": json.dumps(merged_descriptions),
                "chunk_ids": data.get("chunk_ids") or [],
                "raw_types": data.get("raw_types") or [],
                "acl": data.get("acl") or [],
            })
        self._run_batched(
            """
            UNWIND $batch AS row
            MATCH (src:Entity {entity_id: row.src_id})
            MATCH (tgt:Entity {entity_id: row.tgt_id})
            MERGE (src)-[r:EXTRACTED_REL {relationship_type: row.relationship_type}]->(tgt)
            SET r.description = CASE WHEN row.description <> '' THEN row.description
                                     ELSE r.description END,
                r.descriptions_json = row.descriptions_json, r.tenant_id = row.tenant_id
            WITH r, row,
                 coalesce(r.chunk_ids, []) + [c IN row.chunk_ids WHERE NOT c IN coalesce(r.chunk_ids, [])]
                 AS merged_chunk_ids,
                 coalesce(r.raw_types, []) + [t IN row.raw_types WHERE NOT t IN coalesce(r.raw_types, [])]
                 AS merged_raw_types,
                 coalesce(r.allow_access_control, []) + [a IN row.acl WHERE NOT a IN coalesce(r.allow_access_control, [])]
                 AS merged_acl
            SET r.chunk_ids = merged_chunk_ids, r.support = size(merged_chunk_ids),
                r.raw_types = merged_raw_types, r.allow_access_control = merged_acl
            """,
            rows, "Relationships",
        )

    def mark_vector_missing(self, tenant_id: str, entity_ids: list[str]) -> int:
        """Dual-write rollback: corrects e.has_vector back to false for
        entities whose Qdrant push exhausted its retries, so Neo4j never
        claims a vector Qdrant doesn't have. Safe to call even if Qdrant
        didn't actually lose it — the next mention just re-embeds and
        re-upserts."""
        if not entity_ids:
            return 0
        response = self._execute(
            "UNWIND $entity_ids AS eid MATCH (e:Entity {entity_id: eid, tenant_id: $tenant_id}) "
            "SET e.has_vector = false RETURN count(e) AS n",
            {"tenant_id": tenant_id, "entity_ids": entity_ids},
        )
        return _scalar(response, "n") or 0

    def recompute_degree(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> int:
        """e.degree: how many DISTINCT other entities this entity is connected to
        via EXTRACTED_REL. Recomputed fresh (not incrementally tracked) so it's
        always exactly right.

        Distinct neighbours, NOT edge count: a pair joined by two relationship
        types (Omaha `located_in` Nebraska and Omaha `part_of` Nebraska) is one
        connection, not two. Edge counting inflated hubs badly — "Globex" scored
        7,284 by edges vs 4,955 real neighbours.

        ai-graph-retrieval consumes this as a ranking prior in entity resolution
        (`entity_w_degree * log10(1 + degree)`) and as the traversal tiebreak
        (`ORDER BY score DESC, coalesce(nbr.degree, 0) DESC`), so the definition
        here is a retrieval contract — changing it changes ranking.

        entity_ids scopes to one document/run's touched entities; None
        recomputes the whole tenant (bulk's fresh-run case).

        Batched (client-side over entity_ids, or server-side IN TRANSACTIONS for
        the whole-tenant case) — a single autocommit SET across a large corpus
        (seen on Confluence-sized tenants, 150k+ entities) can exhaust the
        server's heap and silently no-op every entity, since the caller
        (core/pipeline.py) treats this as non-fatal and just logs."""
        if entity_ids is not None:
            n = 0
            for i in range(0, len(entity_ids), self._batch_size):
                batch = entity_ids[i:i + self._batch_size]
                response = self._execute(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})
                    WHERE e.entity_id IN $entity_ids
                    SET e.degree = COUNT { MATCH (e)-[:EXTRACTED_REL]-(nbr:Entity) RETURN DISTINCT nbr }
                    RETURN count(e) AS n
                    """,
                    {"tenant_id": tenant_id, "entity_ids": batch},
                )
                n += _scalar(response, "n") or 0
            return n
        response = self._execute(
            f"""
            CALL {{
              MATCH (e:Entity {{tenant_id: $tenant_id}})
              SET e.degree = COUNT {{ MATCH (e)-[:EXTRACTED_REL]-(nbr:Entity) RETURN DISTINCT nbr }}
              RETURN 1 AS touched
            }} IN TRANSACTIONS OF {self._batch_size} ROWS
            RETURN count(touched) AS n
            """,
            {"tenant_id": tenant_id},
        )
        return _scalar(response, "n") or 0

    def recompute_claim_support(self, tenant_id: str, claim_ids: Optional[list[str]] = None) -> int:
        """cl.support: how many distinct Chunks corroborate this claim, via
        SUPPORTED_BY. Recomputed fresh (not incrementally tracked) — write_claims
        MERGEs the same claim_id again whenever the identical (subject,
        predicate, object, valid_from, valid_to) triple turns up in another
        document, adding another SUPPORTED_BY edge each time, so a plain
        increment-on-write would double count on retries/redelivery.

        ai-graph-retrieval reads this as the claim's evidence weight
        (`support_weight * log10(1 + support)`), coalescing to 1 for claims
        written before this field existed.

        claim_ids scopes to one document/run's touched claims; None recomputes
        the whole tenant (bulk's fresh-run case). Batched the same way as
        recompute_degree, for the same reason — a single autocommit SET across
        a large tenant's claims can exhaust the server's heap."""
        if claim_ids is not None:
            n = 0
            for i in range(0, len(claim_ids), self._batch_size):
                batch = claim_ids[i:i + self._batch_size]
                response = self._execute(
                    """
                    MATCH (cl:Claim {tenant_id: $tenant_id})
                    WHERE cl.claim_id IN $claim_ids
                    SET cl.support = COUNT { MATCH (cl)-[:SUPPORTED_BY]->(:Chunk) }
                    RETURN count(cl) AS n
                    """,
                    {"tenant_id": tenant_id, "claim_ids": batch},
                )
                n += _scalar(response, "n") or 0
            return n
        response = self._execute(
            f"""
            CALL {{
              MATCH (cl:Claim {{tenant_id: $tenant_id}})
              SET cl.support = COUNT {{ MATCH (cl)-[:SUPPORTED_BY]->(:Chunk) }}
              RETURN 1 AS touched
            }} IN TRANSACTIONS OF {self._batch_size} ROWS
            RETURN count(touched) AS n
            """,
            {"tenant_id": tenant_id},
        )
        return _scalar(response, "n") or 0

    # ------------------------------------------------------------------ People / org graph
    # Deterministic (no-LLM) post-processing turning the HR fields write_entities
    # stamped into native graph structure (mirrors graph_writer.py). entity_ids
    # scopes to a run's affected persons; None means the whole tenant.

    def label_person_records(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> int:
        """Adds the secondary :Person label to persons their source named with a
        key (those carrying a source_key) so directory/org queries can scope to
        (:Person) with an index, distinct from a bare-name PERSON mention that
        was never confirmed against a real record.

        No pipeline calls this any more — write_entities labels a node as it
        writes it, and both paths write persons through MATERIALIZE now. Kept as
        the repair for an older graph: source_key present, label missing."""
        response = self._execute(
            """
            MATCH (e:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE e.source_key IS NOT NULL AND e.source_key <> ''
              AND ($entity_ids IS NULL OR e.entity_id IN $entity_ids)
            SET e:Person
            RETURN count(e) AS n
            """,
            {"tenant_id": tenant_id, "entity_ids": entity_ids},
        )
        return _scalar(response, "n") or 0

    def materialize_reporting_edges(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> int:
        """(:Person employee)-[:REPORTS_TO]->(:Person manager), resolving
        person_manager_id -> the manager's own source_key within the same
        connector_type. Exact and idempotent (no LLM). Was person_user_id, an HR
        field name only Zeus populates, so no other connector got edges."""
        response = self._execute(
            """
            MATCH (e:Person {tenant_id: $tenant_id})
            WHERE e.person_manager_id IS NOT NULL AND e.person_manager_id <> ''
              AND ($entity_ids IS NULL OR e.entity_id IN $entity_ids)
            MATCH (m:Person {tenant_id: $tenant_id, connector_type: e.connector_type,
                             source_key: e.person_manager_id})
            WHERE m.entity_id <> e.entity_id
            MERGE (e)-[:REPORTS_TO]->(m)
            RETURN count(*) AS n
            """,
            {"tenant_id": tenant_id, "entity_ids": entity_ids},
        )
        return _scalar(response, "n") or 0

    def retire_person_manages(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> int:
        """Deletes person<->person EXTRACTED_REL{MANAGES} edges — LLM-extracted
        MANAGES is unreliable for org hierarchy; REPORTS_TO above is
        authoritative. A MANAGES edge involving a non-PERSON entity is left
        intact (legitimately semantic)."""
        response = self._execute(
            """
            MATCH (a:Entity {tenant_id: $tenant_id, type: 'PERSON'})
                  -[r:EXTRACTED_REL {relationship_type: 'MANAGES'}]->
                  (b:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE $entity_ids IS NULL OR a.entity_id IN $entity_ids OR b.entity_id IN $entity_ids
            DELETE r
            RETURN count(r) AS n
            """,
            {"tenant_id": tenant_id, "entity_ids": entity_ids},
        )
        return _scalar(response, "n") or 0

    def materialize_org_membership(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> dict:
        """(:Person)-[:MEMBER_OF]->(:Department|:Team) from the People
        department/team fields. Deterministic, no LLM; org nodes keyed by
        name within the tenant."""
        params = {"tenant_id": tenant_id, "entity_ids": entity_ids}
        dep = self._execute(
            """
            MATCH (e:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE e.person_department IS NOT NULL AND e.person_department <> ''
              AND ($entity_ids IS NULL OR e.entity_id IN $entity_ids)
            MERGE (d:Department {tenant_id: $tenant_id, name: e.person_department})
            MERGE (e)-[:MEMBER_OF]->(d)
            RETURN count(*) AS n
            """,
            params,
        )
        team = self._execute(
            """
            MATCH (e:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE e.person_team IS NOT NULL AND e.person_team <> ''
              AND ($entity_ids IS NULL OR e.entity_id IN $entity_ids)
            MERGE (t:Team {tenant_id: $tenant_id, name: e.person_team})
            MERGE (e)-[:MEMBER_OF]->(t)
            RETURN count(*) AS n
            """,
            params,
        )
        return {"department_edges": _scalar(dep, "n") or 0, "team_edges": _scalar(team, "n") or 0}

    def compute_org_rollups(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> int:
        """Caches direct_reports (|incoming REPORTS_TO|) and org_size
        (|transitive reports| via REPORTS_TO*) on manager nodes — pure
        traversals once the edges exist. On an incremental tick, restricted
        to the managers reachable upward from the run's affected persons."""
        ids = None
        if entity_ids:
            up = self._execute(
                """
                MATCH (e:Entity {tenant_id: $tenant_id}) WHERE e.entity_id IN $entity_ids
                MATCH (e)-[:REPORTS_TO*0..]->(m:Entity)
                RETURN DISTINCT m.entity_id AS id
                """,
                {"tenant_id": tenant_id, "entity_ids": list(entity_ids)},
            )
            ids = [r["id"] for r in _rows(up)]
        response = self._execute(
            """
            MATCH (m:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE ($ids IS NULL OR m.entity_id IN $ids)
            SET m.direct_reports = COUNT { (m)<-[:REPORTS_TO]-() },
                m.org_size = COUNT { MATCH (r:Entity)-[:REPORTS_TO*]->(m) RETURN DISTINCT r }
            RETURN count(m) AS n
            """,
            {"tenant_id": tenant_id, "ids": ids},
        )
        return _scalar(response, "n") or 0

    def find_person_mention_merges(self, tenant_id: str, entity_ids: Optional[list[str]] = None) -> list[dict]:
        """Finds (bare-mention -> canonical :Person) pairs: a bare-name PERSON
        entity with no source_key whose lower-cased name matches EXACTLY ONE
        keyed :Person record. Unique-match only — an ambiguous name (>= 2
        records) is left unmerged, never a blind merge (mirrors
        services/people/resolution.py's own ambiguity guard)."""
        response = self._execute(
            """
            MATCH (m:Entity {tenant_id: $tenant_id, type: 'PERSON'})
            WHERE (m.source_key IS NULL OR m.source_key = '') AND m.name IS NOT NULL
              AND ($entity_ids IS NULL OR m.entity_id IN $entity_ids)
            WITH m, toLower(trim(m.name)) AS mn WHERE mn <> ''
            MATCH (c:Person {tenant_id: $tenant_id})
            WHERE c.source_key IS NOT NULL AND toLower(trim(c.name)) = mn AND c.entity_id <> m.entity_id
            WITH m, collect(DISTINCT c.entity_id) AS cids WHERE size(cids) = 1
            RETURN m.entity_id AS m, cids[0] AS c
            """,
            {"tenant_id": tenant_id, "entity_ids": entity_ids},
        )
        return _rows(response)

    def merge_person_mentions(self, tenant_id: str, pairs: list[dict]) -> dict:
        """Rewires a bare-name PERSON mention's MENTIONS/ABOUT/EXTRACTED_REL
        edges onto its canonical HR :Person node, then DETACH DELETEs the
        mention. pairs: [{"m": entity_id, "c": entity_id}, ...] (from
        find_person_mention_merges).

        Nothing the mention carried may be lost. DETACH DELETE destroys every property on
        the node and on the edges being replaced, so each is migrated first:

          edges  chunk_ids, support, raw_types, allow_access_control (Cypher union),
                 descriptions_json (Python-side, below)
          node   descriptions_json, aliases_json (Python-side), aliases, doc_ids,
                 source_docs, support, allow_access_control, description
                 (Cypher; description by coalesce — the survivor's own wins)

        allow_access_control matters most: write_relationships unions it on every ordinary
        edge write, and skipping it here silently narrowed access — a viewer entitled to the
        mention's source document lost the merged person and that document's relationships.
        Same for aliases, which back the entity_aliases fulltext index: dropping them stopped
        the source document's own spelling from resolving to the person.

        descriptions_json/aliases_json are JSON strings Cypher can't union, so they're read,
        merged with _union_doc_entries, and written back — aggregated per SURVIVING target,
        since two mentions can merge onto the same node or edge and a per-pair write would
        clobber the first contribution.

        e.degree is deliberately NOT recomputed here — core/pipeline.py does it after this
        step, which is why the recompute must follow the merge, not precede it.

        Two deliberate exclusions: an edge between the mention and its canonical node is a
        self-loop after merging, so it's dropped (mirrors edge_pool_service); and edge types
        other than MENTIONS/ABOUT/EXTRACTED_REL go with the node. Of those only
        CONTRIBUTED_TO can attach to a PERSON at all, and only if
        settings.contributor_entity_types is widened to include PERSON, which it isn't.

        Registry redirect + vector cleanup remain the caller's responsibility (this repo's
        registry/vector store aren't reachable from here)."""
        if not pairs:
            return {"merged": 0}
        merged = 0
        # One round per repeated canonical — see _distinct_canonical_batches for why.
        for batch in _distinct_canonical_batches([{"m": x["m"], "c": x["c"]} for x in pairs]):
            self._merge_person_mention_batch(tenant_id, batch)
            merged += len(batch)
        return {"merged": merged}

    def _merge_person_mention_batch(self, tenant_id: str, p: list[dict]) -> None:
        """One merge statement group, for pairs with distinct canonical ids (the caller
        guarantees this via _distinct_canonical_batches)."""
        params = {"tenant_id": tenant_id, "p": p}

        # --- Read the JSON node properties BEFORE any delete, merged per survivor.
        node_rows = _rows(self._execute(
            "UNWIND $p AS pr "
            "MATCH (m:Entity {entity_id: pr.m, tenant_id: $tenant_id}) "
            "MATCH (c:Entity {entity_id: pr.c}) "
            "RETURN pr.m AS m, pr.c AS c, m.descriptions_json AS m_desc, "
            "       m.aliases_json AS m_aliases, c.descriptions_json AS c_desc, "
            "       c.aliases_json AS c_aliases",
            params,
        ))
        merged_node_json: dict = {}
        for row in node_rows:
            current = merged_node_json.setdefault(
                row["c"], {"descriptions": row.get("c_desc"), "aliases": row.get("c_aliases")})
            current["descriptions"] = json.dumps(
                _union_doc_entries(current["descriptions"], row.get("m_desc")))
            current["aliases"] = json.dumps(
                _union_doc_entries(current["aliases"], row.get("m_aliases")))

        self._execute(
            "UNWIND $p AS pr "
            "MATCH (ch:Chunk {tenant_id: $tenant_id})-[old:MENTIONS]->(:Entity {entity_id: pr.m}) "
            "MATCH (c:Entity {entity_id: pr.c}) MERGE (ch)-[:MENTIONS]->(c) DELETE old",
            params,
        )
        self._execute(
            "UNWIND $p AS pr "
            "MATCH (cl:Claim)-[old:ABOUT]->(:Entity {entity_id: pr.m, tenant_id: $tenant_id}) "
            "MATCH (c:Entity {entity_id: pr.c}) MERGE (cl)-[:ABOUT]->(c) DELETE old",
            params,
        )
        for direction in ("out", "in"):
            match = ("(:Entity {entity_id: pr.m, tenant_id: $tenant_id})-[r:EXTRACTED_REL]->(o:Entity)"
                     if direction == "out" else
                     "(o:Entity)-[r:EXTRACTED_REL]->(:Entity {entity_id: pr.m, tenant_id: $tenant_id})")
            merge = ("(c)-[nr:EXTRACTED_REL {relationship_type: r.relationship_type}]->(o)"
                     if direction == "out" else
                     "(o)-[nr:EXTRACTED_REL {relationship_type: r.relationship_type}]->(c)")
            existing = ("(c2:Entity {entity_id: pr.c})-[nr:EXTRACTED_REL "
                        "{relationship_type: r.relationship_type}]->(o)"
                        if direction == "out" else
                        "(o)-[nr:EXTRACTED_REL {relationship_type: r.relationship_type}]->"
                        "(c2:Entity {entity_id: pr.c})")
            # Read phase: this edge's descriptions_json plus the target edge's (OPTIONAL —
            # the target may not exist until the MERGE below).
            edge_rows = _rows(self._execute(
                "UNWIND $p AS pr MATCH " + match + " WHERE o.entity_id <> pr.c "
                "OPTIONAL MATCH " + existing + " "
                "RETURN pr.c AS c, o.entity_id AS o, r.relationship_type AS rel_type, "
                "       r.descriptions_json AS old_desc, nr.descriptions_json AS existing_desc",
                params,
            ))
            merged_edge_json: dict = {}
            for row in edge_rows:
                key = (row["c"], row["o"], row["rel_type"])
                current = merged_edge_json.setdefault(key, row.get("existing_desc"))
                merged_edge_json[key] = json.dumps(
                    _union_doc_entries(current, row.get("old_desc")))
            self._execute(
                "UNWIND $p AS pr MATCH " + match + " WHERE o.entity_id <> pr.c "
                "MATCH (c:Entity {entity_id: pr.c}) MERGE " + merge + " "
                "WITH nr, r, coalesce(nr.chunk_ids, []) AS ex "
                "WITH nr, r, ex + [x IN coalesce(r.chunk_ids, []) WHERE NOT x IN ex] AS merged "
                "SET nr.chunk_ids = merged, nr.support = size(merged), "
                "    nr.raw_types = coalesce(nr.raw_types, []) "
                "        + [x IN coalesce(r.raw_types, []) WHERE NOT x IN coalesce(nr.raw_types, [])], "
                "    nr.allow_access_control = coalesce(nr.allow_access_control, []) "
                "        + [x IN coalesce(r.allow_access_control, []) "
                "           WHERE NOT x IN coalesce(nr.allow_access_control, [])] "
                "DELETE r",
                params,
            )
            if merged_edge_json:
                self._execute(
                    "UNWIND $rows AS row "
                    "MATCH (c:Entity {entity_id: row.c}) "
                    "MATCH (o:Entity {entity_id: row.o}) "
                    "MATCH " + ("(c)-[nr:EXTRACTED_REL {relationship_type: row.rel_type}]->(o)"
                                if direction == "out" else
                                "(o)-[nr:EXTRACTED_REL {relationship_type: row.rel_type}]->(c)") + " "
                    "SET nr.descriptions_json = row.descriptions_json",
                    {"rows": [{"c": c, "o": o, "rel_type": rel_type, "descriptions_json": desc}
                              for (c, o, rel_type), desc in merged_edge_json.items()]},
                )

        # Node-property migration, immediately before the node is destroyed. Native lists
        # union in Cypher; the JSON ones were merged above. Correct only because this batch
        # holds at most one pair per canonical node (_distinct_canonical_batches) — two rows
        # writing the same node would each read the pre-statement value, last one winning.
        self._execute(
            "UNWIND $rows AS row "
            "MATCH (m:Entity {entity_id: row.m, tenant_id: $tenant_id}) "
            "MATCH (c:Entity {entity_id: row.c}) "
            "WITH m, c, row, "
            "     coalesce(c.aliases, []) + [x IN coalesce(m.aliases, []) "
            "         WHERE NOT x IN coalesce(c.aliases, [])] AS merged_aliases, "
            "     coalesce(c.doc_ids, []) + [x IN coalesce(m.doc_ids, []) "
            "         WHERE NOT x IN coalesce(c.doc_ids, [])] AS merged_doc_ids, "
            "     coalesce(c.source_docs, []) + [x IN coalesce(m.source_docs, []) "
            "         WHERE NOT x IN coalesce(c.source_docs, [])] AS merged_source_docs, "
            "     coalesce(c.allow_access_control, []) + [x IN coalesce(m.allow_access_control, []) "
            "         WHERE NOT x IN coalesce(c.allow_access_control, [])] AS merged_acl "
            "SET c.aliases = merged_aliases, c.doc_ids = merged_doc_ids, "
            "    c.support = size(merged_doc_ids), c.source_docs = merged_source_docs, "
            "    c.allow_access_control = merged_acl, "
            # e.description (the one merged text per entity): the survivor's own wins,
            # the mention's fills the gap — DETACH DELETE below would destroy it.
            "    c.description = coalesce(c.description, m.description), "
            "    c.descriptions_json = row.descriptions_json, c.aliases_json = row.aliases_json",
            {"tenant_id": tenant_id,
             "rows": [{"m": row["m"], "c": row["c"],
                       "descriptions_json": merged_node_json[row["c"]]["descriptions"],
                       "aliases_json": merged_node_json[row["c"]]["aliases"]}
                      for row in node_rows]},
        )
        self._execute(
            "UNWIND $p AS pr MATCH (m:Entity {entity_id: pr.m, tenant_id: $tenant_id}) DETACH DELETE m",
            params,
        )

    def mark_person_departed(self, tenant_id: str, doc_id: str, when: Optional[str] = None) -> int:
        """Flips a departed person's node to inactive (keeps the node + its org
        edges for history) instead of retracting it, found through the document
        that defined them and keyed on (connector_type, source_key). Was
        `person_user_id: $doc_id`, true only of Zeus, whose People doc id IS the
        HR user id. source_docs, not doc_ids: doc_ids accumulates every document
        that merely mentions a person."""
        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})
            MATCH (e:Person {tenant_id: $tenant_id, connector_type: d.connector_type})
            WHERE e.source_key IS NOT NULL AND $doc_id IN e.source_docs
            SET e.person_employment_status = 'inactive',
                e.person_departed_on = coalesce($when, e.person_departed_on)
            RETURN count(e) AS n
            """,
            {"tenant_id": tenant_id, "doc_id": doc_id, "when": when},
        )
        return _scalar(response, "n") or 0

    def doc_object_type(self, doc_id: str, tenant_id: str) -> Optional[str]:
        """The Document's own object_type (stamped as d.source by
        write_documents), or None if the doc doesn't exist. Read before a
        delete event decides how to retract — a People doc needs
        mark_person_departed instead of the normal retract_document (see
        services/incremental_pipeline.py.delete_document)."""
        response = self._execute(
            "MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id}) RETURN d.source AS ot",
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        return _scalar(response, "ot")

    def materialize_authorship_edges(self, tenant_id: str, doc_ids: Optional[list[str]] = None,
                                     contributor_types: tuple = ("PROJECT", "PRODUCT", "PROGRAM",
                                                                  "FEATURE", "INITIATIVE")) -> dict:
        """Materializes authorship as first-class, deterministic edges (no
        LLM), resolving a document's author/editor keys to the :Person node
        that MATERIALIZE minted for the same (connector_type, source_key):

          (:Person)-[:AUTHORED]->(:Document)     — the creator,
          (:Person)-[:EDITED]->(:Document)       — the last editor, when
                                                    different from the creator,
          (:Person)-[:CONTRIBUTED_TO]->(:Entity) — the author linked to the
                                                    project-like entities their
                                                    document is about.

        The join is (connector_type, source_key), both sides of which come from
        the contract: a Document's created_by_id IS its SourceDocument.author's
        source_key, and a :Person's source_key IS its PersonRecord's. It used to
        be person_user_id, an HR field name, which is why this only ever
        produced edges for the intranet — it worked because a People document's
        own id happens to share a namespace with createdbyid, and no other
        source populates either.

        connector_type is part of the key, not decoration: two sources can
        legitimately hand out the same opaque id for two different humans.

        Kept separate from the LLM-extracted EXTRACTED_REL{CONTRIBUTES_TO}
        (prose-derived, noisier). doc_ids scopes to an incremental tick's own
        documents; None is whole-tenant (bulk). Idempotent (MERGE)."""
        scope = "AND d.doc_id IN $doc_ids " if doc_ids else ""
        params = {"tenant_id": tenant_id, "doc_ids": doc_ids, "types": list(contributor_types)}
        authored = self._execute(
            "MATCH (d:Document {tenant_id: $tenant_id}) "
            "WHERE d.created_by_id IS NOT NULL AND d.created_by_id <> '' " + scope +
            "MATCH (p:Person {tenant_id: $tenant_id, connector_type: d.connector_type, "
            "                source_key: d.created_by_id}) "
            "MERGE (p)-[:AUTHORED]->(d) RETURN count(*) AS n",
            params,
        )
        edited = self._execute(
            "MATCH (d:Document {tenant_id: $tenant_id}) "
            "WHERE d.modified_by_id IS NOT NULL AND d.modified_by_id <> '' "
            "AND d.modified_by_id <> coalesce(d.created_by_id, '') " + scope +
            "MATCH (p:Person {tenant_id: $tenant_id, connector_type: d.connector_type, "
            "                source_key: d.modified_by_id}) "
            "MERGE (p)-[:EDITED]->(d) RETURN count(*) AS n",
            params,
        )
        contributed = self._execute(
            "MATCH (d:Document {tenant_id: $tenant_id}) WHERE d.created_by_id IS NOT NULL " + scope +
            "MATCH (p:Person)-[:AUTHORED]->(d)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity) "
            "WHERE e.type IN $types AND e.entity_id <> p.entity_id AND trim(coalesce(e.name, '')) <> '' "
            "MERGE (p)-[:CONTRIBUTED_TO]->(e) RETURN count(DISTINCT [p.entity_id, e.entity_id]) AS n",
            params,
        )
        return {"authored": _scalar(authored, "n") or 0, "edited": _scalar(edited, "n") or 0,
                "contributed_to": _scalar(contributed, "n") or 0}

    def find_person_entities_by_chunks(
        self, tenant_id: str, chunk_ids: list[str],
    ) -> dict[str, list[dict]]:
        """PERSON entities each chunk mentions — {chunk_id: [{entity_id, name}, ...]}."""
        if not chunk_ids:
            return {}
        response = self._execute(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {tenant_id: $tenant_id, chunk_id: chunk_id})-[:MENTIONS]->(e:Entity {type: 'PERSON'})
            RETURN chunk_id, collect(DISTINCT {entity_id: e.entity_id, name: e.name}) AS entities
            """,
            {"tenant_id": tenant_id, "chunk_ids": chunk_ids},
        )
        return {row["chunk_id"]: row["entities"] for row in _rows(response)}

    def write_claims(self, final_claims: list[dict], tenant_id: str) -> list[str]:
        """final_claims: Step 4's output. Returns the written claim_ids so a
        caller can scope a follow-up recompute_claim_support to just this
        batch instead of the whole tenant.

        claim_id hashes in valid_from/valid_to so two claims about the same
        subject/predicate/object but different time windows get distinct
        nodes. subject is an entity_id, so the same real-world subject still
        hashes consistently across claims.

        Matches subject by entity_id. chunk_id is also persisted as a plain
        property, but a claim's real source-chunk link is the
        (:Claim)-[:SUPPORTED_BY]->(:Chunk) edge below — a property alone
        isn't traversable in Cypher the way (:Claim)-[:ABOUT]->(:Entity)
        already is. MATCH (not OPTIONAL MATCH) assumes write_chunks always
        runs before write_claims.

        acl/documented_on are this claim's own document's values, exact and
        never merged — a claim has exactly one chunk_id, so there's no union
        to compute here, unlike Entity/relationship acl.

        documented_on/valid_from/valid_to land as "", never absent — even when
        the source document has no date (every people profile: an HR record
        carries no publish date) or the claim's temporal window is unknown.
        Reverses an earlier decision that deliberately wrote null so these
        properties would be entirely absent (`cl.documented_on IS NULL`) rather
        than an empty string, on the reasoning that absence was a more honest
        "we don't know" than "" and that "" was indistinguishable from a real
        value in queries/dashboards. Product now wants the opposite: the
        property key must always exist on every Claim, so a query can rely on
        `keys(cl)` containing it — Neo4j has no way to store a property with an
        actual null value, so writing "" is the only way to guarantee the key
        exists. "" is still the honest "unknown" (retrieval already treats a
        missing valid_from as equivalent to '' — see cypher.py's own
        `coalesce(valid_from, '')` — and `"" >= "<date>"` is false, same as a
        real absence would compare)."""
        rows = []
        for claim in final_claims:
            claim_id = hashlib.sha1(
                f"{claim['subject']}|{claim['predicate']}|{claim['object']}"
                f"|{claim.get('valid_from') or ''}|{claim.get('valid_to') or ''}".encode("utf-8")
            ).hexdigest()
            rows.append({**claim, "claim_id": claim_id, "acl": claim.get("acl") or [],
                        "documented_on": claim.get("documented_on") or "",
                        "valid_from": claim.get("valid_from") or "",
                        "valid_to": claim.get("valid_to") or "",
                        "tenant_id": tenant_id})
        self._run_batched(
            """
            UNWIND $batch AS row
            MATCH (subj:Entity {entity_id: row.subject})
            MATCH (chunk:Chunk {chunk_id: row.chunk_id})
            MERGE (cl:Claim {claim_id: row.claim_id})
            SET cl.subject = row.subject, cl.predicate = row.predicate, cl.object = row.object,
                cl.statement_type = row.statement_type, cl.temporal_type = row.temporal_type,
                cl.valid_from = row.valid_from, cl.valid_to = row.valid_to,
                cl.chunk_id = row.chunk_id, cl.allow_access_control = row.acl,
                cl.documented_on = row.documented_on, cl.tenant_id = row.tenant_id
            MERGE (cl)-[:ABOUT]->(subj)
            MERGE (cl)-[:SUPPORTED_BY]->(chunk)
            """,
            rows, "Claims",
        )
        return [row["claim_id"] for row in rows]

    # ------------------------------------------------------------------ #
    # Incremental writes: unlike write_entities' bulk path, these track    #
    # doc_ids[]/support per entity so a per-document retraction can        #
    # cleanly undo exactly what that document contributed.                 #
    # ------------------------------------------------------------------ #

    def find_similar_entities(self, vector_store: VectorStore, embedding: list, entity_type: str,
                              tenant_id: str, k: int = 5, min_similarity: float = 0.83) -> list[dict]:
        """Vector-search for likely duplicates of a newly extracted entity,
        same-type and same-tenant only, ordered by similarity descending.
        Used by services/incremental_canonicalization.py to decide
        merge-into-existing vs. treat-as-new.

        Search runs against Qdrant (services/vector_store.py) with
        server-side type/tenant pre-filtering. Qdrant only carries the vector
        + scoping payload, not entity metadata, so this does a follow-up
        Neo4j MATCH by entity_id for descriptions_json — the one field the
        caller's LLM merge-confirmation prompt needs.

        Qdrant down / missing collection: fail-open to no candidates so the
        incremental path can still MERGE Entity + MENTIONS in Neo4j (under-merge
        until vectors are back). Raising here aborted the whole Kafka event after
        Document/Chunk were already written, and redelivery then skipped extraction.
        """
        try:
            hits = vector_store.search(embedding, entity_type, tenant_id, k=k, min_similarity=min_similarity)
        except Exception as exc:  # noqa: BLE001 — ANN is best-effort; empty == treat-as-new
            logger.warning(
                "find_similar_entities: Qdrant search failed (type=%s tenant_id=%s): %s — "
                "treating as no candidates so Neo4j entity writes can proceed",
                entity_type, tenant_id, exc,
            )
            return []
        if not hits:
            return []
        score_by_id = {h["entity_id"]: h["score"] for h in hits}
        response = self._execute(
            "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids "
            "RETURN e.entity_id AS entity_id, e.name AS name, e.descriptions_json AS descriptions_json",
            {"entity_ids": list(score_by_id.keys())},
        )
        results = [
            {"name": r["name"], "descriptions_json": r["descriptions_json"], "score": score_by_id[r["entity_id"]]}
            for r in _rows(response)
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def entity_exists_exact(self, name: str, entity_type: str, tenant_id: str) -> bool:
        """Exact-name fast path: True only when an Entity for this exact
        (name, type, tenant_id) exists AND already has a real vector in
        Qdrant.

        services/incremental_canonicalization.py calls this first — if True,
        `name` is already canonical and the embedding/ANN/LLM-merge calls are
        all skipped, the common case for incremental updates.

        Checks e.has_vector rather than a Qdrant round trip. Requiring
        has_vector (not just node existence) matters: a legacy row with no
        vector yet must still go through the slow path once, or it would stay
        permanently un-embedded and invisible to ANN search."""
        response = self._execute(
            "MATCH (e:Entity {name: $name, type: $type, tenant_id: $tenant_id}) "
            "RETURN coalesce(e.has_vector, false) AS has_vector",
            {"name": name, "type": entity_type, "tenant_id": tenant_id},
        )
        rows = _rows(response)
        return bool(rows and rows[0].get("has_vector"))

    def upsert_entity_incremental(self, name: str, entity_type: str, description: str,
                                  embedding: Optional[list], doc_id: str, acl: list[str],
                                  tenant_id: str, raw_name: Optional[str] = None,
                                  vector_store: Optional[VectorStore] = None) -> str:
        """Returns the entity_id resolved/minted for this call — the caller
        (services/incremental_pipeline.py) needs it for canon_to_id, the same
        way the bulk path's resolve_entity_ids does.

        embedding is pushed to Qdrant, not written onto the node; embedding=None
        (the exact-name fast path) skips the Qdrant push entirely.

        One entity, one document, written incrementally. Idempotent: the
        `$doc_id IN coalesce(e.doc_ids, [])` guard, reused across every ON
        MATCH branch, makes a retried/redelivered Kafka message a no-op for
        support/doc_ids/descriptions/aliases.

        entity_id is resolved via repository/entity_registry.py using the
        same (normalized_name, entity_type, tenant_id) key the bulk path
        mints through, so an entity created by either path always resolves to
        the same entity_id. The MERGE below keys on entity_id alone.

        descriptions_json/aliases_json: this document's own entry, tagged
        with its own acl, is appended to whatever's already on the node.
        Cypher has no JSON append without APOC (not assumed available here),
        so this reads the current JSON, appends in Python, and writes the
        whole array back — one extra round trip, acceptable at this path's
        per-document scale."""
        from repository import entity_registry
        from services.clustering_utils import normalize_name
        normalized = normalize_name(name) or name.strip().lower() or "∅"
        alias_norms = [normalized]
        if raw_name and raw_name != name:
            alias_norms.append(normalize_name(raw_name) or raw_name.strip().lower())
        entity_id = entity_registry.link_or_mint(
            name, entity_type, normalized, tenant_id, aliases=alias_norms,
        )["_id"]

        existing = self._execute(
            "OPTIONAL MATCH (e:Entity {entity_id: $entity_id}) "
            "RETURN e.descriptions_json AS descriptions_json, e.aliases_json AS aliases_json, "
            "$doc_id IN coalesce(e.doc_ids, []) AS already_contributed",
            {"entity_id": entity_id, "doc_id": doc_id},
        )
        existing_row = _rows(existing)
        existing_row = existing_row[0] if existing_row else {}
        already_contributed = bool(existing_row.get("already_contributed"))
        try:
            descriptions = json.loads(existing_row.get("descriptions_json") or "[]")
        except (TypeError, ValueError):
            descriptions = []
        try:
            aliases = json.loads(existing_row.get("aliases_json") or "[]")
        except (TypeError, ValueError):
            aliases = []

        if not already_contributed:
            if description:
                descriptions.append({"text": description, "doc_id": doc_id, "allow_access_control": acl or []})
            aliases.append({"name": raw_name or name, "doc_id": doc_id, "allow_access_control": acl or []})

        # Denormalized plain list of alias strings, kept alongside aliases_json
        # because Neo4j's fulltext index needs a native list property.
        aliases_flat = sorted({a.get("name") for a in aliases if a.get("name")})

        has_new_embedding = bool(embedding)
        response = self._execute(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            ON CREATE SET
                e.name = $name, e.type = $type, e.tenant_id = $tenant_id,
                e.has_vector = $has_new_embedding, e.doc_ids = [$doc_id],
                e.description = $description,
                e.descriptions_json = $descriptions_json, e.aliases_json = $aliases_json,
                e.aliases = $aliases_flat, e.support = 1, e.allow_access_control = $acl
            ON MATCH SET
                e.has_vector = CASE WHEN $has_new_embedding THEN true ELSE coalesce(e.has_vector, false) END,
                e.descriptions_json = $descriptions_json, e.aliases_json = $aliases_json,
                e.aliases = $aliases_flat,
                e.support = CASE WHEN $doc_id IN coalesce(e.doc_ids, [])
                                 THEN coalesce(e.support, 1) ELSE coalesce(e.support, 1) + 1 END,
                e.allow_access_control = CASE WHEN $doc_id IN coalesce(e.doc_ids, [])
                             THEN e.allow_access_control
                             ELSE coalesce(e.allow_access_control, [])
                                  + [t IN $acl WHERE NOT t IN coalesce(e.allow_access_control, [])] END,
                e.doc_ids = CASE WHEN $doc_id IN coalesce(e.doc_ids, [])
                                 THEN e.doc_ids ELSE coalesce(e.doc_ids, []) + $doc_id END
            RETURN e.allow_access_control AS acl
            """,
            {"entity_id": entity_id, "name": name, "type": entity_type,
             "has_new_embedding": has_new_embedding,
             "doc_id": doc_id, "acl": acl or [],
             # ON CREATE only: a new entity's first raw description is better than none,
             # and ON MATCH never touches it — one document's blurb must not clobber
             # bulk's LLM-merged text (write_entities owns refining it).
             "description": (description or "").strip(),
             "descriptions_json": json.dumps(descriptions), "aliases_json": json.dumps(aliases),
             "aliases_flat": aliases_flat, "tenant_id": tenant_id},
        )
        if vector_store is not None and has_new_embedding:
            final_acl = _scalar(response, "acl") or []
            # Dual-write safety: has_vector already committed true above; roll back
            # to false if the Qdrant push exhausts its retries.
            call_with_retry_or_rollback(
                lambda: vector_store.upsert_entity(entity_id, embedding, entity_type, tenant_id, final_acl),
                step="qdrant_entity_upsert", context=f"entity_id={entity_id!r}",
                rollback=lambda: self.mark_vector_missing(tenant_id, [entity_id]),
            )
        return entity_id

    def _retract_chunks_and_dependents(self, doc_id: str, tenant_id: str,
                                       vector_store: Optional[VectorStore] = None,
                                       chunks_vector_store: Optional[VectorStore] = None) -> dict:
        """Undoes everything ONE document's chunks contributed, WITHOUT
        deleting the Document node itself. Shared by retract_document (also
        deletes the Document node) and replace_document_chunks (keeps it, for
        an immediate re-write of fresh chunks).

        Steps: 1) entities this doc's chunks mention lose this doc_id from
        doc_ids, support decrements, acl recomputes as the live union of
        remaining contributors. 1b) this doc_id's own entry is dropped from
        those entities' descriptions_json/aliases_json, or a later re-ingest
        of the same document would append a duplicate entry instead of
        replacing it. 2) entities that hit zero support are DETACH DELETEd.
        3) relationships touching those entities lose this doc's chunk_ids,
        or are deleted if none remain. 4) relationships that survived step 3
        get their acl recomputed the same live-union way. 5) claims tied to
        this doc's chunks lose their SUPPORTED_BY edge to it and are deleted
        only if zero remain — claims CAN be shared across documents (claim_id
        hashes subject/predicate/object/validity only, no doc_id), so an
        unconditional delete would remove a claim a surviving document still
        cites. 6) this doc's Chunk nodes are deleted.

        IMPORTANT: the zero-support check must use
        `e.doc_ids IS NOT NULL AND size(e.doc_ids) = 0`, never
        `size(coalesce(e.doc_ids, [])) = 0` — every entity written before
        doc_ids tracking existed has no doc_ids property at all, and
        coalescing null to [] would match and delete all of them. An earlier
        coalesce-based version of this query deleted 153 legitimate entities
        from the dev graph in one shot; this check must never regress to
        that form.

        vector_store/chunks_vector_store, when given, keep Qdrant in sync:
        narrowed acl is pushed, and anything actually deleted here has its
        Qdrant point deleted too. None skips both syncs; the Neo4j write is
        still correct alone."""
        stats: dict = {}

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e, [x IN coalesce(e.doc_ids, []) WHERE x <> $doc_id] AS remaining_doc_ids
            SET e.doc_ids = remaining_doc_ids, e.support = size(remaining_doc_ids)
            WITH e, remaining_doc_ids
            OPTIONAL MATCH (rd:Document) WHERE rd.doc_id IN remaining_doc_ids AND rd.tenant_id = $tenant_id
            WITH e, collect(DISTINCT coalesce(rd.allow_access_control, [])) AS acls
            SET e.allow_access_control = reduce(allTokens = [], a IN acls | allTokens + [t IN a WHERE NOT t IN allTokens])
            RETURN e.entity_id AS entity_id, e.allow_access_control AS acl
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        decremented_rows = _rows(response)
        stats["entities_decremented"] = len(decremented_rows)
        if vector_store is not None:
            # Keep Qdrant's payload in sync with the just-narrowed acl, or a
            # retrieval query filtering on Qdrant would keep honoring a
            # revoked access grant.
            sync_acl_for_each(
                vector_store, [(row["entity_id"], row.get("acl") or []) for row in decremented_rows],
                step="qdrant_acl_sync",
            )

        # Prune this doc_id's entry from descriptions_json/aliases_json (see
        # _prune_doc_entries). No JSON-array filter without APOC, hence read +
        # Python-filter + batched write.
        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            RETURN DISTINCT e.entity_id AS entity_id,
                   e.descriptions_json AS descriptions_json, e.aliases_json AS aliases_json
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        pruned_rows = [{
            "entity_id": row["entity_id"],
            "descriptions_json": json.dumps(_prune_doc_entries(row.get("descriptions_json"), doc_id)),
            "aliases_json": json.dumps(_prune_doc_entries(row.get("aliases_json"), doc_id)),
        } for row in _rows(response)]
        if pruned_rows:
            self._execute(
                """
                UNWIND $batch AS row
                MATCH (e:Entity {entity_id: row.entity_id})
                SET e.descriptions_json = row.descriptions_json, e.aliases_json = row.aliases_json
                """,
                {"batch": pruned_rows},
            )

        response = self._execute(
            "MATCH (e:Entity) WHERE e.tenant_id = $tenant_id AND e.doc_ids IS NOT NULL "
            "AND size(e.doc_ids) = 0 WITH e, e.entity_id AS entity_id DETACH DELETE e RETURN entity_id",
            {"tenant_id": tenant_id},
        )
        deleted_ids = [row["entity_id"] for row in _rows(response)]
        stats["entities_deleted"] = len(deleted_ids)
        if vector_store is not None:
            # Delete the corresponding Qdrant point too, or it's an orphaned
            # vector forever invisible from the Neo4j side.
            delete_vectors(vector_store, deleted_ids, step="qdrant_entity_delete", id_label="entity_ids")

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
            WITH d, collect(DISTINCT c.chunk_id) AS doc_chunk_ids
            MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e, doc_chunk_ids
            MATCH (e)-[r]-()
            WHERE r.chunk_ids IS NOT NULL AND any(cid IN r.chunk_ids WHERE cid IN doc_chunk_ids)
            WITH DISTINCT r, [cid IN r.chunk_ids WHERE NOT cid IN doc_chunk_ids] AS remaining
            FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END | SET r.chunk_ids = remaining)
            FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END | DELETE r)
            RETURN count(r) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        stats["relationships_pruned"] = _scalar(response, "n") or 0

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            MATCH (e)-[r]-()
            WHERE r.chunk_ids IS NOT NULL AND size(r.chunk_ids) > 0
            WITH DISTINCT r
            UNWIND r.chunk_ids AS cid
            MATCH (rd:Document {tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk {chunk_id: cid})
            WITH r, collect(DISTINCT coalesce(rd.allow_access_control, [])) AS acls
            SET r.allow_access_control = reduce(allTokens = [], a IN acls | allTokens + [t IN a WHERE NOT t IN allTokens])
            RETURN count(r) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        stats["relationships_acl_recomputed"] = _scalar(response, "n") or 0

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
                  <-[sb:SUPPORTED_BY]-(cl:Claim)
            WITH DISTINCT cl, collect(sb) AS this_docs_edges
            FOREACH (r IN this_docs_edges | DELETE r)
            WITH DISTINCT cl
            OPTIONAL MATCH (cl)-[remaining:SUPPORTED_BY]->(:Chunk)
            WITH cl, count(remaining) AS remaining_count
            WHERE remaining_count = 0
            DETACH DELETE cl
            RETURN count(cl) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        stats["claims_deleted"] = _scalar(response, "n") or 0

        response = self._execute(
            "MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk) "
            "WITH c, c.chunk_id AS chunk_id DETACH DELETE c RETURN collect(chunk_id) AS chunk_ids",
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        deleted_chunk_ids = _scalar(response, "chunk_ids") or []
        if chunks_vector_store is not None:
            # Nothing to roll back — the Chunk node is already gone.
            delete_vectors(chunks_vector_store, deleted_chunk_ids, step="qdrant_chunk_delete", id_label="chunk_ids")

        return stats

    def replace_document_chunks(self, doc_id: str, tenant_id: str,
                                vector_store: Optional[VectorStore] = None,
                                chunks_vector_store: Optional[VectorStore] = None) -> dict:
        """Whole-document hard-replace: undoes everything this document's
        PREVIOUS chunks contributed, keeping the Document node. Safe to call
        on a brand-new doc_id (all stats come back 0).

        Superseded on the hot path by retract_chunks' narrower diff; kept as
        a general full-reset primitive."""
        return self._retract_chunks_and_dependents(doc_id, tenant_id, vector_store, chunks_vector_store)

    def get_document_chunk_rids(self, doc_id: str, tenant_id: str) -> dict[str, str]:
        """{recursive_chunk_id: chunk_id} for every chunk currently on this
        document — the diff key process_document uses to work out which
        chunks are new/dropped/unchanged on an update. Empty dict for an
        unknown doc_id."""
        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.recursive_chunk_id AS rid, c.chunk_id AS chunk_id
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        return {row["rid"]: row["chunk_id"] for row in _rows(response) if row.get("rid")}

    def retract_chunks(self, doc_id: str, tenant_id: str, chunk_ids: list[str],
                       vector_store: Optional[VectorStore] = None,
                       chunks_vector_store: Optional[VectorStore] = None) -> dict:
        """Retracts a specific subset of a document's chunks — used when an
        update drops some chunks but keeps others (unlike
        replace_document_chunks, which retracts all of them). Does NOT touch
        entity doc_ids/support/acl — the owning Document isn't being removed
        as a contributor, only these chunks are.

        1. Relationships whose chunk_ids include one of these chunks: strip
           them, delete the edge if none remain.
        2. Claims tied to these chunks: same strip-then-delete-if-empty.
        3. These Chunk nodes: DETACH DELETE.
        4. Entities these chunks mentioned (captured before step 3 removes
           the MENTIONS edges) are deleted only if they now have zero
           relationships of any kind — a live check, since another
           document's chunk or a surviving claim may still reference them.
           vector_store deletes the corresponding Qdrant point for anything
           actually deleted."""
        stats: dict = {}
        if not chunk_ids:
            return {"relationships_pruned": 0, "claims_deleted": 0, "chunks_deleted": 0, "entities_deleted": 0}

        candidates_response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IN $chunk_ids
            RETURN DISTINCT e.entity_id AS entity_id
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "chunk_ids": chunk_ids},
        )
        candidate_entity_ids = [row["entity_id"] for row in _rows(candidates_response)]

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.chunk_id IN $chunk_ids
            WITH DISTINCT e
            MATCH (e)-[r]-()
            WHERE r.chunk_ids IS NOT NULL AND any(cid IN r.chunk_ids WHERE cid IN $chunk_ids)
            WITH DISTINCT r, [cid IN r.chunk_ids WHERE NOT cid IN $chunk_ids] AS remaining
            FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
                     SET r.chunk_ids = remaining, r.support = size(remaining))
            FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END | DELETE r)
            RETURN count(r) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "chunk_ids": chunk_ids},
        )
        stats["relationships_pruned"] = _scalar(response, "n") or 0

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
                  <-[sb:SUPPORTED_BY]-(cl:Claim)
            WHERE c.chunk_id IN $chunk_ids
            WITH DISTINCT cl, collect(sb) AS these_chunks_edges
            FOREACH (r IN these_chunks_edges | DELETE r)
            WITH DISTINCT cl
            OPTIONAL MATCH (cl)-[remaining:SUPPORTED_BY]->(:Chunk)
            WITH cl, count(remaining) AS remaining_count
            WHERE remaining_count = 0
            DETACH DELETE cl
            RETURN count(cl) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "chunk_ids": chunk_ids},
        )
        stats["claims_deleted"] = _scalar(response, "n") or 0

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            DETACH DELETE c
            RETURN count(c) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "chunk_ids": chunk_ids},
        )
        stats["chunks_deleted"] = _scalar(response, "n") or 0
        if chunks_vector_store is not None:
            delete_vectors(chunks_vector_store, chunk_ids, step="qdrant_chunk_delete", id_label="chunk_ids")

        deleted_ids: list[str] = []
        if candidate_entity_ids:
            response = self._execute(
                """
                MATCH (e:Entity) WHERE e.entity_id IN $candidate_ids AND NOT (e)--()
                WITH e, e.entity_id AS entity_id
                DETACH DELETE e
                RETURN entity_id
                """,
                {"candidate_ids": candidate_entity_ids},
            )
            deleted_ids = [row["entity_id"] for row in _rows(response)]
        stats["entities_deleted"] = len(deleted_ids)
        if vector_store is not None:
            delete_vectors(vector_store, deleted_ids, step="qdrant_entity_delete", id_label="entity_ids")

        return stats

    def update_document_acl(self, doc_id: str, tenant_id: str, acl: list[str],
                            vector_store: Optional[VectorStore] = None,
                            chunks_vector_store: Optional[VectorStore] = None) -> dict:
        """ACL-only refresh (permissions changed, content did not) — updates
        existing graph nodes without touching text or triggering
        re-extraction.

        1. Document/Chunk/Claim get this document's new acl exactly (never
           merged).
        2. Entities mentioned by this document's chunks get
           allow_access_control recomputed as a live union over every
           document currently in their doc_ids (picking up the new value
           from step 1).
        3. Relationships touching those entities get the same live-union
           recompute.

        A no-op if doc_id doesn't exist yet (event arrived before its upsert).

        vector_store, when given, pushes step 2's recomputed acl to Qdrant
        too — a revocation would otherwise correct Neo4j but leave Qdrant
        over-granting access forever."""
        stats: dict = {}

        response = self._execute(
            "MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id}) "
            "SET d.allow_access_control = $acl RETURN count(d) AS n",
            {"doc_id": doc_id, "tenant_id": tenant_id, "acl": acl},
        )
        stats["document_updated"] = bool(_scalar(response, "n"))

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk)
            SET c.allow_access_control = $acl
            RETURN collect(c.chunk_id) AS chunk_ids
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "acl": acl},
        )
        updated_chunk_ids = _scalar(response, "chunk_ids") or []
        stats["chunks_updated"] = len(updated_chunk_ids)
        if chunks_vector_store is not None:
            # Same revocation-leak concern as the entity acl sync below, at the chunk level.
            sync_acl_for_each(
                chunks_vector_store, [(chunk_id, acl) for chunk_id in updated_chunk_ids],
                step="qdrant_chunk_acl_sync", id_label="chunk_id",
            )

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk)
                  <-[:SUPPORTED_BY]-(cl:Claim)
            SET cl.allow_access_control = $acl
            RETURN count(cl) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id, "acl": acl},
        )
        stats["claims_updated"] = _scalar(response, "n") or 0

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            OPTIONAL MATCH (rd:Document) WHERE rd.doc_id IN coalesce(e.doc_ids, []) AND rd.tenant_id = $tenant_id
            WITH e, collect(DISTINCT coalesce(rd.allow_access_control, [])) AS acls
            SET e.allow_access_control = reduce(allTokens = [], a IN acls | allTokens + [t IN a WHERE NOT t IN allTokens])
            RETURN e.entity_id AS entity_id, e.allow_access_control AS acl
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        recomputed_rows = _rows(response)
        stats["entities_acl_recomputed"] = len(recomputed_rows)
        if vector_store is not None:
            # Dual-write safety: retry, log durably on exhaustion rather than
            # leaving Qdrant with a stale acl.
            sync_acl_for_each(
                vector_store, [(row["entity_id"], row.get("acl") or []) for row in recomputed_rows],
                step="qdrant_acl_sync",
            )

        response = self._execute(
            """
            MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            MATCH (e)-[r]-()
            WHERE r.chunk_ids IS NOT NULL AND size(r.chunk_ids) > 0
            WITH DISTINCT r
            UNWIND r.chunk_ids AS cid
            MATCH (rd:Document {tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk {chunk_id: cid})
            WITH r, collect(DISTINCT coalesce(rd.allow_access_control, [])) AS acls
            SET r.allow_access_control = reduce(allTokens = [], a IN acls | allTokens + [t IN a WHERE NOT t IN allTokens])
            RETURN count(r) AS n
            """,
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        stats["relationships_acl_recomputed"] = _scalar(response, "n") or 0

        return stats

    def count_documents(self, tenant_id: str) -> int:
        """Total Document nodes for one tenant — the denominator the
        mass-retraction guard checks a prospective delete's fraction against
        before calling retract_document below."""
        response = self._execute(
            "MATCH (d:Document {tenant_id: $tenant_id}) RETURN count(d) AS n",
            {"tenant_id": tenant_id},
        )
        return _scalar(response, "n") or 0

    def retract_document(self, doc_id: str, tenant_id: str,
                         vector_store: Optional[VectorStore] = None,
                         chunks_vector_store: Optional[VectorStore] = None) -> dict:
        """Undoes everything one document contributed, INCLUDING the Document
        node itself (the incremental hard-delete path). See
        _retract_chunks_and_dependents for the entity/relationship/claim/chunk
        cleanup; this adds deleting the Document node itself."""
        stats = self._retract_chunks_and_dependents(doc_id, tenant_id, vector_store, chunks_vector_store)

        response = self._execute(
            "MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id}) DETACH DELETE d "
            "RETURN count(d) AS n",
            {"doc_id": doc_id, "tenant_id": tenant_id},
        )
        stats["doc_deleted"] = bool(_scalar(response, "n"))

        return stats

    def write_all(self, docs: list, final_entities: dict, final_edges: dict,
                 final_claims: list[dict], mentions: list[tuple], tenant_id: str,
                 doc_acl_by_doc: Optional[dict] = None) -> None:
        """The whole Step 5 write, in dependency order — each step MATCHes nodes the
        previous step created, so order matters: documents/chunks/entities must exist
        before mentions/relationships/claims can link to them.

        tenant_id: one bulk run writes exactly one tenant's data — see
        core/pipeline.py's run_bulk_async / core/entrypoint.py's --tenant-id.

        docs: SourceDocuments, same as write_documents/write_chunks above. This
        method has no callers — run_bulk_async drives the same steps itself so
        it can interleave extraction between them — and is kept only as the
        readable statement of the ordering."""
        self.create_constraints()
        self.write_documents(docs, tenant_id, doc_acl_by_doc)
        self.write_chunks(docs, tenant_id, doc_acl_by_doc)
        self.write_entities(final_entities, tenant_id)
        self.write_mentions(mentions, tenant_id)
        self.write_relationships(final_edges, tenant_id)
        self.write_claims(final_claims, tenant_id)
        logger.info("=== Step 5 done: %d docs, %d entities, %d edges, %d claims, %d mentions written ===",
                    len(docs), len(final_entities), len(final_edges), len(final_claims), len(mentions))
