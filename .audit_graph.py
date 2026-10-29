"""Throwaway: dump what a run actually wrote into Neo4j, stage by stage."""
import json
import sys

import requests

from config.settings import settings

DB = sys.argv[1]
AUTH = (settings.neo4j_username, settings.neo4j_password)
URL = f"{settings.neo4j_http_url}/db/{DB}/query/v2"


def q(statement, **params):
    r = requests.post(URL, auth=AUTH,
                      headers={"Content-Type": "application/json", "Accept": "application/json"},
                      json={"statement": statement, "parameters": params})
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RuntimeError(d["errors"])
    fields = d["data"]["fields"]
    return [dict(zip(fields, v)) for v in d["data"]["values"]]


def show(title, rows, limit=None):
    print(f"\n--- {title} ---")
    for row in rows[:limit]:
        print("   ", json.dumps(row, ensure_ascii=False, default=str)[:400])


print(f"==================== {DB} ====================")
show("node counts", q("MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC"))
show("relationship counts", q("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC"))

show("documents", q("""
    MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title, d.url AS url,
      d.source AS source, d.connector_type AS connector_type, d.published_at AS published_at,
      d.tenant_id AS tenant_id, d.allow_access_control AS acl,
      d.author AS author, d.editor AS editor
"""))

show("chunk property completeness", q("""
    MATCH (c:Chunk) RETURN count(*) AS chunks,
      sum(CASE WHEN c.doc_id IS NULL THEN 1 ELSE 0 END) AS missing_doc_id,
      sum(CASE WHEN c.tenant_id IS NULL THEN 1 ELSE 0 END) AS missing_tenant,
      sum(CASE WHEN c.text IS NULL OR c.text = '' THEN 1 ELSE 0 END) AS empty_text,
      sum(CASE WHEN c.allow_access_control IS NULL THEN 1 ELSE 0 END) AS missing_acl,
      sum(CASE WHEN c.recursive_chunk_id IS NULL THEN 1 ELSE 0 END) AS missing_rid,
      sum(CASE WHEN c.url IS NULL THEN 1 ELSE 0 END) AS missing_url
"""))

show("orphan chunks (no HAS_CHUNK parent)", q(
    "MATCH (c:Chunk) WHERE NOT ()-[:HAS_CHUNK]->(c) RETURN c.chunk_id AS chunk_id"))
show("chunks whose doc_id disagrees with their parent Document", q("""
    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE c.doc_id <> d.doc_id
    RETURN c.chunk_id AS chunk_id, c.doc_id AS chunk_doc_id, d.doc_id AS parent
"""))
show("heading-only chunks (wikitext '== ... ==')", q(
    r"MATCH (c:Chunk) WHERE c.text =~ '^\\s*=+[^=]+=+\\s*$' RETURN c.chunk_id AS chunk_id, c.text AS text"))
show("shortest chunks", q(
    "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, size(c.text) AS len, c.text AS text ORDER BY len ASC LIMIT 5"))

show("entity type vocabulary", q("""
    MATCH (e:Entity) RETURN e.type AS type, count(*) AS n ORDER BY n DESC
"""))
show("non-UPPER_SNAKE entity types", q(r"""
    MATCH (e:Entity) WHERE NOT e.type =~ '^[A-Z][A-Z0-9_]*$'
    RETURN DISTINCT e.type AS type
"""))
show("entity property completeness", q("""
    MATCH (e:Entity) RETURN count(*) AS entities,
      sum(CASE WHEN e.entity_id IS NULL THEN 1 ELSE 0 END) AS missing_entity_id,
      sum(CASE WHEN e.name IS NULL OR e.name = '' THEN 1 ELSE 0 END) AS missing_name,
      sum(CASE WHEN e.description IS NULL OR e.description = '' THEN 1 ELSE 0 END) AS missing_desc,
      sum(CASE WHEN e.tenant_id IS NULL THEN 1 ELSE 0 END) AS missing_tenant,
      sum(CASE WHEN e.allow_access_control IS NULL THEN 1 ELSE 0 END) AS missing_acl,
      sum(CASE WHEN e.degree IS NULL THEN 1 ELSE 0 END) AS missing_degree
"""))
show("entities with no MENTIONS edge (unsupported by any chunk)", q("""
    MATCH (e:Entity) WHERE NOT ()-[:MENTIONS]->(e)
    RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type
"""), limit=10)
show("sample entities", q("""
    MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.degree AS degree,
      left(e.description, 150) AS description ORDER BY e.degree DESC LIMIT 8
"""))

show("relationship type vocabulary", q("""
    MATCH ()-[r:EXTRACTED_REL]->() RETURN r.relationship_type AS type, count(*) AS n ORDER BY n DESC LIMIT 20
"""))
show("non-UPPER_SNAKE relationship types", q(r"""
    MATCH ()-[r:EXTRACTED_REL]->() WHERE NOT r.relationship_type =~ '^[A-Z][A-Z0-9_]*$'
    RETURN DISTINCT r.relationship_type AS type
"""))
show("self-loop relationships", q(
    "MATCH (a:Entity)-[r:EXTRACTED_REL]->(a) RETURN a.name AS name, r.relationship_type AS type"))
show("sample relationships", q("""
    MATCH (a:Entity)-[r:EXTRACTED_REL]->(b:Entity)
    RETURN a.name AS src, r.relationship_type AS rel, b.name AS tgt,
      left(r.descriptions_json, 120) AS descriptions LIMIT 8
"""))

show("claims", q("""
    MATCH (c:Claim) RETURN count(*) AS claims,
      sum(CASE WHEN NOT (c)-[:ABOUT]->() THEN 1 ELSE 0 END) AS claims_without_subject,
      sum(CASE WHEN NOT ()-[:SUPPORTED_BY]->(c) AND NOT (c)-[:SUPPORTED_BY]->() THEN 1 ELSE 0 END) AS claims_without_support
"""))
show("sample claims", q("""
    MATCH (cl:Claim) OPTIONAL MATCH (cl)-[:ABOUT]->(e:Entity)
    RETURN e.name AS about, cl.predicate AS predicate, cl.object AS object,
      cl.statement_type AS statement_type, cl.temporal_type AS temporal_type LIMIT 8
"""))
