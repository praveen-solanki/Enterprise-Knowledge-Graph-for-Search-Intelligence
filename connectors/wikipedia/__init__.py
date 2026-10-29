"""Wikipedia connector — the zero-local-dependency connector.

Reads live from the public, no-auth MediaWiki Action API
(https://www.mediawiki.org/wiki/API:Search): no Elasticsearch, no download or
indexing step, no credentials. --index is the search phrase (or
"Category:<name>" to walk a category instead); settings.wikipedia_api_base_url
can point at any MediaWiki install, so a self-hosted mirror works unchanged.

Implements the same `ConnectorSpec` contract as Zeus and Confluence
(connectors/base.py) — nothing downstream of fetch (extraction,
canonicalization, Neo4j/Qdrant write) knows or cares that this connector has
no site/audience/identity system behind it.
"""
