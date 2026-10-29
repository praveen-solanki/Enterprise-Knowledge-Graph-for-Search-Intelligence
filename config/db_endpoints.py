"""The single place EC2-vs-primary datastore selection is decided.

Historically this was a process-global choice (settings.use_ec2_databases →
Settings._apply_ec2_databases, which swaps Mongo+Neo4j+Qdrant endpoints at
startup). The incremental Kafka worker now needs it PER MESSAGE — a
`use_ec2_database` header routes one event to a specific environment — so the
choice is factored out here into a pure resolver returning an immutable bundle.

Both the primary fields (settings.mongo_uri / neo4j_http_url / qdrant_url …)
and the EC2 fields (settings.ec2_mongo_uri / ec2_neo4j_http_url / …) already
live in Settings and are sourced from the Vault `.env` config — nothing here
is hardcoded, and this module never mutates settings.

Deployment note: the incremental worker should leave the global
USE_EC2_DATABASES flag UNSET and provide both primary and EC2 credentials, so
the header decides per message. When the global flag IS set, Settings.
_apply_ec2_databases overwrites the primary fields with the EC2 ones at
startup, so a header explicitly asking for primary can no longer recover them —
that one combination is unsupported (and unneeded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import settings


@dataclass(frozen=True)
class DbEndpoints:
    """One environment's Mongo + Neo4j + Qdrant connection settings. Frozen so a
    bound per-message selection can't be mutated mid-event."""
    use_ec2: bool
    mongo_uri: str
    mongo_db: str
    neo4j_http_url: str
    neo4j_username: str
    neo4j_password: str
    qdrant_url: str
    qdrant_api_key: str


def resolve_db_endpoints(use_ec2: Optional[bool] = None) -> DbEndpoints:
    """Resolve the datastore endpoints for one unit of work.

    use_ec2 = None (the default) falls back to the process-global
    settings.use_ec2_databases, so a caller that passes nothing — and every
    path that existed before per-message routing — behaves exactly as before.
    """
    if use_ec2 is None:
        use_ec2 = settings.use_ec2_databases

    if use_ec2:
        return DbEndpoints(
            use_ec2=True,
            mongo_uri=settings.ec2_mongo_uri,
            mongo_db=settings.ec2_mongo_db,
            neo4j_http_url=settings.ec2_neo4j_http_url,
            neo4j_username=settings.ec2_neo4j_username,
            neo4j_password=settings.ec2_neo4j_password,
            qdrant_url=settings.ec2_qdrant_url,
            qdrant_api_key=settings.ec2_qdrant_api_key,
        )
    return DbEndpoints(
        use_ec2=False,
        mongo_uri=settings.mongo_uri,
        mongo_db=settings.mongo_db,
        neo4j_http_url=settings.neo4j_http_url,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )
