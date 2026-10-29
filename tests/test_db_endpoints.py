"""resolve_db_endpoints: the single EC2-vs-primary selection used per Kafka
message (config/db_endpoints.py)."""
from __future__ import annotations

from unittest.mock import patch

from config import db_endpoints
from config.db_endpoints import resolve_db_endpoints
from config.settings import settings


def test_resolve_true_returns_ec2_endpoints():
    ep = resolve_db_endpoints(True)
    assert ep.use_ec2 is True
    assert ep.mongo_uri == settings.ec2_mongo_uri
    assert ep.mongo_db == settings.ec2_mongo_db
    assert ep.neo4j_http_url == settings.ec2_neo4j_http_url
    assert ep.neo4j_username == settings.ec2_neo4j_username
    assert ep.neo4j_password == settings.ec2_neo4j_password
    assert ep.qdrant_url == settings.ec2_qdrant_url
    assert ep.qdrant_api_key == settings.ec2_qdrant_api_key


def test_resolve_false_returns_primary_endpoints():
    ep = resolve_db_endpoints(False)
    assert ep.use_ec2 is False
    assert ep.mongo_uri == settings.mongo_uri
    assert ep.mongo_db == settings.mongo_db
    assert ep.neo4j_http_url == settings.neo4j_http_url
    assert ep.qdrant_url == settings.qdrant_url


def test_resolve_none_follows_the_global_setting():
    """A header-less message (use_ec2=None) must behave exactly as before
    per-message routing existed — i.e. follow settings.use_ec2_databases."""
    with patch.object(db_endpoints.settings, "use_ec2_databases", True):
        assert resolve_db_endpoints(None).use_ec2 is True
    with patch.object(db_endpoints.settings, "use_ec2_databases", False):
        assert resolve_db_endpoints(None).use_ec2 is False
