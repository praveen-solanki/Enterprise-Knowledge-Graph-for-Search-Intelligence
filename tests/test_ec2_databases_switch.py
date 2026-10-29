"""USE_EC2_DATABASES switch: EC2 docker vs cloud Neo4j/Mongo/Qdrant."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def test_use_ec2_databases_false_keeps_cloud_endpoints(monkeypatch):
    monkeypatch.setenv("USE_EC2_DATABASES", "false")
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:17474")
    monkeypatch.setenv("NEO4J_PASSWORD", "cloud-neo4j")
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27018")
    monkeypatch.setenv("MONGO_DB", "cloud_db")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:16333")
    monkeypatch.setenv("QDRANT_API_KEY", "cloud-qdrant")
    # EC2 values present but must be ignored when switch is false
    monkeypatch.setenv("EC2_NEO4J_HTTP_URL", "http://127.0.0.1:7474")
    monkeypatch.setenv("EC2_NEO4J_PASSWORD", "local-neo4j")
    monkeypatch.setenv("EC2_MONGO_URI", "mongodb://root:local@127.0.0.1:27017")
    monkeypatch.setenv("EC2_QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("EC2_QDRANT_API_KEY", "local-qdrant")

    s = Settings(_env_file=None)
    assert s.use_ec2_databases is False
    assert s.neo4j_http_url == "http://localhost:17474"
    assert s.neo4j_password == "cloud-neo4j"
    assert s.mongo_uri == "mongodb://localhost:27018"
    assert s.mongo_db == "cloud_db"
    assert s.qdrant_url == "http://localhost:16333"
    assert s.qdrant_api_key == "cloud-qdrant"


def test_use_ec2_databases_true_overrides_with_ec2_endpoints(monkeypatch):
    monkeypatch.setenv("USE_EC2_DATABASES", "true")
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:17474")
    monkeypatch.setenv("NEO4J_PASSWORD", "cloud-neo4j")
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27018")
    monkeypatch.setenv("MONGO_DB", "cloud_db")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:16333")
    monkeypatch.setenv("QDRANT_API_KEY", "cloud-qdrant")
    monkeypatch.setenv("EC2_NEO4J_HTTP_URL", "http://127.0.0.1:7474")
    monkeypatch.setenv("EC2_NEO4J_PASSWORD", "local-neo4j")
    monkeypatch.setenv("EC2_MONGO_URI", "mongodb://root:local@127.0.0.1:27017/?authSource=admin")
    monkeypatch.setenv("EC2_MONGO_DB", "local_db")
    monkeypatch.setenv("EC2_QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("EC2_QDRANT_API_KEY", "local-qdrant")

    s = Settings(_env_file=None)
    assert s.use_ec2_databases is True
    assert s.neo4j_http_url == "http://127.0.0.1:7474"
    assert s.neo4j_password == "local-neo4j"
    assert s.mongo_uri.startswith("mongodb://root:local@127.0.0.1:27017")
    assert s.mongo_db == "local_db"
    assert s.qdrant_url == "http://127.0.0.1:6333"
    assert s.qdrant_api_key == "local-qdrant"


def test_use_ec2_databases_true_requires_ec2_password(monkeypatch):
    monkeypatch.setenv("USE_EC2_DATABASES", "true")
    monkeypatch.setenv("EC2_NEO4J_HTTP_URL", "http://127.0.0.1:7474")
    monkeypatch.delenv("EC2_NEO4J_PASSWORD", raising=False)
    monkeypatch.setenv("EC2_MONGO_URI", "mongodb://root:local@127.0.0.1:27017")
    monkeypatch.setenv("EC2_QDRANT_URL", "http://127.0.0.1:6333")

    with pytest.raises(ValidationError, match="EC2_NEO4J_PASSWORD"):
        Settings(_env_file=None)
