"""Unit tests for repository/neo4j_repository.py's per-run database creation
(ensure_database, and Neo4jRepository's database= override).
Mocks requests.post directly; nothing here touches a real Neo4j server."""
from unittest.mock import MagicMock, patch

import pytest

from repository.neo4j_repository import Neo4jRepository, ensure_database


def _mock_settings(mock_settings):
    mock_settings.neo4j_http_url = "http://localhost:7474"
    mock_settings.neo4j_username = "neo4j"
    mock_settings.neo4j_password = "pw"
    mock_settings.verify_certs = True
    # Real (non-MagicMock) values — ensure_database's polling loop does real
    # arithmetic/comparisons against these (deadline = monotonic() + this, etc.).
    mock_settings.neo4j_database_online_max_wait_seconds = 30.0
    mock_settings.neo4j_database_online_poll_interval_seconds = 0.5
    mock_settings.neo4j_http_timeout_seconds = 30.0


def _response(fields, values, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": {"fields": fields, "values": values}}
    return resp


def test_ensure_database_issues_create_database_against_system_db():
    with patch("repository.neo4j_repository.settings") as mock_settings, \
         patch("repository.neo4j_repository.requests.post") as mock_post:
        _mock_settings(mock_settings)
        mock_post.side_effect = [
            _response([], []),  # CREATE DATABASE ... IF NOT EXISTS
            _response(["currentStatus"], [["online"]]),  # SHOW DATABASE
        ]
        ensure_database("g-acme-intranet-ab12")

    create_call = mock_post.call_args_list[0]
    assert create_call.args[0] == "http://localhost:7474/db/system/query/v2"
    assert "CREATE DATABASE `g-acme-intranet-ab12` IF NOT EXISTS" in create_call.kwargs["json"]["statement"]


def test_ensure_database_polls_until_online():
    with patch("repository.neo4j_repository.settings") as mock_settings, \
         patch("repository.neo4j_repository.requests.post") as mock_post, \
         patch("repository.neo4j_repository.time.sleep") as mock_sleep:
        _mock_settings(mock_settings)
        mock_post.side_effect = [
            _response([], []),  # CREATE DATABASE
            _response(["currentStatus"], [["allocating"]]),  # not online yet
            _response([], [], status_code=503),  # ... and not answering queries either
            _response(["currentStatus"], [["online"]]),  # now online
        ]
        ensure_database("g-acme-intranet-ab12")
    assert mock_post.call_count == 4
    mock_sleep.assert_called_once()


def test_ensure_database_times_out_if_never_online():
    with patch("repository.neo4j_repository.settings") as mock_settings, \
         patch("repository.neo4j_repository.requests.post") as mock_post, \
         patch("repository.neo4j_repository.time.sleep"), \
         patch("repository.neo4j_repository.time.monotonic", side_effect=[0, 0, 100]):
        _mock_settings(mock_settings)
        mock_post.side_effect = [
            _response([], []),  # CREATE DATABASE
            _response(["currentStatus"], [["allocating"]]),  # iteration 1: not online yet
            _response([], [], status_code=503),  # ... nor answering queries
            _response(["currentStatus"], [["allocating"]]),  # iteration 2: still not online -> times out
            _response([], [], status_code=503),
        ]
        with pytest.raises(TimeoutError, match="did not come online"):
            ensure_database("g-acme-intranet-ab12")


def test_ensure_database_requires_neo4j_http_url():
    with patch("repository.neo4j_repository.settings") as mock_settings:
        mock_settings.neo4j_http_url = ""
        with pytest.raises(ValueError, match="NEO4J_HTTP_URL"):
            ensure_database("g-acme-intranet-ab12")


def test_neo4j_repository_uses_the_explicit_database():
    with patch("repository.neo4j_repository.settings") as mock_settings:
        _mock_settings(mock_settings)
        repo = Neo4jRepository(database="g-acme-intranet-ab12")
    assert repo._url == "http://localhost:7474/db/g-acme-intranet-ab12/query/v2"


def test_neo4j_repository_requires_an_explicit_database():
    """PR-41 review point 7: there is no settings default anymore — the run's own
    RunContext.neo4j_database is the only source, so constructing without one must
    raise rather than silently target a shared 'neo4j'."""
    with patch("repository.neo4j_repository.settings") as mock_settings:
        _mock_settings(mock_settings)
        with pytest.raises(ValueError, match="requires an explicit database"):
            Neo4jRepository()
