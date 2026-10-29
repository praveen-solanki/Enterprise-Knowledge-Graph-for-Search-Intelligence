"""Precedence: .env beats stale shell; local_vault Vault beats other stale shell keys."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from connectors.vault_manager import (
    _apply_dotenv,
    _inject_secret_keys,
)


def test_apply_dotenv_overwrites_stale_shell_for_keys_in_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NEO4J_HTTP_URL=http://localhost:7474\n"
        "KAFKA_CONSUMER_GROUP=graphrag-ingestion-service-group-ravi-local-dev\n"
        "# COMMENTED_OUT=ignore-me\n"
    )
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:17474")
    monkeypatch.setenv("KAFKA_CONSUMER_GROUP", "graphrag-ingestion-service-group-anything-dev")
    monkeypatch.setenv("ONLY_IN_SHELL", "stale-shell-value")

    keys = _apply_dotenv(str(env_file))

    assert "NEO4J_HTTP_URL" in keys
    assert "KAFKA_CONSUMER_GROUP" in keys
    assert "COMMENTED_OUT" not in keys
    assert os.environ["NEO4J_HTTP_URL"] == "http://localhost:7474"
    assert os.environ["KAFKA_CONSUMER_GROUP"] == "graphrag-ingestion-service-group-ravi-local-dev"
    # Keys absent from .env are left alone (shell or later Vault owns them).
    assert os.environ["ONLY_IN_SHELL"] == "stale-shell-value"


def test_inject_local_vault_overwrites_shell_except_dotenv_keys(monkeypatch):
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:7474")  # from .env
    monkeypatch.setenv("ELASTICSEARCH_HOST", "localhost:19200")

    env_file_keys = frozenset({"NEO4J_HTTP_URL"})
    _inject_secret_keys(
        {
            "NEO4J_HTTP_URL": "http://localhost:27474",
            "ELASTICSEARCH_HOST": "localhost:29200",
        },
        env_file_keys=env_file_keys,
        local_vault=True,
    )

    assert os.environ["NEO4J_HTTP_URL"] == "http://localhost:7474"
    assert os.environ["ELASTICSEARCH_HOST"] == "localhost:29200"


def test_inject_deployed_keeps_configmap_over_vault(monkeypatch):
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:37474")

    _inject_secret_keys(
        {"NEO4J_HTTP_URL": "http://localhost:27474", "NEW_FROM_VAULT": "v"},
        env_file_keys=frozenset(),
        local_vault=False,
    )

    assert os.environ["NEO4J_HTTP_URL"] == "http://localhost:37474"
    assert os.environ["NEW_FROM_VAULT"] == "v"


def test_apply_dotenv_noop_when_file_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEO4J_HTTP_URL", "http://localhost:37474")
    missing = tmp_path / "no-such.env"

    keys = _apply_dotenv(str(missing))

    assert keys == frozenset()
    assert os.environ["NEO4J_HTTP_URL"] == "http://localhost:37474"
