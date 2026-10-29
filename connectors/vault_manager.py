"""HashiCorp Vault secret retrieval + startup env injection.

Three ENV-routed modes, KV v2, secrets fetched once at startup and injected
into os.environ so the existing Settings (config/settings.py) resolves them
exactly like a locally-set env var — Settings never talks to Vault directly.

  ENV=local_env   -> no Vault; read plain .env (the local-dev default)
  ENV=local_vault -> hvac + AppRole, bootstrapped from AWS Secrets Manager
  ENV=<deployed>  -> the internal Vault SDK (ai_infra_vault.VaultClient)

Path layout: one secret per concern, e.g. SERVICE_CONFIG_VAULT_PATH ->
"ai-graph-ingestion/<env>/static/service-config", holding flags like
USE_IN_HOUSE_MODEL. Helm supplies the concrete path via each *_VAULT_PATH env
var below; this module only reads whatever of those are actually set and fans
their contents into os.environ. Adding a new secret later is a one-line
addition to _PATH_ENV_VARS below, no other code changes.

Secret keys are copied out VERBATIM (no renaming): a secret key
USE_IN_HOUSE_MODEL lands as the env var USE_IN_HOUSE_MODEL, which Settings
reads directly.

Precedence (laptop vs deploy):
  * Keys present in ``.env`` always win over ambient shell exports. Long-lived
    terminals keep stale ``NEO4J_HTTP_URL`` / ``KAFKA_CONSUMER_GROUP`` from prior
    environment switches; ``load_dotenv(override=False)`` left those stuck
    forever. Deployed images ship no ``.env``, so this is a no-op there.
  * ``ENV=local_vault``: Vault overwrites other stale shell keys, but never a
    key the ``.env`` already set (so ``NEO4J_HTTP_URL=http://localhost:7474``
    in ``.env`` still beats whatever Vault's neo4j secret holds).
  * Deployed: ``setdefault`` — ConfigMap/Helm wins over Vault for keys already
    present (same as before).

This module must be imported and called (setup_environment_variables())
BEFORE config.settings is imported anywhere in the process — Settings is
built once, eagerly, the moment it's first imported, so Vault's values have
to already be in os.environ by then. Both core/entrypoint.py and
core/kafka_worker_entrypoint.py call this as their very first line, ahead of
every other project-local import.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol
from urllib.parse import quote_plus

# TLS trust: make every later `requests`/`hvac`/`urllib3` HTTPS call (Vault,
# Wikipedia, arXiv, NVIDIA NIM, ...) verify against the OS-native certificate
# store instead of the `certifi` bundle `requests` ships with. This is what
# lets the process work behind a corporate TLS-inspecting proxy (e.g.
# Zscaler) whose root CA is already trusted by macOS/Windows/Linux itself
# (and therefore by `curl`) but is NOT in `certifi` — without ever exporting
# or handling the proxy's certificate material ourselves. Must happen before
# any ssl.SSLContext is created, so this sits at the top of the file that is
# already documented as "import this before anything else" (see module
# docstring above). A laptop with no such proxy is unaffected either way.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - truststore ships in requirements.txt
    pass

logger = logging.getLogger("graphrag.connectors.vault")

_DOTENV_PATH = ".env"

# Vault path env var -> read once at startup if set; a secret is a flat
# {KEY: value} dict, each KEY copied into os.environ under its own name.
# Grouped by concern (config/settings.py's own section headers), one KV v2
# secret per group, mirroring ai-graph-retrieval's vault_manager.py:
#   NEO4J_VAULT_PATH            -> neo4j_username/neo4j_password/neo4j_uri
#   MONGO_VAULT_PATH            -> mongo_uri (checkpoints)
#   ELASTIC_VAULT_PATH          -> es_host/es_username/es_password
#   QDRANT_VAULT_PATH           -> qdrant_url/qdrant_api_key
#   LLM_VAULT_PATH              -> azure_openai_{endpoint,api_key} +
#                                  azure_openai_embedd_{endpoint,api_key}
#   KAFKA_VAULT_PATH            -> kafka_bootstrap_servers (shared infra secret —
#                                  path is "infra/<env>/static/kafka", NOT under
#                                  this service's own "ai-graph-ingestion/" prefix,
#                                  since the broker cluster is shared company-wide)
#   SERVICE_CONFIG_VAULT_PATH   -> USE_IN_HOUSE_MODEL / USE_IN_HOUSE_EMBEDDINGS /
#                                  CONSUME_UPSTREAM_VECTOR (bools; Settings maps
#                                  USE_IN_HOUSE_EMBEDDINGS → embedding_backend) AND
#                                  USE_EC2_DATABASES + EC2_NEO4J_HTTP_URL /
#                                  EC2_NEO4J_USERNAME / EC2_NEO4J_PASSWORD /
#                                  EC2_MONGO_URI / EC2_MONGO_DB / EC2_QDRANT_URL /
#                                  EC2_QDRANT_API_KEY — when USE_EC2_DATABASES=true
#                                  Settings (_apply_ec2_databases) swaps Neo4j/Mongo/
#                                  Qdrant to these EC2 docker endpoints. All fanned
#                                  verbatim (their key == the field's validation_alias),
#                                  so this path is their single source of truth in
#                                  Vault modes.
_PATH_ENV_VARS = (
    "NEO4J_VAULT_PATH",
    "MONGO_VAULT_PATH",
    "ELASTIC_VAULT_PATH",
    "QDRANT_VAULT_PATH",
    "LLM_VAULT_PATH",
    "KAFKA_VAULT_PATH",
    "SERVICE_CONFIG_VAULT_PATH",
)


class _VaultReader(Protocol):
    def read(self, path: str) -> dict: ...


class LocalVaultManager:
    """hvac + AppRole, bootstrapped from AWS Secrets Manager (ENV=local_vault)."""

    # Helm / ConfigMap supply short paths (``ai-graph-ingestion/dev/static/neo4j``).
    # The platform SDK prepends ``services/data/``; hvac KV v2 uses mount
    # ``services`` and the remainder as the secret path.
    _MOUNT = "services"
    _PREFIX = "services/data/"

    def __init__(self) -> None:
        import hvac

        from connectors.secrets_manager import SecretsManager

        cfg = SecretsManager().get_secret(os.environ["AWS_SM_VAULT_CONFIG"])
        url = cfg.get("url") or cfg.get("host")
        client = hvac.Client(url=url)
        if cfg.get("roleId") and cfg.get("secretId"):
            client.auth.approle.login(role_id=cfg["roleId"], secret_id=cfg["secretId"])
        elif cfg.get("token"):
            client.token = cfg["token"]
        else:
            raise RuntimeError("Vault bootstrap secret has neither AppRole creds nor a token")
        if not client.is_authenticated():
            raise RuntimeError("Vault authentication failed")
        self._client = client

    def read(self, path: str) -> dict:
        secret_path = path[len(self._PREFIX) :] if path.startswith(self._PREFIX) else path
        resp = self._client.secrets.kv.v2.read_secret_version(
            mount_point=self._MOUNT, path=secret_path, raise_on_deleted_version=True
        )
        return resp["data"]["data"]


class VaultManager:
    """Deployed: the internal ai-infra Vault SDK (lazy import; KV v2 services/data/)."""

    _PREFIX = "services/data/"

    def __init__(self) -> None:
        from ai_infra_vault.vault_client import VaultClient  # internal SDK, deploy-image only

        self._client = VaultClient()
        if not self._client.is_available():
            raise RuntimeError("Vault client could not initialize from AWS Secrets Manager config")

    def read(self, path: str) -> dict:
        resolved = path if path.startswith(self._PREFIX) else self._PREFIX + path
        result = self._client.get_secret(resolved, use_cache=True)
        if not result:
            raise RuntimeError(f"Vault returned no data for {path}")
        return result


def _get_reader(run_mode: str) -> _VaultReader:
    return LocalVaultManager() if run_mode == "local_vault" else VaultManager()


def _assemble_mongo_secret(secret: dict) -> dict:
    """Vault's Mongo secret (host/username/password/database — confirmed shape,
    same convention documented in vault-devops' docs/vault-create-secrets.md for
    other services' Mongo secrets) is not a ready connection string: 'host' is a
    TEMPLATE — a mongodb+srv:// URI with two literal placeholder tokens,
    DB_USER_NAME and DB_PASSWORD, sitting where a real username and password
    would normally go, immediately before the '@' that separates credentials
    from the cluster hostname — that must be substituted with the real
    username/password, then the database name appended.
    Returns {"MONGO_URI": ..., "MONGO_DB": ...} — the raw host/username/password/
    database keys are deliberately not injected as their own env vars, since those
    generic names risk colliding with unrelated env vars (e.g. a system $HOST).

    Raises if 'host' or 'database' is missing, or if the host template contains a
    DB_USER_NAME/DB_PASSWORD placeholder that 'username'/'password' doesn't cover —
    matches setup_environment_variables()'s own fail-closed contract. Leaving a
    placeholder unsubstituted would produce a syntactically valid but wrong URI
    that only fails at the first actual Mongo operation, instead of at startup."""
    host = secret.get("host")
    database = secret.get("database")
    if not (host and database):
        raise RuntimeError("Mongo Vault secret is missing required 'host' and/or 'database' fields")
    username = secret.get("username")
    password = secret.get("password")
    if "DB_USER_NAME" in host and not username:
        raise RuntimeError("Mongo Vault secret's host template requires 'username' but it is missing")
    if "DB_PASSWORD" in host and not password:
        raise RuntimeError("Mongo Vault secret's host template requires 'password' but it is missing")
    uri = host
    if username:
        uri = uri.replace("DB_USER_NAME", quote_plus(str(username)))
    if password:
        uri = uri.replace("DB_PASSWORD", quote_plus(str(password)))
    # database is also the ONLY db this Mongo user is authorized against (the
    # standard per-user-per-db MongoDB permission model) — must be propagated as
    # MONGO_DB too, not just embedded in the URI, since mongo_repository.py
    # selects the working db via settings.mongo_db, not the URI's own path segment.
    return {"MONGO_URI": f"{uri}/{database}", "MONGO_DB": database}


# Per-path-var transforms for secrets whose Vault shape isn't a flat
# {ENV_VAR_NAME: value} dict ready for verbatim copy (see the loop below).
_SECRET_TRANSFORMS = {
    "MONGO_VAULT_PATH": _assemble_mongo_secret,
}


def _dotenv_keys(path: str = _DOTENV_PATH) -> frozenset[str]:
    """Return keys with a non-empty value in ``path`` (missing file → empty)."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return frozenset()
    return frozenset(
        k for k, v in dotenv_values(path).items()
        if v is not None and str(v).strip() != ""
    )


def _apply_dotenv(path: str = _DOTENV_PATH) -> frozenset[str]:
    """Load ``.env`` into ``os.environ``, overwriting ambient shell for keys it sets.

    Returns the set of keys the file declared so Vault injection can leave them
    alone. No file / empty file → no-op (deployed images).
    """
    keys = _dotenv_keys(path)
    try:
        from dotenv import load_dotenv
    except ImportError:
        return keys
    # override=True only when the file actually has keys; otherwise leave the
    # process env untouched (ConfigMap / prior exports in deploy).
    load_dotenv(path, override=bool(keys))
    if keys:
        logger.info("dotenv_loaded override=True keys=%d", len(keys))
    return keys


def _inject_secret_keys(
    secret: dict,
    *,
    env_file_keys: frozenset[str],
    local_vault: bool,
) -> None:
    """Copy secret key/values into ``os.environ`` under the precedence rules above."""
    for key, value in secret.items():
        if value is None:
            continue
        if key in env_file_keys:
            continue
        text = str(value)
        if local_vault:
            os.environ[key] = text
        else:
            os.environ.setdefault(key, text)


def setup_environment_variables() -> bool:
    """Fetch every configured secret and inject its keys into os.environ.

    Called ONCE at process startup, before config.settings is imported.
    Returns True on success; a no-op in local mode (secrets come from .env).
    Fails closed — a missing secret at a configured path raises, so the
    service does not boot half-configured — except a missing platform SDK
    (ModuleNotFoundError), which degrades to reading plain env vars instead.

    Loads ``.env`` first (overriding ambient shell for keys the file sets) so
    ``ENV=local_vault``, ``AWS_DEFAULT_REGION``, and ``*_VAULT_PATH`` in the
    file are visible here — the same file pydantic-settings later reads for
    Settings.
    """
    env_file_keys = _apply_dotenv()

    run_mode = os.getenv("ENV", "local_env")
    local_dev = os.getenv("LOCAL_DEV", "").strip().lower() in ("1", "true", "yes")
    if local_dev or run_mode == "local_env":
        logger.info("Vault skipped (run_mode=%s, local_dev=%s) — using plain env/.env", run_mode, local_dev)
        return True

    try:
        reader = _get_reader(run_mode)
    except ModuleNotFoundError as exc:
        logger.warning("Vault SDK not installed (%s) — falling back to plain env vars", exc)
        return True

    local_vault = run_mode == "local_vault"
    loaded = []
    for path_var in _PATH_ENV_VARS:
        path = os.getenv(path_var)
        if not path:
            continue
        secret = reader.read(path)
        transform = _SECRET_TRANSFORMS.get(path_var)
        if transform:
            secret = transform(secret)
            if "MONGO_DB" in secret:
                # MONGO_URI and MONGO_DB must come from the same source: the
                # Vault Mongo user is authorized against exactly one database
                # (see _assemble_mongo_secret), so Vault's MONGO_DB is only
                # valid paired with Vault's own MONGO_URI — never with an
                # operator-supplied one, and never left stale next to Vault's.
                if "MONGO_URI" in os.environ and "MONGO_URI" in env_file_keys:
                    # An explicit .env MONGO_URI already selects a different
                    # Mongo user — MONGO_DB keeps its own independent value.
                    secret.pop("MONGO_DB", None)
                elif "MONGO_URI" in os.environ and not local_vault:
                    # Deployed operator/ConfigMap MONGO_URI — same as before.
                    secret.pop("MONGO_DB", None)
                elif "MONGO_URI" not in env_file_keys:
                    # Vault's own MONGO_URI is about to apply below — force
                    # its MONGO_DB too, even over a stale pre-existing value,
                    # so the pair can never end up mismatched.
                    os.environ.pop("MONGO_DB", None)
        _inject_secret_keys(secret, env_file_keys=env_file_keys, local_vault=local_vault)
        loaded.append(path_var)
    logger.info("Vault secrets loaded for: %s", loaded)
    return True
