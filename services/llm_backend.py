"""Resolves, ONCE per process, whether extraction/gleaning should use the
in-house Ministral model or Azure OpenAI.

Resolved once at process startup (core/entrypoint.py for the bulk job,
core/kafka_worker_entrypoint.py for the incremental worker) and cached for
that process's whole lifetime — never re-checked per call. This is
deliberate: every call in a run must use the same backend throughout. A
single chunk's Ministral call failing mid-run must NOT flip that one call
over to Azure while the rest of the run keeps using Ministral — that would
mean two different models silently contributing to the same run's output.
The only fallback point is the one-time probe below, at startup.

Embeddings are NOT probed: the embedding backend is pure declared config
(settings.embedding_backend, one of the two supported spaces in
config/settings.py's SUPPORTED_EMBEDDING_SPACES). A probe-style fallback
would silently swap the model and land vectors in a different space — for
embeddings an unreachable backend must fail the work, never reroute it.

Plain .env-only local dev (ENV unset/local_env, no Vault — see
connectors/vault_manager.py's own ENV semantics) skips the reachability
probe and resolves to Azure when MINISTRAL_URL is still the in-cluster
service: that hostname does not resolve from a laptop. A local_env whose
MINISTRAL_URL points at localhost (a local vLLM) is probed like any other
environment, so the opt-in actually reaches the box. local_vault and any
real deployed environment always run the probe."""
from __future__ import annotations

import logging
from typing import Literal

import requests

from config.settings import SUPPORTED_EMBEDDING_SPACES, settings
from services.run_context import RunConfig, active_run_config

logger = logging.getLogger("graphrag.services.llm_backend")

LlmBackend = Literal["ministral", "azure"]

# Exported: scripts/check_ministral.py reads this module attribute directly.
_PROBE_TIMEOUT_SECONDS = settings.llm_probe_timeout_seconds

_llm_backend: LlmBackend = "azure"
_resolved = False


def _is_plain_local_env() -> bool:
    """Same ENV semantics as connectors/vault_manager.py's setup_environment_variables:
    local_env is a human on a laptop with no Vault. local_vault (real Vault via
    AppRole) and any other value (a real deployed pod) both get the real probe
    below, not this shortcut."""
    return settings.env == "local_env"


def _ministral_url_is_local() -> bool:
    """True when MINISTRAL_URL is a laptop vLLM (or anything else bound to
    this machine), so the local_env shortcut must not hide it."""
    url = settings.ministral_url.lower()
    return "localhost" in url or "127.0.0.1" in url


def _using_nvidia_nim() -> bool:
    """The explicit switch (settings.inference_provider), not URL sniffing —
    see config/settings.py's _apply_inference_provider."""
    return settings.inference_provider == "nvidia"


def _probe_ministral() -> bool:
    """One minimal real chat-completion request — the cheapest way to confirm the
    whole path (network + endpoint + model actually loaded) works, not just that
    something answers on the port."""
    headers = {}
    if settings.ministral_api_key:
        headers["Authorization"] = f"Bearer {settings.ministral_api_key}"
    try:
        response = requests.post(
            settings.ministral_url,
            json={
                "model": settings.ministral_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            headers=headers,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        return 200 <= response.status_code < 300
    except Exception as exc:  # noqa: BLE001 — any failure here just means "unreachable"
        logger.warning("Chat reachability probe failed — falling back to Azure "
                       "for this run: %s", exc)
        return False


def resolve_llm_backends() -> None:
    """Call once, synchronously, at process startup — before constructing
    ExtractionService or doing any real work. Idempotent: only the first call
    actually probes, every later call is a no-op, so it's safe to call
    defensively from more than one place."""
    global _llm_backend, _resolved
    if _resolved:
        return

    # NIM is the default and is reachable from a laptop, so local_env must not
    # skip it the way it skips an in-cluster hostname. A failed probe does not
    # swap NIM for Azure — that would silently leave the configured backend.
    if _using_nvidia_nim() and settings.use_in_house_model:
        ok = _probe_ministral()
        _llm_backend = "ministral"
        if not ok:
            logger.warning("NVIDIA NIM probe did not succeed — this process still "
                           "uses NIM (the configured default). Set NVIDIA_API_KEY.")
    elif _is_plain_local_env() and not _ministral_url_is_local():
        _llm_backend = "azure"
        logger.info("Plain local_env with a non-local chat URL — skipping the "
                    "reachability probe, using Azure OpenAI for this process")
    else:
        _llm_backend = "ministral" if settings.use_in_house_model and _probe_ministral() else "azure"
    logger.info("Resolved LLM backend for this process: llm=%s", _llm_backend)
    _resolved = True


def get_llm_backend() -> LlmBackend:
    """Resolves lazily on first use if resolve_llm_backends() wasn't already
    called explicitly at startup (e.g. in tests) — still only ever probes once."""
    if not _resolved:
        resolve_llm_backends()
    return _llm_backend


def reset_for_tests() -> None:
    """Test-only: clears the cached resolution so each test starts fresh
    instead of inheriting whatever an earlier test resolved."""
    global _resolved
    _resolved = False


def snapshot_run_config() -> RunConfig:
    """The model/space/flags a run resolves to RIGHT NOW — captured once at bulk
    run start (core/pipeline.py) and persisted in the run registry so an
    incremental event reuses the exact same space and model instead of
    re-resolving against its own worker's process-global settings + backend probe.

    This is the ONE place the LLM backend->model mapping is made for persistence;
    it must mirror the per-call read in services/llm_client.complete()
    (ministral_model vs azure_openai_completion_model). The embedding space is a
    verbatim snapshot of the declared settings — no probe, see the module
    docstring. Everything else is likewise a verbatim snapshot of the settings
    each consumer reads today."""
    llm_backend = get_llm_backend()
    # settings.embedding_backend is the ONE embedding knob; the space's model,
    # dimension, and upstream ES field all resolve through the supported-spaces
    # table, so a mixed space is unrepresentable in config.
    space = SUPPORTED_EMBEDDING_SPACES[settings.embedding_backend]
    return RunConfig(
        embedding_backend=settings.embedding_backend,
        embedding_model=space["model"],
        embedding_dimension=space["dimension"],
        consume_upstream_vector=settings.consume_upstream_vector,
        es_chunk_vector_field=space["es_vector_field"],
        llm_backend=llm_backend,
        llm_model=(settings.ministral_model if llm_backend == "ministral"
                   else settings.azure_openai_completion_model),
        build_authorship_edges=settings.build_authorship_edges,
        contributor_entity_types=tuple(settings.contributor_entity_types),
        people_org_graph_enabled=settings.people_org_graph_enabled,
    )


def effective_run_config() -> RunConfig:
    """The config the current work must use: the run bound for this event
    (incremental — core/kafka_worker_entrypoint.py binds it per message), or a
    live snapshot of process settings when nothing is bound (the bulk path,
    where process settings ARE the run's config). Every model/space/flag
    consumer reads through this so the two paths share one source of truth."""
    return active_run_config() or snapshot_run_config()
