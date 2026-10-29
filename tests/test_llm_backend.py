"""Unit tests for services/llm_backend.py -- resolving, once per process,
whether extraction/gleaning use Ministral or Azure. Embeddings are NOT
probed: the embedding backend is pure declared config (the embedding-space
table), snapshotted into RunConfig -- see the snapshot tests below.
Resolution is cached at module level, so every test resets it first
(reset_for_tests) to avoid depending on whatever an earlier test in the
same session happened to resolve."""
import pytest

import services.llm_backend as llm_backend
from services.llm_backend import get_llm_backend, resolve_llm_backends, snapshot_run_config


@pytest.fixture(autouse=True)
def _reset():
    llm_backend.reset_for_tests()
    yield
    llm_backend.reset_for_tests()


def test_plain_local_env_skips_the_probe_and_uses_azure(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "inference_provider", "inhouse")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://vllm:8000/v1/chat/completions")
    called = {"ministral": False}
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: called.__setitem__("ministral", True) or True)
    resolve_llm_backends()
    assert get_llm_backend() == "azure"
    assert called == {"ministral": False}  # never even probed


def test_local_vault_probes_and_uses_ministral_when_reachable(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_vault")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: True)
    resolve_llm_backends()
    assert get_llm_backend() == "ministral"


def test_deployed_env_falls_back_to_azure_when_ministral_unreachable(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "prod")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    monkeypatch.setattr(llm_backend.settings, "inference_provider", "inhouse")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://vllm:8000/v1/chat/completions")
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: False)
    resolve_llm_backends()
    assert get_llm_backend() == "azure"


def test_nvidia_nim_stays_selected_when_the_probe_fails(monkeypatch):
    """NIM is the default. A failed probe must not silently swap it for Azure."""
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    monkeypatch.setattr(llm_backend.settings, "inference_provider", "nvidia")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "https://integrate.api.nvidia.com/v1/chat/completions")
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: False)
    resolve_llm_backends()
    assert get_llm_backend() == "ministral"


def test_deployed_env_uses_azure_when_use_in_house_model_is_off_regardless_of_reachability(monkeypatch):
    """Operator intent still gates the probe -- a reachable Ministral is not
    used at all if the operator hasn't opted in."""
    monkeypatch.setattr(llm_backend.settings, "env", "prod")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", False)
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: True)
    resolve_llm_backends()
    assert get_llm_backend() == "azure"


def test_resolution_only_happens_once_even_if_called_repeatedly(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_vault")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    call_count = {"n": 0}

    def _probe():
        call_count["n"] += 1
        return True

    monkeypatch.setattr(llm_backend, "_probe_ministral", _probe)
    resolve_llm_backends()
    resolve_llm_backends()
    get_llm_backend()
    get_llm_backend()
    assert call_count["n"] == 1


def test_get_llm_backend_resolves_lazily_if_never_called_explicitly(monkeypatch):
    """A caller that forgets to call resolve_llm_backends() at startup (e.g. a
    test, or some path outside the two real entrypoints) still gets a real,
    one-time-only resolution instead of an unresolved/undefined state."""
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "inference_provider", "inhouse")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://vllm:8000/v1/chat/completions")
    assert get_llm_backend() == "azure"  # resolved on first read, not before


def test_local_env_probes_a_localhost_vllm_and_uses_it_when_reachable(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    monkeypatch.setattr(llm_backend.settings, "inference_provider", "inhouse")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://localhost:8000/v1/chat/completions")
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: True)
    resolve_llm_backends()
    assert get_llm_backend() == "ministral"


# --- the embedding space snapshot: declared config, never probed ---

def test_snapshot_resolves_the_embedding_space_from_the_supported_spaces_table(monkeypatch):
    """settings.embedding_backend is the one knob; model, dimension, and the
    upstream ES field come from the table, so a mixed space is unrepresentable."""
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://vllm:8000/v1/chat/completions")
    monkeypatch.setattr(llm_backend.settings, "embedding_backend", "azure")
    cfg = snapshot_run_config()
    space = llm_backend.SUPPORTED_EMBEDDING_SPACES["azure"]
    assert cfg.embedding_backend == "azure"
    assert cfg.embedding_model == space["model"]
    assert cfg.embedding_dimension == space["dimension"]
    assert cfg.es_chunk_vector_field == space["es_vector_field"]


def test_snapshot_inhouse_space_resolves_through_the_same_table(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_env")
    monkeypatch.setattr(llm_backend.settings, "ministral_url",
                        "http://vllm:8000/v1/chat/completions")
    monkeypatch.setattr(llm_backend.settings, "embedding_backend", "inhouse")
    cfg = snapshot_run_config()
    space = llm_backend.SUPPORTED_EMBEDDING_SPACES["inhouse"]
    assert cfg.embedding_backend == "inhouse"
    assert (cfg.embedding_model, cfg.embedding_dimension) == (space["model"], space["dimension"])


def test_snapshot_records_the_resolved_llm_backend_and_its_model(monkeypatch):
    monkeypatch.setattr(llm_backend.settings, "env", "local_vault")
    monkeypatch.setattr(llm_backend.settings, "use_in_house_model", True)
    monkeypatch.setattr(llm_backend, "_probe_ministral", lambda: True)
    cfg = snapshot_run_config()
    assert cfg.llm_backend == "ministral"
    assert cfg.llm_model == llm_backend.settings.ministral_model
