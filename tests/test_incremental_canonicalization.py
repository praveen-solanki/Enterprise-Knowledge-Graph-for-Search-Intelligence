"""Unit tests for services/incremental_canonicalization.py (AIS-3634)."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from services.incremental_canonicalization import (
    AUTO_LINK_THRESHOLD,
    _candidate_description,
    _get_client,
    _llm_confirms_merge,
    resolve_canonical_name,
)


def _cand(name: str, description: str = "", **overrides) -> dict:
    """A find_similar_entities result row — code review must-fix #8:
    descriptions_json (per-source entries), not a flat description."""
    cand = {"name": name, "score": 0.9,
           "descriptions_json": json.dumps([{"text": description, "doc_id": "d1"}]) if description else None}
    cand.update(overrides)
    return cand


def _embedder(vector=None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.return_value = [vector or [0.1, 0.2, 0.3]]
    return embedder


def _repo(candidates=None, exact_exists=False) -> MagicMock:
    repo = MagicMock()
    repo.find_similar_entities.return_value = candidates or []
    repo.entity_exists_exact.return_value = exact_exists
    return repo


TENANT = "tenant-1"


def test_resolve_canonical_name_returns_own_name_when_no_candidates():
    name, embedding = asyncio.run(resolve_canonical_name(
        _repo(candidates=[]), _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
    ))
    assert name == "Globex"
    assert embedding == [0.1, 0.2, 0.3]


def test_resolve_canonical_name_passes_tenant_id_to_find_similar_entities():
    """Code review must-fix #4: the tenant must reach the vector search so a
    candidate from another tenant is never even considered."""
    repo = _repo(candidates=[])
    asyncio.run(resolve_canonical_name(repo, _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT))
    args, kwargs = repo.find_similar_entities.call_args
    assert TENANT in args or kwargs.get("tenant_id") == TENANT


def test_resolve_canonical_name_passes_the_given_vector_store_through_to_find_similar_entities():
    """The ANN search now runs against Qdrant (services/vector_store.py) --
    this must be the SAME vector_store instance the caller passed in, not a
    fresh one built internally."""
    repo = _repo(candidates=[])
    vector_store = MagicMock()
    asyncio.run(resolve_canonical_name(repo, _embedder(), vector_store, "Globex", "ORGANIZATION",
                                       "A company", TENANT))
    args, _ = repo.find_similar_entities.call_args
    assert args[0] is vector_store


def test_resolve_canonical_name_returns_confirmed_candidate():
    candidates = [_cand("Globex Inc.", "The company")]
    with patch("services.incremental_canonicalization._llm_confirms_merge", new=AsyncMock(return_value=True)):
        name, _ = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
        ))
    assert name == "Globex Inc."


def test_resolve_canonical_name_falls_through_unconfirmed_candidates():
    # Both scores stay below AUTO_LINK_THRESHOLD (0.93) so this exercises the
    # LLM-confirm tier, not the new no-LLM auto-link tier (see the dedicated
    # auto-link tests below).
    candidates = [_cand("Wrong Match", score=0.90), _cand("Right Match", score=0.84)]
    calls = {"Wrong Match": False, "Right Match": True}

    async def fake_confirm(new_name, new_desc, cand_name, cand_desc):
        return calls[cand_name]

    with patch("services.incremental_canonicalization._llm_confirms_merge", side_effect=fake_confirm):
        name, _ = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
        ))
    assert name == "Right Match"


# --- auto-link tier (mirrors entity_linking_service.py's auto_link_threshold) ---

def test_resolve_canonical_name_auto_links_a_near_exact_candidate_without_an_llm_call():
    candidates = [_cand("Globex Inc.", score=AUTO_LINK_THRESHOLD)]
    with patch("services.incremental_canonicalization._llm_confirms_merge",
              new=AsyncMock()) as llm_confirm:
        name, embedding = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION",
            "A company", TENANT,
        ))
    assert name == "Globex Inc."
    assert embedding == [0.1, 0.2, 0.3]
    llm_confirm.assert_not_called()


def test_resolve_canonical_name_below_auto_link_still_uses_llm_confirm():
    candidates = [_cand("Globex Inc.", "The company", score=AUTO_LINK_THRESHOLD - 0.01)]
    with patch("services.incremental_canonicalization._llm_confirms_merge",
              new=AsyncMock(return_value=True)) as llm_confirm:
        name, _ = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION",
            "A company", TENANT,
        ))
    assert name == "Globex Inc."
    llm_confirm.assert_called_once()


def test_resolve_canonical_name_auto_link_check_applies_per_candidate_not_just_the_first():
    """The auto-link check runs on EACH candidate as it's considered, not
    just the top-ranked one -- if an earlier, lower-scored candidate is
    rejected by the LLM, a later candidate that clears the auto-link bar
    must still skip the LLM entirely when its own turn comes."""
    candidates = [_cand("Rejected", score=0.90), _cand("Auto-linked", score=0.95)]
    with patch("services.incremental_canonicalization._llm_confirms_merge",
              new=AsyncMock(return_value=False)) as llm_confirm:
        name, _ = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION",
            "A company", TENANT,
        ))
    assert name == "Auto-linked"
    llm_confirm.assert_called_once()  # only for "Rejected" -- "Auto-linked" skipped it


def test_resolve_canonical_name_treats_all_unconfirmed_as_new():
    candidates = [_cand("Not It")]
    with patch("services.incremental_canonicalization._llm_confirms_merge", new=AsyncMock(return_value=False)):
        name, _ = asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
        ))
    assert name == "Globex"


# --- descriptions_json -> internal signal (code review must-fix #8) ---

def test_candidate_description_picks_first_entry_text():
    descriptions_json = json.dumps([{"text": "first", "doc_id": "d1"}, {"text": "second", "doc_id": "d2"}])
    assert _candidate_description(descriptions_json) == "first"


def test_candidate_description_empty_for_none_or_malformed_json():
    assert _candidate_description(None) == ""
    assert _candidate_description("") == ""
    assert _candidate_description("not json") == ""
    assert _candidate_description("[]") == ""


def test_resolve_canonical_name_passes_parsed_description_to_llm_confirmation():
    candidates = [_cand("Globex Inc.", "The real company")]
    with patch("services.incremental_canonicalization._llm_confirms_merge",
              new=AsyncMock(return_value=True)) as confirm:
        asyncio.run(resolve_canonical_name(
            _repo(candidates=candidates), _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
        ))
    confirm.assert_awaited_once_with("Globex", "A company", "Globex Inc.", "The real company")


# --- exact-name fast path (experiment-parity review #10) ---

def test_resolve_canonical_name_fast_paths_on_exact_existing_entity():
    """If (name, type, tenant_id) already exists in the graph, name is already
    its own canonical form -- no embedding call, no ANN query, no LLM confirm."""
    repo = _repo(exact_exists=True)
    embedder = _embedder()
    name, embedding = asyncio.run(resolve_canonical_name(
        repo, embedder, MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
    ))
    assert name == "Globex"
    assert embedding is None
    embedder.embed_batch.assert_not_called()
    repo.find_similar_entities.assert_not_called()


def test_resolve_canonical_name_checks_exact_path_with_tenant_scoping():
    repo = _repo(exact_exists=False)
    asyncio.run(resolve_canonical_name(repo, _embedder(), MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT))
    args, kwargs = repo.entity_exists_exact.call_args
    assert TENANT in args or kwargs.get("tenant_id") == TENANT


def test_resolve_canonical_name_falls_through_to_ann_when_no_exact_match():
    repo = _repo(candidates=[], exact_exists=False)
    embedder = _embedder()
    name, embedding = asyncio.run(resolve_canonical_name(
        repo, embedder, MagicMock(), "Globex", "ORGANIZATION", "A company", TENANT,
    ))
    assert name == "Globex"
    assert embedding == [0.1, 0.2, 0.3]
    embedder.embed_batch.assert_called_once()


def test_llm_confirms_merge_delegates_to_the_shared_confirm_with_retry():
    """The actual retry/backoff/logging behavior is services/llm_client.
    py's own confirm_with_retry (see tests/test_llm_client.py) -- this
    only needs to prove _llm_confirms_merge builds the right prompt/step/context
    and passes through whatever confirm_with_retry returns, not re-test retry
    logic that's now shared with services/people/disambiguation.py's own caller."""
    with patch("services.incremental_canonicalization.llm_client") as mock_module:
        mock_module.confirm_with_retry = AsyncMock(return_value=True)
        result = asyncio.run(_llm_confirms_merge("New", "new desc", "Existing", "cand desc"))
    assert result is True
    args, kwargs = mock_module.confirm_with_retry.call_args
    assert "New" in args[0] or "New" in args[1]  # prompt mentions the entity names somewhere
    assert args[2] == "merge"
    assert kwargs["step"] == "cluster_merge_confirm_incremental"
    assert kwargs["context"] == "new='New' candidate='Existing'"
    assert kwargs["log_label"] == "Merge-confirmation"


def test_get_client_delegates_to_the_shared_azure_client():
    """The actual singleton-caching behavior is services/llm_client.
    py's own (see tests/test_llm_client.py) -- this only needs to
    prove _get_client() here delegates to it, not re-test the caching."""
    with patch("services.incremental_canonicalization.llm_client") as mock_module:
        mock_module.get_client.return_value = "the-shared-client"
        assert _get_client() == "the-shared-client"
