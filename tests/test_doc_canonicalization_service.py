"""Unit tests for services/doc_canonicalization_service.py's precision holdout
(experiment-parity review C2) — same treatment as Step 3
(graph_canonicalization_service.py), scoped to one document — plus the
per-document description merge and LLM-trusted naming."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from models.extraction import ChunkExtractionResult, ExtractedEntity
from services.doc_canonicalization_service import DOC_SIMILARITY_THRESHOLD, canonicalize_document


def _fake_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    return embedder


def _result(*entities: ExtractedEntity) -> ChunkExtractionResult:
    return ChunkExtractionResult(entities=list(entities))


def _run(coro):
    return asyncio.run(coro)


def _no_description_merge():
    """Patches out the per-document description-merge LLM call for tests that
    aren't about it — the deterministic fallback (shortest_clean name, longest
    description) is what the pre-merge code path effectively did anyway."""
    return patch("services.doc_canonicalization_service.structured_llm_call",
                 new=AsyncMock(return_value=(False, None)))


def _description_merge_returning(*entities: dict):
    """Patches the per-document description-merge call to return a canned
    response. Each entity dict is {id, canonical_name, description}."""
    from services.doc_canonicalization_service import _DocumentDescriptionMerge
    parsed = _DocumentDescriptionMerge(entities=list(entities))
    return patch("services.doc_canonicalization_service.structured_llm_call",
                 new=AsyncMock(return_value=(True, parsed)))


def test_precision_values_never_reach_clustering_or_the_llm():
    chunk_results = [
        _result(
            ExtractedEntity(name="$82k", type="ENTITY", description="ACV"),
            ExtractedEntity(name="$1.2k", type="ENTITY", description="ACV"),
            ExtractedEntity(name="Globex", type="ORGANIZATION", description="d"),
            ExtractedEntity(name="Globex Inc.", type="ORGANIZATION", description="d"),
        ),
    ]

    def _cluster_must_not_see_precision_values(representatives, vectors, threshold):
        assert "$82k" not in representatives and "$1.2k" not in representatives
        return [["Globex", "Globex Inc."]]

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        assert "$82k" not in cluster and "$1.2k" not in cluster
        from services.clustering_utils import ClusterResolution, MergeGroup
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="Globex", canonical_type="ORGANIZATION",
            merged_description="merged", members=list(cluster),
        )])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _no_description_merge(), \
         patch("services.clustering_utils.cluster_by_similarity",
              side_effect=_cluster_must_not_see_precision_values), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.clustering_utils.resolve_cluster",
              side_effect=_fake_resolve_cluster):
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert "$82k" in canonical_entities
    assert "$1.2k" in canonical_entities
    assert merge_map["$82k"] == "$82k"
    assert merge_map["$1.2k"] == "$1.2k"


def test_exact_duplicate_precision_values_still_fold_together():
    chunk_results = [
        _result(ExtractedEntity(name="$82k", type="ENTITY", description="from chunk 1")),
        _result(ExtractedEntity(name="$82k", type="ENTITY", description="from chunk 2")),
    ]
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _no_description_merge():
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))
    assert list(canonical_entities.keys()) == ["$82k"]
    assert merge_map == {"$82k": "$82k"}


# --- Per-document description merge (tickets 8 & 10) ---

def test_every_mentions_description_reaches_the_reduce_call_not_just_the_first():
    """The regression this change exists to prevent: three chunks describe Amelia
    differently and only ONE description used to survive -- both as the cluster-merge
    LLM's identity signal and as the text Step 3 persists. Under-cap multi-desc
    groups now fold in reduce first; all source descriptions must reach that call."""
    chunk_results = [
        _result(ExtractedEntity(name="Amelia", type="PERSON",
                                description="A data scientist who works on enterprise search.")),
        _result(ExtractedEntity(name="Amelia", type="PERSON", description="Lives in Gurgaon.")),
        _result(ExtractedEntity(name="Amelia ", type="PERSON", description="Joined in 2022.")),
    ]
    seen = {}

    async def _capture(system_prompt, user_payload, response_model, **kwargs):
        import json
        from services.doc_canonicalization_service import _DocumentDescriptionMerge, _MergedMention
        payload = json.loads(user_payload)
        if kwargs.get("step") == "doc_description_reduce":
            seen["reduce_payload"] = payload
            return True, _DocumentDescriptionMerge(entities=[_MergedMention(
                id=0, canonical_name="Amelia",
                description="Amelia is a data scientist working on enterprise search, "
                            "based in Gurgaon, who joined in 2022.")])
        seen["merge_payload"] = payload
        return True, _DocumentDescriptionMerge(entities=[_MergedMention(
            id=0, canonical_name="Amelia",
            description="Amelia is a data scientist working on enterprise search, "
                        "based in Gurgaon, who joined in 2022.")])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_capture):
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert seen["reduce_payload"][0]["descriptions"] == [
        "A data scientist who works on enterprise search.", "Lives in Gurgaon.", "Joined in 2022."]
    # Merge sees the already-folded single description.
    assert seen["merge_payload"][0]["descriptions"] == [
        "Amelia is a data scientist working on enterprise search, "
        "based in Gurgaon, who joined in 2022."]
    assert merge_map["Amelia"] == "Amelia" and merge_map["Amelia "] == "Amelia"
    assert canonical_entities["Amelia"]["description"] == (
        "Amelia is a data scientist working on enterprise search, based in Gurgaon, "
        "who joined in 2022.")


def test_description_merge_is_one_call_per_stage_for_the_whole_document():
    """N repeated entities in a document must share ONE reduce call and ONE merge
    call — not N of either."""
    chunk_results = [
        _result(ExtractedEntity(name="Alpha", type="ORG", description="a1"),
                ExtractedEntity(name="Beta", type="ORG", description="b1")),
        _result(ExtractedEntity(name="Alpha", type="ORG", description="a2"),
                ExtractedEntity(name="Beta", type="ORG", description="b2")),
    ]
    steps = []

    async def _spy(system_prompt, user_payload, response_model, **kwargs):
        from services.doc_canonicalization_service import _DocumentDescriptionMerge, _MergedMention
        import json
        steps.append(kwargs.get("step"))
        payload = json.loads(user_payload)
        return True, _DocumentDescriptionMerge(entities=[
            _MergedMention(id=item["id"],
                           canonical_name=item["names"][0],
                           description=f"merged {item['names'][0].lower()}")
            for item in payload])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.clustering_utils.cluster_by_similarity", return_value=[["Alpha"], ["Beta"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_spy):
        canonical_entities, _ = _run(canonicalize_document(chunk_results))

    assert steps.count("doc_description_reduce") == 1
    assert steps.count("doc_description_merge") == 1
    assert canonical_entities["Alpha"]["description"] == "merged alpha"
    assert canonical_entities["Beta"]["description"] == "merged beta"


def test_description_merge_splits_when_char_budget_is_exceeded():
    """Packed by character budget, not a fixed entity count: groups whose combined
    descriptions exceed the budget must land in separate concurrent calls."""
    from services.description_pack_utils import pack_by_char_budget
    from services.doc_canonicalization_service import (
        _DESCRIPTION_MERGE_CHAR_BUDGET,
        _DocumentDescriptionMerge,
        _MergedMention,
        _merge_mention_descriptions,
    )
    long = "x" * 600
    # Two 600-char descriptions -> 1200 chars/group. Budget 8000 fits 6, so 8 -> [6, 2].
    total = 8
    groups = [{"key": f"e{i}", "names": [f"E{i}", f"E{i}."], "type": "ORG",
               "descriptions": [long, long]} for i in range(total)]
    expected_reduce_sizes = [len(b) for b in pack_by_char_budget(
        groups, lambda g: sum(len(d) for d in g["descriptions"]),
        _DESCRIPTION_MERGE_CHAR_BUDGET)]
    assert expected_reduce_sizes == [6, 2], expected_reduce_sizes
    reduce_sizes = []
    merge_sizes = []

    async def _per_batch(system_prompt, user_payload, response_model, **kwargs):
        import json
        payload = json.loads(user_payload)
        if kwargs.get("step") == "doc_description_reduce":
            reduce_sizes.append(len(payload))
        else:
            merge_sizes.append(len(payload))
        return True, _DocumentDescriptionMerge(entities=[
            _MergedMention(id=item["id"], canonical_name=item["names"][0],
                           description="merged") for item in payload])

    with patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_per_batch):
        resolved = _run(_merge_mention_descriptions(groups))

    # Under-cap multi-desc groups fold in reduce first (same packing as the old merge
    # split), then merge sees one short description each and packs into fewer calls.
    assert reduce_sizes == expected_reduce_sizes
    assert sum(merge_sizes) == total
    assert len(resolved) == total
    assert all(r["description"] == "merged" for r in resolved.values())


def test_description_merge_reduces_one_entitys_descriptions_above_the_per_group_cap():
    """The other axis of the same prompt: an entity mentioned in hundreds of chunks
    used to send all of them in the prompt shared by its whole batch, and an oversized
    response fails every group in that batch back to its fallback. No prompt may carry
    more than the per-group cap for one entity, at any recursion depth. After slicing,
    an extra fold round collapses the partials to exactly one before merge."""
    from services.doc_canonicalization_service import (
        _DESCRIPTIONS_PER_GROUP_MAX,
        _DocumentDescriptionMerge,
        _MergedMention,
        _merge_mention_descriptions,
    )
    total = _DESCRIPTIONS_PER_GROUP_MAX * 3 + 1
    groups = [{"key": "amelia", "names": ["Amelia"], "type": "PERSON",
               "descriptions": [f"fact {i}" for i in range(total)]}]
    per_item_counts = []

    async def _spy(system_prompt, user_payload, response_model, **kwargs):
        import json
        payload = json.loads(user_payload)
        per_item_counts.extend(len(item["descriptions"]) for item in payload)
        return True, _DocumentDescriptionMerge(entities=[
            _MergedMention(id=item["id"], canonical_name="Amelia",
                           description=f"partial {item['id']}") for item in payload])

    with patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_spy):
        resolved = _run(_merge_mention_descriptions(groups))

    assert all(c <= _DESCRIPTIONS_PER_GROUP_MAX for c in per_item_counts), per_item_counts
    # 61 -> four slices (20,20,20,1) -> four partials folded as one 4-item piece ->
    # one description into merge.
    assert per_item_counts == (
        [_DESCRIPTIONS_PER_GROUP_MAX] * 3 + [1, 4, 1]
    ), per_item_counts
    assert resolved["amelia"]["description"] == "partial 0"


def test_description_reduce_fails_closed_to_the_richest_description():
    """A reduce whose LLM call fails must still leave exactly one description: the
    single richest, rather than passing the full list through or dropping the group."""
    from services.doc_canonicalization_service import (
        _DESCRIPTIONS_PER_GROUP_MAX,
        _reduce_pending_to_cap,
    )
    descriptions = [f"d{i}" for i in range(_DESCRIPTIONS_PER_GROUP_MAX + 5)] + ["x" * 500]
    group = {"key": "amelia", "names": ["Amelia"], "type": "PERSON",
             "descriptions": descriptions}

    with patch("services.doc_canonicalization_service.structured_llm_call",
               new=AsyncMock(return_value=(False, None))):
        reduced = _run(_reduce_pending_to_cap([group]))["amelia"]

    assert reduced == "x" * 500


def test_description_reduce_pools_over_cap_entities_into_shared_calls():
    """Several over-cap entities with short descriptions share reduce calls. First
    round pools the slices; a second pooled round folds each entity's partials to 1."""
    from services.doc_canonicalization_service import (
        _DESCRIPTIONS_PER_GROUP_MAX,
        _DocumentDescriptionMerge,
        _MergedMention,
        _reduce_pending_to_cap,
    )
    n_entities = 5
    descs_each = _DESCRIPTIONS_PER_GROUP_MAX + 5  # 25 -> 2 pieces each
    groups = [
        {"key": f"e{i}", "names": [f"E{i}"], "type": "ORG",
         "descriptions": [f"e{i}-d{j}" for j in range(descs_each)]}
        for i in range(n_entities)
    ]
    reduce_calls: list[int] = []

    async def _spy(system_prompt, user_payload, response_model, **kwargs):
        import json
        payload = json.loads(user_payload)
        assert kwargs.get("step") == "doc_description_reduce"
        reduce_calls.append(len(payload))
        return True, _DocumentDescriptionMerge(entities=[
            _MergedMention(id=item["id"], canonical_name=item["names"][0],
                           description=f"partial-{item['id']}") for item in payload])

    with patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_spy):
        result = _run(_reduce_pending_to_cap(groups))

    pieces_per_entity = -(-descs_each // _DESCRIPTIONS_PER_GROUP_MAX)
    total_pieces = n_entities * pieces_per_entity
    # Round 1: 10 slices pooled; round 2: 5 single-piece folds pooled.
    assert reduce_calls == [total_pieces, n_entities], reduce_calls
    for group in groups:
        assert isinstance(result[group["key"]], str)
        assert result[group["key"]]  # exactly one non-empty description


def test_description_reduce_splits_long_pieces_by_char_budget():
    """Pieces whose combined character size exceeds the budget must not share a
    single oversized prompt — each over-budget piece goes out alone. The follow-up
    fold-to-one round still runs after that."""
    from services.description_pack_utils import pack_by_char_budget
    from services.doc_canonicalization_service import (
        _DESCRIPTION_MERGE_CHAR_BUDGET,
        _DESCRIPTIONS_PER_GROUP_MAX,
        _DocumentDescriptionMerge,
        _MergedMention,
        _reduce_pending_to_cap,
    )
    # 21 x 600-char descriptions -> 2 pieces (20*600=12000, 1*600=600). Each 12000
    # piece is over budget alone, so two entities -> 4 alone calls in round 1.
    long = "y" * 600
    groups = [
        {"key": f"e{i}", "names": [f"E{i}"], "type": "ORG",
         "descriptions": [long] * (_DESCRIPTIONS_PER_GROUP_MAX + 1)}
        for i in range(2)
    ]
    pieces = []
    for group in groups:
        descs = group["descriptions"]
        slices = [descs[i:i + _DESCRIPTIONS_PER_GROUP_MAX]
                  for i in range(0, len(descs), _DESCRIPTIONS_PER_GROUP_MAX)]
        for ordinal, chunk in enumerate(slices):
            pieces.append((group["key"], ordinal, group, chunk))
    round1_calls = len(pack_by_char_budget(
        pieces, lambda p: sum(len(d) for d in p[3]), _DESCRIPTION_MERGE_CHAR_BUDGET))
    assert round1_calls == 4, round1_calls

    reduce_calls: list[int] = []

    async def _spy(system_prompt, user_payload, response_model, **kwargs):
        import json
        payload = json.loads(user_payload)
        assert kwargs.get("step") == "doc_description_reduce"
        reduce_calls.append(len(payload))
        return True, _DocumentDescriptionMerge(entities=[
            _MergedMention(id=item["id"], canonical_name=item["names"][0],
                           description=f"partial-{item['id']}") for item in payload])

    with patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_spy):
        result = _run(_reduce_pending_to_cap(groups))

    # Round 1: 4 alone slice calls; round 2: 2 entities' 2-partial lists fold (pooled).
    assert reduce_calls[:4] == [1, 1, 1, 1]
    assert sum(reduce_calls[4:]) == 2  # two fold-to-one pieces
    for group in groups:
        assert isinstance(result[group["key"]], str)


def test_description_reduce_progress_guard_is_per_entity_in_a_pooled_call():
    """A pooled reduce that makes no progress for one entity must fall back for that
    entity only — siblings that did progress keep folding to one."""
    from services.doc_canonicalization_service import (
        _DESCRIPTIONS_PER_GROUP_MAX,
        _DocumentDescriptionMerge,
        _MergedMention,
        _reduce_pending_to_cap,
    )
    rich = "x" * 500
    ok_group = {"key": "ok", "names": ["Ok"], "type": "ORG",
                "descriptions": [f"ok-{i}" for i in range(_DESCRIPTIONS_PER_GROUP_MAX + 1)]}
    bad_group = {"key": "bad", "names": ["Bad"], "type": "ORG",
                 "descriptions": [f"bad-{i}" for i in range(_DESCRIPTIONS_PER_GROUP_MAX)]
                 + [rich]}

    async def _spy(system_prompt, user_payload, response_model, **kwargs):
        import json
        payload = json.loads(user_payload)
        entities = []
        for item in payload:
            name = item["names"][0]
            if name == "Ok":
                entities.append(_MergedMention(
                    id=item["id"], canonical_name="Ok",
                    description=f"ok-partial-{item['id']}"))
            else:
                entities.append(_MergedMention(
                    id=item["id"], canonical_name="Bad", description=""))
        return True, _DocumentDescriptionMerge(entities=entities)

    with patch("services.doc_canonicalization_service.structured_llm_call", side_effect=_spy):
        result = _run(_reduce_pending_to_cap([ok_group, bad_group]))

    # Ok: round1 -> 2 partials, round2 folds them to one. Bad: no progress -> richest.
    assert result["ok"] == "ok-partial-0"
    assert result["bad"] == rich

def test_single_mention_entities_cost_no_description_merge_call():
    """One surface form, one description -- nothing to decide, so it must not
    reach the LLM at all."""
    chunk_results = [
        _result(ExtractedEntity(name="Alpha", type="ORG", description="only one"),
                ExtractedEntity(name="Beta", type="ORG", description="only one too")),
    ]
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.clustering_utils.cluster_by_similarity", return_value=[["Alpha"], ["Beta"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.doc_canonicalization_service.structured_llm_call",
              new=AsyncMock()) as merge_call:
        canonical_entities, _ = _run(canonicalize_document(chunk_results))

    merge_call.assert_not_awaited()
    assert canonical_entities["Alpha"]["description"] == "only one"


def test_description_merge_failure_falls_back_to_the_richest_description():
    """Fail-closed must be degraded, never lossy-by-surprise and never an
    exception: the longest available description survives."""
    chunk_results = [
        _result(ExtractedEntity(name="Globex", type="ORG", description="short"),
                ExtractedEntity(name="Globex", type="ORG",
                                description="a considerably richer description of the company")),
    ]
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _no_description_merge():
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert canonical_entities["Globex"]["description"] == \
        "a considerably richer description of the company"
    assert merge_map["Globex"] == "Globex"


# --- LLM-trusted naming, with a hallucination guard (ticket 9) ---

def test_description_merge_may_fix_a_garbled_surface_form():
    """Two casing variants of one name reach the merge call together; the LLM's
    corrected form is trusted, and every raw form still maps to it."""
    chunk_results = [
        _result(ExtractedEntity(name="GLoBEX", type="ORGANIZATION", description="d1"),
                ExtractedEntity(name="globex", type="ORGANIZATION", description="d2")),
    ]
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _description_merge_returning(
             {"id": 0, "canonical_name": "Globex", "description": "merged"}):
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert list(canonical_entities.keys()) == ["Globex"]
    assert merge_map["GLoBEX"] == "Globex"
    assert merge_map["globex"] == "Globex"


def test_description_merge_rejects_a_canonical_name_for_a_different_entity():
    """Trusting the LLM's naming is not the same as accepting a hallucination:
    a name that no longer normalizes to this group's own form is rejected in
    favour of the deterministic pick."""
    chunk_results = [
        _result(ExtractedEntity(name="Globex", type="ORGANIZATION", description="d1"),
                ExtractedEntity(name="Globex.", type="ORGANIZATION", description="d2")),
    ]
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _description_merge_returning(
             {"id": 0, "canonical_name": "Microsoft Corporation", "description": "merged"}):
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert "Microsoft Corporation" not in canonical_entities
    assert list(canonical_entities.keys()) == ["Globex"]
    assert merge_map["Globex."] == "Globex"


def test_cluster_merge_synthesis_is_trusted_over_the_longest_member_description():
    """At the cluster-merge step the LLM already gets each candidate's merged
    description, so its synthesis stands -- best_description must NOT swap in the
    longest member text, which is the single-chunk blob this change removes."""
    chunk_results = [
        _result(
            ExtractedEntity(name="Globex", type="ORGANIZATION",
                            description="Globex acquired Initech in 2024 for $50M, a landmark deal."),
            ExtractedEntity(name="The Company", type="ORGANIZATION", description="A vendor."),
        ),
    ]

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        from services.clustering_utils import ClusterResolution, MergeGroup
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="Globex", canonical_type="ORGANIZATION",
            merged_description="Globex is a vendor that acquired a company.",
            members=list(cluster),
        )])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _no_description_merge(), \
         patch("services.clustering_utils.cluster_by_similarity",
              return_value=[["Globex", "The Company"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.clustering_utils.resolve_cluster", side_effect=_fake_resolve_cluster):
        canonical_entities, _ = _run(canonicalize_document(chunk_results))

    assert canonical_entities["Globex"]["description"] == \
        "Globex is a vendor that acquired a company."


def test_doc_level_cluster_merge_splits_a_cluster_above_the_prompt_cap():
    """Step 2 used to pass no resolve_fn, so a cluster of ANY size went out as ONE
    prompt — and an oversized one fails closed to "every candidate its own group",
    silently stopping merges for the document's largest entity. No single prompt may
    exceed the cap, at any recursion depth."""
    chunk_results = [_result(*[
        ExtractedEntity(name=f"Acme Division {i}", type="ORGANIZATION", description=f"d{i}")
        for i in range(5)
    ])]
    calls = []

    async def _spy(cluster, registry, system_prompt):
        calls.append(list(cluster))
        from services.clustering_utils import ClusterResolution, MergeGroup
        # Each call collapses its slice to one group, so survivors shrink and the
        # recursion converges instead of tripping the no-progress guard.
        return ClusterResolution(groups=[MergeGroup(
            canonical_name=cluster[0], canonical_type="ORGANIZATION",
            merged_description="merged", members=list(cluster))])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.doc_canonicalization_service.DEFAULT_LLM_BATCH_MAX", 2), \
         _no_description_merge(), \
         patch("services.clustering_utils.cluster_by_similarity",
              side_effect=lambda names, v, t: [list(names)]), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.clustering_utils.resolve_cluster", side_effect=_spy):
        _canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    assert len(calls) > 1, "a 5-member cluster at cap=2 must be split, not sent as one prompt"
    assert all(len(c) <= 2 for c in calls), calls
    # Nothing is lost by the slicing: every raw name still resolves through merge_map,
    # which edge_pool_service uses for every relationship endpoint and claim subject.
    assert set(merge_map) == {f"Acme Division {i}" for i in range(5)}


def test_all_precision_values_never_calls_embedder_or_clusters():
    """A doc with only precision-typed entities must skip embedding/clustering
    entirely -- there's no prose left to cluster."""
    chunk_results = [
        _result(
            ExtractedEntity(name="$82k", type="ENTITY", description="a"),
            ExtractedEntity(name="120 Seats", type="ENTITY", description="b"),
        ),
    ]
    embedder = _fake_embedder()
    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=embedder), \
         _no_description_merge(), \
         patch("services.clustering_utils.cluster_by_similarity") as cluster_mock:
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))
    cluster_mock.assert_not_called()
    embedder.embed_batch.assert_not_called()
    assert set(canonical_entities.keys()) == {"$82k", "120 Seats"}


def test_doc_similarity_threshold_matches_the_experiments_real_wired_value():
    """The old value (0.85) traced to a DEAD config field in the experiment
    (config.py's doc_threshold, defined but never read by any real clustering
    call site) -- the actually-wired value, confirmed by tracing
    core/batch_canonicalization.py's real call sites, is 0.82."""
    assert DOC_SIMILARITY_THRESHOLD == 0.82


# --- Multi-round clustering (experiment-parity, mirrors the reference's
# core/batch_canonicalization.py:_canonicalize_iterative) ---

def test_canonicalize_document_catches_a_multi_hop_merge_across_rounds():
    """Exercised through the real canonicalize_document entry point: round 1
    merges 'Acme Corp' and 'Acme Corporation'; round 2 catches that the
    merged entity should also fold in 'The Vendor' -- a single pass would
    never see this, since the merged entity didn't exist yet when clustering
    first ran. Uses three genuinely distinct names (not case/punctuation
    variants of each other) so none collapse via the cheap exact-normalized-
    name pass before clustering even runs."""
    chunk_results = [
        _result(
            ExtractedEntity(name="Acme Corp", type="ORGANIZATION", description="d1"),
            ExtractedEntity(name="Acme Corporation", type="ORGANIZATION", description="d2"),
            ExtractedEntity(name="The Vendor", type="ORGANIZATION", description="d3"),
        ),
    ]
    calls = {"n": 0}

    def _cluster(representatives, vectors, threshold):
        calls["n"] += 1
        reps = set(representatives)
        if calls["n"] == 1:
            assert reps == {"Acme Corp", "Acme Corporation", "The Vendor"}
            return [["Acme Corp", "Acme Corporation"], ["The Vendor"]]
        assert "The Vendor" in reps and len(reps) == 2
        return [list(representatives)]

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        from services.clustering_utils import ClusterResolution, MergeGroup
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="Acme Corp", canonical_type="ORGANIZATION",
            merged_description="merged", members=list(cluster),
        )])

    with patch("services.doc_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         _no_description_merge(), \
         patch("services.clustering_utils.cluster_by_similarity", side_effect=_cluster), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.clustering_utils.resolve_cluster", side_effect=_fake_resolve_cluster):
        canonical_entities, merge_map = _run(canonicalize_document(chunk_results))

    final_name = merge_map["Acme Corp"]
    assert merge_map["Acme Corporation"] == final_name
    assert merge_map["The Vendor"] == final_name
    assert len(canonical_entities) == 1
    assert calls["n"] == 2
