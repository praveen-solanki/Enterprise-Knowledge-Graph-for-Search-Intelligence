"""Unit tests for services/clustering_utils.py's precision-value detection
(experiment-parity review C2) and complete-linkage anti-chaining (review C3).
Precision values (money, counts, dates, versions, IPs, standard clause IDs)
carry no synonymy and must never merge on embedding similarity alone -- only
on an exact normalized-name duplicate."""
import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import inspect

from services.clustering_utils import (
    ClusterResolution,
    MergeGroup,
    best_description,
    cluster_by_similarity,
    cluster_to_fixpoint,
    is_precision,
    resolve_cluster,
    shortest_clean,
    tight_partition,
)


def test_money_is_precision():
    assert is_precision("$82k", "ENTITY") is True
    assert is_precision("€1,200", "ENTITY") is True


def test_seat_count_is_precision():
    assert is_precision("120 Seats", "ENTITY") is True
    assert is_precision("80k users", "ENTITY") is True


def test_percentage_is_precision():
    assert is_precision("80%", "ENTITY") is True


def test_iso_date_is_precision():
    assert is_precision("2026-07-16", "ENTITY") is True


def test_version_is_precision():
    assert is_precision("v13.27", "ENTITY") is True
    assert is_precision("26.01-A.2.0", "ENTITY") is True


def test_ipv4_is_precision():
    assert is_precision("192.168.1.1", "ENTITY") is True


def test_type_hint_marks_precision_regardless_of_name_shape():
    """A METRIC/DATE/VERSION/... typed entity is precision even if its name
    doesn't match the regex shape (e.g. a spelled-out date)."""
    assert is_precision("Q1 2024", "METRIC") is True
    assert is_precision("next Tuesday", "DATE") is True


def test_ordinary_prose_name_is_not_precision():
    assert is_precision("Globex", "ORGANIZATION") is False
    assert is_precision("Initech", "ORGANIZATION") is False


def test_empty_name_is_not_precision():
    assert is_precision("", "ENTITY") is False
    assert is_precision(None, "ENTITY") is False


# --- tight_partition / anti-chaining (experiment-parity review C3) ---

def _unit_vector_at_degrees(deg: float) -> list:
    rad = math.radians(deg)
    return [math.cos(rad), math.sin(rad)]


def test_tight_partition_splits_a_chain_the_bridge_does_not_belong_to_both_ends():
    """A~B and B~C both clear the threshold, but A~C does not — single-linkage
    would chain all three into one cluster via bridge B, but complete linkage
    must never hand A and C to the LLM together as if they were similar."""
    threshold = 0.9
    embeddings = {
        "A": _unit_vector_at_degrees(0),
        "B": _unit_vector_at_degrees(15),   # cos(15) ~= 0.966 with A -- clears threshold
        "C": _unit_vector_at_degrees(30),   # cos(15) ~= 0.966 with B, cos(30) ~= 0.866 with A -- fails
    }
    fragments = tight_partition(["A", "B", "C"], embeddings, threshold)
    fragment_sets = [set(f) for f in fragments]
    # A and C must never end up in the same fragment.
    for frag in fragment_sets:
        assert not ({"A", "C"} <= frag)
    # Nothing was dropped.
    assert set().union(*fragment_sets) == {"A", "B", "C"}


def test_tight_partition_keeps_a_genuine_clique_as_one_fragment():
    """When every pair really is mutually similar, there's nothing to break —
    the whole cluster returns as a single fragment."""
    threshold = 0.9
    embeddings = {
        "A": _unit_vector_at_degrees(0),
        "B": _unit_vector_at_degrees(2),
        "C": _unit_vector_at_degrees(4),
    }
    fragments = tight_partition(["A", "B", "C"], embeddings, threshold)
    assert fragments == [["A", "B", "C"]] or {tuple(sorted(f)) for f in fragments} == {("A", "B", "C")}


def test_tight_partition_single_member_returns_as_is():
    assert tight_partition(["A"], {"A": [1.0, 0.0]}, 0.9) == [["A"]]


def test_tight_partition_empty_returns_empty():
    assert tight_partition([], {}, 0.9) == [[]]


def test_tight_partition_passes_through_unpartitioned_above_the_size_cap(monkeypatch):
    """A pathologically huge cluster is passed through rather than building an
    O(n^2) similarity matrix -- confirmed via a small cap so the test itself
    stays fast."""
    import services.clustering_utils as clustering_utils
    monkeypatch.setattr(clustering_utils, "_TIGHT_PARTITION_MAX", 2)
    members = ["A", "B", "C"]
    embeddings = {m: _unit_vector_at_degrees(i) for i, m in enumerate(members)}
    result = tight_partition(members, embeddings, 0.9)
    assert result == [members]


# --- shortest_clean / best_description (experiment-parity review #14) ---

def test_shortest_clean_prefers_normal_casing_over_a_garble():
    """A lexicographic tie-break would let 'GLoBEX' beat 'Globex' (uppercase
    'L' sorts before lowercase 'l') -- must not happen."""
    assert shortest_clean(["GLoBEX", "Globex"]) == "Globex"


def test_shortest_clean_prefers_shorter_among_equally_clean_names():
    assert shortest_clean(["Globex Inc", "Globex"]) == "Globex"


def test_shortest_clean_preserves_a_short_acronym_over_titlecase():
    assert shortest_clean(["AWS", "Aws"]) == "AWS"


def test_shortest_clean_demotes_an_all_caps_multi_word_phrase():
    assert shortest_clean(["ENTERPRISE SEARCH", "Enterprise Search"]) == "Enterprise Search"


def test_shortest_clean_most_frequent_wins_when_counts_given():
    assert shortest_clean(["Globex", "GLoBEX"], counts={"GLoBEX": 10, "Globex": 1}) == "GLoBEX"


def test_shortest_clean_empty_list_returns_empty_string():
    assert shortest_clean([]) == ""


def test_best_description_falls_back_to_longest_when_no_merged_text():
    assert best_description(None, ["short", "a much longer description"]) == "a much longer description"


def test_best_description_keeps_synthesis_when_it_preserves_the_facts():
    merged = "Globex acquired Initech in 2024 for $50M."
    members = ["Globex acquired Initech.", "The 2024 deal was valued at $50M."]
    assert best_description(merged, members) == merged


def test_best_description_falls_back_when_synthesis_drops_salient_facts():
    """A one-sentence synthesis that drops the concrete number/date must lose
    to the richer member description, not silently ship the lossy rewrite."""
    merged = "Globex acquired a company."
    members = ["Globex acquired Initech in 2024 for $50M, a landmark deal."]
    assert best_description(merged, members) == members[0]


def test_best_description_no_member_descriptions_returns_merged_or_none():
    assert best_description("some text", []) == "some text"
    assert best_description(None, []) is None


# --- resolve_cluster call metrics (experiment-parity review #21) ---

def test_resolve_cluster_records_call_metrics():
    """Each real API attempt must be counted, so a run's final summary log
    reflects actual LLM call volume."""
    from services import run_metrics
    run_metrics.reset()
    registry = {"A": {"type": "ORG", "description": ""}}
    result = ClusterResolution(groups=[MergeGroup(
        canonical_name="A", canonical_type="ORG", merged_description="", members=["A"])])

    with patch("services.clustering_utils.llm_client.complete",
              new=AsyncMock(return_value=result)):
        asyncio.run(resolve_cluster(["A"], registry, "system"))
    assert run_metrics.snapshot().get("merge_calls") == 1


def test_resolve_cluster_uses_parse_structured_for_both_backends():
    """All cluster-merge structured calls go through the shared parse_structured helper."""
    registry = {"A": {"type": "ORG", "description": ""}}
    result = ClusterResolution(groups=[MergeGroup(
        canonical_name="A", canonical_type="ORG", merged_description="", members=["A"])])

    with patch("services.clustering_utils.llm_client.complete",
              new=AsyncMock(return_value=result)) as parse_mock:
        outcome = asyncio.run(resolve_cluster(["A"], registry, "system"))
    assert outcome.groups[0].canonical_name == "A"
    parse_mock.assert_called_once()


def test_resolve_cluster_ministral_failure_retries_then_fails_closed():
    """Structured-call failures go through call_with_retry's transient retry path."""
    registry = {"A": {"type": "ORG", "description": ""}, "B": {"type": "ORG", "description": ""}}

    with patch("services.clustering_utils.llm_client.complete",
              new=AsyncMock(side_effect=ValueError("truncated"))), \
         patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(resolve_cluster(["A", "B"], registry, "system"))
    assert len(result.groups) == 2  # fails closed: every member its own group
    record_call.assert_called_once_with("cluster_merge_confirm", "members=['A', 'B']",
                                        "transient", "truncated", attempts=5)


# --- Full-traceability: an exhausted retry must leave a durable record, not just a log line ---

def _fake_throttle() -> MagicMock:
    throttle = MagicMock()
    throttle.slot.return_value.__aenter__ = AsyncMock(return_value=None)
    throttle.slot.return_value.__aexit__ = AsyncMock(return_value=None)
    throttle.record_success = AsyncMock()
    throttle.record_rate_limit = AsyncMock()
    return throttle


def test_resolve_cluster_logs_call_failure_when_rate_limited_retries_exhaust():
    from openai import RateLimitError
    registry = {"A": {"type": "ORG", "description": ""}, "B": {"type": "ORG", "description": ""}}
    rate_limit_exc = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)

    with patch("services.clustering_utils.llm_client.complete",
              new=AsyncMock(side_effect=rate_limit_exc)), \
         patch("services.llm_client.record_call_failure") as record_call, \
         patch("services.llm_client.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(resolve_cluster(["A", "B"], registry, "system"))
    # Fails closed: every member becomes its own group, never a blind merge.
    assert len(result.groups) == 2
    record_call.assert_called_once()
    args, kwargs = record_call.call_args
    assert args[0] == "cluster_merge_confirm"
    assert kwargs["attempts"] == 5  # _MAX_RETRIES


def test_resolve_cluster_logs_call_failure_on_a_non_retryable_error():
    from botocore.exceptions import ClientError

    registry = {"A": {"type": "ORG", "description": ""}}
    permanent_exc = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {}}, "PutObject")

    with patch("services.clustering_utils.llm_client.complete",
              new=AsyncMock(side_effect=permanent_exc)), \
         patch("services.llm_client.record_call_failure") as record_call:
        result = asyncio.run(resolve_cluster(["A"], registry, "system"))
    assert len(result.groups) == 1
    record_call.assert_called_once_with("cluster_merge_confirm", "members=['A']",
                                        "permanent", str(permanent_exc), attempts=1)


# --- ANN tunables reconciled with the experiment (review #30) ---

def test_cluster_by_similarity_default_k_matches_the_experiment():
    """The experiment's build_ann_clusters defaults to k_neighbors=32 at every
    call site (never overridden) -- this repo's old default of 10 undersampled
    candidates relative to the proven pipeline."""
    assert inspect.signature(cluster_by_similarity).parameters["k"].default == 32


# --- cluster_to_fixpoint (multi-round clustering, mirrors _canonicalize_iterative) ---

def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.0] for _ in texts]  # content irrelevant -- clustering is mocked
    return embedder


def _resolution(*groups: MergeGroup) -> ClusterResolution:
    return ClusterResolution(groups=list(groups))


def _group(canonical_name: str, members: list[str], canonical_type: str = "ORG",
          merged_description: str = "merged") -> MergeGroup:
    return MergeGroup(canonical_name=canonical_name, canonical_type=canonical_type,
                      merged_description=merged_description, members=members)


def _no_partition(cluster, embeddings, threshold):
    """Test double for tight_partition that never splits -- these tests are
    about cluster_to_fixpoint's OWN round/fixpoint logic, not anti-chaining
    (already covered by the tight_partition tests above)."""
    return [cluster]


def test_cluster_to_fixpoint_no_multi_clusters_returns_identity_map():
    registry = {"A": {"type": "ORG", "description": "d1"}, "B": {"type": "ORG", "description": "d2"}}
    with patch("services.clustering_utils.cluster_by_similarity", return_value=[["A"], ["B"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B"], registry, lambda n, r: n, _embedder(), 0.82, 0.86, "system",
        ))
    assert merge_map == {"A": "A", "B": "B"}
    assert set(final.keys()) == {"A", "B"}


def test_cluster_to_fixpoint_single_round_merge_then_reaches_fixpoint():
    """A+B merge in round 1; round 2 finds only singletons remaining and
    stops -- resolve_fn must be called exactly once, not once per
    max_iterations regardless of whether there's anything left to merge."""
    registry = {"A": {"type": "ORG", "description": "d1"}, "B": {"type": "ORG", "description": "d2"},
               "C": {"type": "ORG", "description": "d3"}}
    calls = {"n": 0}

    def fake_cluster_by_similarity(names, vectors, threshold):
        calls["n"] += 1
        if calls["n"] == 1:
            return [["A", "B"], ["C"]]
        return [[n] for n in names]  # round 2: nothing clusters -- fixpoint

    resolve_fn = AsyncMock(return_value=_resolution(_group("A", ["A", "B"])))
    with patch("services.clustering_utils.cluster_by_similarity", side_effect=fake_cluster_by_similarity), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B", "C"], registry, lambda n, r: n, _embedder(), 0.82, 0.86, "system", resolve_fn=resolve_fn,
        ))
    assert merge_map["A"] == merge_map["B"]
    assert merge_map["C"] == "C"
    resolve_fn.assert_called_once()


def test_cluster_to_fixpoint_catches_a_multi_hop_merge_a_single_pass_would_miss():
    """The whole point of this feature: round 1 merges A+B; round 2 re-clusters
    the MERGED entity against C and merges them too -- a single pass could
    never see this, since the merged AB entity didn't exist yet when
    clustering first ran."""
    registry = {"A": {"type": "ORG", "description": "d1"}, "B": {"type": "ORG", "description": "d2"},
               "C": {"type": "ORG", "description": "d3"}}
    calls = {"n": 0}

    def fake_cluster_by_similarity(names, vectors, threshold):
        calls["n"] += 1
        if calls["n"] == 1:
            return [["A", "B"], ["C"]]
        assert "C" in names and len(names) == 2  # round 2: the merged AB entity + C
        return [list(names)]

    # canonical_name is deliberately whatever the "LLM" suggests ("ABC") -- the
    # real code must never trust that verbatim (matches shortest_clean's own
    # documented override), so the assertions below check by SHARED identity,
    # not by a hardcoded expected name.
    def fake_resolve(cluster, reg, system):
        if "C" in cluster and len(cluster) > 1:
            return _resolution(_group("ABC", cluster))
        return _resolution(_group(shortest_clean(cluster), cluster))

    resolve_fn = AsyncMock(side_effect=fake_resolve)
    with patch("services.clustering_utils.cluster_by_similarity", side_effect=fake_cluster_by_similarity), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B", "C"], registry, lambda n, r: n, _embedder(), 0.82, 0.86, "system", resolve_fn=resolve_fn,
        ))
    assert merge_map["A"] == merge_map["B"] == merge_map["C"]
    assert len(final) == 1
    assert resolve_fn.call_count == 2


def test_cluster_to_fixpoint_stops_at_max_iterations_cap_even_if_more_would_merge():
    """With the cap set to 1 round, a merge that WOULD have happened in a
    hypothetical round 2 must never run -- the cap is a hard ceiling, not a
    hint."""
    registry = {n: {"type": "ORG", "description": ""} for n in ["A", "B", "C", "D"]}

    def fake_cluster_by_similarity(names, vectors, threshold):
        names = list(names)
        return [names[:2], names[2:]] if len(names) >= 4 else [names]

    resolve_fn = AsyncMock(side_effect=lambda cluster, reg, system: _resolution(
        _group(shortest_clean(cluster), cluster)))
    with patch("services.clustering_utils.cluster_by_similarity", side_effect=fake_cluster_by_similarity), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B", "C", "D"], registry, lambda n, r: n, _embedder(), 0.82, 0.86, "system",
            resolve_fn=resolve_fn, max_iterations=1,
        ))
    assert len(final) == 2  # 4 -> 2 after exactly one capped round, not all the way to 1
    assert resolve_fn.call_count == 2  # both of round 1's multi-clusters get resolved...
    # ...but a hypothetical round 2 (which would merge those 2 remaining entities
    # into 1) must never run -- confirmed by len(final) == 2 above, not 1.


def test_cluster_to_fixpoint_folds_descriptions_via_best_description():
    """The opted-OUT path (trust_llm_naming=False). No caller uses it today —
    both stages trust the model — but it stays covered so the heuristic
    fallback can't rot silently while it's still the blank-field fallback."""
    registry = {"A": {"type": "ORG", "description": "It merged."},
               "B": {"type": "ORG", "description": "A much longer and more detailed description of 2024."}}
    resolve_fn = AsyncMock(return_value=_resolution(
        _group("A", ["A", "B"], merged_description="It merged.")))
    with patch("services.clustering_utils.cluster_by_similarity", return_value=[["A", "B"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        _, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B"], registry, lambda n, r: n, _embedder(), 0.82, 0.86, "system", resolve_fn=resolve_fn,
            trust_llm_naming=False,
        ))
    # best_description must prefer B's richer text over the LLM's lossy "Short." synthesis.
    assert "2024" in list(final.values())[0]["description"]


def test_cluster_to_fixpoint_embed_text_fn_sees_the_current_rounds_merged_description():
    """Round 2's embedding text must reflect round 1's MERGED description, not
    the original -- otherwise a multi-hop merge could be judged on stale text."""
    registry = {"A": {"type": "ORG", "description": "d1"}, "B": {"type": "ORG", "description": "d2"},
               "C": {"type": "ORG", "description": "d3"}}
    seen_round2_descriptions = []

    def embed_text_fn(name, reg):
        if len(reg) == 2:  # round 2's view: the merged entity + C
            seen_round2_descriptions.append(reg[name].get("description"))
        return name

    calls = {"n": 0}

    def fake_cluster_by_similarity(names, vectors, threshold):
        calls["n"] += 1
        return [["A", "B"], ["C"]] if calls["n"] == 1 else [[n] for n in names]

    resolve_fn = AsyncMock(return_value=_resolution(
        _group("A", ["A", "B"], merged_description="round-1 merged text")))
    with patch("services.clustering_utils.cluster_by_similarity", side_effect=fake_cluster_by_similarity), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        asyncio.run(cluster_to_fixpoint(
            ["A", "B", "C"], registry, embed_text_fn, _embedder(), 0.82, 0.86, "system", resolve_fn=resolve_fn,
        ))
    assert "round-1 merged text" in seen_round2_descriptions


# --- trust_llm_naming (now the default, for both stages; passed explicitly
# here so these stay readable as the tests OF that behaviour) ---

def test_cluster_to_fixpoint_trust_llm_naming_keeps_the_llm_name_and_description():
    """Same inputs as the two heuristic tests above, opted in: the LLM's
    canonical_name and merged_description are used verbatim instead of being
    replaced by shortest_clean / the longest member description."""
    registry = {"Globex Incorporated": {"type": "ORG", "description": "It merged."},
               "Globex Corp": {"type": "ORG", "description": "A much longer, richer description of 2024."}}
    resolve_fn = AsyncMock(return_value=_resolution(_group(
        "Globex", ["Globex Incorporated", "Globex Corp"], merged_description="Globex merged in 2024.")))
    with patch("services.clustering_utils.cluster_by_similarity",
              return_value=[["Globex Incorporated", "Globex Corp"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["Globex Incorporated", "Globex Corp"], registry, lambda n, r: n, _embedder(),
            0.82, 0.86, "system", resolve_fn=resolve_fn, trust_llm_naming=True,
        ))
    assert set(final.keys()) == {"Globex"}  # not shortest_clean's "Globex Corp"
    assert final["Globex"]["description"] == "Globex merged in 2024."
    assert merge_map["Globex Incorporated"] == merge_map["Globex Corp"] == "Globex"


def test_cluster_to_fixpoint_trust_llm_naming_falls_back_on_a_blank_name_or_description():
    """Trusting the model is not the same as accepting an empty field."""
    registry = {"Alpha": {"type": "ORG", "description": "the richer description"},
               "Alpha Inc": {"type": "ORG", "description": "short"}}
    resolve_fn = AsyncMock(return_value=_resolution(
        _group("   ", ["Alpha", "Alpha Inc"], merged_description="")))
    with patch("services.clustering_utils.cluster_by_similarity", return_value=[["Alpha", "Alpha Inc"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        _, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["Alpha", "Alpha Inc"], registry, lambda n, r: n, _embedder(),
            0.82, 0.86, "system", resolve_fn=resolve_fn, trust_llm_naming=True,
        ))
    assert set(final.keys()) == {"Alpha"}  # shortest_clean's pick
    assert final["Alpha"]["description"] == "the richer description"


def test_cluster_to_fixpoint_untouched_singleton_colliding_with_an_llm_name_is_folded_not_dropped():
    """Under trust_llm_naming the canonical name need not be one of the group's own
    members, so it can collide with an untouched singleton's name. That singleton
    must neither overwrite the merged group nor be silently skipped."""
    registry = {n: {"type": "ORG", "description": "d"} for n in ("A", "B", "C")}
    resolve_fn = AsyncMock(return_value=_resolution(
        _group("C", ["A", "B"], merged_description="merged")))  # names the group after singleton C
    with patch("services.clustering_utils.cluster_by_similarity", return_value=[["A", "B"], ["C"]]), \
         patch("services.clustering_utils.tight_partition", side_effect=_no_partition):
        merge_map, final, _emb = asyncio.run(cluster_to_fixpoint(
            ["A", "B", "C"], registry, lambda n, r: n, _embedder(),
            0.82, 0.86, "system", resolve_fn=resolve_fn, trust_llm_naming=True,
        ))
    assert set(final.keys()) == {"C"}
    assert sorted(final["C"]["members"]) == ["A", "B", "C"]
    assert merge_map == {"A": "C", "B": "C", "C": "C"}  # no name lost


# ---------------------------------------------------------------------------
# resolve_cluster_batched — recursion guards (depth cap + shrink floor)
# ---------------------------------------------------------------------------

def test_resolve_cluster_batched_stops_at_max_depth():
    """8 names at batch_max=2 halve each level (4 -> 2 -> 1 survivors), so an
    uncapped run would recurse twice. max_depth=1 must stop after the slice
    calls of depth 0: exactly 4 resolve calls, survivors kept as-is."""
    from services.clustering_utils import resolve_cluster_batched

    names = [f"n{i}" for i in range(8)]
    registry = {n: {"type": "ORG", "description": ""} for n in names}
    calls = []

    async def _halve(cluster, reg, system_prompt):
        calls.append(list(cluster))
        return ClusterResolution(groups=[MergeGroup(
            canonical_name=cluster[0], canonical_type="ORG",
            merged_description="d", members=list(cluster))])

    result = asyncio.run(resolve_cluster_batched(
        names, registry, "system", resolve_fn=_halve, batch_max=2, max_depth=1))

    assert len(calls) == 4  # depth-0 slices only, no second level
    assert len(result.groups) == 4
    assert sorted(m for g in result.groups for m in g.members) == sorted(names)


def test_resolve_cluster_batched_stops_below_the_shrink_floor():
    """20 names at batch_max=10, where the two slices merge only ONE pair
    between them: 19 survivors > 90% of 20, so recursing would drip one merge
    per level. The shrink floor must keep the slice results after the 2 calls."""
    from services.clustering_utils import resolve_cluster_batched

    names = [f"n{i}" for i in range(20)]
    registry = {n: {"type": "ORG", "description": ""} for n in names}
    calls = []

    async def _merge_one_pair_in_first_slice_only(cluster, reg, system_prompt):
        calls.append(list(cluster))
        if len(calls) == 1:  # first slice: fold its first two members together
            groups = [MergeGroup(canonical_name=cluster[0], canonical_type="ORG",
                                 merged_description="d", members=[cluster[0], cluster[1]])]
            groups += [MergeGroup(canonical_name=n, canonical_type="ORG",
                                  merged_description="", members=[n]) for n in cluster[2:]]
        else:  # second slice: nothing merges
            groups = [MergeGroup(canonical_name=n, canonical_type="ORG",
                                 merged_description="", members=[n]) for n in cluster]
        return ClusterResolution(groups=groups)

    result = asyncio.run(resolve_cluster_batched(
        names, registry, "system",
        resolve_fn=_merge_one_pair_in_first_slice_only, batch_max=10))

    assert len(calls) == 2  # the two depth-0 slices, then the floor stops it
    assert len(result.groups) == 19
    assert sorted(m for g in result.groups for m in g.members) == sorted(names)
