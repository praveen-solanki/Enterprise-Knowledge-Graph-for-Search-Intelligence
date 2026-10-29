"""Unit tests for services/graph_canonicalization_service.py (code review
must-fix #8 — entity description/alias ACL leak via union).

canonicalize_corpus itself has no ACL concept (that's core/pipeline.py's job,
tagging each entry with its own doc's acl right before writing) — these tests
just prove the per-source SHAPE survives corpus canonicalization: every
contributing document's own description/alias is kept as its own {text/name,
doc_id} entry, never blended into one string/list, regardless of whether the
entity ends up resolved via the cheap exact-name path or the LLM cluster path.
"""
import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest

from services.clustering_utils import ClusterResolution, MergeGroup
from services.graph_canonicalization_service import _resolve_cluster_batched, canonicalize_corpus, resolve_entity_ids


def _unit_vector_at_degrees(deg: float) -> list:
    rad = math.radians(deg)
    return [math.cos(rad), math.sin(rad)]


def _fake_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    return embedder


def _run(coro):
    return asyncio.run(coro)


def test_two_docs_with_same_exact_name_keep_separate_descriptions_and_aliases():
    """The cheap exact-name-collision path (len(names) < 2 after registry
    build) — two documents both canonicalize to the literal name "Globex"."""
    canonical_by_doc = {
        "doc-1": ({"Globex": {"type": "ORGANIZATION", "description": "Public blurb"}},
                  {"Globex Inc.": "Globex"}),
        "doc-2": ({"Globex": {"type": "ORGANIZATION", "description": "Private detail"}},
                  {"globex corp": "Globex"}),
    }
    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))

    entity = final_entities["Globex"]
    assert entity["descriptions"] == [
        {"text": "Public blurb", "doc_id": "doc-1"},
        {"text": "Private detail", "doc_id": "doc-2"},
    ]
    alias_pairs = {(a["name"], a["doc_id"]) for a in entity["aliases"]}
    assert alias_pairs == {("Globex Inc.", "doc-1"), ("globex corp", "doc-2")}
    # Neither description was dropped or merged into a single string.
    assert len(entity["descriptions"]) == 2


def test_llm_merged_cluster_pools_every_members_own_descriptions_not_the_llm_blend():
    """The multi-cluster (LLM-resolved) path — two DIFFERENT raw names get
    merged by the LLM into one canonical entity. group.merged_description
    (the LLM's blended text) must never end up as a persisted field; each
    member's own per-doc description must survive independently instead."""
    canonical_by_doc = {
        "doc-1": ({"Globex Inc.": {"type": "ORGANIZATION", "description": "From doc 1"}},
                  {"Globex Inc.": "Globex Inc."}),
        "doc-2": ({"Globex Corporation": {"type": "ORGANIZATION", "description": "From doc 2"}},
                  {"Globex Corporation": "Globex Corporation"}),
    }

    from services.clustering_utils import ClusterResolution, MergeGroup

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="Globex Inc.", canonical_type="ORGANIZATION",
            merged_description="An LLM-blended description that must not be persisted",
            members=list(cluster),
        )])

    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.clustering_utils.cluster_by_similarity",
              return_value=[["Globex Inc.", "Globex Corporation"]]), \
         patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))

    entity = final_entities["Globex Inc."]
    texts_by_doc = {d["doc_id"]: d["text"] for d in entity["descriptions"]}
    assert texts_by_doc == {"doc-1": "From doc 1", "doc-2": "From doc 2"}
    assert "must not be persisted" not in str(entity["descriptions"])
    # The blend IS persisted now — as the entity's one plain description (decision
    # 2026-08-05), NEVER inside the per-doc ACL-taggable entries asserted above.
    assert entity["description"] == "An LLM-blended description that must not be persisted"


def test_single_entity_no_description_gets_empty_descriptions_list():
    canonical_by_doc = {
        "doc-1": ({"Globex": {"type": "ORGANIZATION", "description": None}}, {"Globex": "Globex"}),
    }
    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))
    assert final_entities["Globex"]["descriptions"] == []


def test_persisted_descriptions_are_capped_for_a_name_mentioned_in_too_many_docs():
    """Guards the single entity Mongo document against the 16MB BSON limit: a
    name mentioned in an extreme number of documents must still persist, just
    with its per-doc "descriptions" list capped rather than growing unbounded."""
    canonical_by_doc = {
        f"doc-{i}": ({"Globex": {"type": "ORGANIZATION", "description": f"blurb {i}"}},
                    {"Globex": "Globex"})
        for i in range(5)
    }
    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.graph_canonicalization_service._MAX_PERSISTED_DESCRIPTIONS", 2):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))
    assert len(final_entities["Globex"]["descriptions"]) == 2


def test_defensive_completion_reuses_precomputed_embedding_not_empty_list():
    """Code review must-fix #12: live schema diff found 0/2785 entities had an
    embedding. Root cause — a cluster-merge LLM call whose response doesn't
    cover every input name (clustering_utils.py's non-retryable-error /
    empty-parse fallbacks can return groups=[]) used to fall through to this
    module's defensive-completion pass, which hard-coded embedding=[]. Every
    representative is embedded up front before any LLM call, so this path must
    recover that real vector instead of shipping an unsearchable entity."""
    canonical_by_doc = {
        "doc-1": ({"Alpha Corp": {"type": "ORGANIZATION", "description": "d1"}},
                  {"Alpha Corp": "Alpha Corp"}),
        "doc-2": ({"Beta Corp": {"type": "ORGANIZATION", "description": "d2"}},
                  {"Beta Corp": "Beta Corp"}),
    }

    from services.clustering_utils import ClusterResolution

    async def _fake_resolve_cluster_returns_nothing(cluster, registry, system_prompt):
        # Simulates resolve_cluster's fail-closed/empty-parse fallback: the
        # cluster's names never make it into any group.
        return ClusterResolution(groups=[])

    embedder = _fake_embedder()
    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=embedder), \
         patch("services.clustering_utils.cluster_by_similarity",
              return_value=[["Alpha Corp", "Beta Corp"]]), \
         patch("services.graph_canonicalization_service.resolve_cluster",
              side_effect=_fake_resolve_cluster_returns_nothing):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))

    assert final_entities["Alpha Corp"]["embedding"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
    assert final_entities["Beta Corp"]["embedding"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
    # Recovered from the pre-computed embedding_by_rep map, not a fresh embed call
    # beyond the two upfront per-representative calls (name+description batch).
    assert embedder.embed_batch.call_count == 1


# --- Precision holdout (experiment-parity review C2) ---

def test_precision_values_never_reach_clustering_or_the_llm():
    """Two distinct money values that would look near-identical under a name
    embedding must never be sent to cluster_by_similarity/resolve_cluster at
    all -- confirmed by making both mocks raise if called with them, and by
    asserting the two amounts end up as separate final entities regardless."""
    canonical_by_doc = {
        "doc-1": ({"$82k": {"type": "ENTITY", "description": "ACV"}}, {"$82k": "$82k"}),
        "doc-2": ({"$1.2k": {"type": "ENTITY", "description": "ACV"}}, {"$1.2k": "$1.2k"}),
        "doc-3": ({"Globex": {"type": "ORGANIZATION", "description": "d"}}, {"Globex": "Globex"}),
        "doc-4": ({"Globex Inc.": {"type": "ORGANIZATION", "description": "d"}},
                  {"Globex Inc.": "Globex Inc."}),
    }

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

    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()), \
         patch("services.clustering_utils.cluster_by_similarity",
              side_effect=_cluster_must_not_see_precision_values), \
         patch("services.graph_canonicalization_service.resolve_cluster",
              side_effect=_fake_resolve_cluster):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))

    # The two distinct money values survive as their own separate entities.
    assert "$82k" in final_entities
    assert "$1.2k" in final_entities
    assert final_entities["$82k"] is not final_entities["$1.2k"]


def test_anti_chaining_splits_a_real_chain_before_the_llm_ever_sees_it():
    """Alpha~Bravo and Bravo~Charlie both clear GRAPH_SIMILARITY_THRESHOLD, so
    cluster_by_similarity's single-linkage union-find chains all three into
    one cluster via bridge Bravo. But Alpha~Charlie does NOT clear the
    tighter anti-chaining threshold -- they must never be sent to the LLM as
    one blob, and must never end up merged into the same final entity."""
    canonical_by_doc = {
        "doc-1": ({"Alpha": {"type": "ORGANIZATION", "description": None}}, {"Alpha": "Alpha"}),
        "doc-2": ({"Bravo": {"type": "ORGANIZATION", "description": None}}, {"Bravo": "Bravo"}),
        "doc-3": ({"Charlie": {"type": "ORGANIZATION", "description": None}}, {"Charlie": "Charlie"}),
    }
    vectors_by_name = {
        "Alpha": _unit_vector_at_degrees(0),
        "Bravo": _unit_vector_at_degrees(20),    # cos(20) ~= 0.94 with Alpha -- clears both thresholds
        "Charlie": _unit_vector_at_degrees(40),  # cos(20) ~= 0.94 with Bravo, cos(40) ~= 0.77 with Alpha
    }
    embedder = MagicMock()

    def _embed(texts):
        # A later round embeds a MERGED entity's text as "<name>: <description>"
        # (see clustering_utils.cluster_to_fixpoint) -- look up by the name part
        # alone so this stays valid across every round, not just round 1.
        return [vectors_by_name[t.split(":")[0].strip()] for t in texts]

    embedder.embed_batch.side_effect = _embed

    seen_clusters = []

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        seen_clusters.append(set(cluster))
        from services.clustering_utils import ClusterResolution, MergeGroup
        return ClusterResolution(groups=[MergeGroup(
            canonical_name=sorted(cluster)[0], canonical_type="ORGANIZATION",
            merged_description="merged", members=list(cluster),
        )])

    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=embedder), \
         patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        final_entities, final_merge_map_by_doc = _run(canonicalize_corpus(canonical_by_doc))

    # Alpha and Charlie must never appear together in any single group sent to the LLM.
    for cluster in seen_clusters:
        assert not ({"Alpha", "Charlie"} <= cluster)
    # And they must not end up merged into the same final entity.
    alpha_final = final_merge_map_by_doc["doc-1"]["Alpha"]
    charlie_final = final_merge_map_by_doc["doc-3"]["Charlie"]
    assert alpha_final != charlie_final
    assert alpha_final in final_entities and charlie_final in final_entities


# --- Multi-round clustering (experiment-parity, mirrors the reference's
# core/batch_canonicalization.py:_canonicalize_iterative) ---

def test_canonicalize_corpus_catches_a_multi_hop_merge_across_rounds():
    """The actual point of multi-round clustering, exercised through the real
    canonicalize_corpus entry point: round 1 merges 'Globex Inc.' and
    'Globex Corporation'; the merged entity's folded description then lets
    it also merge with 'The Company' in round 2 -- a single pass would never
    see this, since the merged entity didn't exist yet when clustering first
    ran."""
    canonical_by_doc = {
        "doc-1": ({"Globex Inc.": {"type": "ORGANIZATION", "description": "Sells software."}},
                  {"Globex Inc.": "Globex Inc."}),
        "doc-2": ({"Globex Corporation": {"type": "ORGANIZATION", "description": "A tech company."}},
                  {"Globex Corporation": "Globex Corporation"}),
        "doc-3": ({"The Company": {"type": "ORGANIZATION", "description": "Referenced as 'the company'."}},
                  {"The Company": "The Company"}),
    }
    embedder = _fake_embedder()  # constant vector -- every representative looks identical

    calls = {"n": 0}

    def _cluster(representatives, vectors, threshold):
        calls["n"] += 1
        reps = set(representatives)
        if calls["n"] == 1:
            # Round 1: only the two Globex-named entities are similar enough to
            # cluster -- "The Company" doesn't look similar by NAME alone yet.
            assert reps == {"Globex Inc.", "Globex Corporation", "The Company"}
            return [["Globex Inc.", "Globex Corporation"], ["The Company"]]
        # Round 2: the merged entity (now carrying a folded description that
        # mentions "the company") and "The Company" cluster together.
        assert "The Company" in reps and len(reps) == 2
        return [list(representatives)]

    async def _fake_resolve_cluster(cluster, registry, system_prompt):
        if "The Company" in cluster:
            merged_name = next(n for n in cluster if n != "The Company")
            return ClusterResolution(groups=[MergeGroup(
                canonical_name=merged_name, canonical_type="ORGANIZATION",
                merged_description="The final merged entity.", members=list(cluster),
            )])
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="Globex Inc.", canonical_type="ORGANIZATION",
            merged_description="Sells software and is a tech company.", members=list(cluster),
        )])

    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=embedder), \
         patch("services.clustering_utils.cluster_by_similarity", side_effect=_cluster), \
         patch("services.clustering_utils.tight_partition", side_effect=lambda c, e, t: [c]), \
         patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        final_entities, final_merge_map_by_doc = _run(canonicalize_corpus(canonical_by_doc))

    globex_final = final_merge_map_by_doc["doc-1"]["Globex Inc."]
    assert final_merge_map_by_doc["doc-2"]["Globex Corporation"] == globex_final
    assert final_merge_map_by_doc["doc-3"]["The Company"] == globex_final
    assert len(final_entities) == 1
    entity = final_entities[globex_final]
    # Every original doc's own description survived, per-source -- none collapsed
    # into the LLM's blended text at either round.
    texts = {d["text"] for d in entity["descriptions"]}
    assert texts == {"Sells software.", "A tech company.", "Referenced as 'the company'."}
    # And the LLM's final folded text IS persisted, as the entity's one description.
    assert entity["description"] == "The final merged entity."
    assert calls["n"] == 2


# --- large-cluster batching (experiment-parity review #15) ---

def test_resolve_cluster_batched_calls_resolve_cluster_directly_when_under_the_cap():
    names = ["A", "B", "C"]
    registry = {n: {"type": "ORG", "description": ""} for n in names}
    calls = []

    async def _fake_resolve_cluster(cluster, reg, system_prompt):
        calls.append(list(cluster))
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="A", canonical_type="ORG", merged_description="", members=cluster)])

    with patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        result = _run(_resolve_cluster_batched(names, registry, "system"))
    assert calls == [names]  # exactly one call, the whole cluster -- no splitting needed
    assert result.groups[0].members == names


def test_resolve_cluster_batched_splits_above_the_cap_and_consolidates_survivors():
    """A cluster bigger than _LLM_BATCH_MAX must be sliced into multiple LLM
    calls, then the resulting canonical names go through ONE more pass to
    catch anything that should still merge across slices.

    3 survivors need batch_max >= 3 for that consolidation to fit in a single
    call -- otherwise resolve_cluster_batched must recurse and re-slice the
    survivors themselves, which is correct behaviour (point 17), just not
    what this test is isolating."""
    names = [f"n{i}" for i in range(7)]
    registry = {n: {"type": "ORG", "description": f"desc-{n}"} for n in names}
    slice_reps = {"n0", "n3", "n6"}  # the first member of each of the 3 slices (batch size 3)
    calls = []

    async def _fake_resolve_cluster(cluster, reg, system_prompt):
        calls.append(list(cluster))
        if set(cluster) == slice_reps:
            # consolidation pass over the 3 slice survivors -- merge them all into one
            return ClusterResolution(groups=[MergeGroup(
                canonical_name="n0", canonical_type="ORG", merged_description="merged",
                members=list(cluster))])
        # a slice call: every member in the slice merges into ONE group, repped by the first
        return ClusterResolution(groups=[MergeGroup(
            canonical_name=cluster[0], canonical_type="ORG", merged_description="", members=list(cluster))])

    with patch("services.graph_canonicalization_service._LLM_BATCH_MAX", 3), \
         patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        result = _run(_resolve_cluster_batched(names, registry, "system"))

    # 7 names / batch of 3 -> 3 slice calls, then 1 consolidation call over the 3 survivors.
    assert len(calls) == 4
    assert sum(len(c) for c in calls[:3]) == 7
    assert len(calls[3]) == 3
    # Every original name must appear in the final result, expanded back from survivors.
    all_final_members = [m for g in result.groups for m in g.members]
    assert sorted(all_final_members) == sorted(names)


def test_resolve_cluster_batched_skips_consolidation_when_only_one_survivor():
    names = [f"n{i}" for i in range(4)]
    registry = {n: {"type": "ORG", "description": ""} for n in names}
    calls = []

    async def _fake_resolve_cluster(cluster, reg, system_prompt):
        calls.append(list(cluster))
        # Every slice resolves to the SAME canonical name -- only one survivor.
        return ClusterResolution(groups=[MergeGroup(
            canonical_name="shared", canonical_type="ORG", merged_description="d", members=list(cluster))])

    with patch("services.graph_canonicalization_service._LLM_BATCH_MAX", 2), \
         patch("services.graph_canonicalization_service.resolve_cluster", side_effect=_fake_resolve_cluster):
        result = _run(_resolve_cluster_batched(names, registry, "system"))
    assert len(calls) == 2  # only the 2 slice calls -- no extra consolidation call
    assert len(result.groups) == 1
    assert sorted(result.groups[0].members) == sorted(names)


def test_exact_duplicate_precision_values_still_fold_together():
    """Precision values ARE allowed to merge on an exact normalized-name
    duplicate -- that's the one case that's still safe."""
    canonical_by_doc = {
        "doc-1": ({"$82k": {"type": "ENTITY", "description": "from doc 1"}}, {"$82k": "$82k"}),
        "doc-2": ({"$82k": {"type": "ENTITY", "description": "from doc 2"}}, {"$82k": "$82k"}),
    }
    with patch("services.graph_canonicalization_service.get_embedding_service", return_value=_fake_embedder()):
        final_entities, _ = _run(canonicalize_corpus(canonical_by_doc))
    assert list(final_entities.keys()) == ["$82k"]
    assert len(final_entities["$82k"]["descriptions"]) == 2


# --- resolve_entity_ids (experiment-parity review #27, Phase 1) ---

def test_resolve_entity_ids_mints_one_id_per_canonical_name():
    """resolve_entity_ids fires link_or_mint concurrently (bounded by
    settings.corpus_entity_id_resolution_concurrency), so the mock must key its
    answer off the call's own arguments rather than a fixed-order side_effect
    list -- concurrent completion order is not guaranteed to match input order."""
    final_entities = {
        "Globex": {"type": "ORGANIZATION", "aliases": [{"name": "Globex Inc.", "doc_id": "d1"}]},
        "Initech": {"type": "ORGANIZATION", "aliases": [{"name": "Initech", "doc_id": "d2"}]},
    }
    ids_by_name = {"Globex": "ent_globex", "Initech": "ent_initech"}
    with patch("repository.entity_registry.link_or_mint",
              side_effect=lambda canon_name, *a, **k: {"_id": ids_by_name[canon_name]}):
        canon_to_id = _run(resolve_entity_ids(final_entities, "tenant-1"))
    assert canon_to_id == {"Globex": "ent_globex", "Initech": "ent_initech"}


def test_resolve_entity_ids_passes_tenant_id_and_type_through():
    final_entities = {"Globex": {"type": "ORGANIZATION", "aliases": []}}
    with patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent_1"}) as link_or_mint:
        _run(resolve_entity_ids(final_entities, "tenant-1"))
    args, kwargs = link_or_mint.call_args
    assert args[0] == "Globex"
    assert args[1] == "ORGANIZATION"
    assert args[3] == "tenant-1"


def test_resolve_entity_ids_skips_person_entries_entirely():
    """Experiment-parity review #8, Phase 2: PERSON entries are already
    resolved by identity (services/person_corpus_resolver.py), keyed by
    their own entity_id -- re-running plain link_or_mint on that key would
    treat the entity_id string itself as a name and mint a spurious second
    identity. core/pipeline.py adds the canon_to_id[entity_id] = entity_id
    identity mapping directly, not through this function."""
    final_entities = {
        "Globex": {"type": "ORGANIZATION", "aliases": []},
        "ent_person1": {"type": "PERSON", "name": "Jane Doe", "aliases": []},
    }
    with patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent_globex"}) as link_or_mint:
        canon_to_id = _run(resolve_entity_ids(final_entities, "tenant-1"))
    link_or_mint.assert_called_once()
    assert canon_to_id == {"Globex": "ent_globex"}
    assert "ent_person1" not in canon_to_id
