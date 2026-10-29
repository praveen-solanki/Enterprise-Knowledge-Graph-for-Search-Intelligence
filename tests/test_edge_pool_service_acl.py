"""ACL behavior of services/edge_pool_service.py's build_edge_pool (AIS-3628):
claims never merge across documents (acl stays exact), edges DO merge across
documents (acl becomes a union), and per-source descriptions are kept
separately rather than collapsed into one "winning" string.
"""
from models.extraction import ChunkExtractionResult, ExtractedClaim, ExtractedEntity, ExtractedRelationship
from services.edge_pool_service import build_edge_pool


def _result(entities, relationships=None, claims=None) -> ChunkExtractionResult:
    return ChunkExtractionResult(
        entities=entities, relationships=relationships or [], claims=claims or [],
    )


def _two_doc_scenario(doc_dates_by_doc=None):
    entities = [
        ExtractedEntity(name="Globex", type="ORGANIZATION"),
        ExtractedEntity(name="Initech", type="ORGANIZATION"),
    ]
    result_a = _result(
        entities,
        relationships=[ExtractedRelationship(
            source_entity="Globex", target_entity="Initech",
            relationship_type="acquired", description="Globex acquired Initech.",
        )],
        claims=[ExtractedClaim(
            subject="Globex", predicate="acquired", object="Initech",
            statement_type="FACT", temporal_type="STATIC",
        )],
    )
    result_b = _result(
        entities,
        relationships=[ExtractedRelationship(
            source_entity="Globex", target_entity="Initech",
            relationship_type="acquired",
            description="Globex acquired Initech for $50M, confidential deal terms.",
        )],
        claims=[ExtractedClaim(
            subject="Globex", predicate="acquired_for", object="$50M",
            statement_type="FACT", temporal_type="STATIC",
        )],
    )

    identity_map = {"Globex": "Globex", "Initech": "Initech"}
    # canon_to_id is an identity map (name -> itself) here on purpose -- these tests
    # verify ACL/description/date semantics, not entity_id resolution itself (that's
    # covered separately in test_graph_canonicalization_service.py), so keeping
    # final_edges/final_claims keyed by the readable name keeps every assertion below
    # unchanged and legible.
    return build_edge_pool(
        results_by_doc={"doc_a": [result_a], "doc_b": [result_b]},
        chunk_ids_by_doc={"doc_a": ["c1"], "doc_b": ["c2"]},
        doc_merge_maps={"doc_a": identity_map, "doc_b": identity_map},
        final_merge_map_by_doc={"doc_a": identity_map, "doc_b": identity_map},
        canon_to_id=identity_map,
        doc_acl_by_doc={"doc_a": ["public"], "doc_b": ["group:site:deal-room"]},
        doc_dates_by_doc=doc_dates_by_doc,
    )


def test_edge_acl_is_union_of_contributing_documents():
    final_edges, _, _ = _two_doc_scenario()
    edge = final_edges[("Globex", "ACQUIRED", "Initech")]
    assert edge["acl"] == sorted({"public", "group:site:deal-room"})


def test_edge_descriptions_are_kept_per_source_not_collapsed():
    final_edges, _, _ = _two_doc_scenario()
    edge = final_edges[("Globex", "ACQUIRED", "Initech")]
    assert len(edge["descriptions"]) == 2
    by_chunk = {d["chunk_id"]: d for d in edge["descriptions"]}
    assert by_chunk["c1"]["text"] == "Globex acquired Initech."
    assert by_chunk["c1"]["acl"] == ["public"]
    assert by_chunk["c2"]["text"] == "Globex acquired Initech for $50M, confidential deal terms."
    assert by_chunk["c2"]["acl"] == ["group:site:deal-room"]
    # Singular description is independent of the per-chunk list (Entity.description
    # pattern): first non-empty LLM text wins; descriptions stays the full list.
    assert edge["description"] == "Globex acquired Initech."
    assert edge["description"] != by_chunk["c2"]["text"]


def test_claims_never_merge_across_documents_and_keep_their_own_doc_acl():
    _, final_claims, _ = _two_doc_scenario()
    assert len(final_claims) == 2
    by_predicate = {c["predicate"]: c for c in final_claims}
    assert by_predicate["acquired"]["acl"] == ["public"]
    assert by_predicate["acquired_for"]["acl"] == ["group:site:deal-room"]


def test_edge_descriptions_carry_each_source_documents_own_date():
    """Code review must-fix #12: documented_on is per-source (like acl and
    text), never a single date chosen to represent the whole merged edge."""
    final_edges, _, _ = _two_doc_scenario(doc_dates_by_doc={"doc_a": "2025-06-01", "doc_b": "2026-01-15"})
    edge = final_edges[("Globex", "ACQUIRED", "Initech")]
    by_chunk = {d["chunk_id"]: d for d in edge["descriptions"]}
    assert by_chunk["c1"]["documented_on"] == "2025-06-01"
    assert by_chunk["c2"]["documented_on"] == "2026-01-15"


def test_edge_descriptions_documented_on_defaults_to_empty_string_when_not_provided():
    final_edges, _, _ = _two_doc_scenario()
    edge = final_edges[("Globex", "ACQUIRED", "Initech")]
    assert all(d["documented_on"] == "" for d in edge["descriptions"])


def test_claims_carry_their_own_documents_documented_on():
    """Experiment-parity review #36: a claim's documented_on is its one
    contributing document's own date — same per-source treatment as an
    edge description's documented_on above, just without a union/merge
    concern since claims never merge across documents."""
    _, final_claims, _ = _two_doc_scenario(doc_dates_by_doc={"doc_a": "2025-06-01", "doc_b": "2026-01-15"})
    by_predicate = {c["predicate"]: c for c in final_claims}
    assert by_predicate["acquired"]["documented_on"] == "2025-06-01"
    assert by_predicate["acquired_for"]["documented_on"] == "2026-01-15"


def test_claims_documented_on_defaults_to_empty_string_when_not_provided():
    _, final_claims, _ = _two_doc_scenario()
    assert all(c["documented_on"] == "" for c in final_claims)


def test_build_edge_pool_defaults_acl_to_empty_when_not_provided():
    entities = [ExtractedEntity(name="Alpha Corp", type="ORG"), ExtractedEntity(name="Beta Corp", type="ORG")]
    result = _result(
        entities,
        relationships=[ExtractedRelationship(
            source_entity="Alpha Corp", target_entity="Beta Corp", relationship_type="related_to", description="",
        )],
    )
    identity_map = {"Alpha Corp": "Alpha Corp", "Beta Corp": "Beta Corp"}
    final_edges, final_claims, _ = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id=identity_map,
        # doc_acl_by_doc omitted entirely — existing callers that don't care about ACL
        # (e.g. any older test) must not break.
    )
    edge = final_edges[("Alpha Corp", "RELATED_TO", "Beta Corp")]
    assert edge["acl"] == []
    assert edge["descriptions"][0]["acl"] == []


# --- Junk-endpoint gating (experiment-parity review #7) ---

def test_relationship_with_a_pronoun_endpoint_is_dropped():
    """Without gating, ("it")-[RELATED_TO]->("the team") would be written
    straight into the graph as if "it" and "the team" were real entities."""
    entities = [
        ExtractedEntity(name="it", type="ENTITY"),
        ExtractedEntity(name="the team", type="ENTITY"),
        ExtractedEntity(name="Globex", type="ORGANIZATION"),
    ]
    result = _result(
        entities,
        relationships=[
            ExtractedRelationship(source_entity="it", target_entity="the team", relationship_type="related_to"),
            ExtractedRelationship(source_entity="Globex", target_entity="the team", relationship_type="owns"),
        ],
    )
    identity_map = {"it": "it", "the team": "the team", "Globex": "Globex"}
    final_edges, _, _ = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id=identity_map,
    )
    assert final_edges == {}


def test_relationship_between_two_real_entities_survives_gating():
    entities = [ExtractedEntity(name="Globex", type="ORGANIZATION"), ExtractedEntity(name="Initech", type="ORGANIZATION")]
    result = _result(
        entities,
        relationships=[ExtractedRelationship(source_entity="Globex", target_entity="Initech",
                                            relationship_type="acquired")],
    )
    identity_map = {"Globex": "Globex", "Initech": "Initech"}
    final_edges, _, _ = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id=identity_map,
    )
    assert ("Globex", "ACQUIRED", "Initech") in final_edges


def test_relationship_with_a_date_typed_endpoint_is_dropped():
    entities = [
        ExtractedEntity(name="Globex", type="ORGANIZATION"),
        ExtractedEntity(name="2026-03-04", type="DATE"),
    ]
    result = _result(
        entities,
        relationships=[ExtractedRelationship(source_entity="Globex", target_entity="2026-03-04",
                                            relationship_type="released_on")],
    )
    identity_map = {"Globex": "Globex", "2026-03-04": "2026-03-04"}
    final_edges, _, _ = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id=identity_map,
    )
    assert final_edges == {}


# --- entity_id keying (experiment-parity review #27, Phase 1) ---

def test_final_edges_claims_mentions_are_keyed_by_entity_id_not_name():
    entities = [ExtractedEntity(name="Globex", type="ORGANIZATION"), ExtractedEntity(name="Initech", type="ORGANIZATION")]
    result = _result(
        entities,
        relationships=[ExtractedRelationship(source_entity="Globex", target_entity="Initech",
                                            relationship_type="acquired")],
        claims=[ExtractedClaim(subject="Globex", predicate="acquired", object="Initech",
                              statement_type="FACT", temporal_type="STATIC")],
    )
    identity_map = {"Globex": "Globex", "Initech": "Initech"}
    canon_to_id = {"Globex": "ent_globex", "Initech": "ent_initech"}
    final_edges, final_claims, mentions = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id=canon_to_id,
    )
    assert ("ent_globex", "ACQUIRED", "ent_initech") in final_edges
    assert final_claims[0]["subject"] == "ent_globex"
    assert {m[1] for m in mentions} == {"ent_globex", "ent_initech"}


def test_relationship_endpoint_missing_from_canon_to_id_is_dropped_as_unbound():
    """canon_to_id should always have an entry for every final canonical name in
    practice -- but if it somehow doesn't, the relationship must be dropped as
    unbound, never written with a None/missing entity_id."""
    entities = [ExtractedEntity(name="Globex", type="ORGANIZATION"), ExtractedEntity(name="Initech", type="ORGANIZATION")]
    result = _result(
        entities,
        relationships=[ExtractedRelationship(source_entity="Globex", target_entity="Initech",
                                            relationship_type="acquired")],
    )
    identity_map = {"Globex": "Globex", "Initech": "Initech"}
    final_edges, _, _ = build_edge_pool(
        results_by_doc={"doc_x": [result]},
        chunk_ids_by_doc={"doc_x": ["c1"]},
        doc_merge_maps={"doc_x": identity_map},
        final_merge_map_by_doc={"doc_x": identity_map},
        canon_to_id={"Globex": "ent_globex"},  # "Initech" missing on purpose
    )
    assert final_edges == {}
    assert final_edges == {}
