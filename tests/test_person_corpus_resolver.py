"""Unit tests for services/person_corpus_resolver.py (experiment-parity
review #8, Phase 2 -- keeps PERSON entities out of the normal corpus-wide
name clustering, resolving them by identity instead)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.person_corpus_resolver import resolve_persons_for_corpus

TENANT = "tenant-1"


def _run(coro):
    return asyncio.run(coro)


def _embedder():
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.1] * 3 for _ in texts]
    return embedder


def test_non_person_entities_pass_through_untouched():
    canonical_by_doc = {
        "doc-1": {"Globex": {"type": "ORGANIZATION", "description": "A company"}},
    }
    with patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        non_person, person_entities, person_merge_map = _run(
            resolve_persons_for_corpus(canonical_by_doc, {}, TENANT)
        )
    assert non_person == canonical_by_doc
    assert person_entities == {}
    assert person_merge_map == {}


def test_person_entity_split_out_and_resolved_via_bare_person_path():
    canonical_by_doc = {
        "doc-1": {"Jane Doe": {"type": "PERSON", "description": "an engineer"},
                 "Globex": {"type": "ORGANIZATION", "description": "A company"}},
    }
    # resolve_bare_persons is batched: one call for every mention, ids back in input order.
    with patch("services.person_corpus_resolver.resolve_bare_persons",
              new=AsyncMock(return_value=["ent_person1"])), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        non_person, person_entities, person_merge_map = _run(
            resolve_persons_for_corpus(canonical_by_doc, {}, TENANT)
        )
    assert "Jane Doe" not in non_person["doc-1"]
    assert "Globex" in non_person["doc-1"]
    assert "ent_person1" in person_entities
    assert person_entities["ent_person1"]["name"] == "Jane Doe"
    assert person_entities["ent_person1"]["source_docs"] == ["doc-1"]
    assert person_entities["ent_person1"]["descriptions"] == [{"text": "an engineer", "doc_id": "doc-1"}]
    assert person_merge_map == {"doc-1": {"Jane Doe": "ent_person1"}}


def test_person_entities_get_embeddings_for_qdrant_via_write_entities():
    canonical_by_doc = {"doc-1": {"Jane Doe": {"type": "PERSON", "description": "an engineer"}}}
    with patch("services.person_corpus_resolver.resolve_bare_persons",
              new=AsyncMock(return_value=["ent_person1"])), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, _ = _run(resolve_persons_for_corpus(canonical_by_doc, {}, TENANT))
    assert person_entities["ent_person1"]["embedding"] == [0.1, 0.1, 0.1]


def test_materialized_persons_are_carried_through_untouched():
    """MATERIALIZE already minted and shaped these (services/people/
    materialize.py); this stage only merges them so a bare mention of the same
    person resolves onto one of them. It reads no HR field to do it — that loop
    is gone, along with the raw payloads it used to be handed."""
    materialized = {"ent_person1": {
        "type": "PERSON", "name": "Jane Doe", "source_docs": ["doc-people-1"],
        "descriptions": [], "aliases": [],
        "person_fields": {"source_key": "user-99", "connector_type": "zeus",
                          "person_department": "R&D"},
    }}
    with patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, _ = _run(resolve_persons_for_corpus({}, materialized, TENANT))
    assert person_entities["ent_person1"]["source_docs"] == ["doc-people-1"]
    assert person_entities["ent_person1"]["person_fields"]["source_key"] == "user-99"
    assert person_entities["ent_person1"]["person_fields"]["person_department"] == "R&D"


def test_a_materialized_person_is_embedded_like_any_other_entity():
    """Even when this run's extraction never re-mentioned them by name in any
    chunk, the person still gets a vector."""
    materialized = {"ent_person1": {"type": "PERSON", "name": "Jane Doe", "source_docs": [],
                                    "descriptions": [], "aliases": []}}
    with patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, _ = _run(resolve_persons_for_corpus({}, materialized, TENANT))
    assert person_entities["ent_person1"]["embedding"] == [0.1, 0.1, 0.1]


def test_bare_mention_entry_never_carries_person_fields():
    """A bare-name PERSON mention (never materialized from a PersonRecord)
    must never get person_fields -- write_entities must not stamp HR
    identity onto a node that was never confirmed as that real HR record."""
    canonical_entities_by_doc = {"doc-1": {"Jane Doe": {"type": "PERSON", "description": "an engineer"}}}
    with patch("services.person_corpus_resolver.resolve_bare_persons", new=AsyncMock(return_value=["ent_x"])), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, _ = _run(resolve_persons_for_corpus(canonical_entities_by_doc, {}, TENANT))
    assert "person_fields" not in person_entities["ent_x"]


def test_materialized_person_and_bare_mention_of_them_share_one_entity_id():
    """The core cross-source-dedup guarantee: a person's own profile chunk
    re-mentions them by name -- that bare mention must resolve back to the SAME
    entity_id MATERIALIZE already minted, not mint a second node. (What makes it
    resolve is the name/email aliases MATERIALIZE puts on the registry record;
    resolve_bare_person is stubbed here, and tests/test_people.py covers it.)"""
    materialized = {"ent_person1": {"type": "PERSON", "name": "Jane Doe",
                                    "source_docs": ["doc-people-1"],
                                    "descriptions": [], "aliases": []}}
    canonical_by_doc = {"doc-people-1": {"Jane Doe": {"type": "PERSON", "description": "profile mention"}}}
    with patch("services.person_corpus_resolver.resolve_bare_persons",
              new=AsyncMock(return_value=["ent_person1"])), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, person_merge_map = _run(
            resolve_persons_for_corpus(canonical_by_doc, materialized, TENANT)
        )
    assert len(person_entities) == 1
    assert person_merge_map["doc-people-1"]["Jane Doe"] == "ent_person1"


def test_two_different_same_named_people_get_two_different_entity_ids():
    """The whole point of this module: two different real 'Jane Doe's (one
    per document, resolved by the ambiguity guard as NOT the same person)
    must end up as two separate entries, never collapsed into one."""
    canonical_by_doc = {
        "doc-1": {"Jane Doe": {"type": "PERSON", "description": "sales rep"}},
        "doc-2": {"Jane Doe": {"type": "PERSON", "description": "engineer"}},
    }
    # One batched call now, not one per mention: both ids come back in input order.
    async_mock = AsyncMock(return_value=["ent_person1", "ent_person2"])
    with patch("services.person_corpus_resolver.resolve_bare_persons", new=async_mock), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _, person_entities, person_merge_map = _run(
            resolve_persons_for_corpus(canonical_by_doc, {}, TENANT)
        )
    assert set(person_entities.keys()) == {"ent_person1", "ent_person2"}
    assert person_merge_map["doc-1"]["Jane Doe"] == "ent_person1"
    assert person_merge_map["doc-2"]["Jane Doe"] == "ent_person2"


def test_all_mentions_resolve_in_one_batched_call():
    """Point 12: the corpus resolver must not await resolve_bare_person once per mention —
    every PERSON mention across the whole corpus goes through resolve_bare_persons in ONE call."""
    canonical_by_doc = {
        "doc-1": {"Jane Doe": {"type": "PERSON", "description": "sales rep"},
                 "John Roe": {"type": "PERSON", "description": "engineer"}},
        "doc-2": {"Jane Doe": {"type": "PERSON", "description": "sales rep, again"}},
    }
    batched_mock = AsyncMock(return_value=["ent_person1", "ent_person2", "ent_person1"])
    with patch("services.person_corpus_resolver.resolve_bare_persons", new=batched_mock), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=_embedder()):
        _run(resolve_persons_for_corpus(canonical_by_doc, {}, TENANT))
    batched_mock.assert_awaited_once_with(
        [("Jane Doe", "sales rep"), ("John Roe", "engineer"), ("Jane Doe", "sales rep, again")], TENANT,
    )


def test_person_with_multiple_descriptions_is_embedded_from_the_merged_signal_not_the_first():
    """Point 13: embedding must come from a description merged across every document that
    mentioned this person, not entry["descriptions"][0]."""
    canonical_by_doc = {
        "doc-1": {"Jane Doe": {"type": "PERSON", "description": "an engineer"}},
        "doc-2": {"Jane Doe": {"type": "PERSON", "description": "leads the platform team"}},
    }
    embedder = _embedder()
    merge_mock = AsyncMock(return_value={"ent_person1": "engineer who leads the platform team"})
    with patch("services.person_corpus_resolver.resolve_bare_persons",
              new=AsyncMock(return_value=["ent_person1", "ent_person1"])), \
         patch("services.person_corpus_resolver.merge_corpus_descriptions", new=merge_mock), \
         patch("services.person_corpus_resolver.get_embedding_service", return_value=embedder):
        _run(resolve_persons_for_corpus(canonical_by_doc, {}, TENANT))
    merge_mock.assert_awaited_once_with(
        {"ent_person1": ["an engineer", "leads the platform team"]})
    embedder.embed_batch.assert_called_once_with(
        ["Jane Doe: engineer who leads the platform team"])
