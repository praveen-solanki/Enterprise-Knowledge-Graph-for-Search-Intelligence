"""Unit tests for services/people/ — the connector-blind half of the people
phase: MATERIALIZE (identity + the person entry), the bare-mention ambiguity
guard, and the LLM disambiguation call.

What reads `user_*` is the intranet's, and is covered by
tests/test_people_providers.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from connectors.base import PersonRecord
from services.people import (
    materialize_persons,
    person_identity_norm,
    resolve_bare_person,
)

TENANT = "tenant-1"


def _run(coro):
    return asyncio.run(coro)


def _record(**overrides) -> PersonRecord:
    kwargs = {
        "source_key": "user-99", "display_name": "Jane Doe", "email": "Jane@Example.com",
        "attributes": {"job_title": "Engineer", "department": "R&D", "manager": "Alex Manager"},
        "source_doc_ids": ["doc-people-1"],
    }
    kwargs.update(overrides)
    return PersonRecord(**kwargs)


# --- MATERIALIZE ---

def test_identity_is_keyed_on_connector_and_source_key():
    assert person_identity_norm("zeus", "user-99") == "zeus\x1fperson:user-99"


def test_two_different_same_named_people_get_two_different_identities():
    """The whole point: two different real 'Jane Doe's must never collapse
    onto the same registry identity key."""
    assert person_identity_norm("zeus", "user-99") != person_identity_norm("zeus", "user-42")


def test_the_same_source_key_in_two_connectors_is_two_different_people():
    """Two sources can legitimately hand out the same opaque id."""
    assert person_identity_norm("zeus", "u-1") != person_identity_norm("confluence", "u-1")


def test_a_renamed_person_keeps_their_identity():
    """The old key folded the normalized name in, so a display-name change
    minted a second person. The key is the source's own now."""
    assert person_identity_norm("zeus", "user-99") == person_identity_norm("zeus", "user-99")


def _mint(entity_id="ent_1"):
    return patch("repository.entity_registry.link_or_mint",
                 return_value={"_id": entity_id, "canonical_name": "Jane Doe"})


def test_materialize_links_by_key_and_aliases_the_name_and_email():
    """Name and email go in as aliases, which is what lets a bare mention of
    the same person elsewhere in the corpus resolve onto this entity."""
    with _mint() as link_or_mint, patch("repository.entity_registry.enrich") as enrich:
        entries = materialize_persons([_record()], "zeus", TENANT)
    args, kwargs = link_or_mint.call_args
    assert args[:4] == ("Jane Doe", "PERSON", person_identity_norm("zeus", "user-99"), TENANT)
    assert kwargs["aliases"] == ["jane doe", "jane@example.com"]
    # Registry enrichment feeds the ambiguity guard's prompt, unprefixed.
    assert enrich.call_args.args == ("ent_1", _record().attributes)
    assert entries["ent_1"]["name"] == "Jane Doe"
    assert entries["ent_1"]["source_docs"] == ["doc-people-1"]


def test_materialize_stamps_the_connector_blind_node_properties():
    """B9's shape: :Entity:Person with name/email/id, reachable by the standard
    traversal, whatever the connector."""
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons([_record()], "confluence", TENANT)
    fields = entries["ent_1"]["person_fields"]
    assert fields["source_key"] == "user-99"
    assert fields["connector_type"] == "confluence"
    assert fields["email"] == "jane@example.com"
    # ai-graph-retrieval reads person_email (PEOPLE_RESOLVE coalesces person_email
    # then user_email); only `email` was ever written, so contact display was always
    # null. All three names carry the same value — additive, so nothing reading
    # `email` breaks.
    assert fields["person_email"] == "jane@example.com"
    assert fields["user_email"] == "jane@example.com"


def test_materialize_omits_both_email_names_when_the_record_has_no_email():
    """Falsy is dropped, not written blank — for person_email exactly as for email,
    or write_entities' `SET e += person_fields` would wipe what an earlier run knew."""
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons([_record(email=None)], "zeus", TENANT)
    fields = entries["ent_1"]["person_fields"]
    assert "email" not in fields
    assert "person_email" not in fields
    assert "user_email" not in fields


def test_materialize_stamps_url_as_a_plain_property_not_person_url():
    """url is a top-level PersonRecord field, same as email — not part of
    attributes — so it must land as a plain `url`, matching Document.url/
    Chunk.url, not fall through the person_<key> prefixing as person_url."""
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons(
            [_record(url="http://localhost:3000/people/user-99")], "zeus", TENANT,
        )
    fields = entries["ent_1"]["person_fields"]
    assert fields["url"] == "http://localhost:3000/people/user-99"
    assert "person_url" not in fields


def test_materialize_omits_url_when_the_record_has_no_url():
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons([_record(url=None)], "zeus", TENANT)
    assert "url" not in entries["ent_1"]["person_fields"]


def test_materialize_prefixes_the_connectors_attributes_for_the_node():
    """The one convention between the two sides: attributes are unprefixed,
    node properties are person_<key> — which is what Zeus's own Cypher
    (materialize_reporting_edges, materialize_org_membership) reads."""
    record = _record(attributes={"user_id": "user-99", "manager_id": "mgr-1",
                                 "department": "R&D", "team": None})
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons([record], "zeus", TENANT)
    fields = entries["ent_1"]["person_fields"]
    assert fields["person_user_id"] == "user-99"
    assert fields["person_manager_id"] == "mgr-1"
    assert fields["person_department"] == "R&D"
    # Falsy dropped, not written as a blank — write_entities' `SET e +=` would
    # otherwise wipe what a previous run knew.
    assert "person_team" not in fields


def test_materialize_skips_a_record_with_no_key_or_no_name():
    with patch("repository.entity_registry.link_or_mint") as link_or_mint:
        entries = materialize_persons(
            [_record(source_key=""), _record(display_name=None)], "zeus", TENANT,
        )
    assert entries == {}
    link_or_mint.assert_not_called()


def test_materialize_collapses_two_documents_naming_the_same_person():
    with _mint(), patch("repository.entity_registry.enrich"):
        entries = materialize_persons(
            [_record(source_doc_ids=["doc-1"]), _record(source_doc_ids=["doc-2"])], "zeus", TENANT,
        )
    assert list(entries) == ["ent_1"]
    assert entries["ent_1"]["source_docs"] == ["doc-1", "doc-2"]


def test_materialize_reads_nothing_but_the_four_guaranteed_fields():
    """MATERIALIZE is identical for every connector because it never looks
    inside attributes. A record whose attributes are gibberish still
    materializes exactly the same identity and node properties."""
    with _mint(), patch("repository.entity_registry.enrich"):
        opaque = materialize_persons([_record(attributes={"anything": "at all"})], "zeus", TENANT)
        empty = materialize_persons([_record(attributes={})], "zeus", TENANT)
    assert opaque["ent_1"]["person_fields"]["source_key"] == \
        empty["ent_1"]["person_fields"]["source_key"]
    assert opaque["ent_1"]["person_fields"]["person_anything"] == "at all"


# --- resolve_bare_person (ambiguity guard) ---

def test_resolve_bare_person_links_the_single_candidate_no_llm_call():
    with patch("repository.entity_registry.find_all_by_alias", return_value=[{"_id": "ent_1"}]), \
         patch("repository.entity_registry.add_aliases") as add_aliases, \
         patch("services.people.resolution.llm_confirms_same_person") as llm_confirm:
        entity_id = _run(resolve_bare_person("Jane Doe", "an engineer", TENANT))
    assert entity_id == "ent_1"
    add_aliases.assert_called_once_with("ent_1", ["jane doe"])
    llm_confirm.assert_not_called()


def test_resolve_bare_person_mints_when_no_candidates():
    with patch("repository.entity_registry.find_all_by_alias", return_value=[]), \
         patch("repository.entity_registry.link_or_mint",
              return_value={"_id": "ent_new"}) as link_or_mint:
        entity_id = _run(resolve_bare_person("Jane Doe", "an engineer", TENANT))
    assert entity_id == "ent_new"
    link_or_mint.assert_called_once_with("Jane Doe", "PERSON", "jane doe", TENANT, aliases=["jane doe"])


def test_resolve_bare_person_ambiguous_confirms_via_llm_and_links():
    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {"job_title": "Sales"}},
                 {"_id": "ent_2", "canonical_name": "Jane Doe", "enrichment": {"job_title": "Engineer"}}]
    with patch("repository.entity_registry.find_all_by_alias", return_value=candidates), \
         patch("repository.entity_registry.add_aliases") as add_aliases, \
         patch("services.people.resolution.llm_confirms_same_person",
              new=AsyncMock(side_effect=[False, True])) as llm_confirm:
        entity_id = _run(resolve_bare_person("Jane Doe", "works in engineering", TENANT))
    assert entity_id == "ent_2"
    add_aliases.assert_called_once_with("ent_2", ["jane doe"])
    assert llm_confirm.await_count == 2


def test_resolve_bare_person_ambiguous_with_no_confident_match_mints_unresolved_node():
    """Never a blind merge -- if the LLM can't confidently pick one of the
    ambiguous candidates, a separate conservative node is minted instead."""
    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {}},
                 {"_id": "ent_2", "canonical_name": "Jane Doe", "enrichment": {}}]
    with patch("repository.entity_registry.find_all_by_alias", return_value=candidates), \
         patch("repository.entity_registry.link_or_mint",
              return_value={"_id": "ent_unresolved"}) as link_or_mint, \
         patch("services.people.resolution.llm_confirms_same_person",
              new=AsyncMock(return_value=False)):
        entity_id = _run(resolve_bare_person("Jane Doe", "no context", TENANT))
    assert entity_id == "ent_unresolved"
    link_or_mint.assert_called_once_with("Jane Doe", "PERSON", "jane doe", TENANT, aliases=["jane doe"])


def test_resolve_bare_person_pushes_new_mint_to_qdrant():
    vector_store = MagicMock()
    with patch("repository.entity_registry.find_all_by_alias", return_value=[]), \
         patch("repository.entity_registry.link_or_mint", return_value={"_id": "ent_new"}):
        _run(resolve_bare_person("Jane Doe", "an engineer", TENANT,
                                 embedding=[0.1], vector_store=vector_store))
    vector_store.upsert_entity.assert_called_once_with("ent_new", [0.1], "PERSON", TENANT, [])


def test_llm_confirms_same_person_delegates_to_the_shared_confirm_with_retry():
    """The actual retry/backoff/logging behavior is services/llm_client.
    py's own confirm_with_retry (see tests/test_llm_client.py) -- this
    only needs to prove _llm_confirms_same_person builds the right
    prompt/step/context and passes through whatever confirm_with_retry
    returns, not re-test retry logic that's now shared with
    services/incremental_canonicalization.py's own caller."""
    from services.people.disambiguation import llm_confirms_same_person as _llm_confirms_same_person
    with patch("services.people.disambiguation.llm_client") as mock_module:
        mock_module.confirm_with_retry = AsyncMock(return_value=True)
        result = _run(_llm_confirms_same_person("Jane Doe", "ctx", "Jane Doe", "attrs"))
    assert result is True
    args, kwargs = mock_module.confirm_with_retry.call_args
    assert args[2] == "same_person"
    assert kwargs["step"] == "person_disambiguation_confirm"
    assert kwargs["context"] == "new='Jane Doe' candidate='Jane Doe'"
    assert kwargs["log_label"] == "Person disambiguation"


# --- attr_text (connector-agnostic attribute rendering) ---

def test_attr_text_renders_whatever_keys_the_connector_used():
    """The point of the change: a source whose people carry `space` and
    `account_type` instead of the HR export's `job_title`/`department` used to
    render to "" and lose the guard its only evidence beyond a name."""
    from services.people.disambiguation import attr_text
    assert attr_text({"space": "Engineering", "account_type": "atlassian"}) == \
        "space: Engineering; account type: atlassian"


def test_attr_text_still_renders_the_intranet_attributes():
    from services.people.disambiguation import attr_text
    assert attr_text({"job_title": "CEO", "department": "Executive",
                      "manager": "Ann Roe"}) == \
        "job title: CEO; department: Executive; manager: Ann Roe"


def test_attr_text_drops_opaque_identifiers():
    """user_id/manager_id are on the record for write_entities' Cypher joins,
    not for an LLM — they cannot tell it whether two people are the same."""
    from services.people.disambiguation import attr_text
    assert attr_text({"user_id": "005xx", "manager_id": "005yy",
                      "job_title": "Engineer"}) == "job title: Engineer"


def test_attr_text_drops_empty_values_and_an_empty_dict():
    from services.people.disambiguation import attr_text
    assert attr_text({"job_title": "", "department": None, "team": "Core"}) == "team: Core"
    assert attr_text({}) == ""


# --- resolve_bare_persons (batched ambiguity guard, review #12) ---

from services.people import resolve_bare_persons  # noqa: E402


def test_resolve_bare_persons_dedupes_registry_reads_by_name():
    """A person named in N mentions costs one find_all_by_alias, not N."""
    with patch("repository.entity_registry.find_all_by_alias",
              return_value=[{"_id": "ent_1"}]) as find_all, \
         patch("repository.entity_registry.add_aliases"):
        ids = _run(resolve_bare_persons(
            [("Jane Doe", "an engineer"), ("Jane Doe", "still an engineer")], TENANT))
    assert ids == ["ent_1", "ent_1"]
    find_all.assert_called_once_with("jane doe", "PERSON", TENANT)


def test_resolve_bare_persons_mints_once_for_no_candidates_and_shares_the_id():
    with patch("repository.entity_registry.find_all_by_alias", return_value=[]), \
         patch("repository.entity_registry.link_or_mint",
              return_value={"_id": "ent_new"}) as link_or_mint:
        ids = _run(resolve_bare_persons(
            [("Jane Doe", "an engineer"), ("Jane Doe", "an engineer")], TENANT))
    assert ids == ["ent_new", "ent_new"]
    link_or_mint.assert_called_once_with("Jane Doe", "PERSON", "jane doe", TENANT, aliases=["jane doe"])


def test_resolve_bare_persons_returns_ids_in_input_order_across_distinct_names():
    def _find(alias, entity_type, tenant_id):
        return [{"_id": f"ent_{alias.replace(' ', '_')}"}]

    with patch("repository.entity_registry.find_all_by_alias", side_effect=_find), \
         patch("repository.entity_registry.add_aliases"):
        ids = _run(resolve_bare_persons(
            [("Jane Doe", "engineer"), ("John Roe", "sales"), ("Jane Doe", "engineer, again")], TENANT))
    assert ids == ["ent_jane_doe", "ent_john_roe", "ent_jane_doe"]


def test_resolve_bare_persons_ambiguous_confirms_via_batch_disambiguation():
    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {"job_title": "Sales"}},
                 {"_id": "ent_2", "canonical_name": "Jane Doe", "enrichment": {"job_title": "Engineer"}}]
    with patch("repository.entity_registry.find_all_by_alias", return_value=candidates), \
         patch("repository.entity_registry.add_aliases") as add_aliases, \
         patch("services.people.resolution.disambiguate_item_batch",
              new=AsyncMock(return_value={(0, 0): candidates[1]})) as batch_mock:
        ids = _run(resolve_bare_persons([("Jane Doe", "works in engineering")], TENANT))
    assert ids == ["ent_2"]
    add_aliases.assert_called_once_with("ent_2", ["jane doe"])
    batch_mock.assert_awaited_once()


def test_resolve_bare_persons_ambiguous_with_no_confident_match_mints_unresolved_node():
    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {}},
                 {"_id": "ent_2", "canonical_name": "Jane Doe", "enrichment": {}}]
    with patch("repository.entity_registry.find_all_by_alias", return_value=candidates), \
         patch("repository.entity_registry.link_or_mint",
              return_value={"_id": "ent_unresolved"}) as link_or_mint, \
         patch("services.people.resolution.disambiguate_item_batch",
              new=AsyncMock(return_value={(0, 0): None})):
        ids = _run(resolve_bare_persons([("Jane Doe", "no context")], TENANT))
    assert ids == ["ent_unresolved"]
    link_or_mint.assert_called_once_with("Jane Doe", "PERSON", "jane doe", TENANT, aliases=["jane doe"])


def test_resolve_bare_persons_empty_input_makes_no_calls():
    with patch("repository.entity_registry.find_all_by_alias") as find_all:
        assert _run(resolve_bare_persons([], TENANT)) == []
    find_all.assert_not_called()


# --- disambiguate_item_batch (review #12's batched LLM call) ---

def test_disambiguate_item_batch_matches_via_structured_llm_call():
    from services.people.disambiguation import _PersonDisambiguationBatch, disambiguate_item_batch

    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {}}]
    parsed = _PersonDisambiguationBatch.model_validate({"matches": [{"id": 0, "match_index": 0}]})
    with patch("services.people.disambiguation.structured_llm_call",
              new=AsyncMock(return_value=(True, parsed))):
        result = _run(disambiguate_item_batch([("key-0", "Jane Doe", "an engineer", candidates)]))
    assert result == {"key-0": candidates[0]}


def test_disambiguate_item_batch_no_match_returns_none():
    from services.people.disambiguation import _NO_MATCH, _PersonDisambiguationBatch, disambiguate_item_batch

    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {}}]
    parsed = _PersonDisambiguationBatch.model_validate(
        {"matches": [{"id": 0, "match_index": _NO_MATCH}]})
    with patch("services.people.disambiguation.structured_llm_call",
              new=AsyncMock(return_value=(True, parsed))):
        result = _run(disambiguate_item_batch([("key-0", "Jane Doe", "an engineer", candidates)]))
    assert result == {"key-0": None}


def test_disambiguate_item_batch_fails_closed_to_no_match_on_call_failure():
    from services.people.disambiguation import disambiguate_item_batch

    candidates = [{"_id": "ent_1", "canonical_name": "Jane Doe", "enrichment": {}}]
    with patch("services.people.disambiguation.structured_llm_call",
              new=AsyncMock(return_value=(False, None))):
        result = _run(disambiguate_item_batch([("key-0", "Jane Doe", "an engineer", candidates)]))
    assert result == {"key-0": None}
