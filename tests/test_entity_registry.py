"""Unit tests for repository/entity_registry.py (experiment-parity review #27,
Phase 1 -- stable entity_id identity, mirrors the experiment's
storage/mongo_state.py:EntityRegistry, tenant-scoped)."""
from unittest.mock import MagicMock, call, patch

from pymongo.errors import DuplicateKeyError

from repository import entity_registry

TENANT = "tenant-1"


def _fake_db():
    db = MagicMock()
    return db


def setup_function():
    entity_registry._index_ensured = False


def test_find_returns_none_when_never_minted():
    db = _fake_db()
    db["entity_registry"].find_one.return_value = None
    with patch("repository.entity_registry._get_db", return_value=db):
        assert entity_registry.find("globex", "ORGANIZATION", TENANT) is None


def test_find_scopes_the_query_by_tenant():
    db = _fake_db()
    db["entity_registry"].find_one.return_value = {"_id": "ent_1"}
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.find("globex", "ORGANIZATION", TENANT)
    db["entity_registry"].find_one.assert_called_once_with({
        "normalized_name": "globex", "entity_type": "ORGANIZATION", "tenant_id": TENANT,
    })


def test_find_all_by_alias_returns_every_matching_record():
    db = _fake_db()
    db["entity_registry"].find.return_value = [{"_id": "ent_1"}, {"_id": "ent_2"}]
    with patch("repository.entity_registry._get_db", return_value=db):
        result = entity_registry.find_all_by_alias("john smith", "PERSON", TENANT)
    assert result == [{"_id": "ent_1"}, {"_id": "ent_2"}]
    db["entity_registry"].find.assert_called_once_with({
        "aliases": "john smith", "entity_type": "PERSON", "tenant_id": TENANT,
    })


def test_find_all_by_alias_empty_when_no_matches():
    db = _fake_db()
    db["entity_registry"].find.return_value = []
    with patch("repository.entity_registry._get_db", return_value=db):
        assert entity_registry.find_all_by_alias("nobody", "PERSON", TENANT) == []


def test_link_or_mint_mints_a_new_entity_when_none_exists():
    db = _fake_db()
    db["entity_registry"].find_one.return_value = None
    with patch("repository.entity_registry._get_db", return_value=db):
        rec = entity_registry.link_or_mint("Globex", "ORGANIZATION", "globex", TENANT)
    assert rec["_id"].startswith("ent_")
    assert rec["canonical_name"] == "Globex"
    assert rec["tenant_id"] == TENANT
    db["entity_registry"].insert_one.assert_called_once()


def test_link_or_mint_returns_the_existing_record_when_found():
    db = _fake_db()
    existing = {"_id": "ent_existing", "canonical_name": "Globex", "aliases": []}
    db["entity_registry"].find_one.return_value = existing
    with patch("repository.entity_registry._get_db", return_value=db):
        rec = entity_registry.link_or_mint("Globex Inc.", "ORGANIZATION", "globex", TENANT)
    assert rec is existing
    db["entity_registry"].insert_one.assert_not_called()


def test_link_or_mint_adds_aliases_to_an_existing_record():
    db = _fake_db()
    existing = {"_id": "ent_existing", "canonical_name": "Globex", "aliases": []}
    db["entity_registry"].find_one.return_value = existing
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.link_or_mint("Globex", "ORGANIZATION", "globex", TENANT, aliases=["Globex Inc."])
    db["entity_registry"].update_one.assert_called_once_with(
        {"_id": "ent_existing"}, {"$addToSet": {"aliases": {"$each": ["Globex Inc."]}}},
    )


def test_link_or_mint_two_different_normalized_names_get_different_entity_ids():
    db = _fake_db()
    db["entity_registry"].find_one.return_value = None
    with patch("repository.entity_registry._get_db", return_value=db):
        a = entity_registry.link_or_mint("Globex", "ORGANIZATION", "globex", TENANT)
        b = entity_registry.link_or_mint("Initech", "ORGANIZATION", "initech", TENANT)
    assert a["_id"] != b["_id"]


def test_link_or_mint_race_falls_back_to_find_on_duplicate_key_error():
    """Concurrency safety: a racing duplicate insert must collapse to the
    surviving record via the unique index, not silently produce two ids for
    the same identity."""
    db = _fake_db()
    winner = {"_id": "ent_winner", "canonical_name": "Globex", "aliases": []}
    db["entity_registry"].find_one.side_effect = [None, winner]
    db["entity_registry"].insert_one.side_effect = DuplicateKeyError("dup")
    with patch("repository.entity_registry._get_db", return_value=db):
        rec = entity_registry.link_or_mint("Globex", "ORGANIZATION", "globex", TENANT)
    assert rec == winner


def test_link_or_mint_scopes_identity_by_tenant():
    """Two different tenants must never collide onto the same entity_id even
    with identical normalized_name/entity_type."""
    db = _fake_db()
    db["entity_registry"].find_one.side_effect = [None, {"tenant_id": "tenant-1"}, None]
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.link_or_mint("Globex", "ORGANIZATION", "globex", "tenant-1")
        entity_registry.find("globex", "ORGANIZATION", "tenant-2")
    calls = db["entity_registry"].find_one.call_args_list
    assert calls[0].args[0]["tenant_id"] == "tenant-1"
    assert calls[-1].args[0]["tenant_id"] == "tenant-2"


def test_get_returns_record_by_entity_id():
    db = _fake_db()
    db["entity_registry"].find_one.return_value = {"_id": "ent_1", "canonical_name": "Globex"}
    with patch("repository.entity_registry._get_db", return_value=db):
        rec = entity_registry.get("ent_1")
    db["entity_registry"].find_one.assert_called_once_with({"_id": "ent_1"})
    assert rec["canonical_name"] == "Globex"


def test_add_aliases_is_a_no_op_for_empty_list():
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.add_aliases("ent_1", [])
    db["entity_registry"].update_one.assert_not_called()


def test_enrich_sets_each_field_individually():
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.enrich("ent_1", {"job_title": "Engineer", "department": "R&D"})
    db["entity_registry"].update_one.assert_called_once_with(
        {"_id": "ent_1"}, {"$set": {"enrichment.job_title": "Engineer", "enrichment.department": "R&D"}},
    )


def test_enrich_drops_falsy_values_rather_than_overwriting():
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.enrich("ent_1", {"job_title": "Engineer", "manager": None, "team": ""})
    db["entity_registry"].update_one.assert_called_once_with(
        {"_id": "ent_1"}, {"$set": {"enrichment.job_title": "Engineer"}},
    )


def test_enrich_is_a_no_op_when_everything_is_falsy():
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.enrich("ent_1", {"job_title": None, "manager": ""})
    db["entity_registry"].update_one.assert_not_called()


def test_delete_merged_removes_the_registry_row_by_entity_id():
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry.delete_merged("ent_mention1")
    db["entity_registry"].delete_one.assert_called_once_with({"_id": "ent_mention1"})


def test_ensure_index_creates_both_indexes_once():
    """Two indexes now: the unique identity index, and the aliases index the
    PERSON ambiguity guard's find_all_by_alias needs. Both cached process-wide
    behind _index_ensured, so a second _ensure_index() call is a no-op."""
    db = _fake_db()
    with patch("repository.entity_registry._get_db", return_value=db):
        entity_registry._ensure_index()
        entity_registry._ensure_index()
    assert db["entity_registry"].create_index.call_args_list == [
        call([("normalized_name", 1), ("entity_type", 1), ("tenant_id", 1)], unique=True),
        call([("aliases", 1), ("entity_type", 1), ("tenant_id", 1)]),
    ]
