"""Unit tests for services/edge_gate.py (experiment-parity review #7) --
a heuristic filter for relationship endpoints that are pronouns, generic
references, or bare values, not real graph entities."""
from services.edge_gate import is_junk_endpoint


def test_pronoun_is_junk():
    assert is_junk_endpoint("it") is True
    assert is_junk_endpoint("They") is True


def test_generic_reference_is_junk():
    assert is_junk_endpoint("the team") is True
    assert is_junk_endpoint("The Company") is True


def test_bare_value_is_junk():
    assert is_junk_endpoint("120") is True
    assert is_junk_endpoint("$82k") is True
    assert is_junk_endpoint("80%") is True


def test_date_like_is_junk():
    assert is_junk_endpoint("2026-03-04") is True
    assert is_junk_endpoint("March 2026") is True
    assert is_junk_endpoint("Q1 2026") is True


def test_time_like_is_junk():
    assert is_junk_endpoint("3:45pm") is True


def test_value_typed_entity_is_junk_regardless_of_name():
    assert is_junk_endpoint("Anything", entity_type="DATE") is True
    assert is_junk_endpoint("Anything", entity_type="money") is True


def test_single_character_is_junk():
    assert is_junk_endpoint("A") is True


def test_empty_name_is_junk():
    assert is_junk_endpoint("") is True
    assert is_junk_endpoint(None) is True


def test_punctuation_only_is_junk():
    assert is_junk_endpoint("---") is True


def test_acronym_is_not_junk_despite_being_short_and_uppercase():
    """Short all-caps tokens are acronyms (US, HR, AI), not function words --
    must be exempt from the pronoun/determiner/generic checks."""
    assert is_junk_endpoint("HR") is False
    assert is_junk_endpoint("AI") is False


def test_real_entity_name_is_not_junk():
    assert is_junk_endpoint("Globex", entity_type="ORGANIZATION") is False
    assert is_junk_endpoint("Recognition Hub", entity_type="PRODUCT") is False


def test_month_prefix_alone_is_not_a_date():
    """A month-name PREFIX with no actual date digits must not false-positive
    (e.g. a person or product literally named starting with a month)."""
    assert is_junk_endpoint("Marketing", entity_type="PRODUCT") is False
    assert is_junk_endpoint("Maria", entity_type="PERSON") is False
