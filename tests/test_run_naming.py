"""Unit tests for services/run_naming.py (AIS-4086 — deterministic naming for
per-run Neo4j databases / Mongo collections / Qdrant collections)."""
import re
import string

import pytest

from services.run_naming import build_resource_name, generate_run_suffix

_VALID_NEO4J_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def test_generate_run_suffix_is_four_lowercase_alnum_chars():
    suffix = generate_run_suffix()
    assert len(suffix) == 4
    assert all(c in string.ascii_lowercase + string.digits for c in suffix)


def test_generate_run_suffix_varies_across_calls():
    suffixes = {generate_run_suffix() for _ in range(50)}
    assert len(suffixes) > 1  # would only fail on an astronomically unlucky collision


def test_build_resource_name_basic_format():
    name = build_resource_name("acme", "confluence", "a3f9")
    assert name == "g-acme-confluence-a3f9"


def test_build_resource_name_with_kind():
    name = build_resource_name("acme", "confluence", "a3f9", kind="entities")
    assert name == "g-acme-confluence-entities-a3f9"


def test_build_resource_name_is_valid_neo4j_database_name():
    # tenant_id shaped like a real UUID -- would start with a digit without the "g-" prefix
    name = build_resource_name("191e03de-fb5e-4b01-b9fd-faf1873292a6", "confluence", "a3f9", kind="entities")
    assert _VALID_NEO4J_NAME.match(name)
    assert len(name) <= 63


def test_build_resource_name_sanitizes_unsafe_characters():
    name = build_resource_name("Acme Corp!", "My Connector", "a3f9")
    assert _VALID_NEO4J_NAME.match(name)
    assert "acme-corp" in name
    assert "my-connector" in name


def test_build_resource_name_truncates_long_inputs_without_losing_suffix_or_kind():
    long_tenant = "t" * 100
    long_connector = "c" * 100
    name = build_resource_name(long_tenant, long_connector, "a3f9", kind="entities")
    assert len(name) <= 63
    assert name.endswith("entities-a3f9")


def test_build_resource_name_two_runs_same_tenant_and_connector_differ():
    name1 = build_resource_name("acme", "confluence", generate_run_suffix())
    name2 = build_resource_name("acme", "confluence", generate_run_suffix())
    assert name1 != name2


@pytest.mark.parametrize("tenant_id,connector_type,suffix", [
    ("", "confluence", "a3f9"),
    ("acme", "", "a3f9"),
    ("acme", "confluence", ""),
])
def test_build_resource_name_rejects_empty_required_fields(tenant_id, connector_type, suffix):
    with pytest.raises(ValueError):
        build_resource_name(tenant_id, connector_type, suffix)
