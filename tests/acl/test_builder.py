"""Tests for services.acl.builder (connector pass-through)."""
from services.acl import (
    AclContext,
    DocumentAclBuilder,
    build_document_acl,
)
from services.acl.builder import ConnectorStrategy


def test_connector_reads_gate2_from_object_type_metadata():
    """Prefer object_type_metadata.allow_access_control over a top-level gate object."""
    document = {
        "allow_access_control": {"audience_filter_type": "inclusion", "audiences": ["aud-1"]},
        "object_type_metadata": {
            "allow_access_control": ["group_id:g1", "account_id:acc1", "access:anonymous"],
        },
    }
    decision = build_document_acl(document, AclContext(connector_type="confluence"))
    assert not decision.dropped
    assert decision.tokens == ["access:anonymous", "account_id:acc1", "group_id:g1"]


def test_connector_bulk_reads_raw_allow_access_control():
    """Bulk connector docs read tokens from _allow_access_control."""
    document = {"_allow_access_control": ["group_id:g2", "account_id:acc2"]}
    decision = build_document_acl(document, AclContext(connector_type="confluence"))
    assert decision.tokens == ["account_id:acc2", "group_id:g2"]


def test_connector_dropped_when_no_document_tokens():
    document = {"object_type_metadata": {"allow_access_control": []}}
    decision = build_document_acl(document, AclContext(connector_type="confluence"))
    assert decision.dropped and decision.tokens is None


def test_connector_ignores_top_level_gate_object():
    """Top-level audience-gate dict is not treated as tokens (doc dropped)."""
    document = {"allow_access_control": {"audience_filter_type": "everyone"}}
    decision = build_document_acl(document, AclContext(connector_type="confluence"))
    assert decision.dropped


def test_new_connector_uses_default_passthrough_without_registration():
    """Unregistered sources use connector pass-through by default."""
    document = {"object_type_metadata": {"allow_access_control": ["slack:channel:C1"]}}
    decision = build_document_acl(document, AclContext(connector_type="slack"))
    assert decision.tokens == ["slack:channel:C1"]


def test_register_custom_strategy_for_a_source():
    class FixedStrategy:
        def build(self, document, context):
            from services.acl import AclDecision
            return AclDecision(["custom:token"], False, "custom")

    try:
        DocumentAclBuilder.register(["widget"], FixedStrategy())
        decision = build_document_acl({}, AclContext(connector_type="widget"))
        assert decision.tokens == ["custom:token"]
    finally:
        DocumentAclBuilder._registry.pop("widget", None)


def test_empty_source_defaults_to_wikipedia():
    import connectors.registry  # noqa: F401 — registers the wikipedia strategy
    assert type(DocumentAclBuilder.strategy_for("")) is type(
        DocumentAclBuilder.strategy_for("wikipedia"))
    assert type(DocumentAclBuilder.strategy_for(None)) is type(
        DocumentAclBuilder.strategy_for("wikipedia"))


def test_unknown_source_defaults_to_connector():
    assert isinstance(DocumentAclBuilder.strategy_for("google_drive"), ConnectorStrategy)


def test_tokens_are_cleaned_sorted_deduped():
    document = {
        "object_type_metadata": {"allow_access_control": ["  b ", "a", "a", "", None]},
    }
    decision = build_document_acl(document, AclContext(connector_type="confluence"))
    assert decision.tokens == ["a", "b"]
