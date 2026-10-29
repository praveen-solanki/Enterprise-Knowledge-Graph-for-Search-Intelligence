"""Unit tests for models/extraction.py's date cleaning (experiment-parity
review C4). The model emits the literal string "null", out-of-range years,
and far-future placeholders for date fields -- without cleaning, these slip
into claim properties where "null" sorts after every real date."""
from models.extraction import ChunkExtractionResult, ExtractedClaim, _clean_date


def test_failed_field_is_invisible_to_the_llm_facing_schema():
    # exclude=True alone only hides `failed` from model_dump(), not from the
    # LLM-facing schema -- without SkipJsonSchema, the model could set it itself.
    schema = ChunkExtractionResult.model_json_schema()
    assert "failed" not in schema.get("properties", {})


def test_failed_field_still_settable_from_python():
    assert ChunkExtractionResult(failed=True).failed is True
    assert ChunkExtractionResult().failed is False


def _claim(**overrides) -> ExtractedClaim:
    fields = {
        "subject": "Recognition Hub", "predicate": "reached_status", "object": "Generally Available",
        "statement_type": "FACT", "temporal_type": "STATIC", "valid_from": None, "valid_to": None,
    }
    fields.update(overrides)
    return ExtractedClaim(**fields)


def test_clean_date_accepts_a_real_iso_date():
    assert _clean_date("2026-07-16") == "2026-07-16"


def test_clean_date_rejects_the_literal_string_null():
    assert _clean_date("null") is None


def test_clean_date_rejects_out_of_range_year():
    assert _clean_date("1901-01-01") is None
    assert _clean_date("1978-06-15") is None


def test_clean_date_rejects_far_future_year():
    assert _clean_date("2099-01-01") is None


def test_clean_date_rejects_non_string():
    assert _clean_date(None) is None
    assert _clean_date(12345) is None


def test_clean_date_accepts_date_with_trailing_time_component():
    assert _clean_date("2026-07-16T00:00:00Z") == "2026-07-16"


def test_claim_valid_from_is_cleaned_on_construction():
    claim = _claim(valid_from="null")
    assert claim.valid_from is None


def test_claim_valid_to_is_cleaned_on_construction():
    claim = _claim(valid_to="1901-01-01")
    assert claim.valid_to is None


def test_claim_real_dates_survive_construction():
    claim = _claim(valid_from="2024-11-11", valid_to="2025-01-01")
    assert claim.valid_from == "2024-11-11"
    assert claim.valid_to == "2025-01-01"
