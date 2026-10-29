"""
Step 1 output shapes. What the LLM hands back for one chunk: entities, the
relationships between them, and atomic factual claims. Nothing here talks to
an LLM or ES — this file only defines the shape of the data.

Field descriptions below are a steering surface for the model (Structured
Output reads them), not just documentation for humans — the nano model in
particular benefits from the full temporal decision procedure and
statement-type rules living here rather than only in the system prompt.
"""
from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

# LLM output occasionally contains stray control characters (encoding artifacts) —
# strip them so a bad name/description never breaks a downstream Neo4j write.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_control(v: Optional[str]) -> Optional[str]:
    return _CONTROL_CHARS.sub("", v) if v is not None else None


# type / relationship_type are documented to the model as UPPER_SNAKE_CASE, and
# the bigger models comply, but a small one mixes casings inside a single run:
# one real run produced ORGANIZATION and PRODUCT alongside "system_name",
# "domain_name" and "programming language" — the same category reachable under
# two spellings, so a type filter matches one and misses the other (including
# settings.contributor_entity_types, which is matched as UPPER_SNAKE_CASE).
# Normalising here makes the graph's type vocabulary consistent no matter which
# model produced it. Names are deliberately NOT normalised (they are surface
# forms and must stay as written).
_TYPE_SEPARATORS = re.compile(r"[\s\-/]+")
_TYPE_UNDERSCORES = re.compile(r"_{2,}")


def _upper_snake(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str):
        return v
    cleaned = _TYPE_UNDERSCORES.sub("_", _TYPE_SEPARATORS.sub("_", v.strip())).strip("_")
    return cleaned.upper() or None


# The model emits the literal string "null", out-of-range years (1901, 1978), and
# far-future placeholders for date fields — without cleaning these, they slip into
# claim properties (and any date-based index/sort) where "null" sorts after every
# real date. Coerces to a clean ISO YYYY-MM-DD within a sane window, else None.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _clean_date(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str):
        return None
    m = _ISO_DATE.match(v.strip())
    if not m:
        return None
    d = m.group(0)
    y, mo, dy = int(d[:4]), int(d[5:7]), int(d[8:10])
    if 1990 <= y <= 2027 and 1 <= mo <= 12 and 1 <= dy <= 31:
        return d
    return None


class ExtractedEntity(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Canonical name in its NATURAL surface form (e.g. 'Policy Center', 'Google Analytics 4', "
            "'Contoso' not 'Contoso Inc.'). NOT UPPER_SNAKE_CASE — that is only for the type field; no "
            "underscores unless the source text uses them. One real thing per entity, never 'A + B'."
        ),
    )
    type: str = Field(
        ...,
        description=(
            "Entity type in UPPER_SNAKE_CASE at NATURAL granularity — the general category a person "
            "would file it under (PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, METRIC, PROCESS, ROLE, "
            "LOCATION, POLICY, PRODUCT, FEATURE, EVENT, DATE, …). Use a well-known type where it fits "
            "or introduce a new one; do NOT invent hyper-specific compound types."
        ),
    )
    description: Optional[str] = Field(
        None,
        description=(
            "A rich 2-3 sentence profile of this entity grounded in the chunk: what it is, what it does "
            "or is used for, and any notable attribute the text supports."
        ),
    )

    # No temporal attributes on entities by design: all dates/timestamps/validity live on
    # Claims, so the entity layer stays the static graph structure — see ExtractedClaim.

    @field_validator("name", "type", "description", mode="after")
    @classmethod
    def _clean(cls, v):
        return _strip_control(v)

    @field_validator("type", mode="after")
    @classmethod
    def _normalize_type(cls, v):
        return _upper_snake(v)


class ExtractedRelationship(BaseModel):
    source_entity: str = Field(..., description="Exact name of the source entity (must match an extracted entity name).")
    target_entity: str = Field(..., description="Exact name of the target entity (must match an extracted entity name).")
    relationship_type: str = Field(
        ...,
        description=(
            "Relationship label in UPPER_SNAKE_CASE at NATURAL granularity — the general relation a "
            "person would name (INTEGRATES_WITH, DEPENDS_ON, PART_OF, MANAGES, REPLACES, ADDS_FEATURE, …). "
            "Use a precise verb or define a new one; avoid hyper-specific one-off verbs."
        ),
    )
    description: Optional[str] = Field(
        None, description="A 1-2 sentence explanation of how source and target are related, grounded in the chunk.",
    )

    # No temporal attributes on relationships by design: when a relationship/fact is
    # time-bound, that's expressed as a Claim instead, keeping the edge layer static.

    @field_validator("source_entity", "target_entity", "relationship_type", "description", mode="after")
    @classmethod
    def _clean(cls, v):
        return _strip_control(v)

    @field_validator("relationship_type", mode="after")
    @classmethod
    def _normalize_type(cls, v):
        return _upper_snake(v)


class ExtractedClaim(BaseModel):
    """An atomic, optionally-dated factual claim — subject/predicate/object with statement and temporal typing."""
    subject: str = Field(..., description="The entity the claim is about (should match an extracted entity name).")
    predicate: str = Field(
        ...,
        description=(
            "The asserted relation or property, concise (e.g. 'reached_status', 'supports', "
            "'integrates_with'). Do NOT encode a date in the predicate, and avoid predicates that exist "
            "only to attach a date (e.g. 'has_GA_date') — express the fact and put the date in valid_from."
        ),
    )
    object: str = Field(
        ...,
        description=(
            "The value, target, or state asserted — the WHAT. NEVER a date: any 'when' date belongs in "
            "valid_from / valid_to, not here. (e.g. object='Generally Available', not object='2024-11-11')."
        ),
    )
    statement_type: Literal["FACT", "OPINION", "PREDICTION"] = Field(
        ...,
        description=(
            "Required. Epistemic status (not a time): FACT = an objective, verifiable assertion about what "
            "is/was the case; OPINION = a subjective judgment, evaluation, or recommendation; PREDICTION = a "
            "future, planned, or expected outcome (will / plan to / upcoming / approaching GA). Future-tense "
            "or planned actions are PREDICTION, not FACT."
        ),
    )
    temporal_type: Literal["ATEMPORAL", "STATIC", "DYNAMIC"] = Field(
        ...,
        description=(
            "Required, the fact's INHERENT temporal nature, decided from MEANING not from whether a date is "
            "present. ATEMPORAL = a timeless/definitional truth not tied to any event or release (a "
            "permanent capability or category). STATIC = a one-time event or a state reached/holding from a "
            "point in time (released, created, reached, added, a completed action, a stable current state); "
            "use STATIC even with no date (set valid_from if one is given). DYNAMIC = a fact that genuinely "
            "changes/varies over time or recurs (grows, updates regularly, gradual rollout). Decide STATIC "
            "vs DYNAMIC by fixed point-event/state vs actively varies over a period. Do NOT default to "
            "ATEMPORAL."
        ),
    )
    valid_from: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 (YYYY-MM-DD) date the fact starts being true / the event occurred. Put ANY 'when' "
            "date HERE — never in object or predicate. Normalize to the document publish date. Null only "
            "when the text gives no time."
        ),
    )
    valid_to: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 (YYYY-MM-DD) date the fact stops being true (the end of a DYNAMIC interval). "
            "Null if the fact is point-in-time (STATIC) or open-ended."
        ),
    )

    @field_validator("subject", "predicate", "object", mode="after")
    @classmethod
    def _clean(cls, v):
        return _strip_control(v)

    @field_validator("valid_from", "valid_to", mode="after")
    @classmethod
    def _date(cls, v):
        return _clean_date(v)


class ChunkExtractionResult(BaseModel):
    """Everything extracted from one text chunk.

    failed=True means the extraction call errored out (rate-limit exhaustion,
    timeout, content-filter rejection, etc.). A failed result is intentionally
    NOT saved to the checkpoint cache — the next run will retry the chunk from
    scratch. An empty result with failed=False means the chunk genuinely had
    nothing to extract (e.g. a boilerplate footer) and IS cached so we never
    re-pay for a known-empty chunk. Never conflate the two (N2).
    """
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    claims: List[ExtractedClaim] = Field(default_factory=list)
    # exclude=True keeps it out of model_dump() (never persisted to Mongo);
    # SkipJsonSchema keeps it out of the LLM-facing schema too -- exclude=True
    # alone does NOT do that, so the model was seeing and setting this
    # internal-only flag itself, discarding otherwise-good extractions.
    failed: SkipJsonSchema[bool] = Field(default=False, exclude=True)
