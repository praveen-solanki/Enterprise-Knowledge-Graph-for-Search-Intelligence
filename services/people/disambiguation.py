"""Deciding whether two same-named people are the same person.

Connector-blind: it sees two names and two attribute strings, never a document.
Whoever fills the attribute dict decides what counts as an attribute.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from config.settings import settings
from services import llm_client
from services.clustering_utils import structured_llm_call
from services.llm_profiles import CallKind

logger = logging.getLogger("graphrag.services.people.disambiguation")

DISAMBIGUATION_SYSTEM = (
    "You disambiguate two people who share the same name in a company knowledge graph. "
    "Given a NEW mention and the attributes an EXISTING person's source records about "
    "them, decide if they are the SAME real person. The attributes vary by source and "
    "may be sparse. Respond with strict JSON: "
    '{"same_person": true or false}.'
)


def attr_text(enrichment: dict) -> str:
    """Compact structured-attribute string for the disambiguation prompt.

    Renders whatever the connector put in PersonRecord.attributes, in the order
    it put it there. It used to read exactly job_title, department, team,
    organization and manager — the intranet HR export's own spelling — so a
    source with any other vocabulary rendered to "" and the guard silently fell
    back to comparing names alone, which is the one thing it exists not to do.

    Keys ending in `_id` are skipped. An opaque identifier cannot tell an LLM
    whether two people are the same person, and identifiers that CAN settle it
    never reach here: a person their source named with a key is materialized,
    not disambiguated.
    """
    if not enrichment:
        return ""
    return "; ".join(f"{k.replace('_', ' ')}: {v}"
                     for k, v in enrichment.items()
                     if v and not k.endswith("_id"))


async def llm_confirms_same_person(new_name: str, new_context: Optional[str],
                                   cand_name: str, cand_attrs: str) -> bool:
    """Fail-closed ("not a match") on error or exhausted retries: under-merging
    is cheap to fix later, over-merging two different people is what this
    guard exists to prevent."""
    user = (f"NEW mention: {new_name} — {new_context or '(no context)'}\n"
            f"EXISTING person: {cand_name} — {cand_attrs or '(no attributes on file)'}\n"
            "Are these the same real person?")
    return await llm_client.confirm_with_retry(
        DISAMBIGUATION_SYSTEM, user, "same_person",
        step="person_disambiguation_confirm",
        context=f"new={new_name!r} candidate={cand_name!r}",
        log_label="Person disambiguation",
    )


# --------------------------------------------------------------------------
# Batched disambiguation — many (mention, candidate-chunk) questions per call.
# The one-pair path above asks the model about a single candidate at a time,
# which both serialises the corpus and under-discriminates: shown one candidate
# in isolation the model says yes to the first plausible one. Everything below
# replaces it for the corpus stage (services/person_corpus_resolver.py).
# --------------------------------------------------------------------------

# No candidate is the same person. An integer, not a nullable field, keeps the schema strict.
_NO_MATCH = -1

BATCH_DISAMBIGUATION_SYSTEM = (
    "You disambiguate people who share the same name in a company knowledge graph.\n\n"
    "Each item below is ONE new mention of a person, together with its OWN numbered list of "
    "EXISTING people who already share that name, each with their real attributes. The items "
    "are independent: judge each one only against its own candidate list, and never carry a "
    "decision from one item to another.\n\n"
    "For each item, return:\n"
    "- id: echo the item's id back unchanged.\n"
    "- match_index: the 0-based index, into THAT item's own candidate list, of the existing "
    f"person who is the SAME real human as the mention, or {_NO_MATCH} if none of them is.\n\n"
    f"Return {_NO_MATCH} unless the attributes positively identify one of them. A shared name "
    "is NOT evidence — two different people sharing a name is the exact situation you are being "
    "asked about, so it is the expected case, not the exception. Do not pick the closest "
    f"option; pick only a certain one. When in doubt, return {_NO_MATCH}.\n\n"
    'Respond with strict JSON: {"matches": [{"id": ..., "match_index": ...}]}'
)

# Candidates per ITEM. Usually 2-3 registry rows share a name, but nothing bounds it. Above
# the cap a mention splits across consecutive chunks; the first confident match wins.
# Exported: services/people/resolution.py imports this name directly.
DISAMBIGUATION_BATCH_MAX = settings.disambiguation_batch_max_candidates

# Items (mention x candidate chunk) per LLM call — the other axis. Deliberately NOT recursive
# like resolve_cluster_batched: each item is an independent 1:1 question with nothing to
# consolidate, so a second pass would re-ask the same questions forever.
# Exported: services/people/resolution.py imports this name directly.
DISAMBIGUATION_ITEMS_PER_CALL = settings.disambiguation_items_per_call


class _PersonMatch(BaseModel):
    id: int = Field(..., description="The id of the input item this verdict is for, echoed back "
                    "exactly as given.")
    match_index: int = Field(
        ..., description="The 0-based index, into THIS item's own candidate list, of the existing "
        "person who is positively the same real human as the mention — or -1 when none of them "
        "is. A shared name alone is never enough.")


class _PersonDisambiguationBatch(BaseModel):
    matches: list[_PersonMatch]


def _candidate_line(index: int, cand: dict) -> str:
    attrs = attr_text(cand.get("enrichment") or {})
    return f"[{index}] {cand.get('canonical_name')} — {attrs or '(no attributes on file)'}"


async def disambiguate_item_batch(items: list[tuple]) -> dict:
    """ONE LLM call deciding several (mention, candidate-chunk) items at once.

    items: [(key, name, context, candidates_chunk), ...] — `key` is opaque and handed back.
    Returns {key: candidate dict or None}.

    A mention's candidates are always shown side by side: asked in isolation the model
    under-discriminates and says yes to the first plausible one, and find_all_by_alias
    returns them in arbitrary Mongo order. Different MENTIONS may share a call — each item
    is fixed to its own candidate list, so they stay independent.

    Fails closed to None (mint a separate node) per item on error, unparseable response,
    missing id, or out-of-range index: over-merging is what this guard prevents."""
    payload = [
        {"id": i, "mention": name, "context": context or "(no context)",
         "candidates": [_candidate_line(j, cand) for j, cand in enumerate(chunk)]}
        for i, (_key, name, context, chunk) in enumerate(items)
    ]
    ok, parsed = await structured_llm_call(
        BATCH_DISAMBIGUATION_SYSTEM, json.dumps(payload), _PersonDisambiguationBatch,
        step="person_disambiguation_confirm",
        context=f"mentions={[name for _k, name, _c, _ch in items][:5]} items={len(items)}",
        # Own metric, so this volume stays visible in the "LLM calls=" run summary.
        calls_metric="person_disambiguation_calls",
        failures_metric="person_disambiguation_call_failures",
        kind=CallKind.DISAMBIGUATION,
    )
    by_id = {m.id: m.match_index for m in parsed.matches} if ok and parsed is not None else {}
    if not by_id:
        logger.warning("Person disambiguation returned nothing usable for %d item(s) — treating "
                       "them all as no match", len(items))

    resolved: dict = {}
    for i, (key, name, _context, chunk) in enumerate(items):
        match_index = by_id.get(i, _NO_MATCH)
        if match_index == _NO_MATCH:
            resolved[key] = None
        elif 0 <= match_index < len(chunk):
            resolved[key] = chunk[match_index]
        else:
            logger.warning("Person disambiguation returned out-of-range index %s for %r with %d "
                           "candidate(s) — treating as no match", match_index, name, len(chunk))
            resolved[key] = None
    return resolved
