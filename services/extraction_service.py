"""
Step 1 — extraction. Sends one prepared chunk to the resolved LLM backend
(Azure OpenAI or in-house Ministral) and gets back a ChunkExtractionResult
(entities, relationships, claims), validated against models/extraction.py.

Every call goes through services/llm_client.py: complete() is the one door
(Pydantic response_format, backend resolved per call) and call_with_retry()
is the one retry policy — fatal re-raises (a config-level error stalls the
run instead of silently failing thousands of chunks), permanent fails closed
immediately, transient (including a bad generation: invalid JSON,
schema-validation failure — a regenerate can succeed) consumes a bounded
retry attempt, on both backends.

On failure, returns ChunkExtractionResult(failed=True) — `failed` is excluded
from model_dump(), so a failed chunk is never persisted and is retried
automatically on the next run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Callable, Optional

from config.settings import settings
from models.extraction import (
    ChunkExtractionResult,
    ExtractedClaim,
    ExtractedEntity,
    ExtractedRelationship,
)
from services import llm_client
from services.llm_backend import get_llm_backend
from services.llm_profiles import CallKind
from utils.error_classification import LlmTruncationError

logger = logging.getLogger("graphrag.services.extraction")

_SYSTEM_PROMPT = """\
You are an enterprise knowledge-graph extraction engine.

Given a text chunk, extract every meaningful entity, every relationship between those entities, and \
every atomic factual claim that is explicitly stated or strongly implied by the text.

Entity types: NATURAL-granularity UPPER_SNAKE_CASE categories — the general bucket a person would file \
the entity under (PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, METRIC, PROCESS, ROLE, LOCATION, POLICY, \
PRODUCT, FEATURE, EVENT, DATE, …). Use a well-known type where it fits or introduce a new one; do NOT \
invent hyper-specific compound types.

Relationship types: NATURAL-granularity UPPER_SNAKE_CASE verbs (MANAGES, PART_OF, DEPENDS_ON, \
INTEGRATES_WITH, REPLACES, ADDS_FEATURE, HOSTED_AT, …) — precise but not one-off.

Descriptions: write a rich 2-3 sentence profile for each entity (what it is, what it does, notable \
attributes) and a 1-2 sentence explanation for each relationship, grounded in the chunk.

Claims: extract atomic (subject, predicate, object) assertions the chunk supports. Each carries \
statement_type (FACT | OPINION | PREDICTION) and temporal_type (ATEMPORAL | STATIC | DYNAMIC).

Dates: temporal information lives ONLY on claims (NEVER on entities or relationships), and within a \
claim it goes in the STRUCTURED fields valid_from / valid_to — NEVER in the object. The object is the \
WHAT; the date is WHEN.
- Put any date saying WHEN a fact holds, started, became true, was released, or occurred into \
  valid_from (ISO-8601 YYYY-MM-DD). Put the end of a time-bounded fact into valid_to.
- NEVER make a date the object of a claim. If the only thing you'd put in the object is a date, the \
  claim is mis-modelled: move the date to valid_from and put the real value/state in the object.
- temporal_type = the fact's INHERENT temporal nature, decided from MEANING, not from whether a date is \
  present: ATEMPORAL = a timeless/definitional truth not tied to any event or release (a permanent \
  capability or category); STATIC = a one-time event or a state reached/holding from a point in time \
  (released, created, reached, added, a completed action, a stable current state) — use STATIC even with \
  no date (set valid_from if one is given); DYNAMIC = a fact that genuinely changes/varies over time or \
  recurs (grows, updates regularly, gradual rollout). Decide STATIC vs DYNAMIC by fixed point-event/state \
  (STATIC) vs actively varies over a period (DYNAMIC). Do NOT default to ATEMPORAL. temporal_type is NEVER \
  a statement_type value.
- statement_type: FACT = objective/verifiable; OPINION = subjective judgment or recommendation; \
  PREDICTION = future/planned/expected (will, plan to, upcoming, approaching GA) — future-tense or \
  planned actions are PREDICTION, not FACT.
- Normalise every date to ISO-8601 against the document's publish date shown at the top of the message. \
  Leave valid_from / valid_to null when the text gives no date — never guess.
- Example — "Policy Center became generally available on 11.11.2024":
    GOOD -> subject="Policy Center", predicate="reached_status", object="Generally Available", \
            temporal_type="STATIC", valid_from="2024-11-11"
    BAD  -> subject="Policy Center", predicate="has_GA_date", object="2024-11-11"  (date in object)

Rules:
- Names are the entity's NATURAL surface form as written (e.g. "Policy Center", "Google Analytics \
4"), NOT UPPER_SNAKE_CASE — UPPER_SNAKE_CASE is ONLY for the `type` and `relationship_type` fields. \
Never put underscores in a name unless the source text literally uses them.
- Canonical names: normalise near-duplicates to one name (e.g. "Contoso" not "Contoso Inc.")
- One entity = one real thing. NEVER concatenate multiple items into a single entity (no "A + B", no \
"A and B" merged into one name).
- Structured / tabular text: do NOT extract column headers, row labels, table captions, form or field \
labels, or UI boilerplate as entities (e.g. skip a header like "Releases Version GA Date Release \
Notes"). Do NOT invent a relationship or claim from an empty or placeholder cell ("-", "", "N/A", \
"TBD") — only extract what the text actually asserts.
- relationship_type: reuse a small, general verb vocabulary; prefer a general verb over a \
hyper-specific one-off (use ADDS_FEATURE, not ADDS_BRANDED_HYPERLINK_COLOR_SUPPORT).
- Do NOT extract generic stop-words or meaningless fragments
- An entity is a SPECIFIC named thing (a product, feature, team, org, person, named concept). Do NOT make \
an entity out of a bare value or metadata label: a date, a clock time, a version/package code (e.g. \
"23.85.3"), a URL, or a status/column label (e.g. "General Availability", "Package Version", "Release \
Notes"). Dates/times belong on claims (valid_from), not as entities.
- Every relationship's source_entity and target_entity MUST exactly match a name in entities
- Be comprehensive — missing an entity is worse than including a borderline one
"""

# Second-pass "gleaning" prompt: recovers entities/relationships/claims missed on the first
# pass (~2x LLM calls for this step). Toggle: settings.extraction_gleaning_enabled.
_GLEAN_PROMPT = """\
You already extracted from this passage once. Given the passage and the list of entity names already \
found, extract ONLY the entities, relationships, and claims that were MISSED the first time — no repeats. \
Be exhaustive, especially about relationships between already-known entities. Use the same typing, \
description, and claim rules (dates only on claims, never on entities or relationships). Return empty \
lists if nothing was missed.
"""

EXTRACTION_VERSION = hashlib.sha1(
    f"{_SYSTEM_PROMPT}|{_GLEAN_PROMPT}|{settings.azure_openai_completion_model}".encode("utf-8")
).hexdigest()[:12]


def _drop_dangling_relationships(result: ChunkExtractionResult) -> ChunkExtractionResult:
    names = {e.name for e in result.entities}
    result.relationships = [
        r for r in result.relationships
        if r.source_entity in names and r.target_entity in names
    ]
    return result


def _merge_extraction_results(base: ChunkExtractionResult,
                              extra: ChunkExtractionResult) -> ChunkExtractionResult:
    """Fold `extra` into `base` with no repeats and no dangling relationship
    endpoints — entities dedupe by name, relationships by (source, target,
    type), claims by (subject, predicate, object). One merge rule shared by
    the gleaning pass and split-on-truncation, so the two can never drift."""
    have_names = {e.name for e in base.entities}
    new_entities = [e for e in extra.entities if e.name not in have_names]
    all_names = have_names | {e.name for e in new_entities}

    have_rels = {(r.source_entity, r.target_entity, r.relationship_type) for r in base.relationships}
    new_relationships = [
        r for r in extra.relationships
        if (r.source_entity, r.target_entity, r.relationship_type) not in have_rels
        and r.source_entity in all_names and r.target_entity in all_names
    ]

    have_claims = {(c.subject, c.predicate, c.object) for c in base.claims}
    new_claims = [c for c in extra.claims if (c.subject, c.predicate, c.object) not in have_claims]

    return ChunkExtractionResult(
        entities=base.entities + new_entities,
        relationships=base.relationships + new_relationships,
        claims=base.claims + new_claims,
    )


def _repair_truncated_json(partial: str) -> Optional[dict]:
    """Cut a truncated ChunkExtractionResult JSON back to its last COMPLETE
    array element and close the open brackets, so everything the model
    finished before hitting the ceiling is recovered instead of re-paid.

    A tiny state machine (string/escape-aware bracket stack) finds the last
    '}' that closed a direct child of one of the root object's arrays — i.e.
    a whole entity/relationship/claim — discards the mid-flight tail after
    it, and appends ']}' to close the array and the root. Returns the parsed
    dict, or None when the stump has no complete element to salvage (cut
    inside the very first object) or is malformed beyond this repair."""
    stack: list[str] = []
    in_string = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(partial):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None  # more closers than openers — not our truncation shape
            stack.pop()
            # Back to [root object, array] right after a '}' == one whole
            # entity/relationship/claim just completed.
            if ch == "}" and stack == ["{", "["]:
                last_complete = i
    if last_complete < 0:
        return None
    try:
        return json.loads(partial[: last_complete + 1] + "]}")
    except ValueError:
        return None


def _salvage_extraction(partial: Optional[str]) -> ChunkExtractionResult:
    """Everything usable from a truncated extraction response: repair the
    JSON, validate each item INDIVIDUALLY (one mangled item must not void its
    complete siblings), drop relationships left dangling by the cut. Returns
    an empty result when nothing is salvageable — never raises."""
    if not partial:
        return ChunkExtractionResult()
    data = _repair_truncated_json(partial)
    if not isinstance(data, dict):
        return ChunkExtractionResult()

    def _validated(key: str, model) -> list:
        items = []
        for item in data.get(key) or []:
            if not isinstance(item, dict):
                continue
            try:
                items.append(model(**item))
            except Exception:  # noqa: BLE001 — a cut-mangled item is dropped, not fatal
                continue
        return items

    return _drop_dangling_relationships(ChunkExtractionResult(
        entities=_validated("entities", ExtractedEntity),
        relationships=_validated("relationships", ExtractedRelationship),
        claims=_validated("claims", ExtractedClaim),
    ))


class ExtractionService:
    async def extract_from_chunk(self, prepared_text: str, chunk_id: str = "") -> ChunkExtractionResult:
        """Base extraction plus an optional gleaning pass for missed entities
        (settings.extraction_gleaning_enabled). Gleaning is best-effort — any
        failure falls back to the base result."""
        base = await self._extract_base(prepared_text, chunk_id=chunk_id)
        if not settings.extraction_gleaning_enabled or base.failed or not base.entities:
            return base
        try:
            return await self._glean_pass(prepared_text, base, chunk_id=chunk_id)
        except Exception:  # noqa: BLE001 — gleaning is a bonus pass, not load-bearing
            logger.exception("chunk %s: gleaning pass failed — keeping the base result", chunk_id)
            return base

    async def _structured_call(
        self,
        *,
        system_prompt: str,
        user_content: str,
        chunk_id: str,
        kind: CallKind,
        metric_name: str,
    ) -> tuple[Optional[ChunkExtractionResult], Optional[str]]:
        """One structured extraction/gleaning call under the shared llm_client
        retry policy: fatal re-raises (stall the run), permanent fails closed
        immediately, transients (bad generations included) consume attempts.

        Returns (result, truncated_partial). result is None once failed
        closed — the caller decides what that means (base extraction marks
        the chunk failed=True; gleaning keeps base). truncated_partial is the
        raw partial output when the failure was a context-window overflow
        (classified permanent — deterministic at temperature 0 — so it costs
        one attempt, not three): the one failure the caller can actually FIX,
        by salvaging the partial and asking a follow-up call for the rest."""
        truncated_partial: Optional[str] = None

        async def _attempt() -> ChunkExtractionResult:
            nonlocal truncated_partial
            try:
                result = await llm_client.complete(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=ChunkExtractionResult,
                    temperature=settings.azure_llm_temperature,
                    kind=kind,
                )
            except LlmTruncationError as exc:
                truncated_partial = exc.partial_content
                raise
            # NOT filtered for dangling relationships here: for the gleaning/
            # continuation shape a relationship's endpoints legitimately live
            # in the BASE result's entity list ("be exhaustive, especially
            # about relationships between already-known entities" — the
            # prompt's own words), which this call can't see. The base path
            # filters in _extract_base; the merge validates against the union
            # of names (_merge_extraction_results).
            return result

        result = await llm_client.call_with_retry(
            _attempt, call_name=metric_name.removesuffix("_calls"), context=chunk_id,
            fail_closed=lambda: None, kind=kind,
            metric_calls=metric_name,
            metric_failures="extraction_failures" if metric_name == "extraction_calls" else None,
        )
        return result, truncated_partial

    async def _glean_pass(self, prepared_text: str, base: ChunkExtractionResult,
                          chunk_id: str = "",
                          pass_label: str = "gleaning pass") -> ChunkExtractionResult:
        """One "what did we miss" call, merged into `base` with no repeats and no
        dangling relationship endpoints. Failures are caught by the caller, which
        falls back to `base`.

        pass_label names this pass in the logs — "gleaning pass" for the normal
        bonus sweep, or the truncation-recovery follow-up, which reuses this
        exact call shape (full chunk + already-found names) as its step 2."""
        names = sorted({e.name for e in base.entities})
        user = (f"{prepared_text}\n\nAlready extracted entities: {', '.join(names)}\n"
               "Extract ONLY entities, relationships, and claims that were MISSED above.")

        gleaned, _truncated = await self._structured_call(
            system_prompt=_GLEAN_PROMPT,
            user_content=user,
            chunk_id=chunk_id,
            kind=CallKind.EXTRACTION_GLEANING,
            metric_name="gleaning_calls",
        )
        if gleaned is None:
            # A truncated or failed second pass keeps the base result — this
            # pass only ever adds, never invalidates what the base found.
            logger.info("chunk %s: %s produced nothing usable — keeping the prior result "
                        "(%d entities, %d relationships, %d claims)",
                        chunk_id, pass_label, len(base.entities), len(base.relationships),
                        len(base.claims))
            return base

        merged = _merge_extraction_results(base, gleaned)
        recovered = (len(merged.entities) - len(base.entities),
                     len(merged.relationships) - len(base.relationships),
                     len(merged.claims) - len(base.claims))
        logger.info("chunk %s: %s added +%d entities, +%d relationships, +%d claims "
                    "-> total %d entities, %d relationships, %d claims",
                    chunk_id, pass_label, *recovered,
                    len(merged.entities), len(merged.relationships), len(merged.claims))
        return merged

    async def _extract_base(self, prepared_text: str, chunk_id: str = "") -> ChunkExtractionResult:
        """The single base extraction call. Retries on rate limits with adaptive
        backoff. On failure, returns ChunkExtractionResult(failed=True) — never cached.

        Truncation recovery (salvage + continuation): a chunk whose extraction
        output overflows the model's context window fails DETERMINISTICALLY —
        temperature 0 means every retry and every --retry-failed run truncates
        at the identical spot, so without intervention that chunk is a
        permanent hole in the graph. Verbatim ChatGPT-style continuation can't
        work either: the window is already full, so replaying the partial
        output leaves no room to continue. Instead, the partial output's
        COMPLETE objects are salvaged by repair-parsing (only the one object
        mid-flight at the cut is lost), and the model is asked for what's
        missing via the gleaning shape — full chunk + already-found entity
        names (~a hundred tokens of state instead of thousands), one
        schema-enforced follow-up call. When the normal gleaning pass is
        enabled it IS that follow-up (extract_from_chunk runs it on whatever
        this returns); when disabled, one explicit top-up runs here, because
        for a truncated chunk the second pass is recovery, not a bonus."""
        if not prepared_text:
            return ChunkExtractionResult()

        result, truncated_partial = await self._structured_call(
            system_prompt=_SYSTEM_PROMPT,
            user_content=prepared_text,
            chunk_id=chunk_id,
            kind=CallKind.EXTRACTION,
            metric_name="extraction_calls",
        )
        if result is not None:
            result = _drop_dangling_relationships(result)
            logger.info("chunk %s: base extraction: %d entities, %d relationships, %d claims",
                        chunk_id, len(result.entities), len(result.relationships), len(result.claims))
            return result

        if truncated_partial is None:
            return ChunkExtractionResult(failed=True)

        salvaged = _salvage_extraction(truncated_partial)
        if not salvaged.entities:
            logger.warning("chunk %s: truncation recovery ABANDONED — output overflowed the "
                           "context window and nothing was salvageable from the partial "
                           "(%d chars, cut inside the first object) — failing the chunk",
                           chunk_id, len(truncated_partial))
            return ChunkExtractionResult(failed=True)

        logger.warning("chunk %s: truncation recovery step 1/2 (salvage): output overflowed "
                       "the context window; repair-parsed %d entities, %d relationships, "
                       "%d claims out of the partial output",
                       chunk_id, len(salvaged.entities), len(salvaged.relationships),
                       len(salvaged.claims))
        if settings.extraction_gleaning_enabled:
            # extract_from_chunk's own gleaning pass runs next on this result —
            # that call IS the continuation (full chunk + salvaged names).
            logger.info("chunk %s: truncation recovery step 2/2 (follow-up) runs as the "
                        "gleaning pass next — full chunk + %d salvaged entity names",
                        chunk_id, len(salvaged.entities))
            return salvaged
        try:
            return await self._glean_pass(
                prepared_text, salvaged, chunk_id=chunk_id,
                pass_label="truncation recovery step 2/2 (follow-up)")
        except Exception:  # noqa: BLE001 — salvage is already better than a failed chunk
            logger.exception("chunk %s: truncation recovery step 2/2 (follow-up) failed — "
                             "keeping step 1's salvaged result", chunk_id)
            return salvaged

    async def _extract_tagged(self, text: str, chunk_id: str) -> tuple[str, ChunkExtractionResult]:
        result = await self.extract_from_chunk(text, chunk_id=chunk_id)
        return chunk_id, result

    async def extract_chunks_batch(
        self,
        prepared_texts: list[str],
        chunk_ids: list[str],
        on_result: Optional[Callable] = None,
    ) -> list[ChunkExtractionResult]:
        """Extract many chunks concurrently. Concurrency is self-tuning via
        AdaptiveThrottle — no fixed limit to configure.

        on_result(chunk_id, result) is called (and awaited if async) the moment
        each chunk finishes — making checkpointing truly incremental.
        """
        throttle = llm_client.get_throttle_for(CallKind.EXTRACTION)
        logger.info("extract_chunks_batch: %d chunk(s), starting concurrency=%d (will auto-tune)",
                    len(prepared_texts), throttle.current_concurrency)
        tasks = [self._extract_tagged(text, cid) for text, cid in zip(prepared_texts, chunk_ids)]

        results: dict = {}
        for coro in asyncio.as_completed(tasks):
            cid, result = await coro
            results[cid] = result
            if on_result:
                import inspect
                if inspect.iscoroutinefunction(on_result):
                    await on_result(cid, result)
                else:
                    on_result(cid, result)

        ordered = [results[cid] for cid in chunk_ids]
        total_entities = sum(len(r.entities) for r in ordered)
        failed_count = sum(1 for r in ordered if r.failed)
        if get_llm_backend() == "ministral":
            llm_client.log_ministral_stats()
        else:
            throttle.log_stats(label=CallKind.EXTRACTION.value)
        logger.info("extract_chunks_batch done: %d chunks → %d entities | %d failed (will retry)",
                    len(ordered), total_entities, failed_count)
        return ordered
