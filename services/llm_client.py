"""The one place this repo talks to an LLM.

Two OpenAI-compatible backends, resolved per call via get_llm_backend():
Azure OpenAI (AsyncAzureOpenAI) and the in-house Ministral gateway
(AsyncOpenAI with base_url=MINISTRAL_URL minus the /chat/completions
suffix). One client instance per backend, built once per process.

Three layers, strictly stacked:

  complete()          — ONE door for every LLM call. Resolves the backend,
                        takes that backend's own throttle slot, makes exactly
                        one attempt, records success/failure/rate-limit on
                        the throttle that was actually used, runs the
                        honesty checks, and RAISES on any failure.
  call_with_retry()   — ONE retry loop around an attempt() closure.
                        fatal -> re-raise (stall the run), permanent -> fail
                        closed immediately, transient (bad generations and
                        rate limits included) -> bounded retry with backoff,
                        failing closed on exhaustion. Never touches a
                        throttle; the door already did.
  confirm_with_retry()— convenience for yes/no confirms, built on the two
                        layers above. Fail-closed value is False (no-match).

Structured output uses the SDK's Pydantic response_format on both backends —
ChunkExtractionResult.model_json_schema() works on the current Ministral
deployment (verified 2026-08-04 against the dev gateway).

Throttles are per-backend and deliberately different species: Azure's shared
AdaptiveThrottle reacts to 429s and response headers, while the
LatencyAdaptiveThrottle for Ministral reacts to observed latency against a
target (a vLLM box signals overload by slowing down, not by 429ing). A
RateLimitError is recorded against the backend that actually raised it —
never cross-charged.

Azure gets one throttle per (backend, kind) — quota is per-deployment and a
slow kind must not shrink a fast kind's concurrency. Ministral gets ONE
throttle for the whole backend, shared by every kind: the pod is a single
GPU with one measured capacity knee (~7 concurrent — see the constant-VU
load-test numbers on settings.ministral_max_concurrency), and per-kind pools
stack past it — a real run (x9j2) had the extraction pool at 5 and the
gleaning pool grown to 6, 11 in-flight against a 7-request knee.

The Ministral throttle only backs off on OVERLOAD signals (timeout,
connection error, 5xx, 429). A content failure — truncated output, schema
validation, bad JSON — says nothing about pod load, and halving on it lets
one doomed chunk throttle the whole healthy pipeline (x9j2: one
deterministically-truncating chunk halved extraction 5 -> 2 -> 1)."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar, Union

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel

from config.settings import settings
from repository.mongo_repository import record_call_failure
from services.adaptive_controller import AdaptiveThrottle, LatencyAdaptiveThrottle
from services.llm_backend import effective_run_config
from services.llm_profiles import PROFILES, CallKind
from services.run_metrics import record
from utils.error_classification import LlmTruncationError, classify_llm_retry_error

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

logger = logging.getLogger("graphrag.services.llm_client")

_azure_client: Optional[AsyncAzureOpenAI] = None
_in_house_client: Optional[AsyncOpenAI] = None

_SLOW_CALL_WARN_SECONDS = settings.llm_slow_call_warn_seconds

# Azure: one throttle per (backend, kind) — quota is per-deployment, and a
# slow kind (extraction) must not shrink a fast kind's (confirm) concurrency.
# Ministral: ONE throttle for the whole backend (key (backend, None)) — the
# pod has a single measured capacity knee, and separate per-kind pools stack
# past it whenever two stages overlap (see module docstring).
_throttles: dict[tuple[str, Optional[CallKind]], "AdaptiveThrottle | LatencyAdaptiveThrottle"] = {}


def get_throttle_for(kind: CallKind, backend: Optional[str] = None):
    """The one throttle instance for this (backend, kind) — which for
    Ministral is the backend's single shared pool, whatever `kind` is.
    backend defaults to this process's resolved backend when not given, so
    external callers (e.g. extraction_service.py's own logging) don't need
    to resolve it themselves."""
    backend = backend or effective_run_config().llm_backend
    if backend == "ministral":
        key = (backend, None)
        if key not in _throttles:
            # Target latency comes from EXTRACTION — the slowest kind the pool
            # serves. A faster kind's call simply counts as a fast success;
            # using a fast kind's target would make every extraction look
            # "slow" and freeze the pool's growth at its floor.
            _throttles[key] = LatencyAdaptiveThrottle(
                initial=settings.ministral_max_concurrency,
                target_latency_seconds=PROFILES[CallKind.EXTRACTION].target_latency_seconds,
            )
        return _throttles[key]
    key = (backend, kind)
    if key not in _throttles:
        _throttles[key] = AdaptiveThrottle(initial=settings.azure_throttle_initial_concurrency)
    return _throttles[key]


def _ministral_base_url() -> str:
    url = settings.ministral_url.rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url


def get_client() -> AsyncAzureOpenAI:
    """Azure deployment client — for call sites that only ever hit Azure."""
    global _azure_client
    if _azure_client is None:
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")
        _azure_client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=settings.azure_openai_request_timeout,
        )
    return _azure_client


def get_in_house_client() -> AsyncOpenAI:
    """OpenAI-compatible client for the in-house Ministral model-orchestrator.

    Ministral gets its OWN timeout, not Azure's 120s: extraction p90 was
    measured at 313s on real prod chunks, and a client that gives up while
    the pod keeps generating burns the pod's full generation for zero output
    — the exact goodput-collapse mechanism the load test documented."""
    global _in_house_client
    if _in_house_client is None:
        _in_house_client = AsyncOpenAI(
            base_url=_ministral_base_url(),
            api_key=settings.ministral_api_key,
            timeout=settings.ministral_request_timeout,
            # call_with_retry already retries. The SDK's own retries (default 2)
            # sit silent for the full request timeout on each one, so one dropped
            # connection costs three timeouts before this process even hears about it.
            max_retries=0,
        )
    return _in_house_client


def log_ministral_stats() -> None:
    """End-of-run summary for the shared Ministral throttle, if it was used."""
    for (backend, kind), throttle in _throttles.items():
        if backend == "ministral":
            throttle.log_stats(label=kind.value if kind is not None else "shared")


async def complete(
    *,
    messages: list[dict],
    kind: CallKind,
    response_format: Optional[type[ModelT]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Union[ModelT, dict]:
    """One LLM call on the resolved backend. The only function that touches a
    client or a throttle.

    kind identifies which call shape this is (services/llm_profiles.py) —
    every kind gets its own throttle, so a slow kind never shrinks a fast
    kind's concurrency.

    response_format is a Pydantic model class for a schema-constrained
    structured call (returns the validated model instance), or None for a
    plain json_object completion (returns the parsed dict). Raises on every
    failure — truncation, None/empty output, transport errors, rate limits —
    so call_with_retry() can classify; it never retries and never returns a
    sentinel.

    max_tokens is translated to the wire parameter each backend accepts:
    vLLM wants max_tokens, Azure's gpt-5.x family rejects it and wants
    max_completion_tokens (verified live 2026-08-05 on both).

    Ministral is deliberately sent NO max_tokens unless a caller passes one
    explicitly: vLLM's chat API defaults max_tokens to (context window −
    prompt tokens) computed with the REAL tokenizer. The old client-side
    char-based estimate could only be wrong in two ways, both observed at
    scale in one bulk run: under-request (86 wasted correct-and-retry round
    trips) and over-request (67 outright 400 rejections — "requested 6697
    output tokens and your prompt contains 1597 input tokens" — with no
    retry path). Omitting the parameter deletes both failure classes; a
    genuine window overflow still surfaces as finish_reason='length', which
    extraction recovers from by salvage + follow-up."""
    # The running run's recorded backend + model (settings on the bulk path; the
    # bulk run's own recorded values on an incremental event), so an update
    # extracts with the same model as the run it's updating even if THIS worker's
    # startup probe resolved a different backend. Forcing the run's backend when
    # its gateway is unreachable here surfaces as a transient retry, which is the
    # correct outcome — never a silent model swap.
    cfg = effective_run_config()
    backend = cfg.llm_backend
    throttle = get_throttle_for(kind, backend)
    if backend == "ministral":
        client = get_in_house_client()
        model = cfg.llm_model
        cap_param = "max_tokens"
    else:
        client = get_client()
        model = cfg.llm_model
        cap_param = "max_completion_tokens"
    cap_kwargs = {cap_param: max_tokens} if max_tokens is not None else {}

    async with throttle.slot():
        call_start = time.monotonic()
        try:
            if response_format is not None:
                # with_raw_response so the Azure path can feed its header-driven
                # throttle; .parse() validates against the Pydantic schema.
                raw = await client.beta.chat.completions.with_raw_response.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    **cap_kwargs,
                )
            else:
                raw = await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
        except RateLimitError:
            # Recorded against the backend that actually 429'd — the Azure
            # throttle halves its concurrency; the latency throttle has no
            # rate-limit notion, a failure is the honest equivalent.
            if backend == "ministral":
                await throttle.record_failure()
            else:
                await throttle.record_rate_limit()
            raise
        except APIStatusError as exc:
            # Only a server-side error (5xx) is an overload signal worth
            # halving for. A 4xx (bad request, content filter, context
            # overflow) is a property of THIS request — the pod is fine, and
            # halving on it would let one bad chunk throttle the healthy
            # pipeline.
            if backend == "ministral" and exc.status_code >= 500:
                await throttle.record_failure()
            raise
        except APIConnectionError:
            # Covers APITimeoutError (a subclass): the pod stopped answering
            # in time — the clearest overload signal this backend emits.
            if backend == "ministral":
                await throttle.record_failure()
            raise
        # Anything else (SDK/validation errors) propagates without touching
        # the throttle: it is not evidence about pod load either way.
        call_elapsed = time.monotonic() - call_start
        if backend == "ministral":
            await throttle.record_success(call_elapsed)
        else:
            await throttle.record_success(headers=dict(raw.headers))
        if call_elapsed > _SLOW_CALL_WARN_SECONDS:
            logger.warning("LLM call took %.1fs (> %.0fs) — succeeded, but unusually slow",
                           call_elapsed, _SLOW_CALL_WARN_SECONDS)

    try:
        response = raw.parse()
    except LengthFinishReasonError as exc:
        # Normalized to the shared truncation type so the retry classifier
        # (utils/error_classification.py) can mark it permanent: at
        # temperature 0 the regenerate is byte-identical, so retrying a
        # truncation is deterministic waste. The partial output rides along —
        # recovery (salvaging its complete objects + one follow-up call for
        # the rest) is the CALLER's move; see extraction_service.py.
        partial = None
        try:
            partial = exc.completion.choices[0].message.content
        except (AttributeError, IndexError):
            pass  # no partial available — the caller falls back to failing the input
        raise LlmTruncationError(
            f"LLM output truncated at the context ceiling: {exc}",
            partial_content=partial) from exc
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        if choice.finish_reason == "length":
            raise LlmTruncationError(
                f"LLM response truncated (finish_reason={choice.finish_reason!r})",
                partial_content=choice.message.content)
        raise ValueError(f"LLM response truncated (finish_reason={choice.finish_reason!r})")
    if response_format is not None:
        parsed = choice.message.parsed
        if parsed is None:
            raise ValueError("structured output returned None")
        return parsed
    return json.loads(choice.message.content or "{}")


async def call_with_retry(
    attempt: Callable[[], Awaitable[T]], *, call_name: str, context: str,
    fail_closed: Callable[[], T], kind: CallKind, metric_calls: Optional[str] = None,
    metric_failures: Optional[str] = None,
) -> T:
    """THE retry loop for an LLM call. attempt() is one try (normally a
    complete() call, already backend-resolved) that raises on any failure.

    kind's profile (services/llm_profiles.py) supplies max_retries/base_backoff/
    max_backoff — one number per call shape instead of a value plumbed in from
    each call site.

    Verdicts via classify_llm_retry_error():
      fatal     -> RE-RAISE. A fatal is a global config problem (bad API
                   key/endpoint, 401/403) that fails identically for every
                   future call — failing closed here thousands of times is
                   silent mass corruption, so the run stalls instead.
      permanent -> fail closed immediately (this input will never work).
      transient -> bounded retry with backoff — includes bad generations
                   (invalid JSON, schema validation, empty choices: a
                   regenerate can succeed) and rate limits (already recorded
                   on the right throttle inside complete()).

    fail_closed() is called (and its result returned) only after
    record_call_failure writes a durable record — never silently.
    metric_calls/metric_failures are services/run_metrics.py counter names,
    recorded once per real attempt / once on the terminal failure."""
    profile = PROFILES[kind]
    max_retries, base_backoff, max_backoff = (
        profile.max_retries, profile.base_backoff_seconds, profile.max_backoff_seconds)
    delay = 0.0
    for i in range(max_retries):
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            if metric_calls:
                record(metric_calls)
            return await attempt()
        except Exception as exc:  # noqa: BLE001
            error_class = "transient" if isinstance(exc, RateLimitError) \
                else classify_llm_retry_error(exc)
            if error_class == "fatal":
                logger.error("%s: FATAL error (global config problem, not specific to "
                             "this call) — re-raising to stall the run: %s", call_name, exc)
                record_call_failure(call_name, context, "fatal", str(exc), attempts=i + 1)
                raise
            if error_class == "permanent" or i == max_retries - 1:
                if isinstance(exc, LlmTruncationError):
                    # Not a scary traceback: truncation is deterministic for
                    # this input (hence exactly one attempt) and the CALL SITE
                    # owns recovery — extraction salvages the partial output
                    # and runs a follow-up pass. The ledger record below still
                    # keeps the durable audit trail.
                    logger.warning("%s for %s: output truncated at the context ceiling after "
                                   "%d attempt(s) — this call fails closed (truncation is "
                                   "deterministic, blind retries can't fix it); recovery, if "
                                   "any, happens at the call site: %s",
                                   call_name, context, i + 1, exc)
                else:
                    logger.exception("%s failed for %s after %d attempt(s) (%s) — failing closed",
                                     call_name, context, i + 1, error_class)
                if metric_failures:
                    record(metric_failures)
                record_call_failure(call_name, context, error_class, str(exc), attempts=i + 1)
                return fail_closed()
            delay = min(base_backoff * (2 ** i) + random.uniform(0, 1), max_backoff)
            logger.warning("%s failed (transient, attempt %d/%d), retrying in %.1fs: %s",
                           call_name, i + 1, max_retries, delay, exc)
    return fail_closed()


async def confirm_with_retry(system_prompt: str, user_prompt: str, result_field: str, *,
                             step: str, context: str, log_label: str) -> bool:
    """Yes/no LLM confirmation: one json_object complete() under the standard
    retry loop. Fail-closed value is False (no-match). Fatal re-raises, like
    every other call site."""
    async def _ask() -> bool:
        decision = await complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            kind=CallKind.CONFIRM,
        )
        return bool(decision.get(result_field))

    return await call_with_retry(
        _ask, call_name=step, context=context, fail_closed=lambda: False, kind=CallKind.CONFIRM,
    )
