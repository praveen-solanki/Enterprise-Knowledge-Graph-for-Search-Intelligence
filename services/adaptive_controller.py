"""
Self-calibrating controllers for the two main bottlenecks in the pipeline:

  1. AdaptiveThrottle  — controls how many Azure OpenAI calls fire simultaneously.
  2. AdaptiveBatchSizer — controls how many chunks per ES batch.

WHY NUMBERS CANNOT BE HARDCODED
─────────────────────────────────
Every deployment is different:
  - Azure RPM/TPM quotas vary by subscription tier, region, and model.
  - Chunk processing time varies by chunk length, model version, and Azure load.
  - The "right" concurrency for one run may be wrong the next day if Azure
    is under load or your quota tier changes.

Hardcoded numbers are guesses. These controllers find the right number
automatically, at runtime, for YOUR deployment, on THAT day.

HOW THEY WORK
─────────────

AdaptiveThrottle — AIMD (Additive Increase / Multiplicative Decrease)
  This is the same algorithm TCP uses for congestion control. It provably
  converges to maximum sustainable throughput without needing to know
  the limit upfront:

  • Start at a conservative concurrency (default: 10 simultaneous calls).
  • Every time INCREASE_AFTER requests complete successfully → add 1 slot.
    This is "additive increase" — we probe gently upward.
  • Every time Azure says "429 Too Many Requests" → halve the concurrency.
    This is "multiplicative decrease" — we back off hard and fast.
  • The algorithm oscillates around the true maximum, always staying close
    without needing you to know the limit in advance.

  Reads Azure response headers (x-ratelimit-limit-requests,
  x-ratelimit-remaining-requests, x-ratelimit-limit-tokens,
  x-ratelimit-remaining-tokens) when available to converge faster —
  if Azure tells us "you have 300 RPM", we can jump straight to that
  instead of probing up one slot at a time.

AdaptiveBatchSizer — Timing-based PID-lite
  Target: each batch of chunks finishes in approximately TARGET_SECONDS.
  • If the last batch finished faster than target → increase the next batch.
  • If the last batch finished slower than target → shrink the next batch.
  • Adjustment is capped at ±25% per batch to avoid oscillation.
  • Hard lower and upper bounds prevent runaway growth or starvation.

  This means:
  - On a fast Azure day (low latency, no rate limits) → large batches,
    maximum throughput.
  - On a slow Azure day (high latency, rate-limited) → small batches,
    stable progress without scroll/PIT expiry pressure.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from typing import Optional

from config.settings import settings

logger = logging.getLogger("graphrag.adaptive_controller")


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveThrottle — AIMD concurrency control for Azure OpenAI calls. One
# instance per CallKind (services/llm_client.py's get_throttle_for), not a
# process-wide singleton — see services/llm_profiles.py.
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveThrottle:
    """
    AIMD concurrency controller. Drop-in replacement for asyncio.Semaphore
    that adjusts its own limit based on observed successes and rate-limit errors.

    Usage inside an async function:
        async with throttle.slot():
            result = await azure_call(...)
        await throttle.record_success(headers=response_headers)   # or
        await throttle.record_rate_limit()
    """

    def __init__(self, initial: int = settings.azure_throttle_initial_concurrency,
                 increase_after: int = settings.azure_throttle_increase_after,
                 min_concurrency: int = settings.azure_throttle_min_concurrency,
                 max_concurrency: int = settings.azure_throttle_max_concurrency):
        self._target = initial
        # Per-instance, not class-level — each CallKind needs its own tuning.
        self.INCREASE_AFTER = increase_after  # consecutive successes before +1 slot
        self.MIN_CONCURRENCY = min_concurrency  # forward progress floor
        self.MAX_CONCURRENCY = max_concurrency  # safety ceiling regardless of headers
        self._in_flight = 0
        self._successes_since_increase = 0
        self._total_successes = 0
        self._total_rate_limits = 0
        self._condition = asyncio.Condition()
        self._start_time = time.monotonic()
        # Set when we learn the real limit from Azure response headers.
        self._azure_rpm_limit: Optional[int] = None
        self._azure_tpm_limit: Optional[int] = None
        logger.info("AdaptiveThrottle starting at concurrency=%d (will auto-tune)", initial)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def current_concurrency(self) -> int:
        return self._target

    @asynccontextmanager
    async def slot(self):
        """Acquire one concurrency slot. Blocks if at the current limit."""
        async with self._condition:
            while self._in_flight >= self._target:
                await self._condition.wait()
            self._in_flight += 1
        try:
            yield
        finally:
            async with self._condition:
                self._in_flight -= 1
                self._condition.notify_all()

    async def record_success(self, headers: Optional[dict] = None) -> None:
        """Call after every successful Azure response. Pass HTTP response
        headers if available — we use Azure's quota headers to converge faster."""
        if headers:
            self._ingest_azure_headers(headers)

        async with self._condition:
            self._total_successes += 1
            self._successes_since_increase += 1

            if self._successes_since_increase >= self.INCREASE_AFTER:
                self._successes_since_increase = 0
                # Never exceed what Azure told us our limit is.
                hard_cap = self._azure_rpm_limit or self.MAX_CONCURRENCY
                new_target = min(self._target + 1, hard_cap, self.MAX_CONCURRENCY)
                if new_target != self._target:
                    self._target = new_target
                    elapsed = time.monotonic() - self._start_time
                    rps = self._total_successes / max(1.0, elapsed)
                    logger.info(
                        "Throttle ↑ concurrency=%d | %d successes | %.1f req/s | "
                        "azure_rpm_limit=%s",
                        self._target, self._total_successes, rps,
                        self._azure_rpm_limit or "unknown",
                    )
                    self._condition.notify_all()

    async def record_rate_limit(self) -> None:
        """Call when Azure returns a 429. Halves the concurrency immediately."""
        async with self._condition:
            self._total_rate_limits += 1
            self._successes_since_increase = 0
            new_target = max(self._target // 2, self.MIN_CONCURRENCY)
            logger.warning(
                "Throttle ↓ rate-limit #%d — concurrency %d → %d",
                self._total_rate_limits, self._target, new_target,
            )
            self._target = new_target
            # No notify needed — reducing target; tasks drain naturally.

    def log_stats(self, label: str = "") -> None:
        elapsed = time.monotonic() - self._start_time
        rps = self._total_successes / max(1.0, elapsed)
        logger.info(
            "AdaptiveThrottle%s final stats: concurrency=%d | %d successes | "
            "%d rate-limits | %.1f req/s | azure_rpm=%s | azure_tpm=%s",
            f"[{label}]" if label else "", self._target, self._total_successes,
            self._total_rate_limits, rps,
            self._azure_rpm_limit or "unknown", self._azure_tpm_limit or "unknown",
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _ingest_azure_headers(self, headers: dict) -> None:
        """Read Azure OpenAI rate-limit headers and update our knowledge of the
        actual quota. If Azure tells us the limit, we can jump to it instead
        of probing one slot at a time."""
        try:
            rpm = headers.get("x-ratelimit-limit-requests")
            if rpm and self._azure_rpm_limit is None:
                self._azure_rpm_limit = int(rpm)
                logger.info("Azure tells us RPM limit = %d (via response header)",
                            self._azure_rpm_limit)
            tpm = headers.get("x-ratelimit-limit-tokens")
            if tpm and self._azure_tpm_limit is None:
                self._azure_tpm_limit = int(tpm)
                logger.info("Azure tells us TPM limit = %d (via response header)",
                            self._azure_tpm_limit)
        except (ValueError, TypeError):
            pass  # malformed headers — ignore, AIMD still works without them


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveBatchSizer — timing-based chunk-budget control for ES batches
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveBatchSizer:
    """
    Adjusts how many chunks flow through each ES batch based on how long the
    previous batch actually took to process.

    Target: each batch finishes in approximately TARGET_SECONDS.
    Starting conservatively and growing means:
      - First batch is safe (small).
      - Subsequent batches grow toward maximum throughput.
      - A slow batch (rate-limited, slow Azure) automatically shrinks the next.

    This removes the need to guess MAX_CHUNKS_PER_BATCH. The pipeline
    self-tunes to whatever rate Azure can sustain today.
    """

    # Aim for this many seconds per batch. 45s gives comfortable margin even
    # if Azure is slow, while keeping batches large enough to be efficient.
    TARGET_SECONDS: float = settings.bulk_batch_target_seconds
    # Max single-step change per batch (±25%) — prevents oscillation.
    MAX_CHANGE_FACTOR: float = settings.bulk_batch_max_change_factor
    # Absolute bounds — never go below 10 chunks or above 5000.
    MIN_CHUNKS: int = settings.bulk_batch_min_chunks
    MAX_CHUNKS: int = settings.bulk_batch_max_chunks

    def __init__(self, initial: int = settings.bulk_batch_initial_chunks):
        """Start conservatively — first batch is always fast and safe."""
        self._size = initial
        self._batch_num = 0
        logger.info("AdaptiveBatchSizer starting at %d chunks/batch (will auto-tune "
                    "toward %.0fs per batch)", initial, self.TARGET_SECONDS)

    @property
    def current_size(self) -> int:
        return self._size

    def record_batch(self, actual_seconds: float, chunks_processed: int) -> None:
        """Call after every batch with the wall-clock time it took.
        Adjusts the next batch's chunk budget accordingly."""
        self._batch_num += 1
        if actual_seconds <= 0 or chunks_processed <= 0:
            return

        throughput = chunks_processed / actual_seconds
        ratio = self.TARGET_SECONDS / actual_seconds

        # Cap adjustment to ±25% so we don't oscillate wildly.
        ratio = max(1.0 - self.MAX_CHANGE_FACTOR,
                    min(1.0 + self.MAX_CHANGE_FACTOR, ratio))
        new_size = int(self._size * ratio)
        new_size = max(self.MIN_CHUNKS, min(self.MAX_CHUNKS, new_size))

        direction = "↑" if new_size > self._size else ("↓" if new_size < self._size else "=")
        logger.info(
            "BatchSizer batch#%d %s %d → %d chunks "
            "(took %.1fs / target %.1fs | %.1f chunks/s)",
            self._batch_num, direction, self._size, new_size,
            actual_seconds, self.TARGET_SECONDS, throughput,
        )
        self._size = new_size


# ─────────────────────────────────────────────────────────────────────────────
# Neo4jBatchCalibrator — auto-discovers safe write batch sizes at startup
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jBatchCalibrator:
    """
    Probes the Neo4j server at startup to find the maximum safe batch size
    for entity writes (the ones that carry large embedding vectors).

    Strategy — doubling probe with timeout guard:
      Start at probe_start (e.g. 50 entities).
      Send a test UNWIND batch and measure how long it takes.
      If response < TARGET_SECONDS → safe; double the probe size.
      If response >= TARGET_SECONDS or error → back off to last safe size.
      Stop when: two consecutive timeouts, or probe exceeds max_probe.

    Returns the largest batch size that completed in < TARGET_SECONDS.
    If the server can't even handle probe_start rows, returns probe_start
    (we still make progress, just slowly).
    """

    TARGET_SECONDS: float = settings.neo4j_calibration_target_seconds   # any single batch taking longer than this is risky
    MIN_BATCH: int = settings.neo4j_calibration_min_batch
    MAX_PROBE: int = settings.neo4j_calibration_max_probe

    def calibrate(self, execute_fn, sample_rows: list, label: str = "entity") -> int:
        """
        execute_fn: callable(cypher, {"batch": rows}) -> dict (same as _execute in Neo4jRepository)
        sample_rows: a small list of real rows to use as the test payload.
        Returns the recommended batch size.
        """
        if not sample_rows:
            logger.warning("Neo4jBatchCalibrator: no sample rows — using default 100")
            return 100

        # Probe sequence: 10, 20, 50, 100, 200, 500, 1000, 2000 — each doubles
        probe_sizes = []
        size = self.MIN_BATCH
        while size <= self.MAX_PROBE:
            probe_sizes.append(size)
            size = size * 2

        last_safe = self.MIN_BATCH
        logger.info("Neo4jBatchCalibrator: probing %s batch sizes %s", label, probe_sizes)

        for probe_size in probe_sizes:
            # Build a test payload by repeating sample rows up to probe_size.
            test_rows = (sample_rows * ((probe_size // len(sample_rows)) + 1))[:probe_size]
            try:
                t0 = time.monotonic()
                execute_fn(test_rows)
                elapsed = time.monotonic() - t0
                logger.info("  probe size=%d → %.2fs %s",
                            probe_size, elapsed, "✓" if elapsed < self.TARGET_SECONDS else "SLOW")
                if elapsed < self.TARGET_SECONDS:
                    last_safe = probe_size
                else:
                    # Server is getting slow — stop probing, use last safe size.
                    logger.info("Neo4jBatchCalibrator: %s batch size set to %d "
                                "(probe %d took %.2fs > %.1fs target)",
                                label, last_safe, probe_size, elapsed, self.TARGET_SECONDS)
                    return last_safe
            except Exception as exc:
                logger.warning("Neo4jBatchCalibrator: probe size=%d failed (%s) — "
                               "using last safe size %d", probe_size, exc, last_safe)
                return last_safe

        logger.info("Neo4jBatchCalibrator: %s batch size set to %d (all probes passed)",
                    label, last_safe)
        return last_safe


# ─────────────────────────────────────────────────────────────────────────────
# LatencyAdaptiveThrottle — AIMD for backends with no rate-limit signal of
# their own (a self-hosted vLLM box, or any future model on any other serving
# framework). AdaptiveThrottle above keys off Azure's specific 429s/quota
# headers, which only Azure sends — this one keys off latency and outright
# failures instead, since those are observable no matter what's actually
# running behind a given endpoint.
#
# A single self-hosted box also has no equivalent of Azure's published
# quota — there's nothing to "jump to" the way Azure's headers let
# AdaptiveThrottle converge faster. This one always probes gradually, one
# slot at a time, and can only ever infer a slowdown from what it directly
# observes (elevated latency, a failed call) — never from a number the
# server volunteers.
# ─────────────────────────────────────────────────────────────────────────────

class LatencyAdaptiveThrottle:
    """
    AIMD concurrency controller keyed on latency/failure rather than a
    vendor-specific rate-limit signal.

    Usage inside an async function:
        async with throttle.slot():
            t0 = time.monotonic()
            try:
                result = await call(...)
            except Exception:
                await throttle.record_failure()
                raise
            else:
                await throttle.record_success(time.monotonic() - t0)
    """

    def __init__(
        self,
        initial: int = settings.ministral_max_concurrency,
        target_latency_seconds: float = settings.llm_default_target_latency_seconds,
        increase_after: int = settings.ministral_throttle_increase_after,
        min_concurrency: int = settings.ministral_throttle_min_concurrency,
        max_concurrency: int = settings.ministral_throttle_max_concurrency,
        slow_call_multiplier: float = settings.ministral_throttle_slow_call_multiplier,
        recent_window: int = settings.ministral_throttle_recent_window,
    ):
        self._target = initial
        self._target_latency = target_latency_seconds
        # Per-instance, not class-level — each CallKind needs its own tuning.
        self.INCREASE_AFTER = increase_after  # consecutive fast successes before +1 slot
        self.MIN_CONCURRENCY = min_concurrency
        # No vendor-published quota to bound this against — a generous but finite
        # ceiling so a runaway feedback loop can't grow unbounded.
        self.MAX_CONCURRENCY = max_concurrency
        # A success slower than target_latency_seconds * this multiplier is treated
        # as a back-off signal, same as an outright failure.
        self.SLOW_CALL_MULTIPLIER = slow_call_multiplier
        self._RECENT_WINDOW = recent_window
        self._in_flight = 0
        self._fast_successes_since_increase = 0
        self._total_successes = 0
        self._total_failures = 0
        self._recent_latencies: list[float] = []
        self._condition = asyncio.Condition()
        self._start_time = time.monotonic()
        logger.info("LatencyAdaptiveThrottle starting at concurrency=%d, target_latency=%.1fs",
                    initial, target_latency_seconds)

    @property
    def current_concurrency(self) -> int:
        return self._target

    @asynccontextmanager
    async def slot(self):
        """Acquire one concurrency slot. Blocks if at the current limit."""
        async with self._condition:
            while self._in_flight >= self._target:
                await self._condition.wait()
            self._in_flight += 1
        try:
            yield
        finally:
            async with self._condition:
                self._in_flight -= 1
                self._condition.notify_all()

    async def record_success(self, latency: float, target_latency_seconds: Optional[float] = None) -> None:
        """Call after every successful response, with how long it took.
        target_latency_seconds here is a one-off override — each instance
        already has the right default from construction (services/llm_profiles.py)."""
        target_latency = target_latency_seconds if target_latency_seconds is not None else self._target_latency
        async with self._condition:
            self._total_successes += 1
            self._recent_latencies = (self._recent_latencies + [latency])[-self._RECENT_WINDOW:]

            if latency > target_latency * self.SLOW_CALL_MULTIPLIER:
                # Succeeded, but slow enough to be the same signal a failure
                # would be — back off exactly like record_failure does.
                self._fast_successes_since_increase = 0
                new_target = max(self._target // 2, self.MIN_CONCURRENCY)
                if new_target != self._target:
                    logger.warning(
                        "Throttle ↓ slow call (%.1fs > %.1fs target×%.0f) — concurrency %d → %d",
                        latency, target_latency, self.SLOW_CALL_MULTIPLIER,
                        self._target, new_target,
                    )
                    self._target = new_target
                return

            if latency <= target_latency:
                self._fast_successes_since_increase += 1
                if self._fast_successes_since_increase >= self.INCREASE_AFTER:
                    self._fast_successes_since_increase = 0
                    new_target = min(self._target + 1, self.MAX_CONCURRENCY)
                    if new_target != self._target:
                        self._target = new_target
                        logger.info(
                            "Throttle ↑ concurrency=%d | %d successes | avg_latency=%.1fs",
                            self._target, self._total_successes,
                            statistics.mean(self._recent_latencies),
                        )
                        self._condition.notify_all()
            # latency between target and target*SLOW_CALL_MULTIPLIER: neither
            # fast enough to grow nor slow enough to back off — hold steady.

    async def record_failure(self) -> None:
        """Call when a request fails outright (connection error, 5xx, timeout).
        Halves concurrency immediately, same as a rate-limit response would."""
        async with self._condition:
            self._total_failures += 1
            self._fast_successes_since_increase = 0
            new_target = max(self._target // 2, self.MIN_CONCURRENCY)
            logger.warning("Throttle ↓ failure #%d — concurrency %d → %d",
                           self._total_failures, self._target, new_target)
            self._target = new_target

    def log_stats(self, label: str = "") -> None:
        elapsed = time.monotonic() - self._start_time
        avg_latency = statistics.mean(self._recent_latencies) if self._recent_latencies else 0.0
        logger.info(
            "LatencyAdaptiveThrottle%s final stats: concurrency=%d | %d successes | "
            "%d failures | avg_latency=%.1fs | elapsed=%.1fs",
            f"[{label}]" if label else "", self._target, self._total_successes,
            self._total_failures, avg_latency, elapsed,
        )
