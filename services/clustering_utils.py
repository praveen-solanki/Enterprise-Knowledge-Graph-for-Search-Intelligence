"""
Shared clustering + LLM-merge-resolution mechanics for Step 2 (doc-level) and
Step 3 (corpus-wide) canonicalization; the two stages differ only in
similarity threshold, embed mode, and merge-decision system prompt.

Uses FAISS approximate nearest-neighbor search so cost stays near O(n log n)
instead of quadratic as entity counts grow.

Every structured LLM call here goes through structured_llm_call, which sits on
llm_client: that resolves the backend (Ministral or Azure), takes that call
kind's own throttle slot (services/llm_profiles.py — its own pool, not shared
with extraction or any other kind), and shares one retry/backoff/fail-closed
ladder via llm_client.call_with_retry, tuned per kind.

A cluster larger than DEFAULT_LLM_BATCH_MAX is sliced and resolved recursively by
resolve_cluster_batched rather than overflowing a single prompt.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import time
from typing import Optional, Union

import faiss
import numpy as np
from pydantic import BaseModel, Field

from config.settings import settings
from services import llm_client
from services.llm_profiles import CallKind

logger = logging.getLogger("graphrag.services.clustering_utils")

# Cap FAISS/OpenMP parallelism so large corpus rounds (100k+ vectors) don't
# multiply peak RSS under memory pressure / swap thrashing.
try:
    _faiss_threads = max(1, min(2, int(faiss.omp_get_max_threads())))
    faiss.omp_set_num_threads(_faiss_threads)
except Exception:  # pragma: no cover — faiss builds without OMP still work
    _faiss_threads = 1

# Bump when a change would invalidate previously-cached cluster resolutions.
CLUSTERING_VERSION = "faiss-ann-k32-v6-depth-capped-mem"

# Above this many members, the dense NxN matrix tight_partition needs is too expensive
# to build; a cluster this size is passed through unpartitioned instead (and logged).
# 512 → ~1 MiB float32 sims matrix (was 1500 → ~9 MiB, and many large clusters stacked).
_TIGHT_PARTITION_MAX = 512

# Reclaim Python list[list[float]] garbage after converting large embedding batches.
_GC_AFTER_EMBED_N = 10_000


def _as_float32_matrix(vectors) -> np.ndarray:
    """One contiguous float32 matrix — not nested Python floats.

    Azure returns list[list[float]]. At corpus scale (~150k × 3k dims) that
    Python object graph was the Step 3 RSS spike (~5–8× the float32 size).
    """
    if isinstance(vectors, np.ndarray):
        return np.ascontiguousarray(vectors, dtype=np.float32)
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))


def embedding_as_list(vector: Optional[Union[list, np.ndarray]]) -> list[float]:
    """Normalize an embedding to list[float] for Neo4j / Mongo persistence."""
    if vector is None:
        return []
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32, copy=False).tolist()
    return list(vector)


class MergeGroup(BaseModel):
    canonical_name: str = Field(
        ..., description="The single most representative surface-form name for this merged group "
        "(not a garbled casing variant or an invented blend of member names).")
    canonical_type: str = Field(..., description="The shared entity type for every member of this group.")
    merged_description: str = Field(
        ..., description="A synthesized description covering every member's description, preserving "
        "concrete facts (numbers, dates, versions) rather than a vague generic summary.")
    members: list[str] = Field(
        ..., description="Every candidate name that refers to the same real-world entity as "
        "canonical_name, including canonical_name itself.")


class ClusterResolution(BaseModel):
    groups: list[MergeGroup] = Field(
        ..., description="The real groups within this cluster of candidate names -- candidates that "
        "are NOT the same real thing must end up in separate groups, each with a single member.")


def normalize_name(name: str) -> str:
    """Cheap surface-form cleanup so obvious duplicates ('Globex', 'globex ')
    never need an embedding or LLM call."""
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


# Precision values (money, counts, dates, versions, IPs, clause IDs) look near-identical
# under name embeddings, so they must merge only on an exact normalized-name match, never
# on similarity (otherwise e.g. "$82k" and "$124k" could collapse together).
_PRECISION_TYPE_HINTS = ("METRIC", "DATE", "VERSION", "RELEASE", "IP_ADDRESS",
                        "SEAT", "PRICE", "CURRENCY", "IDENTIFIER", "CRITERION")
_PRECISION_NAME = re.compile(
    r"^\s*(?:[$€£]\s?[\d,.]+\w*"          # money: $82k, €1,200
    r"|[\d,.]+\s*(?:seats?|users?|k|m|gb|mb)\b"  # counts: 120 Seats, 80k
    r"|[\d,.]+\s*%"                        # percentages: 80%
    r"|\d{4}-\d{2}-\d{2}"                  # ISO date
    r"|v?\d+(?:[.\-][\w]+){1,}"            # versions/releases: v13.27, 26.01-A.2.0
    r"|\d{1,3}(?:\.\d{1,3}){3}"            # IPv4
    r"|EN\s?\d|WCAG)",                     # standard clause ids
    re.IGNORECASE)


def is_precision(name: str, entity_type: Optional[str]) -> bool:
    """True if this entity is a precision value that must not be merged on
    similarity alone — only ever on an exact normalized-name duplicate."""
    if entity_type and any(h in entity_type.upper() for h in _PRECISION_TYPE_HINTS):
        return True
    return bool(_PRECISION_NAME.match(name or ""))


def _casing_garble(x: str) -> int:
    """0 for normal casing (all-lower, Title Case, or a short ALL-CAPS
    acronym), 1 for a garble with irregular interior capitals like
    'GLoBEX'. Without this, lexicographic tie-break would let 'GLoBEX'
    beat 'Globex' (uppercase 'L' sorts before lowercase 'l')."""
    if not x:
        return 1
    if x.islower() or x.istitle() or (x.isupper() and len(x) <= 5 and " " not in x):
        return 0
    return 1


def shortest_clean(names: list[str], counts: Optional[dict] = None) -> str:
    """Pick a canonical surface form deterministically, for when no LLM answer is
    available or usable: exact-normalized-name duplicates before any call, and the
    fallback for a blank/off-group name. A real canonical_name is trusted instead
    (see cluster_to_fixpoint's trust_llm_naming).

    Prefers the most frequent form (if `counts` given); ties fall back to
    normal casing over garble, then shortest, then demoting an ALL-CAPS
    multi-word phrase, then lexicographic — so 'AWS' beats 'Aws' and
    'Enterprise Search' beats 'ENTERPRISE SEARCH'."""
    def _key(x: str):
        freq = -(counts.get(x, 0)) if counts else 0
        allcaps_phrase = x.isupper() and (len(x) > 5 or " " in x)
        return (freq, _casing_garble(x), len(x), allcaps_phrase, x)
    return sorted(names, key=_key)[0] if names else ""


_SALIENT_FACT = re.compile(r"\d[\d.\-/]*|\b[A-Z][A-Za-z0-9]{2,}\b")


def _salient_facts(text: str) -> set:
    """Concrete, checkable facts in a description: numbers/versions/dates
    and proper-noun-ish tokens — exactly what a one-sentence LLM synthesis
    tends to drop."""
    return {t.lower() for t in _SALIENT_FACT.findall(text or "")}


def best_description(merged: Optional[str], member_descs: list[str]) -> Optional[str]:
    """Guard against the LLM's merged_description silently dropping facts a
    member's own description preserved: compares salient facts (numbers,
    dates, proper nouns) covered by the merged text against the richest
    member description, and falls back to that member's text when the
    synthesis loses facts — catching lossy rewrites, not just truncation."""
    descs = [d for d in member_descs if d]
    if not descs:
        return merged or None
    longest = max(descs, key=len)
    if not merged:
        return longest
    terms = set().union(*(_salient_facts(d) for d in descs))
    if not terms:  # nothing concrete to verify -> length guard only
        return longest if len(merged) < 0.6 * len(longest) else merged
    ml, ll = merged.lower(), longest.lower()
    cov_merged = sum(1 for t in terms if t in ml) / len(terms)
    cov_long = sum(1 for t in terms if t in ll) / len(terms)
    # Keep the synthesis unless the richest member clearly preserves more facts,
    # or the synthesis is also just truncated.
    if cov_merged + 1e-9 < cov_long and (cov_merged < 0.8 or len(merged) < 0.6 * len(longest)):
        return longest
    return merged


def cluster_by_similarity(representatives: list[str], vectors, threshold: float,
                          k: int = 32) -> list[list[str]]:
    """Single-linkage clustering via FAISS approximate nearest-neighbor search:
    each entity is compared only against its top-k most-similar candidates
    (~O(n log n) instead of O(n²) brute-force). Union-find merges any pair
    that clears `threshold` transitively, so A-B-C ends up one cluster even
    if A and C weren't each other's nearest neighbors.

    `vectors` may be list[list[float]] or a float32 ndarray; normalized in-place
    on a float32 working copy so callers can keep a separate raw matrix."""
    n = len(representatives)
    if n <= 1:
        return [[r] for r in representatives]

    # Own a float32 buffer and L2-normalize in place (no second full copy).
    norm = _as_float32_matrix(vectors)
    faiss.normalize_L2(norm)

    index = faiss.IndexFlatIP(norm.shape[1])
    index.add(norm)
    k_search = min(k + 1, n)
    similarities, neighbor_idx = index.search(norm, k_search)
    # FAISS holds its own copy after add(); drop our search matrix ASAP.
    del norm
    del index

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for sim, j in zip(similarities[i], neighbor_idx[i]):
            if j != i and j >= 0 and sim >= threshold:
                union(i, int(j))

    del similarities, neighbor_idx

    clusters_by_root: dict = {}
    for i, rep in enumerate(representatives):
        clusters_by_root.setdefault(find(i), []).append(rep)
    return list(clusters_by_root.values())


def tight_partition(members: list[str], embeddings: dict, threshold: float) -> list[list[str]]:
    """Anti-chaining: splits a (possibly single-linkage-chained) cluster from
    cluster_by_similarity into complete-linkage fragments, where every pair
    within a fragment clears `threshold` (i.e. each fragment is a clique).

    A single weak bridge entity can transitively chain two unrelated entities
    into one single-linkage cluster; re-running single-linkage at a tighter
    threshold does NOT reliably fix this, since the bridge can still survive
    the tighter cut. Complete linkage can't chain, since a member only joins a
    fragment once it's close to every existing member of it.

    Greedy and order-stable (deterministic across re-runs on the same input).
    O(n^2); a cluster larger than _TIGHT_PARTITION_MAX is passed through
    unpartitioned instead (logged)."""
    n = len(members)
    if n <= 1:
        return [list(members)]
    if n > _TIGHT_PARTITION_MAX:
        logger.warning("tight_partition: cluster of %d members exceeds the %d anti-chaining "
                       "cap — passing it through unpartitioned rather than building an O(n^2) "
                       "matrix; members=%s", n, _TIGHT_PARTITION_MAX, list(members)[:3])
        return [list(members)]

    M = np.ascontiguousarray(
        np.vstack([np.asarray(embeddings[m], dtype=np.float32) for m in members]),
        dtype=np.float32,
    )
    faiss.normalize_L2(M)
    sims = M @ M.T
    del M
    degree = (sims >= threshold).sum(axis=1)
    order = sorted(range(n), key=lambda i: (-int(degree[i]), members[i]))
    assigned = [False] * n
    fragments: list[list[str]] = []
    for i in order:
        if assigned[i]:
            continue
        members_idx = [i]
        assigned[i] = True
        for j in order:
            if assigned[j]:
                continue
            if all(sims[j, k] >= threshold for k in members_idx):
                members_idx.append(j)
                assigned[j] = True
        fragments.append([members[k] for k in members_idx])
    return fragments


# ---------------------------------------------------------------------------
# Cluster resolution
# ---------------------------------------------------------------------------

def _ungrouped_fallback(names: list[str], registry: dict) -> ClusterResolution:
    """Every candidate stays its own group — the safe fail-closed result."""
    return ClusterResolution(groups=[
        MergeGroup(
            canonical_name=n,
            canonical_type=registry[n]["type"],
            merged_description=registry[n].get("description") or "",
            members=[n],
        )
        for n in names
    ])


async def structured_llm_call(system_prompt: str, user_payload: str, response_model,
                              *, step: str, context: str,
                              calls_metric: str, failures_metric: str,
                              kind: CallKind,
                              slow_call_context: str = "",
                              ) -> tuple[bool, Optional[object]]:
    """One structured-output LLM call on the resolved backend (Ministral or Azure),
    behind that CALL KIND's own throttle (services/llm_profiles.py — never shared
    with an unrelated kind), with retry/backoff, run metrics and a durable
    call_failures record when it gives up. Shared by every structured call site so
    they all get the same ladder, tuned per kind.

    Transport, throttling, backoff and the durable failure record all live in
    llm_client (complete + call_with_retry); this wrapper exists for the
    (ok, parsed) contract its call sites are written against.

    Returns (ok, parsed):
      (True, model)  — succeeded and parsed.
      (True, None)   — succeeded but returned nothing parseable (not an error).
                       llm_client.complete treats a None parse as a bad generation
                       and retries it, so this is no longer reachable in practice;
                       call sites still handle it.
      (False, None)  — failed after retries, or a permanent error. The CALLER
                       owns the fail-closed fallback: "safe degraded" differs per
                       site (unmerged cluster vs. each mention's own description).

    A FATAL error (bad credentials/endpoint) propagates instead of failing closed:
    call_with_retry re-raises it so a global misconfiguration stalls the run rather
    than silently degrading every call site thousands of times.

    slow_call_context is accepted for call-site compatibility; llm_client.complete
    owns the slow-call warning now."""
    async def _attempt() -> tuple[bool, Optional[object]]:
        parsed = await llm_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format=response_model,
            kind=kind,
        )
        return True, parsed

    return await llm_client.call_with_retry(
        _attempt, call_name=step, context=context,
        fail_closed=lambda: (False, None), kind=kind,
        metric_calls=calls_metric, metric_failures=failures_metric,
    )


async def resolve_cluster(names: list[str], registry: dict, system_prompt: str) -> ClusterResolution:
    """One LLM call (with concurrency cap + rate-limit retry): given a cluster
    of similar-looking entity names, decide the real groups. Uses Ministral when
    get_llm_backend() resolves to it, Azure otherwise -- resolved once for the
    whole process, never re-checked mid-retry (see services/llm_backend.py).

    Fails closed only after all retries are exhausted: each candidate becomes
    its own group (no merge). That degraded result is not cached by the
    caller. Retry/backoff/fail-closed-logging is shared via structured_llm_call
    on top of llm_client.call_with_retry -- this function only builds the
    candidate payload for the one real attempt (Azure or Ministral)."""
    candidates = [
        {"name": n, "type": registry[n]["type"], "description": registry[n].get("description") or ""}
        for n in names
    ]
    ok, parsed = await structured_llm_call(
        system_prompt, json.dumps(candidates), ClusterResolution,
        step="cluster_merge_confirm", context=f"members={names[:5]}",
        calls_metric="merge_calls", failures_metric="merge_call_failures",
        slow_call_context=f"members={names[:2]}",
        kind=CallKind.CLUSTER_MERGE,
    )
    if not ok:
        return _ungrouped_fallback(names, registry)
    # Succeeded but parsed to nothing: an empty resolution, NOT the fail-closed
    # fallback — callers treat "no groups came back" differently from "every name
    # is its own group".
    return parsed if parsed is not None else ClusterResolution(groups=[])


# Cap on candidates in ONE cluster-merge prompt. Both stages resolve clusters of
# unbounded size, and an oversized prompt fails closed to "every candidate its own
# group" — merging stops for the biggest entity while the run still reports success.
# Exported: services/doc_canonicalization_service.py imports this name directly.
DEFAULT_LLM_BATCH_MAX = settings.doc_cluster_merge_batch_max

# A recursion level must shrink the survivor list by at least this fraction for
# resolve_cluster_batched to recurse on it — below this, the level is dripping one
# or two merges, and the slice results are already the answer to within noise.
MIN_SHRINK_FRACTION = 0.1


async def resolve_cluster_batched(cluster: list[str], registry: dict, system_prompt: str,
                                  *, resolve_fn=None, batch_max: int = DEFAULT_LLM_BATCH_MAX,
                                  log_prefix: str = "Canonicalization",
                                  max_depth: Optional[int] = None,
                                  depth: int = 0) -> ClusterResolution:
    """Resolve a cluster of ANY size within the `batch_max` prompt cap: slice it,
    resolve the slices concurrently, then RECURSE on the survivors until they fit one
    call. Recursion rather than one flat pass because ceil(N/batch_max) survivors can
    still exceed the cap (N > 2500 at 50), which tight_partition's 1500-member
    passthrough makes reachable. Each level judges survivors on the level below's
    MERGED description, never the original member text.

    Termination: survivors must shrink by MIN_SHRINK_FRACTION before recursing, so a
    level that merged nothing (or next to nothing) keeps its slice results instead of
    re-sending the same list. max_depth is the hard cost cap on top of that (each
    stage passes its own from settings): a level it forbids also keeps its slice
    results — by then the LLM is judging descriptions-of-descriptions anyway.

    resolve_fn / batch_max are injectable for per-stage caching wrappers, caps, and
    test patch points. Returns resolve_cluster's contract, with members expanded
    back to the original input names."""
    resolve_fn = resolve_fn or resolve_cluster
    if len(cluster) <= batch_max:
        return await resolve_fn(cluster, registry, system_prompt)

    slices = [cluster[i:i + batch_max] for i in range(0, len(cluster), batch_max)]
    logger.info("%s: cluster of %d members split into %d batch(es) of <= %d (depth %d)",
                log_prefix, len(cluster), len(slices), batch_max, depth)
    slice_results = await asyncio.gather(
        *[resolve_fn(s, registry, system_prompt) for s in slices])

    survivor_registry = dict(registry)
    rep_to_members: dict[str, list[str]] = {}
    for result in slice_results:
        for group in result.groups:
            rep_to_members.setdefault(group.canonical_name, []).extend(group.members)
            survivor_registry[group.canonical_name] = {
                "type": group.canonical_type, "description": group.merged_description,
            }
    survivors = list(rep_to_members.keys())

    def _keep_slice_results() -> ClusterResolution:
        """The slice-level merges, with no cross-slice consolidation."""
        return ClusterResolution(groups=[
            MergeGroup(canonical_name=rep, canonical_type=survivor_registry[rep]["type"],
                      merged_description=survivor_registry[rep]["description"], members=members)
            for rep, members in rep_to_members.items()
        ])

    if len(survivors) <= 1:
        final = _keep_slice_results()
    elif len(survivors) > (1 - MIN_SHRINK_FRACTION) * len(cluster):
        logger.warning("%s: the %d-member cluster only merged down to %d survivor(s) at "
                       "depth %d — below the %.0f%% shrink floor, keeping the slice results "
                       "rather than dripping one merge per level",
                       log_prefix, len(cluster), len(survivors), depth,
                       MIN_SHRINK_FRACTION * 100)
        final = _keep_slice_results()
    elif max_depth is not None and depth + 1 >= max_depth:
        logger.warning("%s: %d survivor(s) of the %d-member cluster still exceed one prompt "
                       "at depth %d, but max_depth=%d forbids another level — keeping the "
                       "slice results",
                       log_prefix, len(survivors), len(cluster), depth, max_depth)
        final = _keep_slice_results()
    else:
        final = await resolve_cluster_batched(survivors, survivor_registry, system_prompt,
                                              resolve_fn=resolve_fn, batch_max=batch_max,
                                              log_prefix=log_prefix, max_depth=max_depth,
                                              depth=depth + 1)

    return ClusterResolution(groups=[
        MergeGroup(
            canonical_name=group.canonical_name, canonical_type=group.canonical_type,
            merged_description=group.merged_description,
            # dict.fromkeys, not a plain comprehension: on the keep-slice paths a
            # group's members are already expanded, so re-mapping the rep itself
            # would list its co-members twice.
            members=list(dict.fromkeys(
                m for rep in group.members for m in rep_to_members.get(rep, [rep]))),
        )
        for group in final.groups
    ])


# Cap on how many times cluster_to_fixpoint re-clusters its own merged output: 1 initial
# pass + up to 2 more rounds, stopping early once a round merges nothing.
DEFAULT_MAX_CANON_ITERATIONS = settings.canon_max_iterations


async def cluster_to_fixpoint(
    representatives: list[str],
    registry: dict,
    embed_text_fn,
    embedder,
    threshold: float,
    anti_chain_threshold: float,
    system_prompt: str,
    resolve_fn=None,
    max_iterations: int = DEFAULT_MAX_CANON_ITERATIONS,
    trust_llm_naming: bool = True,
) -> tuple[dict, dict, dict]:
    """Iterates clustering + LLM-merge to a fixpoint: a newly-merged entity
    (with its folded description and a fresh embedding) re-enters clustering
    and can merge further, catching multi-hop merges a single pass would miss
    (e.g. round 1 merges A+B; round 2 merges AB+C, which didn't exist as a
    candidate until round 1 finished).

    representatives: names already deduped by exact normalized form with
        precision values held out — this only ever clusters prose names.
    registry: {name: {"type", "description"}} for every name in
        `representatives`, same shape resolve_cluster expects.
    embed_text_fn(name, registry) -> str: what to embed per name (name-only
        for doc-level, name+description for corpus-level). Re-invoked each
        round against that round's current (possibly-merged) registry, so a
        later round embeds the merged description, not the original.
    resolve_fn: injectable (tests, or large-cluster batching) — defaults to
        resolve_cluster.
    trust_llm_naming: default True for both stages — the merge call already gets
        every candidate's merged description, so its canonical_name and
        merged_description are used as returned; overriding them would restore the
        arbitrary single-source text the description merge replaces. Only an
        empty/whitespace field falls back to the heuristic. Pass False for the old
        deterministic behaviour (shortest_clean + best_description).

    Returns (merge_map, final, first_round_embeddings):
      merge_map: {original_name: final_canonical_name}.
      final: {final_canonical_name: {type, description, members}} — members
        lists every original representative folded into this canonical name.
      first_round_embeddings: {original_name: float32 vector} from round 1, before
        any merging — lets a caller look up a vector for the merged entity
        without a second embedding call. Values are numpy 1-d arrays (not
        nested Python floats) to keep corpus-scale RSS bounded; callers that
        need list[float] should use embedding_as_list(). Empty if
        `representatives` had fewer than 2 names.

    A round's LLM failures already fail closed inside resolve_cluster/
    resolve_fn (never raises); this function adds no try/except of its own,
    so a failure here is a genuine bug."""
    resolve_fn = resolve_fn or resolve_cluster
    current = {n: {"type": registry[n]["type"], "description": registry[n].get("description"),
                   "members": [n]} for n in representatives}
    first_round_embeddings: dict = {}

    for iteration in range(max_iterations):
        names = list(current.keys())
        if len(names) < 2:
            break
        round_registry = {n: {"type": current[n]["type"], "description": current[n].get("description")}
                          for n in names}
        texts = [embed_text_fn(n, round_registry) for n in names]
        # embed_batch blocks (sync Azure client with its own time.sleep backoff).
        # On the event loop it would stall every other coroutine, including the
        # sibling documents Step 2 canonicalizes concurrently.
        logger.info(
            "cluster_to_fixpoint: round %d embedding %d entit(y/ies)",
            iteration + 1, len(names),
        )
        t0 = time.monotonic()
        # Prefer matrix path on real EmbeddingService — avoids materializing
        # ~150k Python float lists that drove the Step 3 memory spike.
        # Check the class (not the instance): MagicMock would otherwise claim
        # every attribute exists via hasattr().
        if callable(getattr(type(embedder), "embed_batch_matrix", None)):
            matrix = await asyncio.to_thread(embedder.embed_batch_matrix, texts)
        else:
            raw = await asyncio.to_thread(embedder.embed_batch, texts)
            matrix = _as_float32_matrix(raw)
            del raw
        matrix = _as_float32_matrix(matrix)
        if len(names) >= _GC_AFTER_EMBED_N:
            gc.collect()
        logger.info(
            "cluster_to_fixpoint: round %d embeddings done in %.1fs "
            "(matrix shape=%s dtype=%s, ~%.2f GiB) — starting FAISS similarity cluster",
            iteration + 1, time.monotonic() - t0,
            tuple(matrix.shape), matrix.dtype,
            matrix.nbytes / (1024 ** 3),
        )
        if iteration == 0:
            # Row views into `matrix` — matrix must stay alive for the return value.
            first_round_embeddings = {n: matrix[i] for i, n in enumerate(names)}
        t1 = time.monotonic()
        # Pass a copy so FAISS in-place L2-normalize does not mutate round-1 rows.
        clusters = await asyncio.to_thread(
            cluster_by_similarity, names, matrix.copy(), threshold,
        )
        logger.info(
            "cluster_to_fixpoint: round %d FAISS produced %d cluster(s) in %.1fs — anti-chain partition",
            iteration + 1, len(clusters), time.monotonic() - t1,
        )
        embedding_by_name = {n: matrix[i] for i, n in enumerate(names)}
        # Partition multi-member clusters with heartbeat logs so long quiet
        # stretches (esp. under memory pressure) are visible in the run log.
        t2 = time.monotonic()
        partitioned: list[list[str]] = []
        multi_src = [c for c in clusters if len(c) > 1]
        singles = [c for c in clusters if len(c) <= 1]
        partitioned.extend(singles)
        del clusters
        for i, cluster in enumerate(multi_src, start=1):
            partitioned.extend(
                await asyncio.to_thread(tight_partition, cluster, embedding_by_name, anti_chain_threshold)
            )
            if i == 1 or i == len(multi_src) or i % 25 == 0:
                logger.info(
                    "cluster_to_fixpoint: round %d anti-chain %d/%d multi-clusters "
                    "(elapsed %.1fs)",
                    iteration + 1, i, len(multi_src), time.monotonic() - t2,
                )
        multi_clusters = [c for c in partitioned if len(c) > 1]
        logger.info(
            "cluster_to_fixpoint: round %d partition done in %.1fs → %d multi-cluster(s) for LLM",
            iteration + 1, time.monotonic() - t2, len(multi_clusters),
        )
        # Drop per-round working refs before the LLM storm (can be long). Keep
        # first_round_embeddings (views into round-1 matrix) for the return value.
        del embedding_by_name
        if iteration > 0:
            del matrix
            if len(names) >= _GC_AFTER_EMBED_N:
                gc.collect()
        elif iteration == 0:
            # Round 0: matrix is retained via first_round_embeddings views.
            pass

        if not multi_clusters:
            break  # only singletons remain -- fixpoint

        logger.info(
            "cluster_to_fixpoint: round %d resolving %d multi-cluster(s) via LLM",
            iteration + 1, len(multi_clusters),
        )
        resolved = await asyncio.gather(*[resolve_fn(c, round_registry, system_prompt) for c in multi_clusters])
        del multi_clusters, partitioned

        new_current: dict = {}
        consumed: set = set()
        merged_any = False
        for resolution in resolved:
            for group in resolution.groups:
                grp = [m for m in group.members if m in current and m not in consumed]
                if not grp:
                    continue
                consumed.update(grp)
                if len(grp) > 1:
                    merged_any = True
                members: list = []
                descs: list = []
                for m in grp:
                    members.extend(current[m]["members"])
                    if current[m].get("description"):
                        descs.append(current[m]["description"])
                if trust_llm_naming:
                    final_name = (group.canonical_name or "").strip() or shortest_clean(members)
                    merged_desc = (group.merged_description or "").strip() or best_description(None, descs)
                else:
                    final_name = shortest_clean(members)
                    merged_desc = best_description(group.merged_description, descs)
                if final_name in new_current:  # two clusters resolved to the same name -> fold together
                    new_current[final_name]["members"].extend(members)
                else:
                    new_current[final_name] = {"type": group.canonical_type, "description": merged_desc,
                                               "members": members}
        for n in names:
            if n in consumed:
                continue
            if n in new_current:
                # An untouched singleton colliding with a merged group's canonical
                # name (only possible under trust_llm_naming). Overwriting would lose
                # the group's members from merge_map, skipping would drop the
                # singleton — so fold it in.
                new_current[n]["members"].extend(current[n]["members"])
                continue
            new_current[n] = current[n]  # carry over untouched singletons unchanged
        current = new_current

        logger.info("cluster_to_fixpoint: round %d -> %d entities%s",
                    iteration + 1, len(current), "" if merged_any else " (no merges -> fixpoint)")
        if not merged_any:
            break

    merge_map = {orig: canon for canon, meta in current.items() for orig in meta["members"]}
    final = {canon: {"type": meta["type"], "description": meta["description"], "members": meta["members"]}
            for canon, meta in current.items()}
    return merge_map, final, first_round_embeddings
