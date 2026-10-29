"""Lightweight in-process LLM-call counters — answers "how many LLM calls did this run
make, and how many failed" for bulk runs and the incremental worker.

Single-process only (never multiprocessing), but genuinely multi-threaded: the
embedding service fans its sub-batches out across a thread pool and each one
counts itself. `_counts[kind] = _counts.get(kind, 0) + n` is a read-modify-write
spanning several bytecodes, so two threads counting the same kind can interleave
between the read and the write and lose one of the increments — an undercount
that looks exactly like a call that never happened. Cheap to just take a lock:
this runs once per API call, never in a hot loop."""
from __future__ import annotations

import threading

_counts: dict[str, int] = {}
_lock = threading.Lock()


def record(kind: str, n: int = 1) -> None:
    with _lock:
        _counts[kind] = _counts.get(kind, 0) + n


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counts)


def reset() -> None:
    with _lock:
        _counts.clear()
