"""Per-LLM-call-kind tuning, in one place. Retuning a kind = one settings.py
edit; adding a kind = one enum value + one PROFILES entry.

target_latency_seconds only feeds Ministral's throttle — Azure's is
header-driven and ignores it (services/llm_client.py). Azure kinds each get
their own throttle instance (get_throttle_for); Ministral kinds all share
ONE throttle for the backend, sized to the pod's measured capacity knee, so
overlapping stages can never stack past it (extraction's target latency
governs that shared pool — see get_throttle_for)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config.settings import settings


class CallKind(str, Enum):
    EXTRACTION = "extraction"
    EXTRACTION_GLEANING = "extraction_gleaning"
    CONFIRM = "confirm"
    CLUSTER_MERGE = "cluster_merge"
    DOC_DESCRIPTION_MERGE = "doc_description_merge"
    CORPUS_DESCRIPTION_MERGE = "corpus_description_merge"
    DISAMBIGUATION = "disambiguation"


@dataclass(frozen=True)
class LlmCallProfile:
    target_latency_seconds: float
    max_retries: int
    base_backoff_seconds: float
    max_backoff_seconds: float
    # A documented per-kind output ceiling. NOT sent on any request by default:
    # Ministral requests omit max_tokens entirely (vLLM derives the real budget
    # from the window and the actual prompt — services/llm_client.complete),
    # and Azure gets no cap unless a caller passes one explicitly.
    max_tokens: int


PROFILES: dict[CallKind, LlmCallProfile] = {
    CallKind.EXTRACTION: LlmCallProfile(
        settings.llm_extraction_target_latency_seconds,
        settings.llm_extraction_max_retries,
        settings.llm_extraction_base_backoff_seconds,
        settings.llm_extraction_max_backoff_seconds,
        settings.llm_extraction_max_tokens,
    ),
    # Same shape as extraction, fewer retries — gleaning is a bonus pass.
    CallKind.EXTRACTION_GLEANING: LlmCallProfile(
        settings.llm_extraction_target_latency_seconds,
        settings.llm_extraction_gleaning_max_retries,
        settings.llm_extraction_base_backoff_seconds,
        settings.llm_extraction_max_backoff_seconds,
        settings.llm_extraction_gleaning_max_tokens,
    ),
    CallKind.CONFIRM: LlmCallProfile(
        settings.llm_confirm_target_latency_seconds,
        settings.llm_confirm_max_retries,
        settings.llm_confirm_base_backoff_seconds,
        settings.llm_confirm_max_backoff_seconds,
        settings.llm_confirm_max_tokens,
    ),
    CallKind.CLUSTER_MERGE: LlmCallProfile(
        settings.llm_cluster_merge_target_latency_seconds,
        settings.llm_cluster_merge_max_retries,
        settings.llm_cluster_merge_base_backoff_seconds,
        settings.llm_cluster_merge_max_backoff_seconds,
        settings.llm_cluster_merge_max_tokens,
    ),
    CallKind.DOC_DESCRIPTION_MERGE: LlmCallProfile(
        settings.llm_doc_description_merge_target_latency_seconds,
        settings.llm_doc_description_merge_max_retries,
        settings.llm_doc_description_merge_base_backoff_seconds,
        settings.llm_doc_description_merge_max_backoff_seconds,
        settings.llm_doc_description_merge_max_tokens,
    ),
    CallKind.CORPUS_DESCRIPTION_MERGE: LlmCallProfile(
        settings.llm_corpus_description_merge_target_latency_seconds,
        settings.llm_corpus_description_merge_max_retries,
        settings.llm_corpus_description_merge_base_backoff_seconds,
        settings.llm_corpus_description_merge_max_backoff_seconds,
        settings.llm_corpus_description_merge_max_tokens,
    ),
    CallKind.DISAMBIGUATION: LlmCallProfile(
        settings.llm_disambiguation_target_latency_seconds,
        settings.llm_disambiguation_max_retries,
        settings.llm_disambiguation_base_backoff_seconds,
        settings.llm_disambiguation_max_backoff_seconds,
        settings.llm_disambiguation_max_tokens,
    ),
}
