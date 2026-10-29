"""Shared greedy char-budget packing for description-merge / reduce prompts.

Used by both doc-level (Step 2) and corpus-level (Step 3) canonicalization so
each stage sizes calls by description volume, not a fixed entity/piece count.
"""


def pack_by_char_budget(items: list, size_of, char_budget: int) -> list[list]:
    """Greedily pack items into batches of at most `char_budget` chars, preserving
    input order so the stage stays deterministic. Sized by characters, not item
    count: the list mixes a full N-description slice with a 2-description entity,
    so no fixed count is safe at both ends. An over-budget item goes out alone."""
    batches: list[list] = []
    batch: list = []
    batch_chars = 0
    for item in items:
        item_chars = size_of(item)
        if batch and batch_chars + item_chars > char_budget:
            batches.append(batch)
            batch, batch_chars = [], 0
        batch.append(item)
        batch_chars += item_chars
    if batch:
        batches.append(batch)
    return batches
