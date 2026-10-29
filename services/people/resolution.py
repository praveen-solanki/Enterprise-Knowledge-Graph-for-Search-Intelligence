"""Resolving a bare-name person mention against the registry.

Connector-blind: a name, an optional description, a tenant. No source ever
named this person with an id — that is what makes it "bare" — so the ambiguity
guard is the whole point.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config.settings import settings
from repository import entity_registry
from services.clustering_utils import normalize_name
from services.people.disambiguation import (
    DISAMBIGUATION_BATCH_MAX,
    DISAMBIGUATION_ITEMS_PER_CALL,
    attr_text,
    disambiguate_item_batch,
    llm_confirms_same_person,
)
from services.vector_store import VectorStore

logger = logging.getLogger("graphrag.services.people.resolution")


def plain_person_name(name: str) -> str:
    """The registry alias/identity key a bare-name mention resolves on. Public because
    person_corpus_resolver.py groups by the same key, and a second copy would drift."""
    return normalize_name(name) or (name or "").strip().lower() or "∅"


async def resolve_bare_person(name: str, description: Optional[str], tenant_id: str,
                              embedding: Optional[list] = None,
                              vector_store: Optional[VectorStore] = None) -> str:
    """Resolve a bare-name PERSON mention (extracted from Content/File, no
    user_id/email) with the ambiguity guard — see module docstring."""
    plain = normalize_name(name) or (name or "").strip().lower() or "∅"
    candidates = entity_registry.find_all_by_alias(plain, "PERSON", tenant_id)

    if len(candidates) == 1:
        entity_registry.add_aliases(candidates[0]["_id"], [plain])
        return candidates[0]["_id"]

    if len(candidates) > 1:
        for cand in candidates:
            same = await llm_confirms_same_person(
                name, description, cand.get("canonical_name"), attr_text(cand.get("enrichment") or {}),
            )
            if same:
                entity_registry.add_aliases(cand["_id"], [plain])
                return cand["_id"]
        logger.warning("Ambiguous PERSON mention %r matched %d existing people, none confirmed — "
                       "minting a separate unresolved node", name, len(candidates))

    # 0 candidates, or ambiguous with no confident match -> mint/attach to the plain-name node
    rec = entity_registry.link_or_mint(name, "PERSON", plain, tenant_id, aliases=[plain])
    entity_id = rec["_id"]
    if vector_store is not None and embedding:
        vector_store.upsert_entity(entity_id, embedding, "PERSON", tenant_id, [])
    return entity_id


async def resolve_bare_persons(mentions: list[tuple], tenant_id: str) -> list[str]:
    """Resolve MANY bare-name PERSON mentions (extracted from Content/File, no
    user_id/email) with the ambiguity guard — see module docstring.

    mentions: [(name, description), ...]. Returns one entity_id per mention, in input order.
    Batched corpus-wide because neither cost needs to scale with mention count:

      * Registry I/O is per NAME — a person named in 800 documents costs one lookup and
        one alias write, not 800 of each.
      * Disambiguation is per BATCH. An ambiguous name needs one judgement per mention
        (the answer depends on its description), but those are independent questions, so
        they travel DISAMBIGUATION_ITEMS_PER_CALL to a call.

    Bounded by settings.person_resolution_concurrency (registry) and the shared throttle
    (LLM). Reading once per name before any write also removes the read-then-write hazard
    the old per-mention path had to serialise around.

    NOTE for callers: an identity materialized upstream (e.g. services/people/materialize.py)
    must already be registered, or a bare mention of that same person mints its own node
    instead of linking."""
    if not mentions:
        return []

    indexes_by_name: dict = {}
    for index, (name, _description) in enumerate(mentions):
        indexes_by_name.setdefault(plain_person_name(name), []).append(index)

    semaphore = asyncio.Semaphore(settings.person_resolution_concurrency)

    async def _registry_call(fn, *args, **kwargs):
        async with semaphore:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # Phase 1 — one registry read per distinct name, all concurrent.
    names = list(indexes_by_name.keys())
    candidates_by_name = dict(zip(names, await asyncio.gather(
        *[_registry_call(entity_registry.find_all_by_alias, plain, "PERSON", tenant_id)
          for plain in names])))

    resolved: list = [""] * len(mentions)

    async def _mint(plain: str, index: int) -> str:
        """The plain-name node, shared by every mention that lands on it. link_or_mint is
        idempotent on the unique index, so once per name == once per mention."""
        rec = await _registry_call(entity_registry.link_or_mint, mentions[index][0], "PERSON",
                                   plain, tenant_id, aliases=[plain])
        return rec["_id"]

    # Phase 2 — unambiguous names (0 or 1 candidate) need no LLM; every mention of such a
    # name gets the same answer.
    async def _settle_unambiguous(plain: str) -> None:
        candidates = candidates_by_name[plain]
        indexes = indexes_by_name[plain]
        if candidates:
            entity_id = candidates[0]["_id"]
            await _registry_call(entity_registry.add_aliases, entity_id, [plain])
        else:
            entity_id = await _mint(plain, indexes[0])
        for index in indexes:
            resolved[index] = entity_id

    unambiguous = [plain for plain in names if len(candidates_by_name[plain]) <= 1]
    ambiguous = [plain for plain in names if len(candidates_by_name[plain]) > 1]
    await asyncio.gather(*[_settle_unambiguous(plain) for plain in unambiguous])

    if not ambiguous:
        return resolved

    # Phase 3 — one item per (mention, candidate chunk), packed across names into concurrent
    # calls. Keyed by (mention index, chunk ordinal) to keep every chunk addressable.
    items: list[tuple] = []
    for plain in ambiguous:
        candidates = candidates_by_name[plain]
        chunks = [candidates[start:start + DISAMBIGUATION_BATCH_MAX]
                  for start in range(0, len(candidates), DISAMBIGUATION_BATCH_MAX)]
        for index in indexes_by_name[plain]:
            name, description = mentions[index]
            for ordinal, chunk in enumerate(chunks):
                items.append(((index, ordinal), name, description, chunk))

    batches = [items[start:start + DISAMBIGUATION_ITEMS_PER_CALL]
               for start in range(0, len(items), DISAMBIGUATION_ITEMS_PER_CALL)]
    logger.info("Person disambiguation: %d ambiguous name(s), %d mention(s) -> %d item(s) in "
                "%d concurrent call(s)", len(ambiguous),
                sum(len(indexes_by_name[plain]) for plain in ambiguous), len(items), len(batches))
    matches: dict = {}
    for batch_result in await asyncio.gather(*[disambiguate_item_batch(b) for b in batches]):
        matches.update(batch_result)

    # Phase 4 — apply. A mention takes the first chunk that matched, in chunk order, so
    # which chunk a candidate landed in can't change the answer.
    async def _settle_ambiguous(plain: str) -> None:
        indexes = indexes_by_name[plain]
        chunk_count = -(-len(candidates_by_name[plain]) // DISAMBIGUATION_BATCH_MAX)
        unmatched: list = []
        linked: list = []
        for index in indexes:
            match = None
            for ordinal in range(chunk_count):
                match = matches.get((index, ordinal))
                if match is not None:
                    break
            if match is None:
                unmatched.append(index)
            else:
                resolved[index] = match["_id"]
                linked.append(match["_id"])
        await asyncio.gather(*[_registry_call(entity_registry.add_aliases, entity_id, [plain])
                               for entity_id in sorted(set(linked))])
        if unmatched:
            logger.warning("Ambiguous PERSON mention %r matched %d existing people, none confirmed "
                           "for %d of its %d mention(s) — minting a separate unresolved node",
                           mentions[unmatched[0]][0], len(candidates_by_name[plain]),
                           len(unmatched), len(indexes))
            entity_id = await _mint(plain, unmatched[0])
            for index in unmatched:
                resolved[index] = entity_id

    await asyncio.gather(*[_settle_ambiguous(plain) for plain in ambiguous])
    return resolved
