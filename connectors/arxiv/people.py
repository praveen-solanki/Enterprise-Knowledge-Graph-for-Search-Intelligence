"""arXiv papers have authors, but no stable identity directory this pipeline
can join on. DISCOVER reports nobody.
"""
from __future__ import annotations

from connectors.base import PeopleContext, PeopleResult, PersonRecord, SourceDocument


class NullPeopleProvider:
    def observe(self, docs: list[SourceDocument]) -> None:
        return None

    async def discover(self, ctx: PeopleContext) -> list[PersonRecord]:
        return []

    async def enrich(self, records: list[PersonRecord], ctx: PeopleContext) -> PeopleResult:
        return PeopleResult()
