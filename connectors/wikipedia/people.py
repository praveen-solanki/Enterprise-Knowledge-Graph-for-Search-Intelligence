"""Wikipedia has no identity system for this pipeline to resolve against:
pages are collaboratively edited with no single stable author, and the public
Action API exposes no per-user directory. A thinner graph here is an absence,
not a failure — see connectors/base.py's PeopleProvider docstring on ENRICH.
"""
from __future__ import annotations

from connectors.base import PeopleContext, PeopleResult, PersonRecord, SourceDocument


class NullPeopleProvider:
    """DISCOVER finds nobody, ENRICH adds nothing — the correct amount for a
    source with no people to report."""

    def observe(self, docs: list[SourceDocument]) -> None:
        return None

    async def discover(self, ctx: PeopleContext) -> list[PersonRecord]:
        return []

    async def enrich(self, records: list[PersonRecord], ctx: PeopleContext) -> PeopleResult:
        return PeopleResult()
