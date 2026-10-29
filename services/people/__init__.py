"""Connector-blind person algorithms.

A library plus one stage. Nothing here runs on its own without a caller,
nothing here reads a source's field names, and nothing here knows a connector
exists. Connectors call in; this package never calls out to one.

Three modules: MATERIALIZE (the shared middle stage — identity, the
:Entity:Person entry), the LLM disambiguation call, and bare-name resolution
against the registry. Every function that reads `user_work_title`,
`user_department` and the rest of the HR export is intranet policy and lives
in connectors/zeus/.
"""
from services.people.disambiguation import (
    DISAMBIGUATION_SYSTEM,
    attr_text,
    llm_confirms_same_person,
)
from services.people.materialize import materialize_persons, person_identity_norm
from services.people.resolution import (
    plain_person_name,
    resolve_bare_person,
    resolve_bare_persons,
)

__all__ = [
    "DISAMBIGUATION_SYSTEM",
    "attr_text",
    "llm_confirms_same_person",
    "materialize_persons",
    "person_identity_norm",
    "plain_person_name",
    "resolve_bare_person",
    "resolve_bare_persons",
]
