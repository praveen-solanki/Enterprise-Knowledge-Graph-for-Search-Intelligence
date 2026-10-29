"""Every registered connector really implements ConnectorSpec.

There is no type checker in CI — requirements-dev.txt is pytest and nothing
else, and none of the workflows run mypy, pyright or ruff. So the
`_SPEC: type[ConnectorSpec] = ZeusSpec` annotation in each spec module is
unverified prose. This checks it for real, and checks the reverse direction
too: that `core/` is not calling anything the protocol does not declare, which
is how the protocol quietly stops describing the contract.
"""
import inspect

import pytest

from connectors.base import ConnectorSpec, PeopleProvider
from connectors.registry import get_connector, known_connector_types


def _protocol_members(protocol) -> set[str]:
    """Every name the protocol declares: annotated data attributes plus methods
    and properties. `__protocol_attrs__` only exists from Python 3.12, so this
    reconstructs it."""
    names = set(getattr(protocol, "__annotations__", {}))
    names |= {name for name, _ in inspect.getmembers(protocol)
              if not name.startswith("_") and name not in dir(object)}
    return names


@pytest.mark.parametrize("connector_type", known_connector_types())
def test_every_registered_connector_has_every_protocol_member(connector_type):
    spec = get_connector(connector_type)
    missing = sorted(m for m in _protocol_members(ConnectorSpec) if not hasattr(type(spec), m))
    assert missing == [], f"{type(spec).__name__} is missing {missing}"


@pytest.mark.parametrize("connector_type", known_connector_types())
def test_every_registered_connector_satisfies_the_runtime_protocol(connector_type):
    assert isinstance(get_connector(connector_type), ConnectorSpec)


def test_the_protocol_covers_everything_core_calls_on_a_spec():
    """If core/ starts calling a new spec method, it belongs on the protocol —
    otherwise 'implement ConnectorSpec' stops being enough to add a connector."""
    import ast
    import pathlib

    declared = _protocol_members(ConnectorSpec)
    called: set[str] = set()
    for path in pathlib.Path("core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "spec":
                called.add(node.attr)
    undeclared = sorted(called - declared)
    assert undeclared == [], (
        f"core/ calls spec.{{{', '.join(undeclared)}}}, which ConnectorSpec does not declare"
    )


@pytest.mark.parametrize("connector_type", known_connector_types())
def test_every_connector_ships_a_real_people_provider(connector_type):
    assert isinstance(get_connector(connector_type).people, PeopleProvider)


def test_core_never_learns_that_a_document_can_be_a_person():
    """B4's actual guarantee. core/ used to ask spec.is_person_document and hold
    a dict of raw people payloads; the provider collects its own now, so no
    object_type at all — least of all "people" — appears in core/."""
    import pathlib

    for path in pathlib.Path("core").rglob("*.py"):
        source = path.read_text()
        assert '"people"' not in source and "'people'" not in source, f"{path} names a people object_type"
        assert "object_type" not in source, f"{path} reads a document's object_type"


def test_core_drives_all_three_people_stages_in_order():
    """MATERIALIZE mints the identity a bare-name mention then resolves onto, so
    it has to run before the corpus stage; ENRICH needs the nodes the write
    created, so it has to run after. Getting this order wrong produces two nodes
    per person and an empty org chart, both silently."""
    import ast
    import pathlib

    stages = ["discover", "materialize_persons", "_canonicalize_corpus_with_cache", "enrich"]
    tree = ast.parse(pathlib.Path("core/pipeline.py").read_text())
    bulk, = [n for n in ast.walk(tree)
             if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_bulk_async"]
    lines: dict = {}
    for node in ast.walk(bulk):
        # Any reference, not specifically a Call — materialize_persons is handed
        # to asyncio.to_thread rather than called inline.
        name = node.attr if isinstance(node, ast.Attribute) else (
            node.id if isinstance(node, ast.Name) else None)
        if name in stages:
            lines.setdefault(name, node.lineno)
    assert sorted(lines) == sorted(stages), f"run_bulk_async drives only {sorted(lines)}"
    assert [lines[s] for s in stages] == sorted(lines[s] for s in stages), \
        f"run_bulk_async runs the people stages out of order: {lines}"


def test_core_calls_the_people_phase_through_the_protocol_and_nothing_else():
    """The people phase used to be twenty-odd lines inside run_bulk_async calling
    repo.label_person_records, repo.materialize_org_membership and the rest.
    core/ must not call any of them any more — if it does, the block was moved
    but the coupling was not."""
    import ast
    import pathlib

    moved = {"label_person_records", "find_person_mention_merges", "merge_person_mentions",
             "materialize_reporting_edges", "retire_person_manages",
             "materialize_org_membership", "compute_org_rollups"}
    for path in pathlib.Path("core").rglob("*.py"):
        called = {node.attr for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.Attribute)}
        leaked = sorted(moved & called)
        assert leaked == [], f"{path} still calls {leaked}"
