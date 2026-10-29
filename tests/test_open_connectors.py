"""Wikipedia and arXiv turn a public API payload into a SourceDocument.
No network: the HTTP response is stubbed.
"""
import asyncio
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest
import requests

from connectors.arxiv.documents import (
    ARXIV_ACL_STRATEGY,
    ATOM,
    _entry_dict as arxiv_entry_dict,
    _fetch as arxiv_fetch,
    _MAX_RETRIES as ARXIV_MAX_RETRIES,
    fetch_batches as arxiv_fetch_batches,
    iter_entries as arxiv_iter_entries,
    prepare_chunk_text as arxiv_prepare_chunk_text,
    search_query as arxiv_search_query,
)
from connectors.arxiv.documents import to_source_document as arxiv_doc
from connectors.arxiv.spec import ArxivSpec
from connectors.base import Chunk, PeopleContext, PeopleResult
from connectors.wikipedia.documents import (
    WIKIPEDIA_ACL_STRATEGY,
    _chunks_from_extract as wiki_chunks_from_extract,
    _complete_pages as wiki_complete_pages,
    _fetch_with_retry as wiki_fetch_with_retry,
    _MAX_IDS_PER_REQUEST as WIKI_MAX_IDS_PER_REQUEST,
    _MAX_RETRIES as WIKI_MAX_RETRIES,
    _pages_from as wiki_pages_from,
    _preview_text as wiki_preview_text,
    _request_params as wiki_request_params,
    _section_heading as wiki_section_heading,
    fetch_batches as wiki_fetch_batches,
    iter_pages as wiki_iter_pages,
    prepare_chunk_text as wiki_prepare_chunk_text,
    search_query as wiki_search_query,
)
from connectors.wikipedia.documents import to_source_document as wiki_doc
from connectors.wikipedia.spec import WikipediaSpec


def _run(coro):
    return asyncio.run(coro)


def _json_response(body: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = body
    return response


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    resp = MagicMock()
    resp.status_code = status_code
    return requests.exceptions.HTTPError(response=resp)


def test_wikipedia_page_becomes_one_chunk_per_paragraph():
    doc = wiki_doc({
        "pageid": 1,
        "title": "Graph database",
        "fullurl": "https://en.wikipedia.org/wiki/Graph_database",
        "touched": "2026-01-02T00:00:00Z",
        "extract": "First paragraph.\n\nSecond paragraph.",
    })
    assert doc is not None
    assert doc.doc_id == "1"
    assert doc.connector_type.value == "wikipedia"
    assert doc.acl.dropped is False
    assert doc.acl.tokens == ["public"]
    assert [c.text for c in doc.chunks] == ["First paragraph.", "Second paragraph."]
    assert doc.url == "https://en.wikipedia.org/wiki/Graph_database"


def test_wikipedia_fetch_reads_the_search_generator(monkeypatch):
    body = {
        "continue": {},
        "query": {"pages": [{
            "pageid": 7, "index": 1, "title": "Ada",
            "extract": "A mathematician.", "fullurl": "https://en.wikipedia.org/wiki/Ada",
            "touched": "2026-01-01T00:00:00Z",
        }]},
    }
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = body
    session.get.return_value = response
    monkeypatch.setattr(
        "connectors.wikipedia.documents.settings.wikipedia_page_size", 5,
    )
    batches = list(WikipediaSpec().fetch(session, "ada lovelace", batch_size=10))
    assert len(batches) == 1
    docs, count = batches[0]
    assert count == 1
    assert docs[0].title == "Ada"
    params = session.get.call_args.kwargs["params"]
    assert params["generator"] == "search"
    assert params["gsrsearch"] == "ada lovelace"


def test_wikipedia_fetch_merges_a_page_split_across_continuation_responses(monkeypatch):
    """MediaWiki caps whole-article extracts at one per request and then drops
    `prop=info` from the continuation responses, so a two-page set arrives as:
    response 1 = both pages with `fullurl`, only the first with an `extract`;
    response 2 = the second page's `extract`, and no `fullurl` for anyone.

    Read per-response that loses the second document entirely (no extract means
    no chunks) and would have stripped its URL even if it survived. Both pages
    must come out whole, exactly once each."""
    first = {
        "continue": {"excontinue": 1, "continue": "||info"},
        "query": {"pages": [
            {"pageid": 7, "index": 1, "title": "Ada", "touched": "2026-01-01T00:00:00Z",
             "fullurl": "https://en.wikipedia.org/wiki/Ada", "extract": "A mathematician."},
            {"pageid": 8, "index": 2, "title": "Grace", "touched": "2026-01-01T00:00:00Z",
             "fullurl": "https://en.wikipedia.org/wiki/Grace"},
        ]},
    }
    second = {
        "query": {"pages": [
            {"pageid": 7, "index": 1, "title": "Ada", "touched": "2026-01-01T00:00:00Z"},
            {"pageid": 8, "index": 2, "title": "Grace", "touched": "2026-01-01T00:00:00Z",
             "extract": "A rear admiral."},
        ]},
    }
    session = MagicMock()
    session.get.side_effect = [_json_response(first), _json_response(second)]
    monkeypatch.setattr("connectors.wikipedia.documents.settings.wikipedia_page_size", 5)

    docs = [d for batch, _count in WikipediaSpec().fetch(session, "women in computing",
                                                         batch_size=10) for d in batch]

    assert [(d.doc_id, d.title, d.url) for d in docs] == [
        ("7", "Ada", "https://en.wikipedia.org/wiki/Ada"),
        ("8", "Grace", "https://en.wikipedia.org/wiki/Grace"),
    ]
    assert [c.text for d in docs for c in d.chunks] == ["A mathematician.", "A rear admiral."]


def test_arxiv_entry_is_one_abstract_chunk():
    doc = arxiv_doc({
        "id": "2401.00001v1",
        "title": "Graphs",
        "summary": "We study graphs.",
        "published": "2024-01-01T00:00:00Z",
        "updated": "2024-02-01T00:00:00Z",
        "url": "https://arxiv.org/abs/2401.00001v1",
    })
    assert doc is not None
    assert doc.connector_type.value == "arxiv"
    assert doc.published_at == "2024-01-01"
    assert doc.chunks[0].text == "We study graphs."
    assert doc.acl.tokens == ["public"]


def test_arxiv_fetch_parses_atom(monkeypatch):
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <title>A paper</title>
        <summary>The abstract.</summary>
        <published>2020-05-01T00:00:00Z</published>
        <updated>2020-06-01T00:00:00Z</updated>
        <link rel="alternate" href="https://arxiv.org/abs/1234.5678v1"/>
      </entry>
    </feed>"""
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.content = xml.encode()
    session.get.return_value = response
    monkeypatch.setattr("connectors.arxiv.documents.settings.arxiv_page_size", 5)
    docs, count = next(ArxivSpec().fetch(session, "graph", batch_size=10))
    assert count == 1
    assert docs[0].doc_id == "1234.5678v1"
    assert docs[0].title == "A paper"
    assert "search_query" in session.get.call_args.kwargs["params"]


# =====================================================================
# Wikipedia — search_query / _request_params
# =====================================================================

def test_wikipedia_search_query_empty_without_doc_ids():
    assert wiki_search_query() == {}
    assert wiki_search_query(None) == {}


def test_wikipedia_search_query_wraps_doc_ids_as_strings():
    assert wiki_search_query([123, "456"]) == {"pageids": ["123", "456"]}


def test_wikipedia_request_params_rejects_empty_topic_with_no_pageids():
    with pytest.raises(ValueError, match="non-empty topic"):
        wiki_request_params("", {}, 20)


def test_wikipedia_request_params_category_prefix_walks_categorymembers():
    params = wiki_request_params("Category:Graph databases", {}, 20)
    assert params["generator"] == "categorymembers"
    assert params["gcmtitle"] == "Category:Graph databases"
    assert params["gcmlimit"] == 20
    assert params["gcmnamespace"] == 0


def test_wikipedia_request_params_plain_topic_uses_the_search_generator():
    params = wiki_request_params("graph database", {}, 20)
    assert params["generator"] == "search"
    assert params["gsrsearch"] == "graph database"
    assert params["gsrlimit"] == 20


def test_wikipedia_request_params_pageids_short_circuits_the_generator():
    params = wiki_request_params("ignored topic", {"pageids": ["1", "2"]}, 20)
    assert params["pageids"] == "1|2"
    assert "generator" not in params


# =====================================================================
# Wikipedia — _fetch_with_retry
# =====================================================================

def test_wikipedia_fetch_with_retry_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr("connectors.wikipedia.documents.time.sleep", lambda *_: None)
    session = MagicMock()
    session.get.side_effect = [requests.exceptions.ConnectionError("blip"),
                               _json_response({"query": {"pages": []}})]
    body = wiki_fetch_with_retry(session, {"action": "query"})
    assert body == {"query": {"pages": []}}
    assert session.get.call_count == 2


def test_wikipedia_fetch_with_retry_does_not_retry_a_permanent_failure(monkeypatch):
    monkeypatch.setattr("connectors.wikipedia.documents.time.sleep", lambda *_: None)
    session = MagicMock()
    bad = MagicMock()
    bad.raise_for_status.side_effect = _http_error(404)
    session.get.return_value = bad
    with pytest.raises(requests.exceptions.HTTPError):
        wiki_fetch_with_retry(session, {"action": "query"})
    assert session.get.call_count == 1


def test_wikipedia_fetch_with_retry_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("connectors.wikipedia.documents.time.sleep", lambda *_: None)
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("down")
    with pytest.raises(requests.exceptions.ConnectionError):
        wiki_fetch_with_retry(session, {"action": "query"})
    assert session.get.call_count == WIKI_MAX_RETRIES


# =====================================================================
# Wikipedia — _pages_from
# =====================================================================

def test_wikipedia_pages_from_drops_missing_pages_and_sorts_by_relevance_index():
    body = {"query": {"pages": [
        {"pageid": 2, "index": 2, "title": "B"},
        {"pageid": 1, "index": 1, "title": "A"},
        {"pageid": 3, "index": 3, "title": "Missing", "missing": True},
    ]}}
    pages = wiki_pages_from(body)
    assert [p["pageid"] for p in pages] == [1, 2]


def test_wikipedia_pages_from_empty_without_a_query_or_pages_key():
    assert wiki_pages_from({}) == []
    assert wiki_pages_from({"query": {}}) == []


# =====================================================================
# Wikipedia — _complete_pages / iter_pages
# =====================================================================

def test_wikipedia_iter_pages_chunks_a_pageids_lookup_at_the_per_request_cap():
    ids = [str(i) for i in range(1, WIKI_MAX_IDS_PER_REQUEST + 5)]

    def fake_get(url, params, timeout):
        requested = params["pageids"].split("|")
        pages = [{"pageid": int(pid), "index": int(pid), "title": f"P{pid}",
                  "extract": "Body.", "fullurl": f"https://x/{pid}",
                  "touched": "2026-01-01T00:00:00Z"} for pid in requested]
        return _json_response({"query": {"pages": pages}})

    session = MagicMock()
    session.get.side_effect = fake_get
    pages = list(wiki_iter_pages(session, "", {"pageids": ids}, 20))
    assert session.get.call_count == 2  # one chunk over the 50-id cap
    assert len(pages) == len(ids)


def test_wikipedia_iter_pages_drops_a_page_that_never_gets_an_extract():
    """A disambiguation stub never grows an `extract` no matter how long the
    continuation chain runs, and is therefore never yielded — the caller
    treats a missing extract and a never-arriving one identically."""
    first = {
        "continue": {"excontinue": 1, "continue": "||info"},
        "query": {"pages": [
            {"pageid": 1, "index": 1, "title": "Normal", "fullurl": "https://x/1",
             "extract": "Body.", "touched": "2026-01-01T00:00:00Z"},
            {"pageid": 2, "index": 2, "title": "Stub", "fullurl": "https://x/2",
             "touched": "2026-01-01T00:00:00Z"},
        ]},
    }
    second = {"query": {"pages": [{"pageid": 2, "index": 2, "title": "Stub"}]}}
    session = MagicMock()
    session.get.side_effect = [_json_response(first), _json_response(second)]
    pages = list(wiki_iter_pages(session, "topic", {}, 5))
    assert [p["pageid"] for p in pages] == [1]


# =====================================================================
# Wikipedia — _section_heading
# =====================================================================

def test_wikipedia_section_heading_recognizes_levels_2_through_5():
    assert wiki_section_heading("== Early history ==") == "Early history"
    assert wiki_section_heading("=== Early history ===") == "Early history"
    assert wiki_section_heading("===== Deep nesting =====") == "Deep nesting"


def test_wikipedia_section_heading_rejects_plain_prose():
    assert wiki_section_heading("This is just a paragraph.") is None
    assert wiki_section_heading("== Starts like a heading but never closes") is None


def test_wikipedia_section_heading_rejects_a_heading_with_no_text_inside():
    assert wiki_section_heading("=====") is None


# =====================================================================
# Wikipedia — _chunks_from_extract
# =====================================================================

def test_wikipedia_chunks_from_extract_empty_text_yields_no_chunks():
    assert wiki_chunks_from_extract("1", "") == []
    assert wiki_chunks_from_extract("1", "\n\n  \n") == []


def test_wikipedia_chunks_from_extract_collapses_consecutive_blank_lines():
    chunks = wiki_chunks_from_extract("1", "First.\n\n\nSecond.")
    assert [c.text for c in chunks] == ["First.", "Second."]


def test_wikipedia_chunks_from_extract_tags_paragraphs_with_their_enclosing_section():
    extract = "== Intro ==\nFirst.\nSecond.\n== History ==\nThird."
    chunks = wiki_chunks_from_extract("1", extract)
    assert [(c.text, c.title) for c in chunks] == [
        ("First.", "Intro"), ("Second.", "Intro"), ("Third.", "History"),
    ]


def test_wikipedia_chunks_from_extract_text_before_any_heading_has_no_section():
    chunks = wiki_chunks_from_extract("1", "Lead paragraph.\n== Intro ==\nBody.")
    assert chunks[0].title is None
    assert chunks[1].title == "Intro"


def test_wikipedia_chunks_from_extract_chunk_ids_are_sequential_per_document():
    chunks = wiki_chunks_from_extract("42", "A.\nB.")
    assert [c.chunk_id for c in chunks] == ["42::p0", "42::p1"]
    assert [c.recursive_chunk_id for c in chunks] == ["42::p0", "42::p1"]


# =====================================================================
# Wikipedia — _preview_text
# =====================================================================

def test_wikipedia_preview_text_none_without_chunks():
    assert wiki_preview_text([]) is None


def test_wikipedia_preview_text_truncates_to_500_chars():
    chunk = Chunk(chunk_id="1::p0", text="x" * 600, title=None, vector=None)
    assert wiki_preview_text([chunk]) == "x" * 500


# =====================================================================
# Wikipedia — to_source_document edge cases
# =====================================================================

def test_wikipedia_to_source_document_none_without_a_pageid():
    assert wiki_doc({"title": "No id", "extract": "Some text."}) is None


def test_wikipedia_to_source_document_none_when_there_is_no_extractable_prose():
    # A disambiguation stub: empty extract.
    assert wiki_doc({"pageid": 5, "title": "Disambiguation", "extract": ""}) is None
    # An extract that is only a heading, with no prose beneath it.
    assert wiki_doc({"pageid": 5, "title": "Stub", "extract": "== See also =="}) is None


def test_wikipedia_to_source_document_derives_updated_at_from_touched_date_only():
    doc = wiki_doc({"pageid": 9, "title": "T", "extract": "Body.",
                    "touched": "2026-03-04T12:34:56Z"})
    assert doc.updated_at == "2026-03-04"
    assert doc.published_at == ""  # no creation date in this API


def test_wikipedia_to_source_document_url_is_none_without_fullurl():
    doc = wiki_doc({"pageid": 9, "title": "T", "extract": "Body."})
    assert doc.url is None


def test_wikipedia_to_source_document_preserves_the_raw_payload():
    hit = {"pageid": 9, "title": "T", "extract": "Body.", "touched": "2026-01-01T00:00:00Z"}
    doc = wiki_doc(hit)
    assert doc.raw == hit


# =====================================================================
# Wikipedia — prepare_chunk_text
# =====================================================================

def test_wikipedia_prepare_chunk_text_without_a_section():
    doc = wiki_doc({"pageid": 1, "title": "Ada", "extract": "A mathematician."})
    text = wiki_prepare_chunk_text(doc, doc.chunks[0])
    assert text == "Document: Ada\n\nText:\nA mathematician."


def test_wikipedia_prepare_chunk_text_includes_the_section_when_present():
    doc = wiki_doc({"pageid": 1, "title": "Ada", "extract": "== Life ==\nBorn in London."})
    text = wiki_prepare_chunk_text(doc, doc.chunks[0])
    assert text == "Document: Ada\nSection: Life\n\nText:\nBorn in London."


# =====================================================================
# Wikipedia — fetch_batches budgeting
# =====================================================================

def test_wikipedia_fetch_batches_skips_pages_with_no_extractable_text(monkeypatch):
    pages = [
        {"pageid": 1, "title": "Stub", "extract": ""},
        {"pageid": 2, "title": "Real", "extract": "Body."},
    ]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    batches = list(wiki_fetch_batches(MagicMock(), "topic", batch_size=100))
    docs = [d for batch, _ in batches for d in batch]
    assert [d.doc_id for d in docs] == ["2"]


def test_wikipedia_fetch_batches_stops_immediately_when_docs_wanted_is_already_zero(monkeypatch):
    pages = [{"pageid": 1, "title": "P1", "extract": "Body."}]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    batches = list(wiki_fetch_batches(MagicMock(), "topic", batch_size=100, docs_wanted=lambda: 0))
    assert batches == []


def test_wikipedia_fetch_batches_stops_mid_stream_once_docs_wanted_is_exhausted(monkeypatch):
    """docs_wanted() is checked once per page, and this connector's own flush
    (not this check) is what gives a caller the chance to decrement it — so
    quota exhaustion is only noticed on the NEXT page after a flush, one page
    later than it would be with an immediate per-doc counter. batch_size=1
    forces a flush per doc so that lag is visible at all; with a wanted-quota
    of 1 (and that one-page-late timing), two of the three available docs come
    through before fetching stops, which is still well short of all three."""
    pages = [{"pageid": i, "title": f"P{i}", "extract": "Body."} for i in range(1, 4)]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    remaining = [1]
    docs = []
    for batch, _count in wiki_fetch_batches(MagicMock(), "topic", batch_size=1,
                                             docs_wanted=lambda: remaining[0]):
        docs.extend(batch)
        remaining[0] -= len(batch)
    assert [d.doc_id for d in docs] == ["1", "2"]


def test_wikipedia_fetch_batches_flushes_once_the_chunk_budget_is_exceeded(monkeypatch):
    pages = [
        {"pageid": 1, "title": "P1", "extract": "A.\nB."},
        {"pageid": 2, "title": "P2", "extract": "C.\nD."},
    ]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    batches = list(wiki_fetch_batches(MagicMock(), "topic", batch_size=3))
    assert [(len(docs), count) for docs, count in batches] == [(1, 2), (1, 2)]


def test_wikipedia_fetch_batches_a_single_oversized_doc_is_still_yielded_alone(monkeypatch):
    pages = [{"pageid": 1, "title": "Huge", "extract": "A.\nB.\nC.\nD.\nE."}]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    batches = list(wiki_fetch_batches(MagicMock(), "topic", batch_size=2))
    assert len(batches) == 1
    docs, count = batches[0]
    assert len(docs) == 1 and count == 5


def test_wikipedia_fetch_batches_uses_the_batch_sizers_current_size(monkeypatch):
    pages = [
        {"pageid": 1, "title": "P1", "extract": "A."},
        {"pageid": 2, "title": "P2", "extract": "B."},
    ]
    monkeypatch.setattr("connectors.wikipedia.documents.iter_pages", lambda *a, **k: iter(pages))
    sizer = MagicMock()
    sizer.current_size = 1  # smaller than batch_size, so each page must flush alone
    batches = list(wiki_fetch_batches(MagicMock(), "topic", batch_size=100, batch_sizer=sizer))
    assert [len(docs) for docs, _ in batches] == [1, 1]


# =====================================================================
# WikipediaSpec plumbing
# =====================================================================

def test_wikipedia_spec_open_transport_sets_the_configured_user_agent(monkeypatch):
    monkeypatch.setattr("connectors.wikipedia.spec.settings.http_user_agent", "test-agent/1.0")
    session = WikipediaSpec().open_transport()
    assert session.headers["User-Agent"] == "test-agent/1.0"


def test_wikipedia_spec_acl_lookup_document_normalizes_pageid_to_id():
    out = WikipediaSpec().acl_lookup_document({"pageid": 7, "title": "Ada"})
    assert out["id"] == "7"
    assert out["title"] == "Ada"


def test_wikipedia_spec_acl_lookup_document_falls_back_to_the_id_field():
    out = WikipediaSpec().acl_lookup_document({"id": 7})
    assert out["id"] == "7"


def test_wikipedia_spec_acl_lookup_document_none_without_either_field():
    assert WikipediaSpec().acl_lookup_document({"title": "No id here"}) is None


def test_wikipedia_spec_acl_scope_ids_and_audiences_are_always_empty():
    spec = WikipediaSpec()
    assert spec.acl_scope_ids([{"id": "1"}]) == []
    assert spec.acl_scope_audiences(None, "idx", ["1"]) == {}


def test_wikipedia_acl_strategy_always_returns_the_public_constant():
    decision = WIKIPEDIA_ACL_STRATEGY.build({"anything": "goes"}, context=None)
    assert decision.tokens == ["public"]
    assert decision.dropped is False


def test_wikipedia_spec_people_is_a_null_provider_that_finds_and_enriches_nobody():
    spec = WikipediaSpec()
    assert spec.people.observe([]) is None
    assert _run(spec.people.discover(PeopleContext())) == []
    assert _run(spec.people.enrich([], PeopleContext())) == PeopleResult()


# =====================================================================
# arXiv — search_query
# =====================================================================

def test_arxiv_search_query_empty_without_doc_ids():
    assert arxiv_search_query() == {}
    assert arxiv_search_query(None) == {}


def test_arxiv_search_query_wraps_doc_ids_as_strings():
    assert arxiv_search_query([1234, "5678v2"]) == {"ids": ["1234", "5678v2"]}


# =====================================================================
# arXiv — _fetch
# =====================================================================

def test_arxiv_fetch_retries_a_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr("connectors.arxiv.documents.time.sleep", lambda *_: None)
    ok = MagicMock()
    ok.raise_for_status.return_value = None
    ok.content = b"<feed/>"
    session = MagicMock()
    session.get.side_effect = [requests.exceptions.ConnectionError("blip"), ok]
    body = arxiv_fetch(session, {"start": 0})
    assert body == b"<feed/>"
    assert session.get.call_count == 2


def test_arxiv_fetch_does_not_retry_a_permanent_failure(monkeypatch):
    monkeypatch.setattr("connectors.arxiv.documents.time.sleep", lambda *_: None)
    bad = MagicMock()
    bad.raise_for_status.side_effect = _http_error(404)
    session = MagicMock()
    session.get.return_value = bad
    with pytest.raises(requests.exceptions.HTTPError):
        arxiv_fetch(session, {"start": 0})
    assert session.get.call_count == 1


def test_arxiv_fetch_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("connectors.arxiv.documents.time.sleep", lambda *_: None)
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("down")
    with pytest.raises(requests.exceptions.ConnectionError):
        arxiv_fetch(session, {"start": 0})
    assert session.get.call_count == ARXIV_MAX_RETRIES


# =====================================================================
# arXiv — _entry_dict
# =====================================================================

def _parse_entry(xml_entry: str) -> dict:
    feed = f'<feed xmlns="http://www.w3.org/2005/Atom">{xml_entry}</feed>'
    root = ET.fromstring(feed)
    return arxiv_entry_dict(root.find(f"{ATOM}entry"))


def test_arxiv_entry_dict_picks_the_alternate_link_over_other_rels():
    entry = _parse_entry("""
        <entry>
          <id>http://arxiv.org/abs/1111.2222v1</id>
          <title>T</title><summary>S</summary>
          <published>2020-01-01T00:00:00Z</published><updated>2020-01-02T00:00:00Z</updated>
          <link rel="related" href="https://example.com/related"/>
          <link rel="alternate" href="https://arxiv.org/abs/1111.2222v1"/>
        </entry>
    """)
    assert entry["url"] == "https://arxiv.org/abs/1111.2222v1"


def test_arxiv_entry_dict_falls_back_to_the_atom_id_when_no_alternate_link():
    entry = _parse_entry("""
        <entry>
          <id>http://arxiv.org/abs/1111.2222v1</id>
          <title>T</title><summary>S</summary>
          <published>2020-01-01T00:00:00Z</published><updated>2020-01-02T00:00:00Z</updated>
          <link rel="related" href="https://example.com/related"/>
        </entry>
    """)
    assert entry["url"] == "http://arxiv.org/abs/1111.2222v1"


def test_arxiv_entry_dict_strips_the_abs_path_prefix_from_the_atom_id():
    entry = _parse_entry("""
        <entry>
          <id>http://arxiv.org/abs/2401.00001v2</id>
          <title>T</title><summary>S</summary>
          <published>2020-01-01T00:00:00Z</published><updated>2020-01-02T00:00:00Z</updated>
        </entry>
    """)
    assert entry["id"] == "2401.00001v2"


# =====================================================================
# arXiv — iter_entries
# =====================================================================

def _feed(entries_xml: str) -> bytes:
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries_xml}</feed>'.encode()


def _ok_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.content = content
    return resp


def test_arxiv_iter_entries_rejects_empty_topic_with_no_ids():
    session = MagicMock()
    with pytest.raises(ValueError, match="non-empty search phrase"):
        list(arxiv_iter_entries(session, "", {}, 5))
    session.get.assert_not_called()


def test_arxiv_iter_entries_stops_on_the_apis_error_sentinel(monkeypatch):
    """arXiv answers a malformed query with HTTP 200 and a single
    <entry><title>Error</title> rather than an HTTP error status."""
    session = MagicMock()
    session.get.return_value = _ok_response(_feed("<entry><title>Error</title></entry>"))
    entries = list(arxiv_iter_entries(session, "nonsense query!!!", {}, 5))
    assert entries == []
    assert session.get.call_count == 1


def test_arxiv_iter_entries_paginates_until_a_short_page(monkeypatch):
    monkeypatch.setattr("connectors.arxiv.documents.time.sleep", lambda *_: None)

    def entry_xml(i):
        return (f"<entry><id>http://arxiv.org/abs/{i}v1</id><title>P{i}</title>"
                f"<summary>S{i}</summary><published>2020-01-01T00:00:00Z</published>"
                f"<updated>2020-01-01T00:00:00Z</updated></entry>")

    page1 = _ok_response(_feed("".join(entry_xml(i) for i in range(2))))  # full page
    page2 = _ok_response(_feed(entry_xml(2)))  # short page -> stop
    session = MagicMock()
    session.get.side_effect = [page1, page2]
    entries = list(arxiv_iter_entries(session, "graph", {}, 2))
    assert [e["id"] for e in entries] == ["0v1", "1v1", "2v1"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["params"]["start"] == 2


def test_arxiv_iter_entries_an_ids_lookup_never_paginates():
    resp = _ok_response(_feed(
        "<entry><id>http://arxiv.org/abs/1v1</id><title>T</title>"
        "<summary>S</summary><published>2020-01-01T00:00:00Z</published>"
        "<updated>2020-01-01T00:00:00Z</updated></entry>"))
    session = MagicMock()
    session.get.return_value = resp
    entries = list(arxiv_iter_entries(session, "", {"ids": ["1"]}, 1))
    assert len(entries) == 1
    # Even though the page is "full" (1 entry == page_size 1), an ids lookup
    # stops after one call instead of asking for a next page.
    assert session.get.call_count == 1


# =====================================================================
# arXiv — to_source_document edge cases
# =====================================================================

def test_arxiv_to_source_document_none_without_an_id():
    assert arxiv_doc({"summary": "Has text but no id."}) is None


def test_arxiv_to_source_document_none_without_a_summary():
    assert arxiv_doc({"id": "1234.5678v1", "summary": "  "}) is None


def test_arxiv_to_source_document_url_is_none_when_missing():
    doc = arxiv_doc({"id": "1234.5678v1", "summary": "Text.", "url": None})
    assert doc.url is None


# =====================================================================
# arXiv — prepare_chunk_text
# =====================================================================

def test_arxiv_prepare_chunk_text_includes_the_publish_date_when_present():
    doc = arxiv_doc({"id": "1", "title": "A paper", "summary": "Abstract text.",
                     "published": "2024-01-01T00:00:00Z"})
    text = arxiv_prepare_chunk_text(doc, doc.chunks[0])
    assert text == "Document published: 2024-01-01\nDocument: A paper\n\nText:\nAbstract text."


def test_arxiv_prepare_chunk_text_omits_the_publish_date_when_absent():
    doc = arxiv_doc({"id": "1", "title": "A paper", "summary": "Abstract text."})
    text = arxiv_prepare_chunk_text(doc, doc.chunks[0])
    assert text == "Document: A paper\n\nText:\nAbstract text."


# =====================================================================
# arXiv — fetch_batches budgeting
# =====================================================================

def test_arxiv_fetch_batches_skips_entries_with_no_summary(monkeypatch):
    entries = [{"id": "1", "title": "No abstract", "summary": ""},
              {"id": "2", "title": "Has abstract", "summary": "Text."}]
    monkeypatch.setattr("connectors.arxiv.documents.iter_entries", lambda *a, **k: iter(entries))
    batches = list(arxiv_fetch_batches(MagicMock(), "topic", batch_size=100))
    docs = [d for batch, _ in batches for d in batch]
    assert [d.doc_id for d in docs] == ["2"]


def test_arxiv_fetch_batches_stops_mid_stream_once_docs_wanted_is_exhausted(monkeypatch):
    """Same one-page-late timing as Wikipedia's equivalent test, since both
    connectors share the same fetch_batches shape — see that test's docstring."""
    entries = [{"id": str(i), "title": f"P{i}", "summary": "Text."} for i in range(1, 4)]
    monkeypatch.setattr("connectors.arxiv.documents.iter_entries", lambda *a, **k: iter(entries))
    remaining = [1]
    docs = []
    for batch, _count in arxiv_fetch_batches(MagicMock(), "topic", batch_size=1,
                                              docs_wanted=lambda: remaining[0]):
        docs.extend(batch)
        remaining[0] -= len(batch)
    assert [d.doc_id for d in docs] == ["1", "2"]


def test_arxiv_fetch_batches_flushes_once_the_chunk_budget_is_exceeded(monkeypatch):
    # arXiv is always one chunk per doc, so batch_size=1 forces a flush per doc.
    entries = [{"id": "1", "title": "P1", "summary": "A."},
              {"id": "2", "title": "P2", "summary": "B."}]
    monkeypatch.setattr("connectors.arxiv.documents.iter_entries", lambda *a, **k: iter(entries))
    batches = list(arxiv_fetch_batches(MagicMock(), "topic", batch_size=1))
    assert [(len(docs), count) for docs, count in batches] == [(1, 1), (1, 1)]


# =====================================================================
# ArxivSpec plumbing
# =====================================================================

def test_arxiv_spec_open_transport_sets_the_configured_user_agent(monkeypatch):
    monkeypatch.setattr("connectors.arxiv.spec.settings.http_user_agent", "test-agent/2.0")
    session = ArxivSpec().open_transport()
    assert session.headers["User-Agent"] == "test-agent/2.0"


def test_arxiv_spec_acl_lookup_document_normalizes_id_to_string():
    out = ArxivSpec().acl_lookup_document({"id": 1234})
    assert out["id"] == "1234"


def test_arxiv_spec_acl_lookup_document_none_without_an_id():
    assert ArxivSpec().acl_lookup_document({"title": "No id"}) is None


def test_arxiv_spec_acl_scope_ids_and_audiences_are_always_empty():
    spec = ArxivSpec()
    assert spec.acl_scope_ids([{"id": "1"}]) == []
    assert spec.acl_scope_audiences(None, "idx", ["1"]) == {}


def test_arxiv_acl_strategy_always_returns_the_public_constant():
    decision = ARXIV_ACL_STRATEGY.build({"anything": "goes"}, context=None)
    assert decision.tokens == ["public"]
    assert decision.dropped is False


def test_arxiv_spec_people_is_a_null_provider_that_finds_and_enriches_nobody():
    spec = ArxivSpec()
    assert spec.people.observe([]) is None
    assert _run(spec.people.discover(PeopleContext())) == []
    assert _run(spec.people.enrich([], PeopleContext())) == PeopleResult()
