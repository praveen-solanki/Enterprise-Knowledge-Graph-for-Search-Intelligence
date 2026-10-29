"""Junk-endpoint gate: flags entity surface forms that are poor graph
endpoints (pronouns, generic references, bare numbers/dates/times, single
characters, empty strings) so relationships like ("it")-[RELATED_TO]->("the
team") are dropped before entering the graph. Heuristic only, no LLM — kept
cheap and deterministic.
"""
from __future__ import annotations

import re

_PRONOUNS = {
    "it", "he", "she", "they", "them", "we", "us", "you", "i", "me", "him", "her", "his",
    "hers", "their", "theirs", "our", "ours", "your", "yours", "this", "that", "these",
    "those", "who", "whom", "which", "what", "whoever", "someone", "anyone", "everyone",
    "one", "ones", "itself", "themselves",
}
_GENERIC = {
    "the company", "the team", "the organization", "the org", "the group", "the system",
    "the platform", "the product", "the service", "the document", "the page", "the file",
    "the user", "the users", "the customer", "the customers", "the client", "the clients",
    "the project", "the process", "the data", "the report", "the meeting", "the following",
    "the above", "the below", "here", "there", "etc", "n/a", "na", "none", "null", "unknown",
    "others", "other", "various", "several", "many", "some", "all", "any",
}
_DETERMINERS = {"a", "an", "the", "and", "or", "but", "of", "to", "for", "in", "on", "at", "by"}

_BARE_VALUE = re.compile(
    r"^\s*[\$€£]?\d[\d,.\s]*\s*(%|percent|usd|eur|gbp|k|m|bn|million|billion)?\s*$", re.I)
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
# A date must carry actual date digits — a month PREFIX alone must NOT match ("Marketing", "Maria").
_DATE_LIKE = re.compile(
    r"^\s*("
    r"\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?"                       # 2026-03-04, 3/4, 04/03/2026
    r"|\d{4}"                                                  # bare 4-digit year
    r"|q[1-4]\s*\d{2,4}"                                       # Q1 2026
    r"|" + _MONTH + r",?\s+\d{1,4}(,?\s*\d{2,4})?"             # March 4, 2026 / March 2026
    r"|\d{1,2}(st|nd|rd|th)?\s+" + _MONTH +                    # 2nd March
    r")\s*$", re.I)
_TIME_LIKE = re.compile(r"^\s*\d{1,2}:\d{2}(\s*[ap]m)?\s*$", re.I)

# Entity types that are values, not prose entities — poor edge endpoints on their own.
_VALUE_TYPES = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "NUMBER"}


def is_junk_endpoint(name: str, entity_type: str = "") -> bool:
    n = (name or "").strip()
    if not n:
        return True
    low = n.lower()
    if len(n) <= 1:
        return True
    if len(n) > 120:                      # runaway/garbled endpoint
        return True
    # Uppercase short tokens are acronyms (US, IT, CA, HR, AI), NOT function words — exempt
    # them from the pronoun/determiner/generic checks below.
    is_acronym = n.isupper() and len(n) <= 5
    if not is_acronym and (low in _PRONOUNS or low in _GENERIC or low in _DETERMINERS):
        return True
    if (entity_type or "").upper() in _VALUE_TYPES:
        return True
    if _BARE_VALUE.match(n) or _DATE_LIKE.match(n) or _TIME_LIKE.match(n):
        return True
    if not re.search(r"[a-z0-9]", low):    # punctuation-only
        return True
    return False
