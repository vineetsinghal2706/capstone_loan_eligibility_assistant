"""redact.py — PII redaction at the boundary (Session 10, reused from Session 4).

The exact regexes from the Session 4 audit-logging lab, packaged as a module.
Used in two places:
  - guardrails.check_input  : detect PII in the incoming question (log redacted)
  - guardrails.check_output : mask PII before the answer leaves the API

Deterministic regex redaction is the lab-grade approach; production systems
typically layer an NER-based engine (e.g. Microsoft Presidio) on top of
patterns like these.
"""

import re

PII_PATTERNS = [
    ("EMAIL",   re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PAN",     re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE",   re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")),
]


def redact(text: str) -> str:
    """Replace every PII match with a typed placeholder, e.g. <PHONE_REDACTED>."""
    for name, pattern in PII_PATTERNS:
        text = pattern.sub(f"<{name}_REDACTED>", text)
    return text


def find_pii(text: str) -> list[str]:
    """Return the list of PII types present in the text (empty if clean)."""
    return [name for name, pattern in PII_PATTERNS if pattern.search(text)]
