"""Lightweight redaction for common PII in MedMate observability text.

This is a best-effort regex-based filter, not a complete de-identification
system. It does not reliably identify unlabelled names or other free-text PII.
"""

import re


_LABELLED_FIELDS = (
    (
        re.compile(
            r"(?P<label>\b(?:patient(?:'s)?\s+name|full\s+name|name)\s*[:=]\s*)"
            r"(?P<value>[^\r\n,;.!?]+)",
            re.IGNORECASE,
        ),
        "[REDACTED_NAME]",
    ),
    (
        re.compile(
            r"(?P<label>\b(?:date\s+of\s+birth|dob)\s*[:=]\s*)"
            r"(?P<value>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
        "[REDACTED_DOB]",
    ),
    (
        re.compile(
            r"(?P<label>\b(?:medical\s+record\s+number|mrn|patient\s+id)\s*[:=]\s*)"
            r"(?P<value>[A-Z0-9][A-Z0-9-]{2,})",
            re.IGNORECASE,
        ),
        "[REDACTED_ID]",
    ),
    (
        re.compile(
            r"(?P<label>\b(?:home|mailing|street)?\s*address\s*[:=]\s*)"
            r"(?P<value>[^\r\n]+)",
            re.IGNORECASE,
        ),
        "[REDACTED_ADDRESS]",
    ),
)

_PATTERNS = (
    (
        re.compile(
            r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
        ),
        "[REDACTED_SSN]",
    ),
    (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
)


def redact_pii(text: str) -> str:
    """Redact common labelled PII, email addresses, phone numbers, and SSNs."""
    for pattern, replacement in _LABELLED_FIELDS:
        text = pattern.sub(
            lambda match: f"{match.group('label')}{replacement}",
            text,
        )

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text
