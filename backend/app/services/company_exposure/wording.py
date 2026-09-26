"""Deterministic wording checks shared by claim and synthesis validation.

Provider output is untrusted; these helpers only ever *downgrade* support:
negated clauses do not affirm a relationship, and contrastive joins split a
sentence so two unrelated statements cannot be read as one link.
"""

from __future__ import annotations

import re

SENTENCES = re.compile(r"(?<=[.!?。！？])\s*")
NEGATION = re.compile(
    r"\b(not|no longer|never|has not|have not|yet to|without)\b|していない|しておらず|未|尚未|沒有|没有|並未|并未",
    re.IGNORECASE,
)

# Contrastive joins separate clauses, so "supports HBM testing but has not
# begun shipments" still affirms the role while negating only the status.
CONTRAST = re.compile(
    r"[;；]|\b(?:but|however|although|though|whereas|while)\b|しかし|但是|然而",
    re.IGNORECASE,
)

# An explicit end of a business, product or relationship.
EXIT = re.compile(
    r"\b(exit(?:ed|ing)?|discontinu(?:e|ed|ing)|divest(?:ed|iture|ing)?|"
    r"dispos(?:ed|al) of|sold (?:the|our|its)|wound down|wind(?:ing)? down|"
    r"ceased|terminat(?:ed|ion)|no longer)\b"
    r"|撤退|売却|終了|停止|退出|出售|終止|终止",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCES.split(text) if s.strip()]


def clauses(quotes: list[str]) -> list[str]:
    """Clauses of the quoted sentences, split at contrastive joins."""

    return [
        clause
        for quote in quotes
        for sentence in sentences(quote)
        for clause in CONTRAST.split(sentence)
        if clause.strip()
    ]


def affirmed(clause: str) -> bool:
    return not NEGATION.search(clause)
