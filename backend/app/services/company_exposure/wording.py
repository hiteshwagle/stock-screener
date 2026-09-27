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


# Wording that asserts a commercial relationship with a counterparty.
CUSTOMER = re.compile(
    r"\b(customers?|clients?|suppl(?:y|ies|ied|ier|iers|ying)|sell(?:s|ing)?|sold|"
    r"sales to|purchas(?:e|es|ed|er|ers|ing)|orders?|ordered|contracts?|contracted|"
    r"agreements?|partner(?:s|ship|ships)?|buy(?:s|ing)?|bought|ship(?:s|ped|ping)? to)\b"
    r"|顧客|取引先|供給|納入|受注|客戶|客户|供應|供应|採購|采购|訂單|订单",
    re.IGNORECASE,
)

# Wording that asserts a product is shipping or in operation.
SHIPPING = re.compile(
    r"\b(ship(?:s|ped|ping|ment|ments)?|deliver(?:s|ed|ing|y|ies)?|"
    r"in (?:volume |mass |commercial )?production|(?:volume|mass) production|"
    r"operat(?:es|ing|ional)|in (?:commercial )?(?:service|operation)|deployed|installed)\b"
    r"|量産|出荷|稼働|量產|出貨|出货|交付|投產|投产",
    re.IGNORECASE,
)

# Wording that asserts a product can be bought.
AVAILABLE = re.compile(
    r"\b(available|availability|launch(?:ed|es)?|released?|on sale|offer(?:s|ed|ing)?|"
    r"sell(?:s|ing)?|sold|introduced)\b"
    r"|販売|発売|提供|上市|推出|銷售|销售|供貨|供货",
    re.IGNORECASE,
)
