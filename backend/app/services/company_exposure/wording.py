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


def mentions(text: str, term: str) -> bool:
    """Whether ``text`` names ``term`` as a word, not inside another word.

    "AI" is not in "available" and "X1" is not in "X100", but "HBM" is in
    "HBM3E" and "tester" in "testers". Scripts without word boundaries
    (CJK) fall back to substring matching.
    """

    return occurrences(text, term) > 0


def occurrences(text: str, term: str) -> int:
    """How often ``text`` names ``term`` as a word (see ``mentions``)."""

    return len(mention_spans(text, term))


def mention_spans(text: str, term: str) -> list[tuple[int, int]]:
    """Where ``text`` names ``term`` as a word (see ``mentions``)."""

    if not term or not text:
        return []
    if not term.isascii():
        pattern = re.escape(term)
    else:
        before = r"(?<![0-9])" if term[0].isdigit() else r"(?<![A-Za-z])"
        after = r"(?![0-9])" if term[-1].isdigit() else r"(?:e?s)?(?![A-Za-z])"
        pattern = before + re.escape(term) + after
    return [m.span() for m in re.finditer(pattern, text, re.IGNORECASE)]


def affirmed(clause: str) -> bool:
    return not NEGATION.search(clause)


_NO_LONGER = re.compile(r"\bno longer\b", re.IGNORECASE)


def affirmed_exit(clause: str) -> bool:
    """An exit the clause asserts rather than denies.

    "no longer offers" is exit wording, not a negation of it; "has not
    discontinued" denies the exit.
    """

    return not NEGATION.search(_NO_LONGER.sub(" ", clause))


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


# A product or activity serving an application ("supports HBM testing",
# "designed for AI servers", "HBM向け"). Co-occurrence alone ("ET-9000 revenue
# and HBM demand both increased") asserts no relationship.
SERVES = re.compile(
    r"\b(support(?:s|ed|ing)?|test(?:s|ed|ing|er|ers)?|(?:designed|built|optimi[sz]ed|"
    r"qualified|certified|intended|used)\s+(?:for|in|with|by)|uses?|enabl(?:e|es|ed|ing)|"
    r"serv(?:e|es|ed|ing)|target(?:s|ed|ing)?|power(?:s|ed|ing)?|deploy(?:s|ed|ing)?|"
    r"appl(?:y|ies|ied)\s+to|compatible\s+with|address(?:es|ed|ing)?|inspect(?:s|ed|ing|ion)?|"
    r"measur(?:e|es|ed|ing)|packag(?:e|es|ed|ing)|assembl(?:e|es|ed|y|ing))\b"
    r"|向け|対応|用途|用於|用于|適用|适用|支持|支援|測試|测试|テスト|検査|檢測|检测",
    re.IGNORECASE,
)

# The issuer offering a product ("We sell the ET-9000", "our ET-9000 line").
OFFERS = re.compile(
    r"\b(offer(?:s|ed|ing)?|sell(?:s|ing)?|sold|ship(?:s|ped|ping)?|provid(?:e|es|ed|ing)|"
    r"suppl(?:y|ies|ied|ying)|market(?:s|ed|ing)?|launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|"
    r"manufactur(?:e|es|ed|ing)|produc(?:e|es|ed|ing|t|ts)|mak(?:e|es|ing)|made|"
    r"develop(?:s|ed|ing)?|deliver(?:s|ed|ing)?|portfolio|our)\b"
    r"|販売|提供|製造|生産|出荷|製品|銷售|销售|製造|生產|生产|產品|产品",
    re.IGNORECASE,
)

# The issuer itself producing or selling something ("We manufacture HBM
# products"): the subject must be the issuer, the object follows the verb.
PRODUCES = re.compile(
    r"\b(manufactur(?:e|es|ed|ing)|produc(?:e|es|ed|ing)|mak(?:e|es|ing)|made|"
    r"fabricat(?:e|es|ed|ing)|develop(?:s|ed|ing)?|design(?:s|ed|ing)?|"
    r"suppl(?:y|ies|ied|ying)|sell(?:s|ing)?|sold|ship(?:s|ped|ping)?|"
    r"offer(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
ISSUER_SUBJECT = re.compile(r"\b(we|our\s+company|the\s+company)\b", re.IGNORECASE)
# CJK wording is verb-final ("当社はHBMを製造"): subject and verb anywhere.
PRODUCES_CJK = re.compile(r"製造|生産|生產|生产|販売|銷售|销售|出荷")
ISSUER_SUBJECT_CJK = re.compile(r"当社|弊社|本公司|我们|我們|本集團|本集团")
# CJK clauses are split into comma-delimited segments; a segment naming
# another party or an application ("customers use ... to produce HBM",
# "equipment for HBM production") is someone else's production.
CJK_SEGMENT = re.compile(r"[，、,;；：:]")
CJK_OTHER_ACTOR = re.compile(
    r"客户|客戶|顧客|お客様|用户|用戶|使用|用于|用於|用来|用來|向け|のための|用の"
)

# Where a verb's subject or object phrase ends: a relative or subordinate
# clause, an infinitive or a preposition introduces someone else's action.
PHRASE_BOUNDARY = re.compile(
    r"\b(?:that|which|who|whom|whose|where|when|while|to|for|used|using|"
    r"with|by|so|because|customers?)\b|[,;:()]",
    re.IGNORECASE,
)
# Where a clause hands over to another predicate ("Acme, which offers
# ET-9000"); prepositions and parentheses stay inside the object phrase
# ("supports testing for high-bandwidth memory (HBM)").
CLAUSE_BOUNDARY = re.compile(
    r"\b(?:that|which|who|whom|whose|where|when|while|so|because)\b|[,;:]",
    re.IGNORECASE,
)

# A segment or subsidiary belonging to the issuer.
PART_OF = re.compile(
    r"\b(segments?|subsidiar(?:y|ies)|divisions?|business\s+units?|units?\s+of|part\s+of|"
    r"wholly[- ]owned|owned\s+by)\b|セグメント|子会社|事業部|部門|分部|子公司",
    re.IGNORECASE,
)
