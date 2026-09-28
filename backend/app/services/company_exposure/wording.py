"""Deterministic wording checks shared by claim and synthesis validation.

Provider output is untrusted; these helpers only ever *downgrade* support:
negated clauses do not affirm a relationship, and contrastive joins split a
sentence so two unrelated statements cannot be read as one link.
"""

from __future__ import annotations

import re

# A sentence ends at CJK terminal punctuation, or at "." "!" "?" followed by
# whitespace; a period inside a number ("20.5") or after an abbreviation
# ("Example Corp. offers ET-9000") does not end one.
SENTENCES = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")
_ABBREVIATIONS = frozenset(
    {
        "corp", "inc", "co", "ltd", "llc", "plc", "bhd", "no", "nos", "vs",
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "e.g", "i.e", "etc",
        "approx", "u.s", "u.k", "fig", "dept", "est",
    }
)  # fmt: skip
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
    parts, start = [], 0
    for match in SENTENCES.finditer(text):
        if (
            match.end() == match.start()
            and text[match.start() - 1 : match.start()] not in "。！？"
        ):
            continue
        head = text[start : match.start()]
        last = head.rsplit(None, 1)[-1] if head.split() else ""
        if last.endswith(".") and (
            last[:-1].casefold().lstrip("(") in _ABBREVIATIONS
            or (len(last) == 2 and last[0].isupper())
        ):
            continue  # "Corp." or an initial, not a sentence end
        parts.append(head)
        start = match.end()
    parts.append(text[start:])
    return [s for s in parts if s.strip()]


# Finite verbs that give each side of an "and" its own predicate.
FINITE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|won|wins|grew|grows|rose|rises|fell|falls|"
    r"increased|decreased|declined|remains?|remained|became|becomes|received|"
    r"launched|reported|supports?|supported|serves?|served|uses?|used|sells?|sold|"
    r"ships?|shipped|offers?|offered|makes?|made|manufactures?|manufactured|"
    r"produces?|produced|delivers?|delivered|targets?|targeted|enables?|enabled|"
    r"powers?|powered|buys?|bought|purchases?|purchased|discontinued|exited|"
    r"ceased|terminated|divested)\b",
    re.IGNORECASE,
)
_AND = re.compile(r",?\s*\band\b\s*", re.IGNORECASE)
# Adverbs that may precede a shared subject's second verb ("and also supports").
_ADVERBS = frozenset(
    {"also", "currently", "now", "further", "additionally", "recently", "still"}
)


def _own_subject(part: str) -> bool:
    """Whether ``part`` names a subject before its first finite verb."""

    verb = FINITE.search(part)
    if verb is None:
        return False
    words = [w for w in part[: verb.start()].split() if w.casefold() not in _ADVERBS]
    return bool(words)


def _coordinated(clause: str) -> list[str]:
    """Split "A rose, and our X200 supports B" into its two predicates.

    An "and" separates clauses only when both sides carry their own finite
    verb and the right side its own subject; "supports DDR5 and HBM
    testing", "NVIDIA and AMD are customers" and "is available and supports
    HBM testing" stay whole.
    """

    parts = _AND.split(clause)
    joined, current = [], parts[0]
    for part in parts[1:]:
        if FINITE.search(current) and _own_subject(part):
            joined.append(current)
            current = part
        else:
            current = f"{current} and {part}"
    return [*joined, current]


def denied_conjuncts(clause: str) -> list[str]:
    """Conjuncts of ``clause`` that a negation or exit governs.

    A conjunct with its own finite verb carries its own polarity ("does not
    support PCIe and supports HBM testing" affirms HBM); one without a verb
    shares the preceding verb's ("does not support PCIe and HBM testing"
    denies both).
    """

    denied, governed = [], False
    for part in _AND.split(clause):
        if FINITE.search(part) or NEGATION.search(part) or EXIT.search(part):
            governed = not affirmed(part) or bool(EXIT.search(part))
        if governed:
            denied.append(part)
    return denied


def clauses(quotes: list[str]) -> list[str]:
    """Clauses of the quoted sentences, split at contrastive joins and at an
    "and" joining two predicates."""

    return [
        part
        for quote in quotes
        for sentence in sentences(quote)
        for clause in CONTRAST.split(sentence)
        for part in _coordinated(clause)
        if part.strip()
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

# Another party: "our supplier Acme's ET-9000", "a competitor's".
THIRD_PARTY = re.compile(
    r"\b(?:suppliers?|vendors?|partners?|competitors?|rivals?|licensors?|"
    r"customers?|peers?)\b",
    re.IGNORECASE,
)
# Words between a named subject and a verb that hand the verb to someone
# else: "Example Corp says Acme offers", "We believe they make HBM".
_HANDOFF = re.compile(
    r"\b(?:says?|said|stat(?:es|ed)|reports?|reported|notes?|noted|announc(?:es|ed)|"
    r"believes?|believed|expects?|expected|claims?|claimed|disclos(?:es|ed)|"
    r"confirms?|confirmed|argues?|argued|according|thinks?|thought|knows?|knew|"
    r"it|its|they|their|he|his|she|her|which|who)\b",
    re.IGNORECASE,
)
# Capitalized words that continue a company name rather than start another.
_NAME_SUFFIXES = frozenset(
    {"inc", "corp", "corporation", "co", "ltd", "llc", "plc", "group", "holdings"}
)


def nearest_subject(subject: str, spans: list[tuple[int, int]]) -> bool:
    """Whether the last of ``spans`` is the verb's own subject.

    ``subject`` runs up to the verb; nothing after the named subject may
    introduce another actor — a reporting verb, a third party, a pronoun or
    another name ("Example Corp says Acme offers ET-9000" is Acme's offer).
    """

    if not spans:
        return False
    gap = subject[max(end for _, end in spans) :]
    if _HANDOFF.search(gap) or THIRD_PARTY.search(gap):
        return False
    return not any(
        word[:1].isupper() and word.casefold().strip(".,'’") not in _NAME_SUFFIXES
        for word in gap.split()
    )


# A segment or subsidiary belonging to the issuer.
PART_OF = re.compile(
    r"\b(segments?|subsidiar(?:y|ies)|divisions?|business\s+units?|units?\s+of|part\s+of|"
    r"wholly[- ]owned|owned\s+by)\b|セグメント|子会社|事業部|部門|分部|子公司",
    re.IGNORECASE,
)


_PASSIVE_AGENT = re.compile(r"\s+by\b", re.IGNORECASE)
_POSSESSOR = re.compile(r"([\w&.-]+)['’]s\s*$")


def own_mention(phrase: str, target: str, source: str) -> bool:
    """``phrase`` names ``target`` other than as a third party's ("Acme's
    ET-9000"); the source's own possessive ("Example Corp's") is fine."""

    for start, _ in mention_spans(phrase, target):
        possessor = _POSSESSOR.search(phrase[:start])
        if possessor is None or mentions(source, possessor.group(1)):
            return True
    return False


def predicated(clause: str, source: str, target: str, wording: re.Pattern) -> bool:
    """Whether ``clause`` states ``wording`` of ``source`` about ``target``.

    The source must be the verb's nearest subject and the target in its
    object phrase (or the passive "ET-9000 is sold by Example Corp"):
    "Example Corp relies on Acme, which offers ET-9000", "Example Corp says
    Acme offers ET-9000" and "Our ET-9000 sales rose and X200 supports HBM"
    name both ends and the verb, but another subject owns the verb.
    Verb-final CJK wording needs the verb and both ends in one
    comma-delimited segment.
    """

    for verb in wording.finditer(clause):
        if not verb.group().isascii():
            continue
        subject = PHRASE_BOUNDARY.split(clause[: verb.start()])[-1]
        rest = clause[verb.end() :]
        if nearest_subject(subject, mention_spans(subject, source)) and own_mention(
            CLAUSE_BOUNDARY.split(rest)[0], target, source
        ):
            return True
        agent = _PASSIVE_AGENT.match(rest)
        if agent and mentions(subject, target):
            by = PHRASE_BOUNDARY.split(rest[agent.end() :])[0]
            if nearest_subject(by, mention_spans(by, source)):
                return True
    return any(
        any(not match.group().isascii() for match in wording.finditer(segment))
        and mentions(segment, source)
        and mentions(segment, target)
        for segment in CJK_SEGMENT.split(clause)
    )


# "owned by", "part of", "a subsidiary of": the owner follows the wording.
_OWNER_FOLLOWS = re.compile(r"\s*(?:of|by)\b", re.IGNORECASE)
_CJK_POSSESSIVE = ("の", "的")


def owned_by(clause: str, owner: str, owned: str, wording: re.Pattern) -> bool:
    """Whether ``clause`` states that ``owned`` belongs to ``owner``.

    "Acme is part of Example Corp" and "Example Corp's subsidiary Acme" put
    Acme under Example Corp; "Example Corp is part of Acme" does not. With
    "of"/"by" after the wording the owned party precedes it and the owner
    follows; otherwise the owner comes first (possessive or adjacent name).
    CJK wording needs the owner marked possessive ("Example Corpの子会社").
    """

    for match in wording.finditer(clause):
        before, rest = clause[: match.start()], clause[match.end() :]
        if not match.group().isascii():
            if any(
                f"{owner}{mark}" in before.replace(" ", "")
                or f"{owner}{mark}" in before
                for mark in _CJK_POSSESSIVE
            ) and mentions(clause, owned):
                return True
            continue
        # "part of" and "owned by" carry their preposition in the match.
        joined = re.search(r"\b(?:of|by)$", match.group(), re.IGNORECASE)
        follows = None if joined else _OWNER_FOLLOWS.match(rest)
        if joined or follows:
            tail = rest if joined else rest[follows.end() :]
            tail = CLAUSE_BOUNDARY.split(tail)[0]
            if mentions(before, owned) and mentions(tail, owner):
                return True
            continue
        owners = mention_spans(before, owner)
        owneds = mention_spans(clause, owned)
        if owners and owneds and owners[0][0] < owneds[0][0]:
            return True
    return False
