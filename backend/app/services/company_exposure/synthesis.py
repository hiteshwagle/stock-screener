"""Bounded primary synthesis (spec §5.4).

A synthesized claim may join at most three original primary premises with
at most two explicit links. Each link must be stated in a premise quote that
names both ends, and only these relationships can carry a product toward a
theme application:

* ``issuer_offers_product`` — the issuer sells/ships/offers the product;
* ``product_supports_application`` — the product is designed for / supports
  / tests the application;
* ``segment_of_issuer`` — a segment/subsidiary belongs to the issuer.

Customer/supplier relationships (``supplies_to``, ``customer_of``,
``manufactures``) never carry an application across companies: "A supplies
B and B makes HBM" does not establish that A's product serves HBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.company_exposure.policy import (
    MAX_SYNTHESIS_LINKS,
    MAX_SYNTHESIS_PRIMARY_PREMISES,
    within_synthesis_bound,
)
from app.services.company_exposure.wording import (
    OFFERS,
    PART_OF,
    SERVES,
    affirmed,
    clauses,
    mentions,
    predicated,
)

APPLICATION_LINKS = frozenset(
    {"issuer_offers_product", "product_supports_application", "segment_of_issuer"}
)
CROSS_COMPANY_LINKS = frozenset({"supplies_to", "customer_of", "manufactures"})
# Wording each carrying relationship must state, not just two co-occurring ends.
_RELATIONSHIP_WORDING = {
    "issuer_offers_product": OFFERS,
    "product_supports_application": SERVES,
    "segment_of_issuer": PART_OF,
}
# Relationships whose wording is a predicate of the link source: "Example
# Corp offers ET-9000", "ET-9000 supports HBM".
_PREDICATED = frozenset({"issuer_offers_product", "product_supports_application"})


@dataclass(frozen=True, slots=True)
class Premise:
    ref: str
    quote: str
    primary: bool


@dataclass(frozen=True, slots=True)
class Link:
    source: str
    target: str
    relationship: str
    premise_ref: str


@dataclass(frozen=True, slots=True)
class SynthesisDecision:
    permitted: bool
    reasons: tuple[str, ...] = ()
    premises: tuple[Premise, ...] = ()
    links: tuple[Link, ...] = ()
    forbidden_extrapolations: tuple[str, ...] = field(
        default=("theme_specific_sales", "named_customers", "revenue_share")
    )


def _mentions(quote: str, entity: str) -> bool:
    return mentions(quote, entity)


def validate_synthesis(
    premises: list[Premise],
    links: list[Link],
    *,
    subject: str,
    application: str,
) -> SynthesisDecision:
    """Permit only a fully evidenced chain from the issuer's subject to the
    application; anything missing or cross-company is held."""

    reasons: list[str] = []
    by_ref = {premise.ref: premise for premise in premises}
    if not within_synthesis_bound([p.ref for p in premises], links):
        reasons.append(
            f"exceeds_bound_{MAX_SYNTHESIS_PRIMARY_PREMISES}_premises_"
            f"{MAX_SYNTHESIS_LINKS}_links"
        )
    if any(not premise.primary for premise in premises):
        reasons.append("non_primary_premise")
    for link in links:
        premise = by_ref.get(link.premise_ref)
        if premise is None:
            reasons.append("link_premise_missing")
            continue
        linking = [
            clause
            for clause in clauses([premise.quote])
            if _mentions(clause, link.source) and _mentions(clause, link.target)
        ]
        wording = _RELATIONSHIP_WORDING.get(link.relationship)
        if not linking:
            reasons.append("link_not_stated_in_premise")
        elif not any(affirmed(clause) for clause in linking):
            # "ET-9000 does not support HBM" names both ends but denies the link.
            reasons.append("link_negated_in_premise")
        elif wording is not None and not any(
            affirmed(clause)
            and (
                predicated(clause, link.source, link.target, wording)
                if link.relationship in _PREDICATED
                else wording.search(clause)
            )
            for clause in linking
        ):
            # "ET-9000 and HBM demand increased" names both ends but states no
            # support, offer or ownership relationship between them.
            reasons.append("link_relationship_not_stated")
        if link.relationship in CROSS_COMPANY_LINKS:
            reasons.append("cross_company_link_cannot_carry_application")
        elif link.relationship not in APPLICATION_LINKS:
            reasons.append("unsupported_link_relationship")
    # The chain must actually connect subject -> application.
    edges = {
        link.source.casefold(): link.target.casefold()
        for link in links
        if link.relationship in APPLICATION_LINKS
    }
    node = subject.casefold()
    seen = set()
    while node in edges and node not in seen:
        seen.add(node)
        node = edges[node]
    if node != application.casefold():
        reasons.append("application_link_missing")
    reasons = list(dict.fromkeys(reasons))
    return SynthesisDecision(
        permitted=not reasons,
        reasons=tuple(reasons),
        premises=tuple(premises),
        links=tuple(links),
    )
