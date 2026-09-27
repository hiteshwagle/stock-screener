"""Disclosed and reproducibly calculated materiality (spec §6).

Materiality is typed evidence: basis (disclosed / calculated / qualitative /
unknown), metric, Decimal value, unit, currency, period, reporting scope,
denominator and operand citations. A disclosed figure's metric, unit,
currency, period and scope must be stated by its quote (or, for the
period, by the cited document). It is never ``exposure_strength`` or a
confidence score, and nothing here estimates a number the source did not
disclose.

V1 calculation: an explicit ratio of two compatible disclosed quantities.
Any mismatch in period,
unit, currency, accounting basis or scope holds the derived value while the
original reported numbers are retained. A segment or subsidiary share is
reported as that scope's share, never relabelled as a theme share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from app.domain.company_exposure.contracts import (
    UNKNOWN_MATERIALITY_WORDING,
    MaterialityBasis,
    QualitativeMateriality,
)
from app.services.company_exposure.wording import mentions

SHARE_METRICS = frozenset(
    {"revenue_share", "profit_share", "capacity_share", "backlog_share", "asset_share"}
)
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Operand:
    """One disclosed quantity with its provenance."""

    value: Decimal
    unit: str
    period: str
    scope: str
    label: str
    currency: str | None = None
    accounting_basis: str | None = None
    passage_id: str | None = None
    quote: str | None = None
    forecast: bool = False
    # The cited document's reporting period/date, grounding an implicit period.
    period_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MaterialityMeasureResult:
    basis: MaterialityBasis
    metric: str | None = None
    value: Decimal | None = None
    value_high: Decimal | None = None
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    reporting_scope: str = "issuer_consolidated"
    scope_label: str | None = None
    denominator_definition: str | None = None
    qualitative_label: QualitativeMateriality | None = None
    formula: dict = field(default_factory=dict)
    operands: tuple[Operand, ...] = ()
    raw_reported: dict = field(default_factory=dict)
    hold_reasons: tuple[str, ...] = ()
    theme_specific: bool = False

    @property
    def held(self) -> bool:
        return bool(self.hold_reasons)

    @property
    def display(self) -> str:
        if self.basis == MaterialityBasis.UNKNOWN or self.held:
            return UNKNOWN_MATERIALITY_WORDING
        if self.value is None:
            return str(self.qualitative_label.value if self.qualitative_label else "")
        scope = f" of {self.scope_label}" if self.scope_label else ""
        return f"{self.metric} {self.value.normalize():f}{scope} ({self.period})"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def parse_decimal(text: str) -> Decimal:
    """Parse a disclosed number exactly (commas removed, never via float)."""

    match = _NUMBER.search(str(text))
    if match is None:
        raise ValueError("no_number")
    try:
        value = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        raise ValueError("invalid_number") from None
    if not value.is_finite():
        raise ValueError("invalid_number")
    return value


def quote_contains_value(quote: str | None, value: Decimal) -> bool:
    """A cited operand must appear in its quote (exact decimal equality)."""

    if not quote:
        return False
    for token in _NUMBER.findall(quote):
        try:
            if Decimal(token.replace(",", "")) == value:
                return True
        except InvalidOperation:
            continue
    return False


# Wording a quote must carry for a model-supplied metric head, unit, scale
# or currency to be taken as stated rather than invented.
_METRIC_WORDS = {
    "revenue": ("revenue", "sales", "turnover", "売上", "營收", "营收", "收入"),
    "sales": ("revenue", "sales", "turnover", "売上", "營收", "营收", "收入"),
    "profit": ("profit", "income", "earnings", "利益", "利潤", "利润"),
    "income": ("profit", "income", "earnings", "利益", "利潤", "利润"),
    "backlog": ("backlog", "order", "受注", "訂單", "订单"),
}
_PERCENT_UNITS = frozenset({"percent", "pct", "%", "percentage"})
_PERCENT_WORDS = ("%", "％", "percent", "per cent")
_SCALE_WORDS = {
    "thousand": (r"thousand", r"\bk\b", "千"),
    "million": (r"million", r"\bmn\b", r"\bmm\b", r"\d\s*m\b", "百万", "百萬"),
    "billion": (r"billion", r"\bbn\b", r"\d\s*b\b", "億", "亿", "十億"),
}
_CURRENCY_WORDS = {
    "USD": ("usd", "$", "dollar"),
    "EUR": ("eur", "€", "euro"),
    "JPY": ("jpy", "¥", "円", "yen"),
    "TWD": ("twd", "nt$", "新台幣", "新台币"),
    "HKD": ("hkd", "hk$"),
    "CNY": ("cny", "rmb", "人民幣", "人民币"),
    "KRW": ("krw", "₩", "won"),
    "GBP": ("gbp", "£", "pound"),
}
# Wording that places a figure inside the consolidated reporting entity.
_CONSOLIDATED = re.compile(
    r"\b(segments?|consolidated)\b|セグメント|連結|分部|合併|合并",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _grounding_holds(
    *,
    metric: str,
    unit: str,
    currency: str | None,
    period: str,
    scope_label: str | None,
    quote: str,
    period_evidence: tuple[str, ...],
) -> list[str]:
    """Metadata the model supplied that the cited wording does not state.

    Only the number was checked before; "USD 20 million in FY2024" must not
    come back as a 20% FY2026 share of "HBM revenue".
    """

    text = _norm(quote)
    holds = []
    head = (metric or "").split("_")[0].casefold()
    if head and not any(word in text for word in _METRIC_WORDS.get(head, (head,))):
        holds.append("metric_not_in_quote")
    unit_folded = (unit or "").casefold()
    if unit_folded in _PERCENT_UNITS and not any(w in text for w in _PERCENT_WORDS):
        holds.append("unit_not_in_quote")
    for scale, patterns in _SCALE_WORDS.items():
        if scale in unit_folded and not any(re.search(p, text) for p in patterns):
            holds.append("unit_not_in_quote")
    if currency:
        words = _CURRENCY_WORDS.get(currency.upper(), (currency.casefold(),))
        if not any(word in text for word in words):
            holds.append("currency_not_in_quote")
    # The period may come from the document itself (its reporting period).
    period_text = " ".join((text, *(_norm(p) for p in period_evidence)))
    digits = re.findall(r"\d+", period or "")
    if not period or any(d not in period_text for d in digits):
        holds.append("period_not_in_quote")
    if scope_label and any(
        word not in text for word in _WORD.findall(scope_label.casefold())
    ):
        holds.append("scope_not_in_quote")
    return list(dict.fromkeys(holds))


def unknown_materiality(reason: str | None = None) -> MaterialityMeasureResult:
    return MaterialityMeasureResult(
        basis=MaterialityBasis.UNKNOWN,
        raw_reported={"note": UNKNOWN_MATERIALITY_WORDING, "reason": reason},
    )


def validate_measure(
    *,
    metric: str,
    value: Decimal,
    unit: str,
    period: str,
    scope: str,
    scope_label: str | None,
    quote: str,
    passage_id: str | None,
    theme_terms: tuple[str, ...] = (),
    currency: str | None = None,
    period_evidence: tuple[str, ...] = (),
) -> MaterialityMeasureResult:
    """A directly disclosed figure (e.g. "Memory test was 20% of revenue").

    ``period_evidence`` is the cited document's own reporting period or date,
    which may ground a period the quoted sentence leaves implicit.
    """

    holds = []
    if not quote_contains_value(quote, value):
        holds.append("value_not_in_quote")
    holds.extend(
        _grounding_holds(
            metric=metric,
            unit=unit,
            currency=currency,
            period=period,
            scope_label=scope_label,
            quote=quote,
            period_evidence=period_evidence,
        )
    )
    # A share quoted as a percentage ("30 percent of revenue") is checked as
    # reported, then stored as a ratio; the original number stays in the
    # operand and raw_reported.
    percent = (unit or "").casefold() in _PERCENT_UNITS
    stored_value, stored_unit = value, unit
    if metric.endswith("_share"):
        bound = Decimal(100) if percent else Decimal(1)
        if not (Decimal(0) <= value <= bound):
            holds.append("share_out_of_range")
        elif percent:
            stored_value, stored_unit = value / Decimal(100), "ratio"
    theme_specific = bool(scope_label) and any(
        mentions(scope_label, term) for term in theme_terms
    )
    return MaterialityMeasureResult(
        basis=MaterialityBasis.DISCLOSED,
        metric=metric,
        value=stored_value,
        unit=stored_unit,
        currency=currency,
        period=period,
        reporting_scope=scope,
        scope_label=scope_label,
        operands=(
            Operand(
                value,
                unit,
                period,
                scope,
                scope_label or "",
                currency,
                None,
                passage_id,
                quote,
            ),
        ),
        raw_reported={"quote": quote, "value": format(value, "f"), "unit": unit},
        hold_reasons=tuple(holds),
        theme_specific=theme_specific and not holds,
    )


def compatible_ratio(
    *,
    numerator: Decimal,
    denominator: Decimal,
    numerator_scope: str,
    denominator_scope: str,
    period: str,
    unit: str,
    metric: str,
    operands_compatible: bool,
) -> MaterialityMeasureResult:
    """Exact ratio after provenance compatibility has been established."""

    holds = []
    if not operands_compatible:
        holds.append("operands_incompatible")
    if denominator <= 0:
        holds.append("nonpositive_denominator_review_required")
    value = None if holds else numerator / denominator
    if (
        value is not None
        and metric in SHARE_METRICS
        and not (Decimal(0) <= value <= Decimal(1))
    ):
        holds.append("share_out_of_range")
        value = None
    return MaterialityMeasureResult(
        # A ratio that could not be computed is unknown, not a calculated
        # value of nothing (a calculated measure always carries its value).
        basis=MaterialityBasis.CALCULATED
        if value is not None
        else MaterialityBasis.UNKNOWN,
        metric=metric,
        value=value,
        unit="ratio",
        period=period,
        reporting_scope="issuer_consolidated",
        scope_label=numerator_scope,
        denominator_definition=denominator_scope,
        formula={"kind": "ratio", "expression": "numerator / denominator"},
        raw_reported={
            "numerator": format(numerator, "f"),
            "denominator": format(denominator, "f"),
            "unit": unit,
        },
        hold_reasons=tuple(holds),
    )


def calculate_materiality(
    *,
    metric: str,
    numerator: Operand,
    denominator: Operand,
    theme_terms: tuple[str, ...] = (),
) -> MaterialityMeasureResult:
    """Validated ratio of two cited disclosed quantities (E05/E06/I03)."""

    holds = []
    for role, operand in (("numerator", numerator), ("denominator", denominator)):
        if not quote_contains_value(operand.quote, operand.value):
            holds.append(f"{role}_value_not_in_quote")
        # Each operand's metadata must be stated by its own quote, or two
        # unrelated figures ("$20m revenue", "100 employees") could be
        # labelled as compatible revenue operands.
        holds.extend(
            f"{role}_{hold}"
            for hold in _grounding_holds(
                metric=metric,
                unit=operand.unit,
                currency=operand.currency,
                period=operand.period,
                scope_label=operand.label,
                quote=operand.quote or "",
                period_evidence=operand.period_evidence,
            )
        )
    if numerator.forecast or denominator.forecast:
        holds.append("forecast_operand")
    if _norm(numerator.period) != _norm(denominator.period):
        holds.append("period_mismatch")
    if _norm(numerator.unit) != _norm(denominator.unit):
        holds.append("unit_mismatch")
    if (numerator.currency or "").upper() != (denominator.currency or "").upper():
        holds.append("currency_mismatch_requires_approved_conversion")
    if _norm(numerator.accounting_basis or "") != _norm(
        denominator.accounting_basis or ""
    ):
        holds.append("accounting_basis_mismatch")
    # A subsidiary/segment figure may only be divided by a denominator of the
    # same reporting entity; never by the consolidated parent (I03).
    if (
        numerator.scope != denominator.scope
        and denominator.scope != "issuer_consolidated"
    ):
        holds.append("scope_mismatch")
    # A segment or subsidiary figure over the consolidated parent needs the
    # numerator's own wording to show it is a consolidated segment; the
    # model-chosen label ("HBM Labs") says nothing about consolidation.
    if (
        numerator.scope == "segment_or_subsidiary"
        and denominator.scope == "issuer_consolidated"
        and not _CONSOLIDATED.search(numerator.quote or "")
    ):
        holds.append("subsidiary_share_of_parent_requires_consolidation_evidence")
    result = compatible_ratio(
        numerator=numerator.value,
        denominator=denominator.value,
        numerator_scope=numerator.label,
        denominator_scope=denominator.label,
        period=numerator.period,
        unit=numerator.unit,
        metric=metric,
        operands_compatible=not holds,
    )
    all_holds = tuple(dict.fromkeys([*holds, *result.hold_reasons]))
    # Held ratios keep their operands and reasons for review but are stored
    # as unknown: only a computed ratio is a calculated measure.
    theme_specific = any(mentions(numerator.label, term) for term in theme_terms)
    return MaterialityMeasureResult(
        basis=MaterialityBasis.UNKNOWN if all_holds else MaterialityBasis.CALCULATED,
        metric=metric,
        value=None if all_holds else result.value,
        unit="ratio",
        currency=numerator.currency,
        period=numerator.period,
        reporting_scope=numerator.scope,
        scope_label=numerator.label,
        denominator_definition=denominator.label,
        formula={
            "kind": "ratio",
            "expression": "numerator / denominator",
            "numerator_passage": numerator.passage_id,
            "denominator_passage": denominator.passage_id,
        },
        operands=(numerator, denominator),
        raw_reported={
            "numerator": {
                "value": format(numerator.value, "f"),
                "unit": numerator.unit,
                "label": numerator.label,
            },
            "denominator": {
                "value": format(denominator.value, "f"),
                "unit": denominator.unit,
                "label": denominator.label,
            },
        },
        hold_reasons=all_holds,
        theme_specific=theme_specific and not all_holds,
    )


_LIMITING_WORDING = re.compile(
    r"\b(immaterial|not\s+(?:material|significant|meaningful)|insignificant|"
    r"negligible|minimal|de\s+minimis|small\s+(?:portion|percentage|part))\b",
    re.IGNORECASE,
)
_SIGNIFICANT_WORDING = re.compile(
    r"\b(material|significant|substantial|core|primary|principal|main|majority|"
    r"substantially\s+all|key|largest|predominant)\b",
    re.IGNORECASE,
)


def qualitative_measure(label: str, quote: str) -> MaterialityMeasureResult:
    """A qualitative label kept only when its cited wording says so.

    The label comes from the model: "core business" over "HBM revenue was
    immaterial" is not supported, so it becomes ``unknown`` with a hold.
    """

    parsed = QualitativeMateriality(label)
    if not quote.strip():
        holds = ("qualitative_label_without_primary_wording",)
    else:
        limiting = bool(_LIMITING_WORDING.search(quote))
        significant = bool(_SIGNIFICANT_WORDING.search(quote)) and not limiting
        supported = {
            QualitativeMateriality.CORE_BUSINESS: significant,
            QualitativeMateriality.EXPLICITLY_MATERIAL: significant,
            QualitativeMateriality.EXPLICITLY_LIMITED: limiting,
        }.get(parsed, True)
        holds = () if supported else ("qualitative_label_not_supported_by_wording",)
        if not supported:
            parsed = QualitativeMateriality.UNKNOWN
    return MaterialityMeasureResult(
        basis=MaterialityBasis.QUALITATIVE,
        qualitative_label=parsed,
        raw_reported={"quote": quote, "proposed_label": label},
        hold_reasons=holds,
    )
