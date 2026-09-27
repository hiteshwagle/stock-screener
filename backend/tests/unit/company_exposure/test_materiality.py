from __future__ import annotations

from decimal import Decimal

import pytest

from app.domain.company_exposure.contracts import MaterialityBasis
from app.services.company_exposure.materiality import (
    Operand,
    calculate_materiality,
    compatible_ratio,
    parse_decimal,
    unknown_materiality,
    validate_measure,
)

THEME = ("HBM", "memory test")


def operand(value, **kw):
    base = dict(
        value=parse_decimal(value),
        unit="USD million",
        period="FY2025",
        scope="issuer_consolidated",
        label="Memory test revenue",
        currency="USD",
        passage_id="p",
    )
    base.update(kw)
    # By default the quote states everything the operand claims.
    base.setdefault(
        "quote", f"{base['label']} (revenue) of USD {value} million in FY2025"
    )
    return Operand(**base)


def test_compatible_disclosure_ratio_is_exact():
    result = compatible_ratio(
        numerator=Decimal(200),
        denominator=Decimal(1000),
        numerator_scope="issuer_consolidated:ai_memory",
        denominator_scope="issuer_consolidated:total",
        period="FY2025",
        unit="USD_million",
        metric="revenue_share",
        operands_compatible=True,
    )
    assert result.value == Decimal("0.2")
    assert result.basis == MaterialityBasis.CALCULATED


@pytest.mark.case("E04")
@pytest.mark.exposure_layer("unit")
def test_e04_segment_not_theme_share():
    result = validate_measure(
        metric="revenue_share",
        value=Decimal("0.30"),
        unit="ratio",
        period="FY2025",
        scope="segment_or_subsidiary",
        scope_label="Server segment",
        quote="The server segment represented 0.30 of revenue.",
        passage_id="p",
        theme_terms=THEME,
    )
    assert result.value == Decimal("0.30")
    assert result.scope_label == "Server segment"
    assert result.theme_specific is False  # never relabelled as an AI Memory share


@pytest.mark.case("E05")
@pytest.mark.exposure_layer("unit")
def test_e05_compatible_disclosed_ratio():
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand(
            "200",
            passage_id="p-num",
            quote="Memory test revenue was USD 200 million in FY2025",
        ),
        denominator=operand(
            "1,000",
            label="Total revenue",
            passage_id="p-den",
            quote="Total revenue was USD 1,000 million in FY2025",
        ),
        theme_terms=THEME,
    )
    assert result.value == Decimal("0.2")
    assert result.hold_reasons == ()
    assert result.formula["numerator_passage"] == "p-num"
    assert result.formula["denominator_passage"] == "p-den"
    assert result.period == "FY2025" and result.theme_specific


@pytest.mark.case("E06")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    ("numerator", "denominator", "hold"),
    [
        (
            operand("200"),
            operand("1000", period="FY2024", label="Total"),
            "period_mismatch",
        ),
        (
            operand("200"),
            operand("1000", currency="JPY", label="Total"),
            "currency_mismatch_requires_approved_conversion",
        ),
        (
            operand("200"),
            operand("0", label="Operating profit", quote="profit of 0"),
            "nonpositive_denominator_review_required",
        ),
        (
            operand("200"),
            operand("-50", label="Operating profit", quote="loss of -50"),
            "nonpositive_denominator_review_required",
        ),
        (
            operand("200", forecast=True),
            operand("1000", label="Total"),
            "forecast_operand",
        ),
        (
            operand("200", quote="no number here"),
            operand("1000", label="Total"),
            "numerator_value_not_in_quote",
        ),
        (
            operand("1500"),
            operand("1000", label="Total"),
            "share_out_of_range",
        ),
    ],
)
def test_e06_incompatible_inputs_held(numerator, denominator, hold):
    result = calculate_materiality(
        metric="revenue_share", numerator=numerator, denominator=denominator
    )
    assert result.value is None
    assert hold in result.hold_reasons
    assert result.raw_reported["numerator"]["value"]  # original numbers retained


@pytest.mark.case("I03")
@pytest.mark.exposure_layer("unit")
def test_i03_subsidiary_not_parent_share():
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand(
            "80", scope="segment_or_subsidiary", label="Subsidiary HBM revenue"
        ),
        denominator=operand("1000", label="Group revenue"),
    )
    assert result.value is None
    assert (
        "subsidiary_share_of_parent_requires_consolidation_evidence"
        in result.hold_reasons
    )


@pytest.mark.case("E08")
@pytest.mark.exposure_layer("unit")
def test_unknown_materiality_wording():
    result = unknown_materiality()
    assert result.basis == MaterialityBasis.UNKNOWN
    assert result.display == "Not separately disclosed in reviewed evidence"


def test_decimals_are_parsed_exactly():
    assert parse_decimal("1,234.50") == Decimal("1234.50")
    with pytest.raises(ValueError):
        parse_decimal("n/a")


@pytest.mark.parametrize(
    ("label", "quote", "kept"),
    [
        ("core_business", "HBM test equipment is our core business.", "core_business"),
        ("core_business", "HBM revenue was immaterial in fiscal 2025.", "unknown"),
        ("explicitly_material", "HBM sales were not material.", "unknown"),
        ("explicitly_limited", "HBM sales were not material.", "explicitly_limited"),
        ("explicitly_limited", "HBM is our largest product line.", "unknown"),
    ],
)
def test_qualitative_label_must_match_its_wording(label, quote, kept):
    from app.services.company_exposure.materiality import qualitative_measure

    measure = qualitative_measure(label, quote)
    assert measure.qualitative_label.value == kept
    held = "qualitative_label_not_supported_by_wording" in measure.hold_reasons
    assert held is (kept != label)


def test_disclosed_metadata_must_be_stated_by_the_quote():
    quote = "Revenue was USD 20 million in FY2024."
    stated = validate_measure(
        metric="revenue",
        value=Decimal(20),
        unit="USD_million",
        currency="USD",
        period="FY2024",
        scope="issuer_consolidated",
        scope_label=None,
        quote=quote,
        passage_id="p",
    )
    assert not stated.held
    invented = validate_measure(
        metric="revenue_percent",
        value=Decimal(20),
        unit="percent",
        period="FY2026",
        scope="segment_or_subsidiary",
        scope_label="HBM revenue",
        quote=quote,
        passage_id="p",
        theme_terms=THEME,
    )
    assert set(invented.hold_reasons) >= {
        "unit_not_in_quote",
        "period_not_in_quote",
        "scope_not_in_quote",
    }
    assert invented.theme_specific is False
    wrong_currency = validate_measure(
        metric="revenue",
        value=Decimal(20),
        unit="million",
        currency="EUR",
        period="FY2024",
        scope="issuer_consolidated",
        scope_label=None,
        quote=quote,
        passage_id="p",
    )
    assert "currency_not_in_quote" in wrong_currency.hold_reasons


def test_document_period_grounds_an_implicit_period():
    kwargs = {
        "metric": "revenue_percent",
        "value": Decimal(20),
        "unit": "percent",
        "period": "FY2025",
        "scope": "segment_or_subsidiary",
        "scope_label": "Memory test",
        "quote": "Memory test was 20% of revenue",
        "passage_id": "p",
    }
    assert validate_measure(**kwargs).hold_reasons == ("period_not_in_quote",)
    assert not validate_measure(**kwargs, period_evidence=("FY2025",)).held


def test_ratio_operands_must_each_state_their_metadata():
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand(
            "20", quote="Memory test revenue was USD 20 million in FY2025"
        ),
        # A headcount relabelled as USD revenue.
        denominator=operand(
            "100", label="Total revenue", quote="We had 100 employees in FY2025"
        ),
        theme_terms=THEME,
    )
    assert result.value is None
    assert {
        "denominator_metric_not_in_quote",
        "denominator_currency_not_in_quote",
        "denominator_scope_not_in_quote",
    } <= set(result.hold_reasons)
    assert not any(h.startswith("numerator_") for h in result.hold_reasons)


def test_percent_share_is_checked_as_reported_and_stored_as_a_ratio():
    result = validate_measure(
        metric="revenue_share",
        value=Decimal(30),
        unit="percent",
        period="FY2025",
        scope="segment_or_subsidiary",
        scope_label="HBM",
        quote="HBM was 30 percent of revenue in FY2025.",
        passage_id="p",
        theme_terms=THEME,
    )
    assert not result.held
    assert (result.value, result.unit) == (Decimal("0.3"), "ratio")
    assert result.raw_reported["value"] == "30"
    too_big = validate_measure(
        metric="revenue_share",
        value=Decimal(130),
        unit="percent",
        period="FY2025",
        scope="issuer_consolidated",
        scope_label=None,
        quote="Revenue grew 130 percent in FY2025.",
        passage_id="p",
    )
    assert "share_out_of_range" in too_big.hold_reasons


@pytest.mark.parametrize(
    ("label", "quote", "held"),
    [
        ("HBM Labs", "HBM Labs revenue was USD 20 million in FY2025", True),
        (
            "Memory test",
            "Memory test segment revenue was USD 20 million in FY2025",
            False,
        ),
    ],
)
def test_subsidiary_share_of_parent_needs_consolidation_wording(label, quote, held):
    result = calculate_materiality(
        metric="revenue_share",
        numerator=operand(
            "20", scope="segment_or_subsidiary", label=label, quote=quote
        ),
        denominator=operand(
            "100",
            label="Total revenue",
            quote="Total revenue was USD 100 million in FY2025",
        ),
    )
    hold = "subsidiary_share_of_parent_requires_consolidation_evidence"
    assert (hold in result.hold_reasons) is held
    assert (result.value is None) is held
