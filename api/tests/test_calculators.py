"""test_calculators.py — unit tests for deterministic financial calculators."""
import math
import sys
from pathlib import Path

# Allow importing from api/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from calculators import (
    LoanCalculations,
    calculate_emi,
    calculate_foir,
    calculate_ltv,
    calculate_max_affordable_loan,
    run_all_calculators,
)


# ── calculate_emi ─────────────────────────────────────────────────────────────

class TestCalculateEMI:
    def test_standard_personal_loan(self):
        """₹5L at 12% for 36 months → known EMI ~₹16,607"""
        emi = calculate_emi(500_000, 12.0, 36)
        assert 16_500 < emi < 16_750, f"Unexpected EMI: {emi}"

    def test_home_loan(self):
        """₹30L at 8.5% for 240 months → known EMI ~₹26,035"""
        emi = calculate_emi(30_00_000, 8.5, 240)
        assert 25_500 < emi < 26_500, f"Unexpected EMI: {emi}"

    def test_zero_interest(self):
        """Zero-interest loan → simple division"""
        emi = calculate_emi(120_000, 0.0, 12)
        assert emi == pytest.approx(10_000.0, rel=1e-3)

    def test_zero_principal(self):
        assert calculate_emi(0, 12.0, 36) == 0.0

    def test_zero_tenure(self):
        assert calculate_emi(500_000, 12.0, 0) == 0.0

    def test_single_month(self):
        """1-month tenure → entire principal + one month's interest"""
        emi = calculate_emi(100_000, 12.0, 1)
        expected = 100_000 * (1 + 12 / 1200)
        assert emi == pytest.approx(expected, rel=1e-3)


# ── calculate_foir ────────────────────────────────────────────────────────────

class TestCalculateFOIR:
    def test_typical_case(self):
        """Existing EMI ₹5k + new EMI ₹10k, income ₹50k → FOIR = 30%"""
        foir = calculate_foir(5_000, 10_000, 50_000)
        assert foir == pytest.approx(0.30, abs=1e-3)

    def test_at_50pct(self):
        foir = calculate_foir(0, 25_000, 50_000)
        assert foir == pytest.approx(0.50, abs=1e-3)

    def test_exceeds_100pct(self):
        """FOIR > 1 is valid (over-leveraged)"""
        foir = calculate_foir(30_000, 30_000, 50_000)
        assert foir > 1.0

    def test_zero_income(self):
        assert calculate_foir(5_000, 10_000, 0) == float("inf")

    def test_no_existing_emi(self):
        foir = calculate_foir(0, 20_000, 80_000)
        assert foir == pytest.approx(0.25, abs=1e-3)


# ── calculate_ltv ─────────────────────────────────────────────────────────────

class TestCalculateLTV:
    def test_80pct_ltv(self):
        ltv = calculate_ltv(40_00_000, 50_00_000)
        assert ltv == pytest.approx(0.80, abs=1e-3)

    def test_85pct_ltv(self):
        ltv = calculate_ltv(25_50_000, 30_00_000)
        assert ltv == pytest.approx(0.85, abs=1e-3)

    def test_100pct_ltv(self):
        ltv = calculate_ltv(50_00_000, 50_00_000)
        assert ltv == pytest.approx(1.0, abs=1e-3)

    def test_zero_property_value(self):
        assert calculate_ltv(10_00_000, 0) == float("inf")


# ── calculate_max_affordable_loan ─────────────────────────────────────────────

class TestCalculateMaxAffordable:
    def test_personal_loan_case(self):
        """Income ₹60k, no existing EMI, FOIR 50%, 12% rate, 48m"""
        max_loan = calculate_max_affordable_loan(60_000, 0, 0.50, 12.0, 48)
        # max_emi = 30k; should give loan ~₹11.4L
        assert 10_00_000 < max_loan < 12_00_000, f"Got {max_loan}"

    def test_no_budget_with_existing_emi(self):
        """Existing EMI already exceeds FOIR limit"""
        max_loan = calculate_max_affordable_loan(50_000, 30_000, 0.50, 12.0, 36)
        assert max_loan == 0.0

    def test_zero_income(self):
        assert calculate_max_affordable_loan(0, 0, 0.50, 12.0, 36) == 0.0

    def test_zero_tenure(self):
        assert calculate_max_affordable_loan(60_000, 0, 0.50, 12.0, 0) == 0.0

    def test_zero_rate(self):
        """Zero-interest: max = max_emi × tenure"""
        max_loan = calculate_max_affordable_loan(100_000, 0, 0.50, 0.0, 12)
        assert max_loan == pytest.approx(50_000 * 12, rel=1e-3)


# ── run_all_calculators ───────────────────────────────────────────────────────

class TestRunAllCalculators:
    def test_complete_personal_loan(self):
        calcs = run_all_calculators(
            requested_amount=5_00_000,
            monthly_income=60_000,
            existing_emi=5_000,
            tenure_months=36,
            annual_rate_pct=12.0,
            max_foir=0.50,
        )
        assert calcs.emi is not None
        assert calcs.foir is not None
        assert calcs.ltv is None          # no property value → no LTV
        assert calcs.max_affordable_loan is not None
        assert calcs.total_interest is not None
        assert calcs.total_payment == pytest.approx(calcs.emi * 36, rel=1e-2)

    def test_complete_home_loan(self):
        calcs = run_all_calculators(
            requested_amount=40_00_000,
            monthly_income=1_00_000,
            existing_emi=10_000,
            tenure_months=240,
            annual_rate_pct=8.5,
            max_foir=0.55,
            property_value=50_00_000,
        )
        assert calcs.ltv is not None
        assert calcs.ltv == pytest.approx(0.80, abs=0.01)

    def test_complete_auto_loan(self):
        calcs = run_all_calculators(
            requested_amount=6_00_000,
            monthly_income=60_000,
            existing_emi=0,
            tenure_months=48,
            annual_rate_pct=9.0,
            max_foir=0.50,
            on_road_price=8_00_000,
        )
        assert calcs.emi is not None
        assert calcs.foir is not None
        assert calcs.ltv is not None
        assert calcs.ltv == pytest.approx(0.75, abs=0.01)
        assert calcs.max_affordable_loan is not None

    def test_missing_inputs(self):
        """Missing amount/tenure → emi and foir are None"""
        calcs = run_all_calculators(
            requested_amount=None,
            monthly_income=60_000,
            existing_emi=5_000,
            tenure_months=None,
            annual_rate_pct=12.0,
            max_foir=0.50,
        )
        assert calcs.emi is None
        assert calcs.foir is None
        assert calcs.max_affordable_loan is None
