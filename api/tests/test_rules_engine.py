"""test_rules_engine.py — unit tests for the deterministic eligibility rules engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from calculators import LoanCalculations, run_all_calculators
from rules_engine import (
    ApplicantProfile,
    Decision,
    EligibilityResult,
    RuleResult,
    evaluate_home_loan,
    evaluate_personal_loan,
    run_rules_engine,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pl_calcs(profile: ApplicantProfile) -> LoanCalculations:
    from rules_engine import PERSONAL_LOAN_CONFIG as cfg
    return run_all_calculators(
        profile.requested_amount, profile.monthly_net_income, profile.existing_emi,
        profile.requested_tenure_months, cfg["ANNUAL_RATE_PCT"], cfg["MAX_FOIR"],
    )


def _hl_calcs(profile: ApplicantProfile) -> LoanCalculations:
    from rules_engine import HOME_LOAN_CONFIG as cfg
    return run_all_calculators(
        profile.requested_amount, profile.monthly_net_income, profile.existing_emi,
        profile.requested_tenure_months, cfg["ANNUAL_RATE_PCT"], cfg["MAX_FOIR"],
        property_value=profile.property_value,
    )


def _full_pl_profile(**overrides) -> ApplicantProfile:
    """Return a fully-eligible personal loan profile with optional overrides."""
    base = dict(
        loan_type="personal_loan",
        age=35,
        employment_type="salaried",
        monthly_net_income=60_000,
        employment_duration_months=24,
        credit_score=750,
        existing_emi=5_000,
        requested_amount=5_00_000,
        requested_tenure_months=36,
    )
    base.update(overrides)
    return ApplicantProfile(**base)


def _full_hl_profile(**overrides) -> ApplicantProfile:
    """Return a fully-eligible home loan profile with optional overrides."""
    base = dict(
        loan_type="home_loan",
        age=35,
        employment_type="salaried",
        monthly_net_income=1_00_000,
        employment_duration_months=36,
        credit_score=760,
        existing_emi=10_000,
        requested_amount=40_00_000,
        requested_tenure_months=240,
        property_value=50_00_000,
        down_payment=10_00_000,
        property_type="apartment",
        property_location="Mumbai",
    )
    base.update(overrides)
    return ApplicantProfile(**base)


# ── Personal Loan — eligible ──────────────────────────────────────────────────

class TestPersonalLoanEligible:
    def test_fully_eligible_profile(self):
        p = _full_pl_profile()
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE
        assert not result.failed_rules
        assert not result.missing_fields

    def test_all_rules_pass(self):
        p = _full_pl_profile()
        result = evaluate_personal_loan(p, _pl_calcs(p))
        pass_ids = {r.rule_id for r in result.rule_checks if r.result == RuleResult.PASS}
        expected = {"PL-AGE-001", "PL-INC-001", "PL-EMP-001", "PL-CRD-001",
                    "PL-FOIR-001", "PL-AMT-001", "PL-TEN-001"}
        assert expected.issubset(pass_ids)

    def test_boundary_age_21(self):
        p = _full_pl_profile(age=21)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_boundary_age_60(self):
        p = _full_pl_profile(age=60)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_boundary_credit_700(self):
        p = _full_pl_profile(credit_score=700)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_self_employed_eligible(self):
        p = _full_pl_profile(employment_type="self_employed", employment_duration_months=30)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE


# ── Personal Loan — not eligible ──────────────────────────────────────────────

class TestPersonalLoanNotEligible:
    def test_age_too_young(self):
        p = _full_pl_profile(age=20)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-AGE-001" for r in result.failed_rules)

    def test_age_too_old(self):
        p = _full_pl_profile(age=61)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE

    def test_income_too_low(self):
        p = _full_pl_profile(monthly_net_income=20_000)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-INC-001" for r in result.failed_rules)

    def test_credit_score_too_low(self):
        p = _full_pl_profile(credit_score=680)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-CRD-001" for r in result.failed_rules)

    def test_foir_too_high(self):
        """High existing EMI + new EMI > 50% FOIR"""
        p = _full_pl_profile(existing_emi=25_000, monthly_net_income=40_000)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-FOIR-001" for r in result.failed_rules)

    def test_loan_amount_too_high(self):
        p = _full_pl_profile(requested_amount=30_00_000)  # > 25L max
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-AMT-001" for r in result.failed_rules)

    def test_tenure_too_long(self):
        p = _full_pl_profile(requested_tenure_months=72)  # > 60 months
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-TEN-001" for r in result.failed_rules)

    def test_salaried_employment_too_short(self):
        p = _full_pl_profile(employment_type="salaried", employment_duration_months=6)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "PL-EMP-001" for r in result.failed_rules)

    def test_multiple_failures(self):
        """Both age and income fail → NOT_ELIGIBLE"""
        p = _full_pl_profile(age=20, monthly_net_income=15_000)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert len(result.failed_rules) >= 2

    def test_income_multiplier_cap(self):
        """Loan > 30× monthly income fails"""
        p = _full_pl_profile(monthly_net_income=30_000, requested_amount=10_00_000)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE


# ── Personal Loan — insufficient information ──────────────────────────────────

class TestPersonalLoanInsufficient:
    def test_missing_age(self):
        p = _full_pl_profile(age=None)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.INSUFFICIENT_INFORMATION
        assert "age" in result.missing_fields

    def test_missing_income(self):
        p = _full_pl_profile(monthly_net_income=None)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.INSUFFICIENT_INFORMATION

    def test_missing_multiple_fields(self):
        p = ApplicantProfile(loan_type="personal_loan", age=35)
        result = evaluate_personal_loan(p, _pl_calcs(p))
        assert result.decision == Decision.INSUFFICIENT_INFORMATION
        assert len(result.missing_fields) > 3


# ── Home Loan — eligible ──────────────────────────────────────────────────────

class TestHomeLoanEligible:
    def test_fully_eligible_profile(self):
        p = _full_hl_profile()
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_boundary_ltv_80pct(self):
        """LTV exactly at 80% (high-value loan > 30L)"""
        p = _full_hl_profile(requested_amount=40_00_000, property_value=50_00_000)
        result = evaluate_home_loan(p, _hl_calcs(p))
        # LTV = 40/50 = 80% → exactly at limit → PASS
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_boundary_age_21(self):
        p = _full_hl_profile(age=21, requested_tenure_months=240)  # closes at 41
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_villa_property(self):
        p = _full_hl_profile(property_type="villa")
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_residential_house_property(self):
        p = _full_hl_profile(property_type="residential_house")
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.POTENTIALLY_ELIGIBLE


# ── Home Loan — not eligible ──────────────────────────────────────────────────

class TestHomeLoanNotEligible:
    def test_ltv_too_high(self):
        """LTV > 80% for high-value loan"""
        p = _full_hl_profile(requested_amount=45_00_000, property_value=50_00_000)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "HL-LTV-001" for r in result.failed_rules)

    def test_commercial_property_ineligible(self):
        p = _full_hl_profile(property_type="commercial")
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "HL-PROP-001" for r in result.failed_rules)

    def test_plot_ineligible(self):
        p = _full_hl_profile(property_type="plot")
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE

    def test_age_loan_closes_after_70(self):
        """Age 50, tenure 360m → closes at 80 → FAIL"""
        p = _full_hl_profile(age=50, requested_tenure_months=360)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE
        assert any(r.rule_id == "HL-AGE-001" for r in result.failed_rules)

    def test_income_too_low(self):
        p = _full_hl_profile(monthly_net_income=30_000)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE

    def test_credit_too_low(self):
        p = _full_hl_profile(credit_score=680)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.NOT_ELIGIBLE


# ── Home Loan — manual review ─────────────────────────────────────────────────

class TestHomeLoanManualReview:
    def test_self_employed_24_35_months(self):
        """Self-employed 24–35 months → MANUAL_REVIEW on employment rule"""
        p = _full_hl_profile(employment_type="self_employed", employment_duration_months=30)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.MANUAL_REVIEW
        assert any(r.rule_id == "HL-EMP-001" for r in result.manual_review_rules)

    def test_self_employed_exactly_24_months(self):
        p = _full_hl_profile(employment_type="self_employed", employment_duration_months=24)
        result = evaluate_home_loan(p, _hl_calcs(p))
        assert result.decision == Decision.MANUAL_REVIEW


# ── run_rules_engine dispatcher ───────────────────────────────────────────────

class TestDispatcher:
    def test_dispatches_to_personal_loan(self):
        p = _full_pl_profile()
        result = run_rules_engine(p, _pl_calcs(p))
        assert result.product == "personal_loan"

    def test_dispatches_to_home_loan(self):
        p = _full_hl_profile()
        result = run_rules_engine(p, _hl_calcs(p))
        assert result.product == "home_loan"

    def test_unknown_loan_type(self):
        p = ApplicantProfile()  # loan_type=None
        result = run_rules_engine(p, LoanCalculations())
        assert result.decision == Decision.INSUFFICIENT_INFORMATION
        assert "loan_type" in result.missing_fields


# ── to_dict serialisation ─────────────────────────────────────────────────────

class TestSerialization:
    def test_eligible_result_to_dict(self):
        p = _full_pl_profile()
        result = evaluate_personal_loan(p, _pl_calcs(p))
        d = result.to_dict()
        assert d["decision"] == "POTENTIALLY_ELIGIBLE"
        assert "rule_checks" in d
        assert "failed_rules" in d
        assert all("rule_id" in r for r in d["rule_checks"])

    def test_profile_to_dict_excludes_none(self):
        p = ApplicantProfile(loan_type="personal_loan", age=35)
        d = p.to_dict()
        assert "age" in d
        assert "monthly_net_income" not in d  # None excluded


# =============================================================================
# Auto Loan Tests
# =============================================================================

def _al_calcs(profile: ApplicantProfile) -> LoanCalculations:
    from rules_engine import AUTO_LOAN_CONFIG as cfg
    return run_all_calculators(
        profile.requested_amount, profile.monthly_net_income, profile.existing_emi,
        profile.requested_tenure_months, cfg["ANNUAL_RATE_PCT"], cfg["MAX_FOIR"],
        on_road_price=profile.on_road_price,
    )


def _full_al_profile(**overrides) -> ApplicantProfile:
    """Return a fully-eligible auto loan profile with optional overrides."""
    base = dict(
        loan_type="auto_loan",
        age=30,
        employment_type="salaried",
        monthly_net_income=60_000,
        employment_duration_months=18,
        credit_score=720,
        existing_emi=0,
        requested_amount=6_00_000,
        requested_tenure_months=48,
        vehicle_type="new",
        on_road_price=8_00_000,
        vehicle_category="four_wheeler",
    )
    base.update(overrides)
    return ApplicantProfile(**base)


class TestAutoLoanEligible:
    def test_fully_eligible_new_car(self):
        p = _full_al_profile()
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.POTENTIALLY_ELIGIBLE
        assert all(rc.result == RuleResult.PASS for rc in r.rule_checks
                   if rc.result != RuleResult.NOT_EVALUATED)

    def test_eligible_two_wheeler(self):
        p = _full_al_profile(
            vehicle_category="two_wheeler",
            on_road_price=1_50_000,
            requested_amount=1_20_000,
            requested_tenure_months=36,
        )
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_eligible_self_employed_sufficient_tenure(self):
        p = _full_al_profile(
            employment_type="self_employed",
            employment_duration_months=30,
        )
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.POTENTIALLY_ELIGIBLE


class TestAutoLoanNotEligible:
    def test_age_below_minimum(self):
        p = _full_al_profile(age=19)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-AGE-001" in rule_ids

    def test_credit_score_below_minimum(self):
        p = _full_al_profile(credit_score=650)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-CRD-001" in rule_ids

    def test_ltv_too_high_for_used_vehicle(self):
        # Used car: on_road_price=500000, loan=400000 -> LTV=80% > 70% limit
        p = _full_al_profile(
            vehicle_type="used",
            on_road_price=5_00_000,
            requested_amount=4_00_000,
            vehicle_age_years=5,
        )
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-LTV-001" in rule_ids

    def test_ltv_acceptable_for_new_vehicle(self):
        # New car: on_road_price=800000, loan=640000 -> LTV=80% < 85% limit
        p = _full_al_profile(
            vehicle_type="new",
            on_road_price=8_00_000,
            requested_amount=6_40_000,
        )
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.POTENTIALLY_ELIGIBLE

    def test_foir_too_high(self):
        # existing_emi=25000 + new EMI >> 50% of income=60000
        p = _full_al_profile(existing_emi=25_000)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-FOIR-001" in rule_ids

    def test_income_below_minimum(self):
        p = _full_al_profile(monthly_net_income=15_000)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-INC-001" in rule_ids

    def test_tenure_exceeds_used_vehicle_max(self):
        # Used vehicle max tenure: 60 months
        p = _full_al_profile(vehicle_type="used", requested_tenure_months=72, vehicle_age_years=3)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-TEN-001" in rule_ids

    def test_used_vehicle_too_old(self):
        p = _full_al_profile(vehicle_type="used", vehicle_age_years=12)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-VEH-001" in rule_ids

    def test_employment_too_short_salaried(self):
        p = _full_al_profile(employment_duration_months=8)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-EMP-001" in rule_ids

    def test_employment_too_short_self_employed(self):
        p = _full_al_profile(employment_type="self_employed", employment_duration_months=18)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.decision == Decision.NOT_ELIGIBLE
        rule_ids = [rc.rule_id for rc in r.failed_rules]
        assert "AL-EMP-001" in rule_ids


class TestAutoLoanInsufficientInfo:
    def test_missing_all_auto_fields(self):
        p = ApplicantProfile(loan_type="auto_loan")
        c = LoanCalculations()
        r = run_rules_engine(p, c)
        assert r.decision == Decision.INSUFFICIENT_INFORMATION

    def test_missing_vehicle_type(self):
        p = _full_al_profile(vehicle_type=None)
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        # vehicle_type is required; profile.missing_fields() will list it
        assert "vehicle_type" in p.missing_fields()

    def test_dispatcher_returns_auto_loan_result(self):
        p = _full_al_profile()
        c = _al_calcs(p)
        r = run_rules_engine(p, c)
        assert r.product == "auto_loan"


class TestAutoLoanProfile:
    def test_required_fields_auto_loan(self):
        p = ApplicantProfile(loan_type="auto_loan")
        req = p.required_fields()
        assert "vehicle_type" in req
        assert "on_road_price" in req
        assert "vehicle_category" in req
        # Property fields should NOT be required for auto loan
        assert "property_value" not in req
        assert "down_payment" not in req

    def test_missing_fields_auto_loan_complete(self):
        p = _full_al_profile()
        assert len(p.missing_fields()) == 0

    def test_missing_fields_auto_loan_incomplete(self):
        p = _full_al_profile(on_road_price=None)
        assert "on_road_price" in p.missing_fields()
