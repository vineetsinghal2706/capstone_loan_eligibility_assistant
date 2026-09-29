"""rules_engine.py — deterministic eligibility rules engine.

The LLM NEVER decides eligibility. This pure-Python engine evaluates each
rule independently and aggregates a final Decision. The LLM only explains
the structured result that this engine produces.

Architecture:
    profile_extractor (LLM)  →  ApplicantProfile
    calculators (Python)      →  LoanCalculations
    rules_engine (Python)     →  EligibilityResult   ← this file
    explanation_agent (LLM)   →  human-readable text

Products: personal_loan, home_loan
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Decision taxonomy (matches requirements §1)
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    POTENTIALLY_ELIGIBLE     = "POTENTIALLY_ELIGIBLE"
    NOT_ELIGIBLE             = "NOT_ELIGIBLE"
    MANUAL_REVIEW            = "MANUAL_REVIEW"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class RuleResult(str, Enum):
    PASS          = "PASS"
    FAIL          = "FAIL"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NOT_EVALUATED = "NOT_EVALUATED"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RuleCheck:
    rule_id:   str
    rule_name: str
    result:    RuleResult
    detail:    str         # human-readable detail, shown in the UI


@dataclass
class ApplicantProfile:
    """All fields collected from the applicant conversation.

    None = not yet provided. The rules engine treats None as NOT_EVALUATED
    and the conversation agent asks the applicant to supply the missing value.
    """
    # Common fields (both products)
    loan_type:                  Optional[str]   = None  # "personal_loan" | "home_loan"
    age:                        Optional[int]   = None
    employment_type:            Optional[str]   = None  # "salaried" | "self_employed" | "business"
    monthly_net_income:         Optional[float] = None  # monthly, INR
    employment_duration_months: Optional[int]   = None
    credit_score:               Optional[int]   = None
    existing_emi:               Optional[float] = None  # total monthly, INR
    requested_amount:           Optional[float] = None  # INR
    requested_tenure_months:    Optional[int]   = None

    # Home Loan additional fields
    property_value:         Optional[float] = None  # INR
    down_payment:           Optional[float] = None  # INR
    property_type:          Optional[str]   = None  # see ELIGIBLE_PROPERTY_TYPES
    property_location:      Optional[str]   = None
    existing_property_loan: Optional[bool]  = None

    # Auto Loan additional fields
    vehicle_type:      Optional[str]   = None  # "new" | "used"
    vehicle_age_years: Optional[int]   = None  # age of used vehicle in years
    on_road_price:     Optional[float] = None  # total on-road price INR
    vehicle_category:  Optional[str]   = None  # "two_wheeler" | "four_wheeler" | "commercial"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def required_fields(self) -> List[str]:
        """Return required field names for the selected loan type."""
        common = [
            "age", "employment_type", "monthly_net_income",
            "employment_duration_months", "credit_score",
            "existing_emi", "requested_amount", "requested_tenure_months",
        ]
        if self.loan_type == "home_loan":
            return common + ["property_value", "down_payment", "property_type"]
        if self.loan_type == "auto_loan":
            return common + ["vehicle_type", "on_road_price", "vehicle_category"]
        return common  # personal_loan

    def missing_fields(self) -> List[str]:
        """Return field names that are required but not yet provided."""
        if self.loan_type is None:
            return ["loan_type"]
        return [f for f in self.required_fields() if getattr(self, f, None) is None]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0


@dataclass
class EligibilityResult:
    decision:            Decision
    product:             str
    rule_checks:         List[RuleCheck] = field(default_factory=list)
    failed_rules:        List[RuleCheck] = field(default_factory=list)
    manual_review_rules: List[RuleCheck] = field(default_factory=list)
    missing_fields:      List[str]       = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _rc(r: RuleCheck) -> dict:
            return {
                "rule_id":   r.rule_id,
                "rule_name": r.rule_name,
                "result":    r.result.value,
                "detail":    r.detail,
            }
        return {
            "decision":            self.decision.value,
            "product":             self.product,
            "rule_checks":         [_rc(r) for r in self.rule_checks],
            "failed_rules":        [_rc(r) for r in self.failed_rules],
            "manual_review_rules": [_rc(r) for r in self.manual_review_rules],
            "missing_fields":      self.missing_fields,
        }


# ---------------------------------------------------------------------------
# Rule configuration — change these, not the engine logic
# ---------------------------------------------------------------------------

PERSONAL_LOAN_CONFIG: Dict[str, Any] = {
    "MIN_AGE":                             21,
    "MAX_AGE":                             60,
    "MIN_MONTHLY_INCOME":                  25_000,
    "MIN_EMP_MONTHS_SALARIED":             12,
    "MIN_EMP_MONTHS_SELF_EMPLOYED":        24,
    "MIN_CREDIT_SCORE":                    700,
    "MAX_FOIR":                            0.50,
    "ANNUAL_RATE_PCT":                     12.0,    # for EMI / FOIR computation
    "MIN_LOAN_AMOUNT":                     50_000,
    "MAX_LOAN_AMOUNT":                     25_00_000,
    "MAX_LOAN_INCOME_MULTIPLIER":          30,      # max loan = 30 × monthly income
    "MIN_TENURE_MONTHS":                   12,
    "MAX_TENURE_MONTHS":                   60,
}

HOME_LOAN_CONFIG: Dict[str, Any] = {
    "MIN_AGE":                             21,
    "MAX_LOAN_CLOSE_AGE":                  70,      # age + tenure/12 ≤ 70
    "MIN_MONTHLY_INCOME":                  40_000,
    "MIN_EMP_MONTHS_SALARIED":             24,
    "MIN_EMP_MONTHS_SELF_EMPLOYED":        36,
    "SELF_EMP_MANUAL_REVIEW_MONTHS":       24,      # 24–35 months → MANUAL_REVIEW
    "MIN_CREDIT_SCORE":                    700,
    "MAX_FOIR":                            0.55,
    "ANNUAL_RATE_PCT":                     8.5,
    "MAX_LTV_HIGH_VALUE":                  0.80,    # loan > ₹30 L
    "MAX_LTV_STANDARD":                    0.85,    # loan ≤ ₹30 L
    "LTV_HIGH_VALUE_THRESHOLD":            30_00_000,
    "ELIGIBLE_PROPERTY_TYPES":             {"apartment", "villa", "residential_house"},
    "MIN_LOAN_AMOUNT":                     5_00_000,
    "MAX_LOAN_AMOUNT":                     5_00_00_000,
    "MIN_TENURE_MONTHS":                   12,
    "MAX_TENURE_MONTHS":                   360,
}

AUTO_LOAN_CONFIG: Dict[str, Any] = {
    "MIN_AGE":                             21,
    "MAX_AGE":                             65,
    "MAX_LOAN_CLOSE_AGE":                  70,
    "MIN_MONTHLY_INCOME":                  20_000,
    "MIN_EMP_MONTHS_SALARIED":             12,
    "MIN_EMP_MONTHS_SELF_EMPLOYED":        24,
    "MIN_CREDIT_SCORE":                    680,
    "MAX_FOIR":                            0.50,
    "ANNUAL_RATE_PCT":                     9.0,
    "MAX_LTV_NEW_VEHICLE":                 0.85,
    "MAX_LTV_USED_VEHICLE":                0.70,
    "MAX_USED_VEHICLE_AGE_YEARS":          10,
    "MAX_VEHICLE_AGE_AT_MATURITY_YEARS":   15,
    "ELIGIBLE_VEHICLE_CATEGORIES":         {"two_wheeler", "four_wheeler", "commercial"},
    "MIN_LOAN_AMOUNT":                     50_000,
    "MAX_LOAN_AMOUNT":                     50_00_000,
    "MIN_TENURE_MONTHS":                   12,
    "MAX_TENURE_MONTHS":                   84,
    "MAX_TENURE_USED_MONTHS":              60,
    "MAX_TENURE_COMMERCIAL_MONTHS":        48,
}


# ---------------------------------------------------------------------------
# Personal Loan rule evaluators
# ---------------------------------------------------------------------------

def _pl_age(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.age is None:
        return RuleCheck("PL-AGE-001", "Age", RuleResult.NOT_EVALUATED, "Age not provided.")
    if p.age < cfg["MIN_AGE"] or p.age > cfg["MAX_AGE"]:
        return RuleCheck("PL-AGE-001", "Age", RuleResult.FAIL,
                         f"Age {p.age} is outside the allowed range "
                         f"{cfg['MIN_AGE']}–{cfg['MAX_AGE']} years.")
    return RuleCheck("PL-AGE-001", "Age", RuleResult.PASS,
                     f"Age {p.age} is within the allowed range "
                     f"{cfg['MIN_AGE']}–{cfg['MAX_AGE']} years.")


def _pl_income(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.monthly_net_income is None:
        return RuleCheck("PL-INC-001", "Minimum Income", RuleResult.NOT_EVALUATED,
                         "Monthly income not provided.")
    minc = cfg["MIN_MONTHLY_INCOME"]
    if p.monthly_net_income < minc:
        return RuleCheck("PL-INC-001", "Minimum Income", RuleResult.FAIL,
                         f"Monthly income ₹{p.monthly_net_income:,.0f} is below "
                         f"minimum ₹{minc:,.0f}.")
    return RuleCheck("PL-INC-001", "Minimum Income", RuleResult.PASS,
                     f"Monthly income ₹{p.monthly_net_income:,.0f} meets "
                     f"minimum ₹{minc:,.0f}.")


def _pl_employment(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.employment_type is None or p.employment_duration_months is None:
        return RuleCheck("PL-EMP-001", "Employment Duration", RuleResult.NOT_EVALUATED,
                         "Employment type or duration not provided.")
    if p.employment_type == "salaried":
        req = cfg["MIN_EMP_MONTHS_SALARIED"]
        if p.employment_duration_months < req:
            return RuleCheck("PL-EMP-001", "Employment Duration", RuleResult.FAIL,
                             f"Salaried employment {p.employment_duration_months} months "
                             f"< required {req} months.")
        return RuleCheck("PL-EMP-001", "Employment Duration", RuleResult.PASS,
                         f"Salaried employment {p.employment_duration_months} months "
                         f"meets the {req}-month requirement.")
    else:  # self_employed / business
        req = cfg["MIN_EMP_MONTHS_SELF_EMPLOYED"]
        if p.employment_duration_months < req:
            return RuleCheck("PL-EMP-001", "Employment Duration", RuleResult.FAIL,
                             f"Self-employed/business {p.employment_duration_months} months "
                             f"< required {req} months.")
        return RuleCheck("PL-EMP-001", "Employment Duration", RuleResult.PASS,
                         f"Business {p.employment_duration_months} months meets "
                         f"the {req}-month requirement.")


def _pl_credit(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.credit_score is None:
        return RuleCheck("PL-CRD-001", "Credit Score", RuleResult.NOT_EVALUATED,
                         "Credit score not provided.")
    if p.credit_score < cfg["MIN_CREDIT_SCORE"]:
        return RuleCheck("PL-CRD-001", "Credit Score", RuleResult.FAIL,
                         f"Credit score {p.credit_score} is below minimum "
                         f"{cfg['MIN_CREDIT_SCORE']}.")
    return RuleCheck("PL-CRD-001", "Credit Score", RuleResult.PASS,
                     f"Credit score {p.credit_score} meets minimum "
                     f"{cfg['MIN_CREDIT_SCORE']}.")


def _pl_foir(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if calcs.foir is None:
        return RuleCheck("PL-FOIR-001", "FOIR", RuleResult.NOT_EVALUATED,
                         "FOIR cannot be computed (missing income/amount/tenure).")
    max_foir = cfg["MAX_FOIR"]
    pct = calcs.foir * 100
    if calcs.foir > max_foir:
        return RuleCheck("PL-FOIR-001", "FOIR", RuleResult.FAIL,
                         f"FOIR {pct:.1f}% exceeds maximum {max_foir * 100:.0f}%.")
    return RuleCheck("PL-FOIR-001", "FOIR", RuleResult.PASS,
                     f"FOIR {pct:.1f}% is within maximum {max_foir * 100:.0f}%.")


def _pl_amount(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if p.requested_amount is None:
        return RuleCheck("PL-AMT-001", "Loan Amount", RuleResult.NOT_EVALUATED,
                         "Requested loan amount not provided.")
    min_a, max_a = cfg["MIN_LOAN_AMOUNT"], cfg["MAX_LOAN_AMOUNT"]
    if p.requested_amount < min_a:
        return RuleCheck("PL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount ₹{p.requested_amount:,.0f} < minimum ₹{min_a:,.0f}.")
    if p.requested_amount > max_a:
        return RuleCheck("PL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount ₹{p.requested_amount:,.0f} > maximum ₹{max_a:,.0f}.")
    if p.monthly_net_income:
        cap = p.monthly_net_income * cfg["MAX_LOAN_INCOME_MULTIPLIER"]
        if p.requested_amount > cap:
            return RuleCheck("PL-AMT-001", "Loan Amount", RuleResult.FAIL,
                             f"Amount ₹{p.requested_amount:,.0f} exceeds "
                             f"{cfg['MAX_LOAN_INCOME_MULTIPLIER']}× monthly income "
                             f"(cap ₹{cap:,.0f}).")
    return RuleCheck("PL-AMT-001", "Loan Amount", RuleResult.PASS,
                     f"Amount ₹{p.requested_amount:,.0f} is within policy limits.")


def _pl_tenure(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.requested_tenure_months is None:
        return RuleCheck("PL-TEN-001", "Loan Tenure", RuleResult.NOT_EVALUATED,
                         "Loan tenure not provided.")
    min_t, max_t = cfg["MIN_TENURE_MONTHS"], cfg["MAX_TENURE_MONTHS"]
    if p.requested_tenure_months < min_t or p.requested_tenure_months > max_t:
        return RuleCheck("PL-TEN-001", "Loan Tenure", RuleResult.FAIL,
                         f"Tenure {p.requested_tenure_months} months is outside "
                         f"allowed range {min_t}–{max_t} months.")
    return RuleCheck("PL-TEN-001", "Loan Tenure", RuleResult.PASS,
                     f"Tenure {p.requested_tenure_months} months is within "
                     f"{min_t}–{max_t} months.")


# ---------------------------------------------------------------------------
# Home Loan rule evaluators
# ---------------------------------------------------------------------------

def _hl_age(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.age is None:
        return RuleCheck("HL-AGE-001", "Age Eligibility", RuleResult.NOT_EVALUATED,
                         "Age not provided.")
    if p.age < cfg["MIN_AGE"]:
        return RuleCheck("HL-AGE-001", "Age Eligibility", RuleResult.FAIL,
                         f"Age {p.age} < minimum {cfg['MIN_AGE']}.")
    max_close = cfg["MAX_LOAN_CLOSE_AGE"]
    if p.requested_tenure_months:
        age_at_close = p.age + p.requested_tenure_months / 12.0
        if age_at_close > max_close:
            return RuleCheck("HL-AGE-001", "Age Eligibility", RuleResult.FAIL,
                             f"Loan would close at age {age_at_close:.1f}, "
                             f"exceeding limit of {max_close}.")
    return RuleCheck("HL-AGE-001", "Age Eligibility", RuleResult.PASS,
                     f"Age {p.age} is eligible and loan closes within age limit of {max_close}.")


def _hl_income(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.monthly_net_income is None:
        return RuleCheck("HL-INC-001", "Minimum Income", RuleResult.NOT_EVALUATED,
                         "Monthly income not provided.")
    minc = cfg["MIN_MONTHLY_INCOME"]
    if p.monthly_net_income < minc:
        return RuleCheck("HL-INC-001", "Minimum Income", RuleResult.FAIL,
                         f"Monthly income ₹{p.monthly_net_income:,.0f} < minimum ₹{minc:,.0f}.")
    return RuleCheck("HL-INC-001", "Minimum Income", RuleResult.PASS,
                     f"Monthly income ₹{p.monthly_net_income:,.0f} meets minimum ₹{minc:,.0f}.")


def _hl_employment(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.employment_type is None or p.employment_duration_months is None:
        return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.NOT_EVALUATED,
                         "Employment type or duration not provided.")
    dur = p.employment_duration_months
    if p.employment_type == "salaried":
        req = cfg["MIN_EMP_MONTHS_SALARIED"]
        if dur < req:
            return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.FAIL,
                             f"Salaried employment {dur} months < required {req} months.")
        return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.PASS,
                         f"Salaried employment {dur} months meets {req}-month requirement.")
    else:  # self_employed / business
        manual_thresh = cfg["SELF_EMP_MANUAL_REVIEW_MONTHS"]
        full_thresh   = cfg["MIN_EMP_MONTHS_SELF_EMPLOYED"]
        if dur < manual_thresh:
            return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.FAIL,
                             f"Business operation {dur} months < minimum {manual_thresh} months.")
        if dur < full_thresh:
            return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.MANUAL_REVIEW,
                             f"Business {dur} months is between {manual_thresh}–{full_thresh} months — "
                             f"additional income documentation required for manual review.")
        return RuleCheck("HL-EMP-001", "Employment Stability", RuleResult.PASS,
                         f"Business operation {dur} months meets {full_thresh}-month requirement.")


def _hl_credit(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.credit_score is None:
        return RuleCheck("HL-CRD-001", "Credit Score", RuleResult.NOT_EVALUATED,
                         "Credit score not provided.")
    if p.credit_score < cfg["MIN_CREDIT_SCORE"]:
        return RuleCheck("HL-CRD-001", "Credit Score", RuleResult.FAIL,
                         f"Credit score {p.credit_score} < minimum {cfg['MIN_CREDIT_SCORE']}.")
    return RuleCheck("HL-CRD-001", "Credit Score", RuleResult.PASS,
                     f"Credit score {p.credit_score} meets minimum {cfg['MIN_CREDIT_SCORE']}.")


def _hl_foir(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if calcs.foir is None:
        return RuleCheck("HL-FOIR-001", "FOIR", RuleResult.NOT_EVALUATED,
                         "FOIR cannot be computed (missing income/amount/tenure).")
    max_foir = cfg["MAX_FOIR"]
    pct = calcs.foir * 100
    if calcs.foir > max_foir:
        return RuleCheck("HL-FOIR-001", "FOIR", RuleResult.FAIL,
                         f"FOIR {pct:.1f}% > maximum {max_foir * 100:.0f}%.")
    return RuleCheck("HL-FOIR-001", "FOIR", RuleResult.PASS,
                     f"FOIR {pct:.1f}% ≤ maximum {max_foir * 100:.0f}%.")


def _hl_ltv(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if calcs.ltv is None:
        return RuleCheck("HL-LTV-001", "Loan-to-Value (LTV)", RuleResult.NOT_EVALUATED,
                         "LTV cannot be computed (missing property value or loan amount).")
    threshold = cfg["LTV_HIGH_VALUE_THRESHOLD"]
    is_high = p.requested_amount and p.requested_amount > threshold
    max_ltv  = cfg["MAX_LTV_HIGH_VALUE"] if is_high else cfg["MAX_LTV_STANDARD"]
    pct = calcs.ltv * 100
    if calcs.ltv > max_ltv:
        return RuleCheck("HL-LTV-001", "Loan-to-Value (LTV)", RuleResult.FAIL,
                         f"LTV {pct:.1f}% > maximum {max_ltv * 100:.0f}% for this loan amount.")
    return RuleCheck("HL-LTV-001", "Loan-to-Value (LTV)", RuleResult.PASS,
                     f"LTV {pct:.1f}% ≤ maximum {max_ltv * 100:.0f}%.")


def _hl_property(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.property_type is None:
        return RuleCheck("HL-PROP-001", "Property Eligibility", RuleResult.NOT_EVALUATED,
                         "Property type not provided.")
    eligible = cfg["ELIGIBLE_PROPERTY_TYPES"]
    pt = p.property_type.lower().replace(" ", "_")
    if pt not in eligible:
        return RuleCheck("HL-PROP-001", "Property Eligibility", RuleResult.FAIL,
                         f"Property type '{p.property_type}' is not eligible. "
                         f"Eligible types: {', '.join(sorted(eligible))}.")
    return RuleCheck("HL-PROP-001", "Property Eligibility", RuleResult.PASS,
                     f"Property type '{p.property_type}' is eligible for a home loan.")


def _hl_amount(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if p.requested_amount is None:
        return RuleCheck("HL-AMT-001", "Loan Amount", RuleResult.NOT_EVALUATED,
                         "Requested loan amount not provided.")
    min_a, max_a = cfg["MIN_LOAN_AMOUNT"], cfg["MAX_LOAN_AMOUNT"]
    if p.requested_amount < min_a:
        return RuleCheck("HL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount ₹{p.requested_amount:,.0f} < minimum ₹{min_a:,.0f}.")
    if p.requested_amount > max_a:
        return RuleCheck("HL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount ₹{p.requested_amount:,.0f} > maximum ₹{max_a:,.0f}.")
    if calcs.max_affordable_loan and p.requested_amount > calcs.max_affordable_loan * 1.01:
        return RuleCheck("HL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount ₹{p.requested_amount:,.0f} exceeds maximum affordable "
                         f"₹{calcs.max_affordable_loan:,.0f} based on income and FOIR.")
    return RuleCheck("HL-AMT-001", "Loan Amount", RuleResult.PASS,
                     f"Amount ₹{p.requested_amount:,.0f} is within policy limits.")


def _hl_tenure(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.requested_tenure_months is None:
        return RuleCheck("HL-TEN-001", "Loan Tenure", RuleResult.NOT_EVALUATED,
                         "Loan tenure not provided.")
    min_t, max_t = cfg["MIN_TENURE_MONTHS"], cfg["MAX_TENURE_MONTHS"]
    if p.requested_tenure_months < min_t or p.requested_tenure_months > max_t:
        return RuleCheck("HL-TEN-001", "Loan Tenure", RuleResult.FAIL,
                         f"Tenure {p.requested_tenure_months} months outside "
                         f"{min_t}–{max_t} months.")
    return RuleCheck("HL-TEN-001", "Loan Tenure", RuleResult.PASS,
                     f"Tenure {p.requested_tenure_months} months is within "
                     f"{min_t}–{max_t} months.")


# ---------------------------------------------------------------------------
# Auto Loan rule evaluators
# ---------------------------------------------------------------------------

def _al_age(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.age is None:
        return RuleCheck("AL-AGE-001", "Age Eligibility", RuleResult.NOT_EVALUATED, "Age not provided.")
    if p.age < cfg["MIN_AGE"]:
        return RuleCheck("AL-AGE-001", "Age Eligibility", RuleResult.FAIL,
                         f"Age {p.age} < minimum {cfg['MIN_AGE']}.")
    if p.age > cfg["MAX_AGE"]:
        return RuleCheck("AL-AGE-001", "Age Eligibility", RuleResult.FAIL,
                         f"Age {p.age} > maximum {cfg['MAX_AGE']}.")
    if p.requested_tenure_months:
        close_age = p.age + p.requested_tenure_months / 12.0
        if close_age > cfg["MAX_LOAN_CLOSE_AGE"]:
            return RuleCheck("AL-AGE-001", "Age Eligibility", RuleResult.FAIL,
                             f"Loan closes at age {close_age:.1f}, exceeds limit {cfg['MAX_LOAN_CLOSE_AGE']}.")
    return RuleCheck("AL-AGE-001", "Age Eligibility", RuleResult.PASS,
                     f"Age {p.age} is within allowed range {cfg['MIN_AGE']}-{cfg['MAX_AGE']}.")


def _al_income(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.monthly_net_income is None:
        return RuleCheck("AL-INC-001", "Minimum Income", RuleResult.NOT_EVALUATED, "Monthly income not provided.")
    minc = cfg["MIN_MONTHLY_INCOME"]
    if p.monthly_net_income < minc:
        return RuleCheck("AL-INC-001", "Minimum Income", RuleResult.FAIL,
                         f"Monthly income Rs.{p.monthly_net_income:,.0f} < minimum Rs.{minc:,.0f}.")
    return RuleCheck("AL-INC-001", "Minimum Income", RuleResult.PASS,
                     f"Monthly income Rs.{p.monthly_net_income:,.0f} meets minimum Rs.{minc:,.0f}.")


def _al_employment(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.employment_type is None or p.employment_duration_months is None:
        return RuleCheck("AL-EMP-001", "Employment Stability", RuleResult.NOT_EVALUATED,
                         "Employment type or duration not provided.")
    if p.employment_type == "salaried":
        req = cfg["MIN_EMP_MONTHS_SALARIED"]
        if p.employment_duration_months < req:
            return RuleCheck("AL-EMP-001", "Employment Stability", RuleResult.FAIL,
                             f"Salaried employment {p.employment_duration_months}m < required {req}m.")
        return RuleCheck("AL-EMP-001", "Employment Stability", RuleResult.PASS,
                         f"Salaried employment {p.employment_duration_months}m meets {req}m requirement.")
    else:
        req = cfg["MIN_EMP_MONTHS_SELF_EMPLOYED"]
        if p.employment_duration_months < req:
            return RuleCheck("AL-EMP-001", "Employment Stability", RuleResult.FAIL,
                             f"Business {p.employment_duration_months}m < required {req}m.")
        return RuleCheck("AL-EMP-001", "Employment Stability", RuleResult.PASS,
                         f"Business {p.employment_duration_months}m meets {req}m requirement.")


def _al_credit(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.credit_score is None:
        return RuleCheck("AL-CRD-001", "Credit Score", RuleResult.NOT_EVALUATED, "Credit score not provided.")
    if p.credit_score < cfg["MIN_CREDIT_SCORE"]:
        return RuleCheck("AL-CRD-001", "Credit Score", RuleResult.FAIL,
                         f"Credit score {p.credit_score} < minimum {cfg['MIN_CREDIT_SCORE']}.")
    return RuleCheck("AL-CRD-001", "Credit Score", RuleResult.PASS,
                     f"Credit score {p.credit_score} meets minimum {cfg['MIN_CREDIT_SCORE']}.")


def _al_foir(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if calcs.foir is None:
        return RuleCheck("AL-FOIR-001", "FOIR", RuleResult.NOT_EVALUATED,
                         "FOIR cannot be computed (missing income/amount/tenure).")
    max_foir = cfg["MAX_FOIR"]
    pct = calcs.foir * 100
    if calcs.foir > max_foir:
        return RuleCheck("AL-FOIR-001", "FOIR", RuleResult.FAIL,
                         f"FOIR {pct:.1f}% > maximum {max_foir * 100:.0f}%.")
    return RuleCheck("AL-FOIR-001", "FOIR", RuleResult.PASS,
                     f"FOIR {pct:.1f}% is within maximum {max_foir * 100:.0f}%.")


def _al_ltv(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if calcs.ltv is None:
        return RuleCheck("AL-LTV-001", "Loan-to-Value (LTV)", RuleResult.NOT_EVALUATED,
                         "LTV cannot be computed (missing on-road price or loan amount).")
    is_used = (p.vehicle_type == "used")
    max_ltv = cfg["MAX_LTV_USED_VEHICLE"] if is_used else cfg["MAX_LTV_NEW_VEHICLE"]
    pct = calcs.ltv * 100
    label = "used" if is_used else "new"
    if calcs.ltv > max_ltv:
        return RuleCheck("AL-LTV-001", "Loan-to-Value (LTV)", RuleResult.FAIL,
                         f"LTV {pct:.1f}% > maximum {max_ltv * 100:.0f}% for {label} vehicle.")
    return RuleCheck("AL-LTV-001", "Loan-to-Value (LTV)", RuleResult.PASS,
                     f"LTV {pct:.1f}% is within {max_ltv * 100:.0f}% limit for {label} vehicle.")


def _al_vehicle(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.vehicle_type is None:
        return RuleCheck("AL-VEH-001", "Vehicle Eligibility", RuleResult.NOT_EVALUATED,
                         "Vehicle type (new/used) not provided.")
    if p.vehicle_type == "used" and p.vehicle_age_years is not None:
        if p.vehicle_age_years > cfg["MAX_USED_VEHICLE_AGE_YEARS"]:
            return RuleCheck("AL-VEH-001", "Vehicle Eligibility", RuleResult.FAIL,
                             f"Used vehicle age {p.vehicle_age_years}y > maximum {cfg['MAX_USED_VEHICLE_AGE_YEARS']}y.")
    if p.vehicle_category is not None:
        eligible = cfg["ELIGIBLE_VEHICLE_CATEGORIES"]
        if p.vehicle_category not in eligible:
            return RuleCheck("AL-VEH-001", "Vehicle Eligibility", RuleResult.FAIL,
                             f"Vehicle category '{p.vehicle_category}' not eligible.")
    return RuleCheck("AL-VEH-001", "Vehicle Eligibility", RuleResult.PASS,
                     f"Vehicle is eligible ({p.vehicle_type or 'unspecified type'}).")


def _al_amount(p: ApplicantProfile, cfg: dict, calcs) -> RuleCheck:
    if p.requested_amount is None:
        return RuleCheck("AL-AMT-001", "Loan Amount", RuleResult.NOT_EVALUATED, "Loan amount not provided.")
    min_a, max_a = cfg["MIN_LOAN_AMOUNT"], cfg["MAX_LOAN_AMOUNT"]
    if p.requested_amount < min_a:
        return RuleCheck("AL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount Rs.{p.requested_amount:,.0f} < minimum Rs.{min_a:,.0f}.")
    if p.requested_amount > max_a:
        return RuleCheck("AL-AMT-001", "Loan Amount", RuleResult.FAIL,
                         f"Amount Rs.{p.requested_amount:,.0f} > maximum Rs.{max_a:,.0f}.")
    return RuleCheck("AL-AMT-001", "Loan Amount", RuleResult.PASS,
                     f"Amount Rs.{p.requested_amount:,.0f} is within policy limits.")


def _al_tenure(p: ApplicantProfile, cfg: dict) -> RuleCheck:
    if p.requested_tenure_months is None:
        return RuleCheck("AL-TEN-001", "Loan Tenure", RuleResult.NOT_EVALUATED, "Loan tenure not provided.")
    min_t = cfg["MIN_TENURE_MONTHS"]
    if p.vehicle_type == "used":
        max_t = cfg["MAX_TENURE_USED_MONTHS"]
    elif p.vehicle_category == "commercial":
        max_t = cfg["MAX_TENURE_COMMERCIAL_MONTHS"]
    else:
        max_t = cfg["MAX_TENURE_MONTHS"]
    if p.requested_tenure_months < min_t or p.requested_tenure_months > max_t:
        return RuleCheck("AL-TEN-001", "Loan Tenure", RuleResult.FAIL,
                         f"Tenure {p.requested_tenure_months}m outside {min_t}-{max_t}m.")
    return RuleCheck("AL-TEN-001", "Loan Tenure", RuleResult.PASS,
                     f"Tenure {p.requested_tenure_months}m within {min_t}-{max_t}m.")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    checks:         List[RuleCheck],
    missing_fields: List[str],
    product:        str,
) -> EligibilityResult:
    failed  = [c for c in checks if c.result == RuleResult.FAIL]
    manual  = [c for c in checks if c.result == RuleResult.MANUAL_REVIEW]
    unevals = [c for c in checks if c.result == RuleResult.NOT_EVALUATED]

    if missing_fields:
        decision = Decision.INSUFFICIENT_INFORMATION
    elif failed:
        decision = Decision.NOT_ELIGIBLE
    elif manual:
        decision = Decision.MANUAL_REVIEW
    elif unevals:
        # All required fields provided but some rules still unresolved — stay cautious
        decision = Decision.INSUFFICIENT_INFORMATION
    else:
        decision = Decision.POTENTIALLY_ELIGIBLE

    return EligibilityResult(
        decision=decision,
        product=product,
        rule_checks=checks,
        failed_rules=failed,
        manual_review_rules=manual,
        missing_fields=missing_fields,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def evaluate_personal_loan(profile: ApplicantProfile, calcs) -> EligibilityResult:
    """Run all Personal Loan rules. Returns EligibilityResult."""
    cfg     = PERSONAL_LOAN_CONFIG
    missing = [f for f in profile.missing_fields() if f != "loan_type"]
    checks  = [
        _pl_age(profile, cfg),
        _pl_income(profile, cfg),
        _pl_employment(profile, cfg),
        _pl_credit(profile, cfg),
        _pl_foir(profile, cfg, calcs),
        _pl_amount(profile, cfg, calcs),
        _pl_tenure(profile, cfg),
    ]
    return _aggregate(checks, missing, "personal_loan")


def evaluate_home_loan(profile: ApplicantProfile, calcs) -> EligibilityResult:
    """Run all Home Loan rules. Returns EligibilityResult."""
    cfg     = HOME_LOAN_CONFIG
    missing = [f for f in profile.missing_fields() if f != "loan_type"]
    checks  = [
        _hl_age(profile, cfg),
        _hl_income(profile, cfg),
        _hl_employment(profile, cfg),
        _hl_credit(profile, cfg),
        _hl_foir(profile, cfg, calcs),
        _hl_ltv(profile, cfg, calcs),
        _hl_property(profile, cfg),
        _hl_amount(profile, cfg, calcs),
        _hl_tenure(profile, cfg),
    ]
    return _aggregate(checks, missing, "home_loan")


def evaluate_auto_loan(profile: ApplicantProfile, calcs) -> EligibilityResult:
    """Run all Auto Loan rules. Returns EligibilityResult."""
    cfg     = AUTO_LOAN_CONFIG
    missing = [f for f in profile.missing_fields() if f != "loan_type"]
    checks  = [
        _al_age(profile, cfg),
        _al_income(profile, cfg),
        _al_employment(profile, cfg),
        _al_credit(profile, cfg),
        _al_foir(profile, cfg, calcs),
        _al_ltv(profile, cfg, calcs),
        _al_vehicle(profile, cfg),
        _al_amount(profile, cfg, calcs),
        _al_tenure(profile, cfg),
    ]
    return _aggregate(checks, missing, "auto_loan")


def run_rules_engine(profile: ApplicantProfile, calcs) -> EligibilityResult:
    """Dispatch to the correct product engine based on profile.loan_type."""
    if profile.loan_type == "personal_loan":
        return evaluate_personal_loan(profile, calcs)
    if profile.loan_type == "home_loan":
        return evaluate_home_loan(profile, calcs)
    if profile.loan_type == "auto_loan":
        return evaluate_auto_loan(profile, calcs)
    return EligibilityResult(
        decision=Decision.INSUFFICIENT_INFORMATION,
        product="unknown",
        missing_fields=["loan_type"],
    )

