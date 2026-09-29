"""calculators.py -- deterministic financial calculators for loan eligibility.

All calculations are pure Python arithmetic. No LLM involvement.
Used by the rules engine and returned in the API response for full transparency.

Formulas:
  EMI  = P x r x (1+r)^n / ((1+r)^n - 1)    (reducing-balance)
  FOIR = (existing_emi + new_emi) / monthly_income
  LTV  = loan_amount / collateral_value  (property OR on-road vehicle price)
  max_loan: solve EMI formula for P given max allowed EMI
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoanCalculations:
    """All calculation results for a loan application.

    None means the field could not be computed (missing inputs).
    """
    emi:                Optional[float] = None   # monthly EMI in INR
    foir:               Optional[float] = None   # Fixed Obligation to Income Ratio (0-1)
    ltv:                Optional[float] = None   # Loan-to-Value ratio (0-1)
    max_affordable_loan: Optional[float] = None  # max loan by income + FOIR + tenure
    total_interest:     Optional[float] = None   # total interest paid over tenure
    total_payment:      Optional[float] = None   # principal + total_interest

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# Individual calculators
# ---------------------------------------------------------------------------

def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Standard reducing-balance EMI.

    EMI = P x r x (1+r)^n / ((1+r)^n - 1)
    where r = monthly rate = annual_rate_pct / (12 x 100)
          n = tenure_months

    Edge cases:
      - principal <= 0  -> 0
      - annual_rate_pct == 0  -> simple division (zero-interest loan)
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    if annual_rate_pct <= 0:
        return round(principal / tenure_months, 2)
    r = annual_rate_pct / (12.0 * 100.0)
    factor = math.pow(1.0 + r, tenure_months)
    emi = principal * r * factor / (factor - 1.0)
    return round(emi, 2)


def calculate_foir(existing_emi: float, new_emi: float, monthly_income: float) -> float:
    """Fixed Obligation to Income Ratio.

    FOIR = (existing_emi + new_emi) / monthly_income

    Returns float 0-inf. Returns float('inf') when monthly_income == 0.
    """
    if monthly_income <= 0:
        return float("inf")
    return round((existing_emi + new_emi) / monthly_income, 4)


def calculate_ltv(loan_amount: float, collateral_value: float) -> float:
    """Loan-to-Value ratio.

    LTV = loan_amount / collateral_value
    Works for both property (home loan) and on-road price (auto loan).

    Returns float 0-inf. Returns float('inf') when collateral_value == 0.
    """
    if collateral_value <= 0:
        return float("inf")
    return round(loan_amount / collateral_value, 4)


def calculate_max_affordable_loan(
    monthly_income: float,
    existing_emi: float,
    max_foir: float,
    annual_rate_pct: float,
    tenure_months: int,
) -> float:
    """Maximum loan amount an applicant can afford given FOIR constraint.

    Step 1: max_allowed_emi = (max_foir x monthly_income) - existing_emi
    Step 2: Solve EMI formula for P:
            P = max_emi x ((1+r)^n - 1) / (r x (1+r)^n)

    Returns 0 when no budget is available.
    """
    if monthly_income <= 0 or tenure_months <= 0:
        return 0.0
    max_allowed_emi = (max_foir * monthly_income) - existing_emi
    if max_allowed_emi <= 0:
        return 0.0
    if annual_rate_pct <= 0:
        return round(max_allowed_emi * tenure_months, 2)
    r = annual_rate_pct / (12.0 * 100.0)
    factor = math.pow(1.0 + r, tenure_months)
    max_loan = max_allowed_emi * (factor - 1.0) / (r * factor)
    return round(max_loan, 2)


# ---------------------------------------------------------------------------
# Composite runner -- used by app.py
# ---------------------------------------------------------------------------

def run_all_calculators(
    requested_amount:    Optional[float],
    monthly_income:      Optional[float],
    existing_emi:        Optional[float],
    tenure_months:       Optional[int],
    annual_rate_pct:     float,
    max_foir:            float,
    property_value:      Optional[float] = None,
    on_road_price:       Optional[float] = None,
) -> LoanCalculations:
    """Compute all applicable calculators and return a LoanCalculations object.

    Silently skips any calculation where required inputs are None.
    Supports home loan (property_value) and auto loan (on_road_price) LTV.
    """
    calcs = LoanCalculations()
    existing = existing_emi or 0.0

    if requested_amount and tenure_months:
        emi_val = calculate_emi(requested_amount, annual_rate_pct, tenure_months)
        calcs.emi = emi_val
        calcs.total_payment = round(emi_val * tenure_months, 2)
        calcs.total_interest = round(calcs.total_payment - requested_amount, 2)

        if monthly_income:
            calcs.foir = calculate_foir(existing, emi_val, monthly_income)

    # LTV: home loan uses property_value; auto loan uses on_road_price
    collateral = property_value or on_road_price
    if collateral and requested_amount:
        calcs.ltv = calculate_ltv(requested_amount, collateral)

    if monthly_income and tenure_months:
        calcs.max_affordable_loan = calculate_max_affordable_loan(
            monthly_income, existing, max_foir, annual_rate_pct, tenure_months
        )

    return calcs
