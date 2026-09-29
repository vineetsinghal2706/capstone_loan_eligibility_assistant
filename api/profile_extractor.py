"""profile_extractor.py -- LLM-based applicant profile extraction.

Extracts structured ApplicantProfile fields from free-form conversation.
The LLM ONLY extracts -- it does NOT make eligibility decisions.

Key features:
  * Semantic loan-type detection (no hard labels required from user)
  * Auto Loan support: vehicle_type, vehicle_age_years, on_road_price, vehicle_category
  * Profile MERGE: prior_profile fields kept; new fields overlay them
    so the assistant never re-asks for already-provided information
  * Document checklist returned only when user explicitly asks
"""
import json
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger("loanassist.profile")

from rules_engine import ApplicantProfile


# ---------------------------------------------------------------------------
# Loan type alias map (Python fallback after LLM normalisation)
# ---------------------------------------------------------------------------

LOAN_TYPE_ALIAS_MAP = {
    # Home loan variants
    "home_loan": "home_loan", "housing_loan": "home_loan", "house_loan": "home_loan",
    "mortgage": "home_loan", "property_loan": "home_loan", "building_loan": "home_loan",
    "construction_loan": "home_loan", "flat_loan": "home_loan", "apartment_loan": "home_loan",
    "real_estate_loan": "home_loan",
    # Personal loan variants
    "personal_loan": "personal_loan", "consumer_loan": "personal_loan", "pl": "personal_loan",
    "instant_loan": "personal_loan", "salary_loan": "personal_loan", "cash_loan": "personal_loan",
    # Auto loan variants
    "auto_loan": "auto_loan", "car_loan": "auto_loan", "vehicle_loan": "auto_loan",
    "scooter_loan": "auto_loan", "bike_loan": "auto_loan", "motorcycle_loan": "auto_loan",
    "two_wheeler_loan": "auto_loan", "four_wheeler_loan": "auto_loan",
    "bus_loan": "auto_loan", "truck_loan": "auto_loan",
    "commercial_vehicle_loan": "auto_loan", "ev_loan": "auto_loan",
    "electric_vehicle_loan": "auto_loan", "automobile_loan": "auto_loan",
    "used_car_loan": "auto_loan", "second_hand_car_loan": "auto_loan",
}

VEHICLE_CATEGORY_HINTS = {
    "two_wheeler":  ["scooter", "bike", "motorcycle", "two wheeler", "two-wheeler", "activa", "pulsar", "splendor", "two_wheeler"],
    "commercial":   ["bus", "truck", "lorry", "van", "tempo", "commercial"],
    "four_wheeler": ["car", "suv", "sedan", "hatchback", "four wheeler", "four-wheeler", "muv", "four_wheeler", "vehicle", "automobile", "auto"],
}

# Keywords that trigger document checklist inclusion in response
DOC_REQUEST_KEYWORDS = {
    "document", "documents", "docs", "doc", "papers", "paperwork",
    "required", "checklist", "what do i need", "what documents",
    "what papers", "what to bring", "what to submit", "submission",
    "application documents", "supporting documents",
}


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a structured data extractor for a loan eligibility assistant.

Given a conversation, extract all applicant-provided information into a
JSON object. Return ONLY valid JSON -- no explanations, no markdown.

Schema (use null for any unknown field):
{
  "loan_type":                  "personal_loan" | "home_loan" | "auto_loan" | null,
  "age":                        integer | null,
  "employment_type":            "salaried" | "self_employed" | "business" | null,
  "monthly_net_income":         float (INR, monthly) | null,
  "employment_duration_months": integer | null,
  "credit_score":               integer | null,
  "existing_emi":               float (INR, total monthly) | null,
  "requested_amount":           float (INR) | null,
  "requested_tenure_months":    integer | null,
  "property_value":             float (INR) | null,
  "down_payment":               float (INR) | null,
  "property_type":              "apartment" | "villa" | "residential_house" | "plot" | "commercial" | null,
  "property_location":          string | null,
  "existing_property_loan":     boolean | null,
  "vehicle_type":               "new" | "used" | null,
  "vehicle_age_years":          integer | null,
  "on_road_price":              float (INR) | null,
  "vehicle_category":           "two_wheeler" | "four_wheeler" | "commercial" | null
}

Semantic loan type detection (infer from context -- do not require exact phrasing):
- home loan, house loan, housing loan, building loan, construction loan, flat loan,
  apartment loan, mortgage, property loan -> "home_loan"
- personal loan, consumer loan, salary loan, cash loan, instant loan -> "personal_loan"
- car loan, vehicle loan, auto loan, scooter loan, bike loan, motorcycle loan,
  two-wheeler loan, four-wheeler loan, bus loan, truck loan, commercial vehicle loan,
  EV loan, electric vehicle loan, used car loan, second-hand car -> "auto_loan"

Vehicle category detection:
- scooter, bike, motorcycle, two-wheeler, Activa, Pulsar -> "two_wheeler"
- car, SUV, sedan, hatchback, MUV, four-wheeler -> "four_wheeler"
- bus, truck, lorry, van, tempo -> "commercial"

Conversion rules:
- If the applicant switches loan product, loan_type MUST be the LATEST requested product.
- personal_loan: clear all property and vehicle fields (set null)
- home_loan: clear all vehicle fields (set null)
- auto_loan: clear all property fields (set null)
- Annual income -> divide by 12 for monthly
- Duration in years -> multiply by 12 for months
- "employment_duration_months" is work/business experience (e.g. "2 years in business" -> employment_duration_months = 24).
- "requested_tenure_months" is ONLY the loan repayment tenure explicitly chosen for the loan (e.g. "36 months tenure", "loan for 5 years"). NEVER assign business or employment duration to requested_tenure_months. If loan tenure is not explicitly mentioned, requested_tenure_months MUST be null.
- Amounts in lakhs: "5 lakhs" = 500000, "1 crore" = 10000000
- CIBIL score = credit score
- on_road_price = ex-showroom price + registration + insurance + taxes

Extract only what is explicitly stated. Do not infer. Use null for unstated fields.
"""


def _build_extraction_messages(history: list[dict]) -> list[dict]:
    transcript_lines = []
    for msg in history:
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines)
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Conversation:\n{transcript}"},
    ]


def _parse_profile_json(raw: str) -> dict:
    stripped = re.sub(r"```(?:json)?", "", raw).strip()
    start = stripped.find("{")
    end   = stripped.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(stripped[start:end])
    except json.JSONDecodeError:
        return {}


def _normalise(data: dict) -> dict:
    """Type coercions, alias normalisation, and cross-field consistency."""
    for float_field in ("monthly_net_income", "existing_emi", "requested_amount",
                        "property_value", "down_payment", "on_road_price"):
        v = data.get(float_field)
        if v is not None:
            try:
                data[float_field] = float(v)
            except (TypeError, ValueError):
                data[float_field] = None

    for int_field in ("age", "credit_score", "employment_duration_months",
                      "requested_tenure_months", "vehicle_age_years"):
        v = data.get(int_field)
        if v is not None:
            try:
                data[int_field] = int(v)
            except (TypeError, ValueError):
                data[int_field] = None

    # Normalise loan_type via alias map
    lt = data.get("loan_type")
    if isinstance(lt, str):
        lt_norm = lt.lower().replace(" ", "_").replace("-", "_")
        canonical = LOAN_TYPE_ALIAS_MAP.get(lt_norm)
        if canonical:
            data["loan_type"] = canonical
        elif lt_norm not in ("personal_loan", "home_loan", "auto_loan"):
            data["loan_type"] = None  # unknown -- conversation agent will ask

    # Normalise property_type and vehicle fields
    pt = data.get("property_type")
    if isinstance(pt, str):
        data["property_type"] = pt.lower().replace(" ", "_")
    vt = data.get("vehicle_type")
    if isinstance(vt, str):
        data["vehicle_type"] = vt.lower().strip()
    vc = data.get("vehicle_category")
    if isinstance(vc, str):
        data["vehicle_category"] = vc.lower().replace(" ", "_").replace("-", "_")

    # Cross-field consistency
    loan_type = data.get("loan_type")
    if loan_type == "personal_loan":
        for f in ("property_value", "down_payment", "property_type", "property_location",
                  "existing_property_loan", "vehicle_type", "vehicle_age_years",
                  "on_road_price", "vehicle_category"):
            data[f] = None
        if data.get("requested_tenure_months") and data["requested_tenure_months"] > 60:
            data["requested_tenure_months"] = None
        if data.get("requested_amount") and data["requested_amount"] > 3_000_000:
            data["requested_amount"] = None
    elif loan_type == "home_loan":
        for f in ("vehicle_type", "vehicle_age_years", "on_road_price", "vehicle_category"):
            data[f] = None
    elif loan_type == "auto_loan":
        for f in ("property_value", "down_payment", "property_type",
                  "property_location", "existing_property_loan"):
            data[f] = None
        if data.get("requested_tenure_months") and data["requested_tenure_months"] > 84:
            data["requested_tenure_months"] = None

    return data


def _detect_vehicle_category_from_text(text: str) -> Optional[str]:
    """Detect vehicle category from free text using keyword hints."""
    text_lower = text.lower()
    for category, keywords in VEHICLE_CATEGORY_HINTS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return None


def _detect_loan_type_from_text(text: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    padded = f"_{normalized}_"
    for alias, product in sorted(LOAN_TYPE_ALIAS_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if f"_{alias}_" in padded:
            return product
    return None


def _merge_profiles(prior: Optional[ApplicantProfile], new_data: dict) -> dict:
    """Merge new data on top of prior profile -- prevents re-asking for known fields."""
    if prior is None:
        return new_data
    prior_dict = prior.to_dict()  # only non-None fields
    merged = dict(prior_dict)
    for k, v in new_data.items():
        if v is not None:
            merged[k] = v  # new data overrides prior
    return merged


def is_document_request(user_message: str) -> bool:
    """Return True if the user is asking about required documents."""
    text_lower = user_message.lower()
    return any(kw in text_lower for kw in DOC_REQUEST_KEYWORDS)


def extract_profile(
    history:       list[dict],
    llm_client,
    model:         str,
    prior_profile: Optional[ApplicantProfile] = None,
) -> Tuple[ApplicantProfile, list[str]]:
    """Extract ApplicantProfile from conversation history using the LLM.

    Uses prior_profile as a seed so the assistant never re-asks for
    fields that were already provided in earlier turns.

    Returns: (profile, missing_fields)
    """
    messages = _build_extraction_messages(history)

    try:
        completion = llm_client.with_options(timeout=45.0, max_retries=0).chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=250,
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("profile_extraction_failed error=%s", type(exc).__name__)
        raw = "{}"

    data = _parse_profile_json(raw)
    data = _normalise(data)

    # Collect all user text across the entire conversation history
    all_user_text = " ".join(
        msg.get("content", "") for msg in history if msg.get("role") == "user"
    )

    if data.get("loan_type") is None:
        detected_loan_type = _detect_loan_type_from_text(all_user_text)
        if detected_loan_type:
            data["loan_type"] = detected_loan_type
            data = _normalise(data)

    if data.get("loan_type") == "auto_loan":
        # Vehicle category
        if data.get("vehicle_category") is None:
            detected_cat = _detect_vehicle_category_from_text(all_user_text)
            data["vehicle_category"] = detected_cat or "four_wheeler"

        # Vehicle condition (new vs used)
        if data.get("vehicle_type") is None:
            if re.search(r"\b(used|second\s*hand|pre-owned|preowned)\b", all_user_text, re.IGNORECASE):
                data["vehicle_type"] = "used"
            elif re.search(r"\b(new|brand\s*new)\b", all_user_text, re.IGNORECASE):
                data["vehicle_type"] = "new"
            else:
                data["vehicle_type"] = "new"

        # Mutual synchronization of requested_amount and on_road_price
        if data.get("on_road_price") and not data.get("requested_amount"):
            data["requested_amount"] = data["on_road_price"]
        elif data.get("requested_amount") and not data.get("on_road_price"):
            data["on_road_price"] = data["requested_amount"]

    # Detect tenure explicitly specified in text if not extracted
    if data.get("requested_tenure_months") is None:
        tenure_year_match = re.search(r"\b(?:loan\s*)?(?:tenure|term)\s*(?:is|of|=|:)?\s*(\d+)\s*(?:years?|yrs?)\b", all_user_text, re.IGNORECASE)
        if tenure_year_match:
            try:
                data["requested_tenure_months"] = int(tenure_year_match.group(1)) * 12
            except ValueError:
                pass
        else:
            tenure_month_match = re.search(r"\b(?:loan\s*)?(?:tenure|term)\s*(?:is|of|=|:)?\s*(\d+)\s*(?:months?|m)\b", all_user_text, re.IGNORECASE)
            if tenure_month_match:
                try:
                    data["requested_tenure_months"] = int(tenure_month_match.group(1))
                except ValueError:
                    pass

    # Merge with prior profile
    merged_data = _merge_profiles(prior_profile, data)

    # Re-apply auto-loan mutual synchronization on merged data
    if merged_data.get("loan_type") == "auto_loan":
        if merged_data.get("on_road_price") and not merged_data.get("requested_amount"):
            merged_data["requested_amount"] = merged_data["on_road_price"]
        elif merged_data.get("requested_amount") and not merged_data.get("on_road_price"):
            merged_data["on_road_price"] = merged_data["requested_amount"]
        if not merged_data.get("vehicle_category"):
            merged_data["vehicle_category"] = "four_wheeler"
        if not merged_data.get("vehicle_type"):
            merged_data["vehicle_type"] = "new"

    profile = ApplicantProfile(
        loan_type                  = merged_data.get("loan_type"),
        age                        = merged_data.get("age"),
        employment_type            = merged_data.get("employment_type"),
        monthly_net_income         = merged_data.get("monthly_net_income"),
        employment_duration_months = merged_data.get("employment_duration_months"),
        credit_score               = merged_data.get("credit_score"),
        existing_emi               = merged_data.get("existing_emi"),
        requested_amount           = merged_data.get("requested_amount"),
        requested_tenure_months    = merged_data.get("requested_tenure_months"),
        property_value             = merged_data.get("property_value"),
        down_payment               = merged_data.get("down_payment"),
        property_type              = merged_data.get("property_type"),
        property_location          = merged_data.get("property_location"),
        existing_property_loan     = merged_data.get("existing_property_loan"),
        vehicle_type               = merged_data.get("vehicle_type"),
        vehicle_age_years          = merged_data.get("vehicle_age_years"),
        on_road_price              = merged_data.get("on_road_price"),
        vehicle_category           = merged_data.get("vehicle_category"),
    )

    missing = profile.missing_fields()
    logger.info("profile_extracted loan_type=%s missing=%d", profile.loan_type, len(missing))
    return profile, missing


# ---------------------------------------------------------------------------
# Field labels and document checklists
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, str] = {
    "loan_type":                  "loan type (Personal Loan, Home Loan, or Auto Loan)",
    "age":                        "your age",
    "employment_type":            "employment type (salaried, self-employed, or business owner)",
    "monthly_net_income":         "monthly net income (take-home pay after deductions)",
    "employment_duration_months": "how long you have been employed or in business",
    "credit_score":               "your credit / CIBIL score",
    "existing_emi":               "total existing monthly EMI obligations (enter 0 if none)",
    "requested_amount":           "loan amount you are requesting",
    "requested_tenure_months":    "preferred loan tenure in months",
    "property_value":             "property market value",
    "down_payment":               "down payment amount",
    "property_type":              "property type (apartment, villa, or residential house)",
    "vehicle_type":               "vehicle condition (new or used)",
    "on_road_price":              "total on-road price (ex-showroom + taxes + insurance)",
    "vehicle_category":           "vehicle category (two-wheeler, four-wheeler, or commercial)",
}

DOC_CHECKLISTS = {
    "personal_loan": [
        "Identity proof: Aadhaar card + PAN card (mandatory)",
        "Address proof: utility bill, bank statement, or rental agreement",
        "Last 3 months pay slips (salaried) / last 2 years ITR (self-employed)",
        "Last 6 months bank statements",
        "Form 16 for the last 2 years (salaried applicants)",
        "Business registration + GST certificate (self-employed/business)",
    ],
    "home_loan": [
        "Identity proof: Aadhaar card + PAN card (mandatory)",
        "Address proof: utility bill, bank statement, or rental agreement",
        "Last 3 months pay slips / last 2 years audited financials",
        "Last 12 months bank statements (salaried) / 24 months (self-employed)",
        "Form 16 for the last 2 years (salaried applicants)",
        "Sale agreement or allotment letter",
        "Title deed and chain of title documents",
        "Approved building plan and NOC from builder or housing society",
        "Property tax receipts (latest)",
    ],
    "auto_loan": [
        "Identity proof: Aadhaar card + PAN card (mandatory)",
        "Address proof: utility bill, bank statement, or rental agreement",
        "Last 3 months pay slips (salaried) / last 2 years ITR (self-employed)",
        "Last 6 months bank statements",
        "Proforma invoice or dealer quotation (new vehicle)",
        "RC book, insurance certificate, and valuation report (used vehicle)",
        "Form 16 for the last 2 years (salaried applicants)",
    ],
}


def describe_missing(missing_fields: list[str]) -> str:
    lines = [f"- {FIELD_LABELS.get(f, f)}" for f in missing_fields]
    return "\n".join(lines)


def get_doc_checklist(loan_type: Optional[str]) -> Optional[str]:
    """Return formatted document checklist, or None if loan_type unknown."""
    if not loan_type or loan_type not in DOC_CHECKLISTS:
        return None
    items = DOC_CHECKLISTS[loan_type]
    label = loan_type.replace("_", " ").title()
    lines = [f"**Documents required for {label}:**"] + [f"- {item}" for item in items]
    return "\n".join(lines)
