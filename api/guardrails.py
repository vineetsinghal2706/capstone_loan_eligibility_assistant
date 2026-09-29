"""guardrails.py — the ClaimAssist API boundary defence (Session 10).

Three-layer defence for an LLM application:
    layer 1  check_input(text)   — before the model ever sees the request
    layer 2  the model itself    — an aligned model with a grounded prompt
    layer 3  check_output(text)  — before the answer ever leaves the API

Layers 1 and 3 live HERE, in ordinary reviewable code at the API boundary.
They are deliberately simple (blocklists + keyword allowlist + regex PII):
the lab teaches the ARCHITECTURE — where the checks sit and what they return —
not state-of-the-art classifiers. Production counterparts: Guardrails AI,
AWS Bedrock Guardrails, Azure AI Content Safety (see README mapping table).

Middleware order in api/app.py:
    input guard  ->  traced LLM call  ->  output guard
Refusals return {answer: REFUSAL, refused: true, reason} — never a 500.
"""

import logging

from redact import find_pii, redact

logger = logging.getLogger("loanassist.guardrails")

# The polite, product-approved refusal. One template, used by both guards,
# so refusal UX is consistent and testable.
REFUSAL = "I can help with loan eligibility and lending questions only."

# ---- layer 1: input checks ---------------------------------------------------

# Prompt-injection blocklist: phrases that try to override the system prompt.
# A blocklist is a FIRST line of defence, not the only one — see the deck:
# boundary defences matter even with aligned models, and blocklists alone
# are bypassable. Grounding + output checks back this up.
INJECTION_BLOCKLIST = [
    "ignore",
    "ignore your instructions",
    "ignore previous",
    "ignore all previous",
    "disregard your instructions",
    "disregard previous",
    "system prompt",
    "you are now",
    "act as if",
    "developer mode",
    "reveal your instructions",
    "Ignore rules",
]

# Topic gate: a loan/eligibility-keyword ALLOWLIST. If a longer question
# contains none of these, it is off-topic for a loan eligibility assistant.
LOAN_KEYWORDS = [
    # Loan / product
    "loan",
    "personal loan",
    "home loan",
    "house loan",        # alias for home loan
    "housing loan",      # alias for home loan
    "property loan",     # alias for home loan
    "plot loan",
    "mortgage",
    "auto loan",
    "car loan",
    "vehicle loan",
    "scooter loan",
    "bike loan",
    "motorcycle loan",
    "two wheeler loan",
    "two-wheeler",
    "four wheeler loan",
    "four-wheeler",
    "ev loan",
    "electric vehicle",
    "used car",
    "commercial vehicle",
    "vehicle finance",
    "on-road price",
    "on road price",
    "ex-showroom",
    "education loan",
    "business loan",
    "consumer loan",
    "credit facility",
    "borrowing",
    "borrower",

    # Eligibility / application
    "eligibility",
    "eligible",
    "pre-qualify",
    "prequalified",
    "pre-qualified",
    "qualification",
    "application",
    "applicant",
    "underwriting",
    "assessment",
    "approval",
    "approved",
    "approve",
    "rejected",
    "reject",
    "sanction",
    "disbursement",

    # Income / employment
    "income",
    "salary",
    "gross income",
    "annual income",
    "monthly income",
    "employment",
    "employed",
    "employer",
    "job",
    "occupation",
    "self-employed",
    "employment duration",
    "work history",

    # Credit
    "credit",
    "credit score",
    "cibil",
    "credit history",
    "credit report",
    "creditworthiness",
    "default",
    "active default",
    "overdue",
    "delinquent",
    "delinquency",
    "missed payment",
    "late payment",

    # Repayment / debt
    "emi",
    "foir",
    "ltv",
    "loan-to-value",
    "fixed obligation",
    "repayment",
    "repay",
    "installment",
    "monthly payment",
    "debt",
    "debt-to-income",
    "dti",
    "debt obligation",
    "outstanding balance",

    # Loan terms
    "interest",
    "interest rate",
    "tenure",
    "loan amount",
    "principal",
    "term",

    # Security / parties
    "collateral",
    "guarantor",
    "co-applicant",
    "coapplicant",
    "secured loan",
    "unsecured loan",

    # Application / risk
    "fraud",
    "misrepresentation",
    "duplicate application",
    "verification",
    "documentation",
    "documents",
    "residency",
    "resident",
    "work authorization",
]

# Questions shorter than this many words pass the topic gate even without a
# keyword ("hello", "thanks", "who are you?") — refusing greetings is bad UX.
OFF_TOPIC_MIN_WORDS = 6

def classify_intent(text: str, client, model: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user message as exactly one of:\n"
                "GENERAL\n"
                "ELIGIBILITY\n\n"
                "GENERAL = conversation, greetings, questions "
                "about the assistant, how to apply, or general "
                "loan information.\n"
                "ELIGIBILITY = asking whether the user qualifies, "
                "or providing personal eligibility information such "
                "as age, income, employment, residency, or credit status.\n\n"
                "Return ONLY GENERAL or ELIGIBILITY."
            ),
        },
        {"role": "user", "content": text},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=3,
            temperature=0,
        )

        result = (
            completion.choices[0].message.content or ""
        ).strip().upper()

        if result in {"GENERAL", "ELIGIBILITY"}:
            return result

    except Exception as exc:
        logger.warning(
            "intent_classifier_failed error=%s",
            type(exc).__name__,
        )

    # Safe fallback
    return "GENERAL"

def check_input(text: str) -> dict:
    lowered = text.lower()

    # Injection remains deterministic
    for phrase in INJECTION_BLOCKLIST:
        if phrase in lowered:
            return {
                "allowed": False,
                "reason": f"prompt_injection: matched '{phrase}'",
                "type": None,
            }

    pii = find_pii(text)

    if pii:
        logger.info(
            "guardrail=input_pii_detected types=%s question=%r",
            pii,
            redact(text),
        )

    return {
        "allowed": True,
        "reason": None,
        "type": None,
    }


# ---- layer 3: output checks ----------------------------------------------------

# Lending-decision blocklist: the assistant can explain eligibility criteria
# but should not make guaranteed approval/denial or instruct users to
# manipulate their financial information.
LENDING_DECISION_BLOCKLIST = [
    "guaranteed approval",
    "guaranteed loan",
    "you will definitely qualify",
    "you are guaranteed to qualify",
    "hide your income",
    "hide your debt",
    "fake your income",
    "falsify your documents",
]


def check_output(text: str) -> dict:
    """Layer 3. Returns {"text": str, "refused": bool, "reason": str | None}.

    1. PII masking (redact.py, the Session 4 regexes): the model must never
       echo an email/phone/Aadhaar/PAN out of the boundary unredacted.
    2. Financial-advice blocklist: a match replaces the entire answer with
       the refusal template — a partially-scrubbed advice answer is still
       advice.
    """
    lowered = text.lower()
    for phrase in LENDING_DECISION_BLOCKLIST:
        if phrase in lowered:
            logger.warning("guardrail=output_lending_decision phrase=%r", phrase)
            return {"text": REFUSAL, "refused": True,
                    "reason": f"lending_decision: matched '{phrase}'"}

    pii = find_pii(text)
    if pii:
        logger.info("guardrail=output_pii_redacted types=%s", pii)
        return {"text": redact(text), "refused": False,
                "reason": f"pii_redacted: {','.join(pii)}"}

    return {"text": text, "refused": False, "reason": None}
