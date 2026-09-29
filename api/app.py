"""LoanAssist API — v2

Endpoints:
    GET  /health                → liveness + RAG collection stats
    POST /chat                  → streaming SSE eligibility chat (primary)
    POST /ask                   → non-streaming (backward-compat)
    GET  /audit/{application_id}→ retrieve full audit record
    POST /scenario              → what-if: override profile fields, re-run engine
    GET  /metrics               → Prometheus scrape target
    GET  /versions              → active policy versions

Streaming architecture (/chat):
    Request
     → Layer 1: input guard (injection / PII / topic)
     → Profile Extractor (LLM, temp=0)  →  ApplicantProfile
     → if incomplete: Conversation Agent (LLM, streaming)
     → if complete:
         Policy RAG (ChromaDB, metadata-filtered by product + active version)
         Deterministic Rules Engine  →  EligibilityResult
         Calculators                 →  LoanCalculations
         Explanation Agent (LLM, streaming, sentence-buffered output guard)
         Audit Log (JSONL + in-memory store)
     → SSE events: token | profile_update | decision | citations | audit_ref | error

Safety:
    Layer 1: input guard  (injection blocklist + PII detection + topic gate)
    Layer 2: grounded LLM (policy context injected; temperature=0 for decisions)
    Layer 3: output guard (lending-decision blocklist + PII masking)
             Applied per sentence during streaming — no token can bypass the guard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

import chromadb
import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from calculators import run_all_calculators
from guardrails import REFUSAL, check_input, check_output
from profile_extractor import describe_missing, extract_profile
from prompts.loader import load_prompt
from redact import redact
from rules_engine import (
    AUTO_LOAN_CONFIG as _AL_CFG,
    HOME_LOAN_CONFIG as _HL_CFG,
    PERSONAL_LOAN_CONFIG as _PL_CFG,
    ApplicantProfile,
    run_rules_engine,
)
from profile_extractor import is_document_request, get_doc_checklist

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loanassist.api")

# ── configuration ─────────────────────────────────────────────────────────────
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",   "http://localhost:4000/v1")
LLM_API_KEY    = os.getenv("LLM_API_KEY",    "local")
LLM_MODEL      = os.getenv("LLM_MODEL",      "openrouter-mini")
API_KEY        = os.getenv("API_KEY",        "local-dev-key")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
CHROMA_DIR     = os.getenv("CHROMA_DIR",     "rag/chroma")
AUDIT_PATH     = os.getenv("AUDIT_PATH",     "/app/logs/audit.jsonl")
TOP_K          = 5

REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
MAX_RETRIES     = 2
BACKOFF_BASE_S  = 0.5

# ── prompt registry ───────────────────────────────────────────────────────────
ACTIVE_PROMPT         = load_prompt("answer_grounded",    PROMPT_VERSION)
CONV_PROMPT           = load_prompt("conversation_agent", PROMPT_VERSION)
EXPL_PROMPT           = load_prompt("explanation_agent",  PROMPT_VERSION)

# ── Langfuse tracing (graceful no-op if keys absent) ─────────────────────────
LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)
if LANGFUSE_ENABLED:
    try:
        from langfuse.decorators import langfuse_context, observe
        from langfuse.openai import OpenAI
        logger.info("Langfuse ENABLED host=%s", os.getenv("LANGFUSE_HOST"))
    except Exception as exc:
        logger.warning("Langfuse unavailable (%s) — tracing disabled", exc)
        LANGFUSE_ENABLED = False

if not LANGFUSE_ENABLED:
    from openai import OpenAI
    langfuse_context = None

    def observe(*_a, **_kw):
        def _d(fn): return fn
        return _d

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    timeout=60.0,
    max_retries=2,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="LoanAssist API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)


def _get_collection(name: str):
    try:
        return _chroma.get_collection(name)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RAG collection '{name}' unavailable: {type(exc).__name__}: {exc}",
        )


def get_active_version(product: str) -> str:
    """Return the active policy version for a product from environment."""
    env_key = f"ACTIVE_{product.upper()}_VERSION"
    return os.getenv(env_key, "v2")   # default to v2 (latest)


def retrieve_for_product(query: str, product: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-K clauses from the product-specific collection.

    Filters by active policy version via ChromaDB metadata `where`.
    Falls back to the generic 'eligibility' collection on error.
    """
    version = get_active_version(product)
    try:
        col = _get_collection(product)
        n = min(top_k, col.count())
        if n == 0:
            raise ValueError("empty collection")
        res = col.query(
            query_texts=[query],
            n_results=n,
            where={"version": version},
        )
    except Exception:
        # Fallback: generic collection, no version filter
        try:
            col = _get_collection("eligibility")
            n = min(top_k, col.count())
            res = col.query(query_texts=[query], n_results=n)
        except Exception as exc2:
            logger.error("RAG retrieval failed: %s", exc2)
            return []

    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "doc":            res["metadatas"][0][i].get("doc", ""),
            "section":        res["metadatas"][0][i].get("section", ""),
            "rule_id":        res["metadatas"][0][i].get("rule_id", "-"),
            "version":        res["metadatas"][0][i].get("version", version),
            "policy_id":      res["metadatas"][0][i].get("policy_id", ""),
            "product":        res["metadatas"][0][i].get("product", product),
            "text":           res["documents"][0][i],
            "distance":       res["distances"][0][i],
        })
    return out


# ── session + audit stores ────────────────────────────────────────────────────
_conv_store:    Dict[str, list]         = {}  # session_id → [{role, content}]
_profile_store: Dict[str, dict]         = {}  # session_id → profile dict
_audit_store:   Dict[str, dict]         = {}  # application_id → audit record

IDEMPOTENCY_TTL_S = 600
_idem_cache: Dict[str, tuple[float, dict]] = {}


def _session_history(sid: str) -> list:
    return _conv_store.get(sid, [])


def _save_msg(sid: str, role: str, content: str):
    _conv_store.setdefault(sid, []).append({"role": role, "content": content})


def _save_profile(sid: str, profile: ApplicantProfile):
    _profile_store[sid] = profile.to_dict()


def _get_profile(sid: str) -> Optional[ApplicantProfile]:
    d = _profile_store.get(sid)
    return ApplicantProfile(**d) if d else None


def _idem_get(key: str) -> Optional[dict]:
    entry = _idem_cache.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.time() - ts > IDEMPOTENCY_TTL_S:
        _idem_cache.pop(key, None)
        return None
    return val


def _idem_put(key: str, val: dict):
    now = time.time()
    stale = [k for k, (t, _) in _idem_cache.items() if now - t > IDEMPOTENCY_TTL_S]
    for k in stale:
        _idem_cache.pop(k, None)
    _idem_cache[key] = (now, val)


# ── audit ─────────────────────────────────────────────────────────────────────
def audit_log(entry: dict):
    """Append audit event to JSONL file + in-memory store."""
    app_id = entry.get("application_id")
    if app_id:
        _audit_store[app_id] = entry
    try:
        Path(AUDIT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.error("audit_log_write_failed: %s", exc)


# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "status"])
LATENCY  = Histogram(
    "llm_request_latency_seconds", "Request latency",
    ["endpoint"], buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)

@app.middleware("http")
async def _metrics_mw(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt = time.perf_counter() - t0
    path = request.url.path
    if path in ("/chat", "/ask", "/health", "/scenario"):
        REQUESTS.labels(endpoint=path, status=str(resp.status_code)).inc()
    if path in ("/chat", "/ask"):
        LATENCY.labels(endpoint=path).observe(dt)
    return resp


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: f"anon-{uuid.uuid4().hex[:8]}")
    message:    str
    history:    Optional[List[ChatMessage]] = None


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    doc:      str
    text:     str
    rule_id:  Optional[str] = None
    version:  Optional[str] = None

class AskResponse(BaseModel):
    """Backward-compatible response for the /ask endpoint."""
    answer:        str
    decision:      Optional[Literal[
        "POTENTIALLY_ELIGIBLE", "NOT_ELIGIBLE",
        "MANUAL_REVIEW",        "INSUFFICIENT_INFORMATION",
        "PRE_QUALIFIED",        "NOT_PRE_QUALIFIED",
        "NEEDS_INFORMATION",
    ]] = None
    citations:     List[Citation] = Field(default_factory=list)
    confidence:    Literal["high", "medium", "low"] = "medium"
    refused:       bool = False
    reason:        Optional[str] = None
    prompt_version: str = PROMPT_VERSION


class ScenarioRequest(BaseModel):
    session_id: str
    overrides:  Dict[str, Any]


# ── streaming helpers ─────────────────────────────────────────────────────────

def _rag_context_str(sources: list[dict]) -> str:
    """Format retrieved policy clauses for the explanation prompt."""
    return "\n\n".join(
        f"[{s.get('rule_id', '-')}] [{s['doc']} v{s.get('version', '?')}]\n{s['text']}"
        for s in sources
    )


def _decision_summary(result) -> str:
    d = result.to_dict()
    return (
        f"Decision: {d['decision']}\n"
        f"Product:  {d['product']}\n"
        f"Failed rules: {[r['rule_id'] for r in d['failed_rules']]}\n"
        f"Manual review: {[r['rule_id'] for r in d['manual_review_rules']]}\n"
        f"Missing fields: {d['missing_fields']}"
    )


def _rule_checks_str(result) -> str:
    lines = []
    for r in result.rule_checks:
        icon = "✓" if r.result.value == "PASS" else ("✗" if r.result.value == "FAIL" else "⚠")
        lines.append(f"  {icon} [{r.rule_id}] {r.rule_name}: {r.detail}")
    return "\n".join(lines)


def _calcs_str(calcs) -> str:
    parts = []
    if calcs.emi          is not None: parts.append(f"EMI: ₹{calcs.emi:,.2f}/month")
    if calcs.foir         is not None: parts.append(f"FOIR: {calcs.foir * 100:.1f}%")
    if calcs.ltv          is not None: parts.append(f"LTV: {calcs.ltv * 100:.1f}%")
    if calcs.max_affordable_loan is not None:
        parts.append(f"Max affordable loan: ₹{calcs.max_affordable_loan:,.0f}")
    if calcs.total_interest is not None:
        parts.append(f"Total interest: ₹{calcs.total_interest:,.0f}")
    return "; ".join(parts) if parts else "N/A"


SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\n)\n")


async def _stream_llm(
    messages: list[dict],
    queue:    asyncio.Queue,
    max_tokens: int = 600,
) -> str:
    """Stream LLM tokens, buffer into sentences, apply output guard per sentence.

    Puts {"type": "token", "content": "..."} events into the queue.
    Returns the full (guard-approved) text.
    """
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    buf = ""
    full_text = ""

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
            async with http.stream(
                "POST",
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data: "):
                        continue
                    payload_str = raw[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        token = chunk["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if not token:
                        continue
                    buf += token
                    # Flush on sentence boundary
                    if SENTENCE_END.search(buf) or len(buf) > 200:
                        safe = check_output(buf)
                        if safe["text"]:
                            await queue.put(json.dumps({
                                "type": "token",
                                "content": safe["text"],
                            }))
                            full_text += safe["text"]
                        buf = ""
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"LLM HTTP {exc.response.status_code}: {exc}")

    # Flush remaining buffer
    if buf.strip():
        safe = check_output(buf)
        if safe["text"]:
            await queue.put(json.dumps({"type": "token", "content": safe["text"]}))
            full_text += safe["text"]

    return full_text


async def _non_stream_llm(messages: list[dict], max_tokens: int = 400) -> str:
    """Non-streaming LLM call with exponential backoff retry."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as http:
                r = await http.post(
                    f"{LLM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_BASE_S * (2 ** attempt))

    raise HTTPException(
        status_code=503,
        detail=f"LLM upstream unavailable after {MAX_RETRIES + 1} attempts: "
               f"{type(last_exc).__name__}",
    )


# ── /chat producer pipeline ───────────────────────────────────────────────────

async def _chat_producer(
    session_id:     str,
    application_id: str,
    message:        str,
    queue:          asyncio.Queue,
    client_history: Optional[List[ChatMessage]] = None,
):
    """Full eligibility pipeline — puts SSE events into the queue."""
    t0 = time.perf_counter()

    try:
        # ── Layer 1: input guard ───────────────────────────────────────────────
        verdict = check_input(message)
        if not verdict["allowed"]:
            audit_log({
                "application_id": application_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": session_id,
                "event": "input_refused",
                "reason": verdict["reason"],
                "prompt_version": PROMPT_VERSION,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })
            await queue.put(json.dumps({"type": "error", "content": REFUSAL}))
            return

        # ── Synchronize client history if provided ────────────────────────────
        if client_history is not None:
            _conv_store[session_id] = [
                {"role": m.role, "content": m.content}
                for m in client_history
                if m.content and m.role in ("user", "assistant")
            ]

        # ── Save user turn and extract profile ────────────────────────────────
        _save_msg(session_id, "user", message)
        history = _session_history(session_id)

        # Fetch prior profile for merge (prevents re-asking known fields)
        prior_profile = _get_profile(session_id)

        profile, missing = await asyncio.to_thread(
            extract_profile, history, client, LLM_MODEL, prior_profile
        )
        _save_profile(session_id, profile)

        await queue.put(json.dumps({
            "type":           "profile_update",
            "profile":        profile.to_dict(),
            "missing_fields": missing,
        }))

        # ── Check if user is asking about documents ───────────────────────────
        if is_document_request(message) and profile.loan_type:
            doc_text = get_doc_checklist(profile.loan_type)
            if doc_text:
                doc_system = (
                    "You are LoanAssist. The user is asking about required documents. "
                    "Include the following document checklist in your response, formatted clearly.\n\n"
                    + doc_text
                )
                doc_messages = [
                    {"role": "system", "content": doc_system},
                    *history,
                ]
                full_answer = await _stream_llm(doc_messages, queue, max_tokens=300)
                _save_msg(session_id, "assistant", full_answer)
                return

        # ── User asking directly for eligibility assessment ───────────────────
        user_wants_decision = any(kw in message.lower() for kw in [
            "eligible", "eligibility", "can i get", "qualify", "status",
            "assessment", "approve", "completed assessment"
        ])

        # ── Branch: incomplete profile → ask for missing fields ───────────────
        if missing:
            product = profile.loan_type.replace("_", " ") if profile.loan_type else ""
            product_label = product.title()
            intro = f"To help with your {product_label}, please provide:" if product_label else "To get started, please provide:"
            full_answer = f"{intro}\n{describe_missing(missing)}"
            safe_answer = check_output(full_answer)
            full_answer = safe_answer["text"]
            if full_answer:
                await queue.put(json.dumps({"type": "token", "content": full_answer}))
            _save_msg(session_id, "assistant", full_answer)
            return  # done; no eligibility decision yet

        # ── Complete profile → full assessment ────────────────────────────────
        product = profile.loan_type
        if product == "personal_loan":
            cfg = _PL_CFG
        elif product == "home_loan":
            cfg = _HL_CFG
        else:
            cfg = _AL_CFG  # auto_loan

        # 1. Policy RAG
        rag_query = message + " " + " ".join(f"{k}={v}" for k, v in profile.to_dict().items())
        rag_sources = retrieve_for_product(rag_query, product)

        # 2. Calculators
        calcs = run_all_calculators(
            requested_amount = profile.requested_amount,
            monthly_income   = profile.monthly_net_income,
            existing_emi     = profile.existing_emi,
            tenure_months    = profile.requested_tenure_months,
            annual_rate_pct  = cfg["ANNUAL_RATE_PCT"],
            max_foir         = cfg["MAX_FOIR"],
            property_value   = profile.property_value if product == "home_loan" else None,
            on_road_price    = profile.on_road_price if product == "auto_loan" else None,
        )

        # 3. Deterministic rules engine
        result = run_rules_engine(profile, calcs)

        # 4. Explanation agent (streaming)
        expl_system = EXPL_PROMPT["text"].format(
            decision_summary = _decision_summary(result),
            rule_checks      = _rule_checks_str(result),
            calculations     = _calcs_str(calcs),
            rag_context      = _rag_context_str(rag_sources),
        )
        expl_messages = [
            {"role": "system", "content": expl_system},
            {"role": "user",   "content": "Please explain my eligibility assessment."},
        ]
        full_answer = await _stream_llm(expl_messages, queue, max_tokens=600)
        _save_msg(session_id, "assistant", full_answer)

        # 5. Decision event with rule version details
        active_ver = get_active_version(product)
        rule_checks_with_version = [
            {
                "rule_id":      r.rule_id,
                "rule_name":    r.rule_name,
                "result":       r.result.value,
                "detail":       r.detail,
                "rule_version": active_ver,
            }
            for r in result.rule_checks
        ]
        decision_dict = result.to_dict()
        decision_dict["policy_version"] = active_ver
        decision_dict["rule_checks"] = rule_checks_with_version

        await queue.put(json.dumps({
            "type": "decision",
            "data": decision_dict,
        }))

        # 6. Calculations event
        await queue.put(json.dumps({
            "type":         "calculations",
            "data":         calcs.to_dict(),
            "annual_rate":  cfg["ANNUAL_RATE_PCT"],
        }))

        # 7. Citations event
        await queue.put(json.dumps({
            "type": "citations",
            "sources": [
                {
                    "doc":       s["doc"],
                    "rule_id":   s.get("rule_id"),
                    "version":   s.get("version", active_ver),
                    "policy_id": s.get("policy_id"),
                    "text":      s["text"][:300],
                }
                for s in rag_sources
            ],
        }))

        # 8. Audit logging with explicit rule versions
        latency_ms = round((time.perf_counter() - t0) * 1000)
        audit_entry = {
            "application_id": application_id,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id":     session_id,
            "product":        product,
            "input_snapshot": profile.to_dict(),
            "policy_version": active_ver,
            "policy_id":      rag_sources[0].get("policy_id", "") if rag_sources else "",
            "rules_applied":  rule_checks_with_version,
            "calculations":   calcs.to_dict(),
            "decision":       result.decision.value,
            "rag_sources":    [
                {
                    "doc":       s["doc"],
                    "rule_id":   s.get("rule_id"),
                    "version":   s.get("version", active_ver),
                    "policy_id": s.get("policy_id", ""),
                }
                for s in rag_sources
            ],
            "prompt_version": PROMPT_VERSION,
            "latency_ms":     latency_ms,
        }
        audit_log(audit_entry)

        await queue.put(json.dumps({
            "type":           "audit_ref",
            "application_id": application_id,
        }))

    except Exception as exc:
        logger.exception("chat_producer_error session=%s", session_id)
        await queue.put(json.dumps({"type": "error", "content": f"Processing error: {str(exc)}"}))
    finally:
        await queue.put(None)   # sentinel — always signal completion


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    counts: dict = {}
    for name in ["personal_loan", "home_loan", "auto_loan", "eligibility"]:
        try:
            counts[name] = _chroma.get_collection(name).count()
        except Exception:
            counts[name] = 0
    return {
        "status":        "ok",
        "version":       "3.0.0",
        "llm_base_url":  LLM_BASE_URL,
        "model":         LLM_MODEL,
        "prompt_version": PROMPT_VERSION,
        "langfuse_enabled": LANGFUSE_ENABLED,
        "rag_collections": counts,
        "active_policy_versions": {
            "personal_loan": get_active_version("personal_loan"),
            "home_loan":     get_active_version("home_loan"),
            "auto_loan":     get_active_version("auto_loan"),
        },
    }


@app.get("/versions")
async def versions():
    """Return active policy versions and their metadata."""
    return {
        "personal_loan": {
            "active": get_active_version("personal_loan"),
            "env_var": "ACTIVE_PERSONAL_LOAN_VERSION",
        },
        "home_loan": {
            "active": get_active_version("home_loan"),
            "env_var": "ACTIVE_HOME_LOAN_VERSION",
        },
        "auto_loan": {
            "active": get_active_version("auto_loan"),
            "env_var": "ACTIVE_AUTO_LOAN_VERSION",
        },
        "prompt": {
            "active": PROMPT_VERSION,
            "env_var": "PROMPT_VERSION",
        },
    }


@app.post("/chat")
async def chat_endpoint(
    req:              ChatRequest,
    x_api_key:        Optional[str] = Header(default=None, alias="X-API-Key"),
    idempotency_key:  Optional[str] = Header(default=None, alias="Idempotency-Key"),
    x_session_id:     Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    """Primary streaming endpoint. Returns Server-Sent Events.

    SSE event types:
        token           — streamed explanation text
        profile_update  — current extracted profile + missing fields
        decision        — structured EligibilityResult
        calculations    — EMI, FOIR, LTV, max_affordable_loan
        citations       — retrieved policy clauses with Rule IDs
        audit_ref       — application_id for GET /audit/{id}
        error           — error message
    """
    session_id     = x_session_id or req.session_id
    application_id = f"APP-{uuid.uuid4().hex[:12].upper()}"

    queue: asyncio.Queue = asyncio.Queue(maxsize=30)

    async def _event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(
            _chat_producer(session_id, application_id, req.message, queue, req.history)
        )
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield f"data: {item}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            yield f'data: {{"type": "error", "content": "Stream error: {str(exc)}"}}\n\n'
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "X-Application-Id": application_id,
        },
    )


@app.get("/audit/{application_id}")
async def get_audit(application_id: str):
    """Retrieve the full audit record for a given application ID.

    Powers the UI's 'Why did you give me this result?' feature.
    """
    record = _audit_store.get(application_id)
    if not record:
        # Try reading from JSONL file
        try:
            with open(AUDIT_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("application_id") == application_id:
                            return entry
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        raise HTTPException(
            status_code=404,
            detail=f"Audit record '{application_id}' not found.",
        )
    return record


@app.post("/scenario")
async def scenario(req: ScenarioRequest):
    """What-if simulator: override profile fields and re-run the rules engine.

    Returns a new EligibilityResult without streaming (instant deterministic).
    Use this from the UI scenario sliders.
    """
    profile = _get_profile(req.session_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="No active session profile. Start a /chat session first.",
        )

    # Apply overrides
    for field, value in req.overrides.items():
        if hasattr(profile, field):
            setattr(profile, field, value)

    if profile.loan_type is None:
        raise HTTPException(status_code=400, detail="Loan type not determined.")

    product = profile.loan_type
    if product == "personal_loan":
        cfg = _PL_CFG
    elif product == "home_loan":
        cfg = _HL_CFG
    else:
        cfg = _AL_CFG  # auto_loan

    calcs = run_all_calculators(
        requested_amount = profile.requested_amount,
        monthly_income   = profile.monthly_net_income,
        existing_emi     = profile.existing_emi,
        tenure_months    = profile.requested_tenure_months,
        annual_rate_pct  = cfg["ANNUAL_RATE_PCT"],
        max_foir         = cfg["MAX_FOIR"],
        property_value   = profile.property_value if product == "home_loan" else None,
        on_road_price    = profile.on_road_price if product == "auto_loan" else None,
    )

    result = run_rules_engine(profile, calcs)

    return {
        "decision":      result.to_dict(),
        "calculations":  calcs.to_dict(),
        "profile_used":  profile.to_dict(),
        "policy_version": get_active_version(product),
    }


@app.post("/ask", response_model=AskResponse)
@observe(name="loanassist-ask")
async def ask(
    req:             AskRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    x_session_id:    Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    """Non-streaming endpoint (backward compatible).

    Runs the same pipeline as /chat but buffers the full response.
    Used by the legacy Streamlit UI's blocking HTTP call.
    """
    session_id = x_session_id or f"anon-{uuid.uuid4().hex[:8]}"
    application_id = f"APP-{uuid.uuid4().hex[:12].upper()}"
    t0 = time.perf_counter()

    if idempotency_key and (cached := _idem_get(idempotency_key)):
        return AskResponse(**cached)

    # Layer 1: input guard
    verdict = check_input(req.question)
    if not verdict["allowed"]:
        audit_log({
            "application_id": application_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "event": "input_refused",
            "reason": verdict["reason"],
            "prompt_version": PROMPT_VERSION,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        })
        return AskResponse(
            answer=REFUSAL, refused=True, reason=verdict["reason"], confidence="high"
        )

    _save_msg(session_id, "user", req.question)
    history = _session_history(session_id)

    # Profile extraction with merge (prevents re-asking known fields)
    prior_profile = _get_profile(session_id)
    profile, missing = extract_profile(history, client, LLM_MODEL, prior_profile)
    _save_profile(session_id, profile)

    if missing:
        # Conversation agent
        profile_summary = "\n".join(f"  {k}: {v}" for k, v in profile.to_dict().items()) or "  (nothing yet)"
        system_text = CONV_PROMPT["text"].format(
            profile_summary=profile_summary,
            missing_fields=describe_missing(missing),
        )
        messages  = [{"role": "system", "content": system_text}, *history]
        answer    = await _non_stream_llm(messages, max_tokens=150)
        safe      = check_output(answer)
        _save_msg(session_id, "assistant", safe["text"])
        latency_ms = round((time.perf_counter() - t0) * 1000)
        audit_log({
            "application_id": application_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "event": "conversation",
            "missing_fields": missing,
            "latency_ms": latency_ms,
            "prompt_version": PROMPT_VERSION,
        })
        return AskResponse(
            answer=safe["text"] or REFUSAL,
            refused=safe["refused"],
            reason=safe["reason"],
            confidence="medium",
            prompt_version=PROMPT_VERSION,
        )

    # Full assessment
    product = profile.loan_type
    if product == "personal_loan":
        cfg = _PL_CFG
    elif product == "home_loan":
        cfg = _HL_CFG
    else:
        cfg = _AL_CFG  # auto_loan

    rag_sources = retrieve_for_product(
        req.question + " " + " ".join(f"{k}={v}" for k, v in profile.to_dict().items()),
        product,
    )

    calcs  = run_all_calculators(
        profile.requested_amount, profile.monthly_net_income, profile.existing_emi,
        profile.requested_tenure_months, cfg["ANNUAL_RATE_PCT"], cfg["MAX_FOIR"],
        property_value=profile.property_value if product == "home_loan" else None,
        on_road_price=profile.on_road_price if product == "auto_loan" else None,
    )
    result = run_rules_engine(profile, calcs)

    expl_system = EXPL_PROMPT["text"].format(
        decision_summary=_decision_summary(result),
        rule_checks=_rule_checks_str(result),
        calculations=_calcs_str(calcs),
        rag_context=_rag_context_str(rag_sources),
    )
    expl_messages = [
        {"role": "system", "content": expl_system},
        {"role": "user",   "content": "Explain my eligibility assessment."},
    ]
    raw_answer = await _non_stream_llm(expl_messages, max_tokens=600)
    safe       = check_output(raw_answer)
    answer_text = safe["text"] or "I could not generate an explanation. Please contact the branch."

    _save_msg(session_id, "assistant", answer_text)

    latency_ms = round((time.perf_counter() - t0) * 1000)
    audit_entry = {
        "application_id": application_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "product": product,
        "input_snapshot": profile.to_dict(),
        "policy_version": get_active_version(product),
        "rules_applied": [
            {"rule_id": r.rule_id, "result": r.result.value, "detail": r.detail}
            for r in result.rule_checks
        ],
        "calculations": calcs.to_dict(),
        "decision": result.decision.value,
        "rag_sources": [
            {"doc": s["doc"], "rule_id": s.get("rule_id"), "version": s.get("version")}
            for s in rag_sources
        ],
        "prompt_version": PROMPT_VERSION,
        "latency_ms": latency_ms,
    }
    audit_log(audit_entry)

    resp_dict = dict(
        answer=answer_text,
        decision=result.decision.value,
        citations=[
            Citation(doc=s["doc"], text=s["text"][:200],
                     rule_id=s.get("rule_id"), version=s.get("version"))
            for s in rag_sources
        ],
        confidence="high" if not missing else "medium",
        refused=safe["refused"],
        reason=safe["reason"],
        prompt_version=PROMPT_VERSION,
    )
    if idempotency_key:
        _idem_put(idempotency_key, resp_dict)

    return AskResponse(**resp_dict)


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
