"""LoanAssist — Professional Chat UI (ChatGPT / Claude Architecture)

Features:
  * Persistent left sidebar (always available) with brand header, product chips,
    API status, policy versions, and persistent conversation history.
  * Multi-turn chat persistence (saved to disk via ui/conversation_manager.py)
    with the ability to switch between chats or start a new chat.
  * Full-context forwarding: sends entire chat history with follow-up prompts
    so the LLM has multi-turn memory.
  * Token streaming with live typing cursor and clean mid-stream error recovery.
  * Comprehensive audit logging with explicit policy and rule versions (v2, etc.).
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure both project root and ui/ directory are in sys.path
_UI_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _UI_DIR.parent
for _p in [str(_ROOT_DIR), str(_UI_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import ui.conversation_manager as cm
except ImportError:
    import conversation_manager as cm

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="LoanAssist — Eligibility Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8001")
API_KEY = os.getenv("API_KEY", "local-dev-key")

# ── Custom CSS for ChatGPT / Claude Aesthetic ──────────────────────────────────
st.markdown("""
<style>
/* ── Typography & Background ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f8fafc;
    color: #0f172a;
}

/* ── Hide default Streamlit clutter ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 860px !important;
}

/* ── Left Sidebar (Dark ChatGPT/Claude style) ── */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebarUserContent"] {
    padding: 1rem 0.85rem !important;
}

.sidebar-brand-box {
    padding: 0.4rem 0.2rem 0.8rem 0.2rem;
}
.sidebar-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
}
.sidebar-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(96, 165, 250, 0.3);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-top: 4px;
}
.sidebar-status-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    color: #94a3b8;
    margin-top: 6px;
}
.status-dot-green {
    width: 8px; height: 8px;
    border-radius: 50%;
    background-color: #22c55e;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.6);
}
.status-dot-red {
    width: 8px; height: 8px;
    border-radius: 50%;
    background-color: #ef4444;
}

.version-strip {
    background: #1e293b;
    border-radius: 8px;
    padding: 6px 10px;
    margin: 8px 0;
    font-size: 0.72rem;
    color: #cbd5e1;
    display: flex;
    justify-content: space-between;
    border: 1px solid #334155;
}
.version-tag {
    color: #38bdf8;
    font-weight: 600;
}

.product-pill-box {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 8px 0;
}
.product-pill {
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.73rem;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid #334155;
}

.history-section-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin: 1rem 0 0.5rem 0.2rem;
}

.conv-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 0.6rem;
    border-radius: 8px;
    margin-bottom: 3px;
    font-size: 0.83rem;
    color: #cbd5e1;
    background: transparent;
    transition: background 0.15s ease;
}
.conv-item.active {
    background: #1e293b;
    color: #ffffff;
    font-weight: 600;
    border-left: 3px solid #3b82f6;
}

/* ── Main Chat Area ── */
.chat-container {
    max-width: 820px;
    margin: 0 auto;
    padding-bottom: 2rem;
}

/* ── Hero / Welcome Screen (Claude/ChatGPT style) ── */
.hero-box {
    text-align: center;
    padding: 3.5rem 1.5rem 2rem 1.5rem;
    margin: 0 auto 1.5rem auto;
    max-width: 680px;
}
.hero-avatar {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #f8fafc;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    margin: 0 auto 1.2rem auto;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 2rem;
}

/* ── Starter prompt cards ── */
.prompt-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: left;
    transition: all 0.15s ease-in-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    cursor: pointer;
    margin-bottom: 0.5rem;
}
.prompt-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
    transform: translateY(-1px);
}
.prompt-title {
    font-weight: 600;
    font-size: 0.86rem;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 2px;
}
.prompt-desc {
    font-size: 0.77rem;
    color: #64748b;
}

/* ── Messages ── */
.msg-wrapper {
    margin-bottom: 1.25rem;
    display: flex;
    flex-direction: column;
}
.msg-user-row {
    display: flex;
    justify-content: flex-end;
    margin: 0.5rem 0;
}
.msg-user-bubble {
    background: #2563eb;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.15rem;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
    word-break: break-word;
}

.msg-assistant-row {
    display: flex;
    justify-content: flex-start;
    gap: 12px;
    margin: 0.6rem 0;
    align-items: flex-start;
}
.assistant-avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: #0f172a;
    color: #f8fafc;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 5px rgba(15, 23, 42, 0.15);
}
.msg-assistant-bubble {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.25rem;
    max-width: 84%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    word-break: break-word;
}

/* ── Typing Indicator ── */
.typing-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 0;
}
.typing-dots span {
    width: 6px;
    height: 6px;
    background-color: #94a3b8;
    border-radius: 50%;
    animation: typingBounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-5px); opacity: 1; }
}

/* ── Badges & Metric Chips ── */
.decision-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.4rem 0.85rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.83rem;
    margin: 0.6rem 0 0.4rem 0;
}
.badge-eligible {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.badge-notelig {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.badge-manual {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-info {
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
}
.version-pill-inline {
    background: rgba(0,0,0,0.06);
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 4px;
}

.metric-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.6rem 0;
}
.metric-chip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.79rem;
    color: #475569;
}
.metric-chip strong { color: #0f172a; font-weight: 600; }

/* ── Rule Breakdown Table ── */
.rule-table {
    width: 100%;
    margin-top: 0.4rem;
    font-size: 0.81rem;
    border-collapse: collapse;
}
.rule-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 0.35rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.rule-row:last-child { border-bottom: none; }
.rule-icon { font-size: 0.9rem; width: 18px; flex-shrink: 0; }
.rule-id-tag {
    font-family: monospace;
    font-size: 0.75rem;
    background: #f1f5f9;
    color: #4338ca;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    white-space: nowrap;
}
.rule-version-tag {
    font-family: monospace;
    font-size: 0.7rem;
    background: #e0f2fe;
    color: #0369a1;
    padding: 1px 4px;
    border-radius: 4px;
    margin-left: 2px;
}
.rule-desc { color: #334155; flex: 1; line-height: 1.4; }

/* ── Error Banner ── */
.error-banner {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #9f1239;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Management ──────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = f"conv-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

if "last_decision" not in st.session_state:
    st.session_state.last_decision = None

if "last_calcs" not in st.session_state:
    st.session_state.last_calcs = None

if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

if "last_app_id" not in st.session_state:
    st.session_state.last_app_id = None

if "api_status_cache" not in st.session_state:
    st.session_state.api_status_cache = None


# ── API Health & Versions ──────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_api_health() -> tuple[bool, dict]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, {}
    except Exception:
        return False, {}


api_healthy, health_data = get_api_health()
active_versions = health_data.get("active_policy_versions", {
    "personal_loan": "v2",
    "home_loan": "v1",
    "auto_loan": "v1",
})


# ── Helpers ───────────────────────────────────────────────────────────────────
DECISION_CONFIG = {
    "POTENTIALLY_ELIGIBLE":     ("✅", "badge-eligible", "Potentially Eligible"),
    "NOT_ELIGIBLE":             ("❌", "badge-notelig",  "Not Eligible"),
    "MANUAL_REVIEW":            ("⚠️", "badge-manual",  "Manual Review Required"),
    "INSUFFICIENT_INFORMATION": ("ℹ️", "badge-info",    "More Information Needed"),
}

PRODUCT_DISPLAY = {
    "personal_loan": "Personal Loan",
    "home_loan":     "Home Loan",
    "auto_loan":     "Auto Loan",
}


def _fmt_inr(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"₹{float(v):,.0f}"
    except (TypeError, ValueError):
        return str(v)


def _decision_badge_html(decision: str, policy_ver: str | None = None) -> str:
    icon, css_class, label = DECISION_CONFIG.get(
        decision, ("ℹ️", "badge-info", decision)
    )
    ver_html = f"<span class='version-pill-inline'>{policy_ver}</span>" if policy_ver else ""
    return f'<div class="decision-badge {css_class}">{icon} {label} {ver_html}</div>'


def _metric_strip_html(calcs: dict, annual_rate: float | None = None) -> str:
    chips = []
    if "emi" in calcs and calcs["emi"] is not None:
        chips.append(f"<span class='metric-chip'>EMI: <strong>{_fmt_inr(calcs['emi'])}/mo</strong></span>")
    if "foir" in calcs and calcs["foir"] is not None:
        chips.append(f"<span class='metric-chip'>FOIR: <strong>{calcs['foir']*100:.1f}%</strong></span>")
    if "ltv" in calcs and calcs["ltv"] is not None:
        chips.append(f"<span class='metric-chip'>LTV: <strong>{calcs['ltv']*100:.1f}%</strong></span>")
    if "max_affordable_loan" in calcs and calcs["max_affordable_loan"] is not None:
        chips.append(f"<span class='metric-chip'>Max Loan: <strong>{_fmt_inr(calcs['max_affordable_loan'])}</strong></span>")
    if "total_interest" in calcs and calcs["total_interest"] is not None:
        chips.append(f"<span class='metric-chip'>Interest: <strong>{_fmt_inr(calcs['total_interest'])}</strong></span>")
    if annual_rate is not None:
        chips.append(f"<span class='metric-chip'>Rate: <strong>{annual_rate:.1f}% p.a.</strong></span>")
    if not chips:
        return ""
    return "<div class='metric-strip'>" + "".join(chips) + "</div>"


def _rule_rows_html(rule_checks: list) -> str:
    rows = []
    for r in rule_checks:
        res = r.get("result", "")
        if res == "PASS":
            icon = "✅"
        elif res == "FAIL":
            icon = "❌"
        elif res == "MANUAL_REVIEW":
            icon = "⚠️"
        else:
            icon = "⬜"
        rid  = r.get("rule_id", "-")
        name = r.get("rule_name", "")
        det  = r.get("detail", "")
        rver = r.get("rule_version", "")
        ver_span = f"<span class='rule-version-tag'>{rver}</span>" if rver else ""
        rows.append(
            f"<div class='rule-row'>"
            f"<span class='rule-icon'>{icon}</span>"
            f"<span class='rule-id-tag'>{rid}</span>{ver_span}"
            f"<span class='rule-desc'><strong>{name}</strong> — {det}</span>"
            f"</div>"
        )
    return "<div class='rule-table'>" + "".join(rows) + "</div>"


# ── Full-Context Streaming Sender ─────────────────────────────────────────────
def _send_message(user_msg: str):
    """Stream from /chat SSE, accumulate events, preserve tokens on mid-stream error."""
    session_id = st.session_state.session_id

    # Construct complete multi-turn context
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("content")
    ]

    msg_placeholder = st.empty()

    # Show animated typing indicator before first token arrives
    msg_placeholder.markdown(
        """
        <div class="msg-assistant-row">
            <div class="assistant-avatar">🏦</div>
            <div class="msg-assistant-bubble">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    full_text   = ""
    decision    = None
    calcs       = None
    annual_rate = None
    citations   = []
    app_id      = None
    rule_checks = []
    error_msg   = None

    try:
        with requests.post(
            f"{API_URL}/chat",
            json={
                "session_id": session_id,
                "message": user_msg,
                "history": history_payload,
            },
            headers={"X-API-Key": API_KEY, "Accept": "text/event-stream"},
            stream=True,
            timeout=90,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                try:
                    event = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "token":
                    token = event.get("content", "")
                    full_text += token
                    # Live streaming bubble with cursor
                    msg_placeholder.markdown(
                        f"""
                        <div class="msg-assistant-row">
                            <div class="assistant-avatar">🏦</div>
                            <div class="msg-assistant-bubble">
                                {full_text.replace('\n', '<br>')}▌
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif etype == "decision":
                    decision = event.get("data", {})
                    rule_checks = decision.get("rule_checks", [])

                elif etype == "calculations":
                    calcs = event.get("data", {})
                    annual_rate = event.get("annual_rate")

                elif etype == "citations":
                    citations = event.get("sources", [])

                elif etype == "audit_ref":
                    app_id = event.get("application_id")

                elif etype == "error":
                    error_msg = event.get("content", "Server error encountered.")

    except requests.Timeout:
        error_msg = "⏱️ Request timed out while streaming. The local LLM may still be generating."
    except requests.ConnectionError:
        error_msg = f"❌ Lost connection to API at `{API_URL}`. Please check server status."
    except Exception as exc:
        error_msg = f"⚠️ Unexpected stream error: {exc}"

    # Update session state with decision metadata
    st.session_state.last_decision  = decision
    st.session_state.last_calcs     = calcs
    st.session_state.last_citations = citations
    st.session_state.last_app_id    = app_id

    # Render final completed assistant bubble
    final_parts = [
        "<div class='msg-assistant-row'>",
        "<div class='assistant-avatar'>🏦</div>",
        "<div class='msg-assistant-bubble'>",
        full_text.replace("\n", "<br>") if full_text else "<em>(No text generated)</em>",
    ]

    # Display error notice inline if mid-stream interruption happened
    if error_msg:
        final_parts.append(f"<div class='error-banner'>⚠️ {error_msg}</div>")

    # Inline decision badge with active version
    if decision and decision.get("decision"):
        dec = decision["decision"]
        pver = decision.get("policy_version")
        final_parts.append(_decision_badge_html(dec, pver))

    # Metric strip
    if calcs:
        final_parts.append(_metric_strip_html(calcs, annual_rate))

    final_parts.append("</div></div>")
    msg_placeholder.markdown("".join(final_parts), unsafe_allow_html=True)

    # Save to session messages
    meta = {}
    if decision:
        meta["decision"] = decision
    if calcs:
        meta["calcs"] = calcs
        if annual_rate is not None:
            meta["annual_rate"] = annual_rate
    if citations:
        meta["citations"] = citations
    if app_id:
        meta["app_id"] = app_id
    if rule_checks:
        meta["rule_checks"] = rule_checks
    if error_msg:
        meta["error"] = error_msg

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_text,
        "meta":    meta,
    })

    # Persist conversation to disk
    product = decision.get("product") if decision else None
    cm.save_conversation(
        session_id=session_id,
        messages=st.session_state.messages,
        last_decision=decision,
        last_calcs=calcs,
        last_citations=citations,
        last_app_id=app_id,
        product=product,
    )


# ── Message Renderer (Historical / Non-Streaming) ──────────────────────────────
def _render_message(msg: dict):
    role    = msg.get("role", "assistant")
    content = msg.get("content", "")
    meta    = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f"""
            <div class="msg-user-row">
                <div class="msg-user-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        decision    = meta.get("decision", {})
        calcs       = meta.get("calcs", {})
        annual_rate = meta.get("annual_rate")
        citations   = meta.get("citations", [])
        app_id      = meta.get("app_id")
        rule_checks = meta.get("rule_checks", [])
        error_msg   = meta.get("error")

        parts = [
            "<div class=\"msg-assistant-row\">",
            "<div class=\"assistant-avatar\">🏦</div>",
            "<div class=\"msg-assistant-bubble\">",
            content.replace("\n", "<br>"),
        ]

        if error_msg:
            parts.append(f"<div class='error-banner'>⚠️ {error_msg}</div>")

        if decision and decision.get("decision"):
            pver = decision.get("policy_version")
            parts.append(_decision_badge_html(decision["decision"], pver))

        if calcs:
            parts.append(_metric_strip_html(calcs, annual_rate))

        parts.append("</div></div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

        # Expandable rule-by-rule breakdown
        if rule_checks:
            with st.expander("📋 Evaluated Rules Breakdown (with Rule IDs & Versions)", expanded=False):
                st.markdown(_rule_rows_html(rule_checks), unsafe_allow_html=True)

        # Policy citations
        if citations:
            with st.expander(f"📄 Grounded Policy Citations ({len(citations)} clauses)", expanded=False):
                for c in citations:
                    rid = c.get("rule_id", "-")
                    doc = c.get("doc", "-")
                    ver = c.get("version", "")
                    st.markdown(f"**`{rid}`** · `{doc}` `{ver}`")
                    if c.get("text"):
                        st.caption(f'"{c["text"][:280]}..."')

        # Audit application reference
        if app_id:
            st.caption(f"🔍 Application Audit ID: `{app_id}` (logged with rule versions)")


# ── Left Sidebar: Brand Banner, New Chat, and Persistent History ───────────────
with st.sidebar:
    # 1. Header Banner & Branding
    st.markdown(
        f"""
        <div class="sidebar-brand-box">
            <h1 class="sidebar-title">🏦 LoanAssist</h1>
            <span class="sidebar-badge">v3 · RAG + Deterministic Engine</span>
            <div class="sidebar-status-chip">
                <span class="{'status-dot-green' if api_healthy else 'status-dot-red'}"></span>
                <span>{'API Online & Ready' if api_healthy else 'API Disconnected'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Policy Versions Strip
    st.markdown(
        f"""
        <div class="version-strip">
            <span>Personal: <strong class="version-tag">{active_versions.get('personal_loan','v2')}</strong></span>
            <span>Home: <strong class="version-tag">{active_versions.get('home_loan','v1')}</strong></span>
            <span>Auto: <strong class="version-tag">{active_versions.get('auto_loan','v1')}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Product Pills
    st.markdown(
        """
        <div class="product-pill-box">
            <span class="product-pill">🏠 Home Loan</span>
            <span class="product-pill">💳 Personal Loan</span>
            <span class="product-pill">🚗 Auto Loan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Primary "+ New Chat" Button
    if st.button("＋ New Conversation", use_container_width=True, type="primary"):
        new_sid = f"conv-{uuid.uuid4().hex[:8]}"
        st.session_state.session_id = new_sid
        st.session_state.messages = []
        st.session_state.last_decision = None
        st.session_state.last_calcs = None
        st.session_state.last_citations = []
        st.session_state.last_app_id = None
        st.session_state.pending_input = None
        st.rerun()

    st.markdown("<div class='history-section-title'>Recent Chats</div>", unsafe_allow_html=True)

    # 5. Conversation History List (Persistent across sessions)
    saved_convs = cm.list_conversations()
    current_sid = st.session_state.session_id

    if not saved_convs:
        st.caption("No saved conversations yet.")
    else:
        for conv in saved_convs[:15]:  # show up to 15 recent chats
            sid = conv["session_id"]
            title = conv.get("title", "Conversation")
            is_active = (sid == current_sid)

            col_btn, col_del = st.columns([0.84, 0.16])
            with col_btn:
                btn_label = f"💬 {title}" if not is_active else f"👉 {title}"
                if st.button(
                    btn_label,
                    key=f"load_{sid}",
                    use_container_width=True,
                    help=f"Load session {sid}",
                ):
                    # Load conversation from disk
                    full_conv = cm.get_conversation(sid)
                    if full_conv:
                        st.session_state.session_id     = sid
                        st.session_state.messages       = full_conv.get("messages", [])
                        st.session_state.last_decision  = full_conv.get("last_decision")
                        st.session_state.last_calcs     = full_conv.get("last_calcs")
                        st.session_state.last_citations = full_conv.get("last_citations", [])
                        st.session_state.last_app_id    = full_conv.get("last_app_id")
                        st.session_state.pending_input  = None
                        st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{sid}", help="Delete chat"):
                    cm.delete_conversation(sid)
                    if sid == current_sid:
                        st.session_state.session_id = f"conv-{uuid.uuid4().hex[:8]}"
                        st.session_state.messages = []
                        st.session_state.last_decision = None
                        st.session_state.last_calcs = None
                        st.session_state.last_citations = []
                        st.session_state.last_app_id = None
                    st.rerun()

    st.divider()

    # 6. Sidebar Footer Disclaimer
    st.markdown(
        """
        <div style="font-size:0.73rem; color:#64748b; line-height: 1.4;">
            ⚠️ <strong>Pre-qualification only</strong><br>
            Not a commitment to lend. All assessments are deterministic and logged for compliance audit.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Chat Area: Empty State vs Message Feed ────────────────────────────────
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Hero greeting if no messages yet
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-avatar">🏦</div>
            <h1 class="hero-title">How can I assist with your loan today?</h1>
            <p class="hero-subtitle">
                Instant pre-qualification across Home, Personal, and Auto loans.
                Grounded in versioned banking policies and evaluated by deterministic financial rules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Starter Recommendation Cards
    starters = [
        ("🏠", "Home Loan Pre-Check", "Check eligibility for ₹50L loan with ₹1.2L net income",
         "I want to check my eligibility for a home loan of ₹50,00,000. My monthly net income is ₹1,20,000, age 34, salaried, 5 years employment, credit score 760, existing EMI ₹10,000, property value ₹70,00,000, down payment ₹20,00,000, apartment, 240 months tenure."),
        ("💳", "Personal Loan Quick Assessment", "Check eligibility for ₹5L personal loan for 36 months",
         "I want a personal loan of ₹5,00,000. Age 32, salaried, monthly income ₹75,000, 3 years at company, credit score 760, no existing EMIs, tenure 36 months."),
        ("🚗", "Auto Loan Check", "Pre-qualify for a new car loan with ₹80K income",
         "I need an auto loan for a new four-wheeler. On-road price is ₹12,00,000, loan amount ₹9,00,000, monthly salary ₹80,000, age 29, credit score 730, tenure 60 months, no existing EMIs."),
        ("📋", "Required Documents Checklist", "See mandatory paperwork required for home loans",
         "What documents do I need to prepare for a home loan application?"),
    ]

    col1, col2 = st.columns(2)
    for i, (emoji, title, desc, prompt_text) in enumerate(starters):
        with (col1 if i % 2 == 0 else col2):
            if st.button(
                f"{emoji} {title}\n\n{desc}",
                key=f"starter_card_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_input = prompt_text

else:
    # Render all historical messages
    for msg in st.session_state.messages:
        _render_message(msg)

st.markdown("</div>", unsafe_allow_html=True)


# ── Chat Input & Action Trigger ────────────────────────────────────────────────
user_input = st.chat_input("Ask about home, personal, or auto loan eligibility…")

# If a starter card was clicked, prioritize it
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

if user_input:
    # 1. Append user prompt to messages
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Persist turn immediately to conversation store
    cm.save_conversation(
        session_id=st.session_state.session_id,
        messages=st.session_state.messages,
        last_decision=st.session_state.last_decision,
        last_calcs=st.session_state.last_calcs,
        last_citations=st.session_state.last_citations,
        last_app_id=st.session_state.last_app_id,
    )

    # 3. Re-render prior messages + new user message
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages[:-1]:
        _render_message(msg)
    _render_message(st.session_state.messages[-1])

    # 4. Stream response and update state
    _send_message(user_input)

    st.markdown("</div>", unsafe_allow_html=True)
    st.rerun()