"""conversation_manager.py — Persistent conversation storage for LoanAssist UI.

Saves and loads chat sessions from a local JSON store (ui/data/conversations.json),
enabling ChatGPT/Claude-style conversation history on the left sidebar.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("loanassist.conversations")

DATA_DIR = Path(__file__).parent / "data"
STORAGE_FILE = DATA_DIR / "conversations.json"


def _ensure_storage_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw() -> Dict[str, dict]:
    _ensure_storage_dir()
    if not STORAGE_FILE.exists():
        return {}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s. Reinitializing store.", STORAGE_FILE, exc)
        return {}


def _save_raw(data: Dict[str, dict]):
    _ensure_storage_dir()
    tmp_file = STORAGE_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_file.replace(STORAGE_FILE)
    except Exception as exc:
        logger.error("Failed to write conversations to %s: %s", STORAGE_FILE, exc)
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


def generate_title(first_message: str, product: Optional[str] = None) -> str:
    """Generate a concise, human-friendly title from the first message."""
    clean = re.sub(r"\s+", " ", first_message.strip())
    # Remove common conversational prefixes
    clean = re.sub(r"^(hi|hello|hey|i am looking for|i need|tell me about|can you check)\s+", "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    if not clean:
        clean = "Loan Pre-Qualification"

    # Capitalize first letter
    clean = clean[0].upper() + clean[1:]
    if len(clean) > 36:
        clean = clean[:33].rsplit(" ", 1)[0] + "…"

    if product:
        prod_map = {
            "personal_loan": "💳 Personal Loan",
            "home_loan": "🏠 Home Loan",
            "auto_loan": "🚗 Auto Loan",
        }
        prod_label = prod_map.get(product)
        if prod_label and prod_label not in clean:
            clean = f"{clean}"

    return clean


def list_conversations() -> List[Dict[str, Any]]:
    """List summary metadata for all saved conversations, sorted newest first."""
    data = _load_raw()
    items = []
    for sid, conv in data.items():
        items.append({
            "session_id": sid,
            "title": conv.get("title", "New Conversation"),
            "created_at": conv.get("created_at", time.time()),
            "updated_at": conv.get("updated_at", time.time()),
            "message_count": len(conv.get("messages", [])),
            "last_decision": conv.get("last_decision"),
            "product": conv.get("product"),
        })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def get_conversation(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full conversation details by session_id."""
    data = _load_raw()
    return data.get(session_id)


def save_conversation(
    session_id: str,
    messages: List[Dict[str, Any]],
    title: Optional[str] = None,
    last_decision: Optional[Dict[str, Any]] = None,
    last_calcs: Optional[Dict[str, Any]] = None,
    last_citations: Optional[List[Dict[str, Any]]] = None,
    last_app_id: Optional[str] = None,
    product: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a conversation record."""
    data = _load_raw()
    now = time.time()

    existing = data.get(session_id, {})
    created_at = existing.get("created_at", now)

    # Derive title if not provided
    if not title:
        title = existing.get("title")
        if not title and messages:
            # find first user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                title = generate_title(user_msgs[0].get("content", ""), product)
        if not title:
            title = "New Conversation"

    # Derive product from last_decision if available
    if not product and last_decision and "product" in last_decision:
        product = last_decision.get("product")
    elif not product and existing.get("product"):
        product = existing.get("product")

    conv_record = {
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "messages": messages,
        "last_decision": last_decision if last_decision is not None else existing.get("last_decision"),
        "last_calcs": last_calcs if last_calcs is not None else existing.get("last_calcs"),
        "last_citations": last_citations if last_citations is not None else existing.get("last_citations", []),
        "last_app_id": last_app_id if last_app_id is not None else existing.get("last_app_id"),
        "product": product,
    }

    data[session_id] = conv_record
    _save_raw(data)
    return conv_record


def delete_conversation(session_id: str) -> bool:
    """Delete a conversation by session_id."""
    data = _load_raw()
    if session_id in data:
        del data[session_id]
        _save_raw(data)
        return True
    return False
