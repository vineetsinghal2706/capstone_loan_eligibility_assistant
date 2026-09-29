"""prompts/loader.py — load versioned prompts from the registry (Session 10).

The registry (registry.yaml, beside this file) is the single source of truth
for prompt text. The ACTIVE version is selected by the environment variable
PROMPT_VERSION (default: v1) — so promoting a new prompt is an .env change,
i.e. a release, not a code edit.

Usage:
    from prompts.loader import load_prompt
    p = load_prompt("answer_grounded")            # version from PROMPT_VERSION
    p = load_prompt("answer_grounded", "v2")      # explicit version (A/B tests)
    p["text"], p["version"], p["owner"], p["date"], p["changelog"]
"""

import os
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.yaml"
DEFAULT_VERSION = "v1"

_registry_cache: dict | None = None


def _registry() -> dict:
    """Read and cache registry.yaml (parsed once per process)."""
    global _registry_cache
    if _registry_cache is None:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            _registry_cache = yaml.safe_load(f)
    return _registry_cache


def load_prompt(name: str, version: str | None = None) -> dict:
    """Return the prompt entry {version, date, owner, changelog, text}.

    version resolution order: explicit argument > PROMPT_VERSION env > v1.
    Unknown names/versions raise a clear error listing what IS available —
    a silent fallback here would mean silently shipping the wrong prompt.
    """
    prompts = _registry().get("prompts", {})
    if name not in prompts:
        raise KeyError(
            f"Unknown prompt name '{name}'. "
            f"Registered prompts: {sorted(prompts)}"
        )
    versions = prompts[name]
    resolved = version or os.environ.get("PROMPT_VERSION", DEFAULT_VERSION)
    if resolved not in versions:
        raise KeyError(
            f"Unknown version '{resolved}' for prompt '{name}'. "
            f"Registered versions: {sorted(versions)}. "
            f"Check PROMPT_VERSION in your .env."
        )
    entry = dict(versions[resolved])
    entry["name"] = name
    return entry


if __name__ == "__main__":
    # Quick smoke test:  python prompts/loader.py
    for v in ("v1", "v2"):
        p = load_prompt("answer_grounded", v)
        print(f"--- {p['name']} {p['version']} ({p['date']}, {p['owner']})")
        print(p["text"])
