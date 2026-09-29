"""ingest.py — LoanAssist v2 RAG ingestion.

Reads all policy markdown files from rag/policies/ and ingests them into
per-product ChromaDB collections with full metadata (policy_id, version,
effective_from, product, rule_id).

Key design decisions
─────────────────────
• Clause-level chunking: one chunk = one numbered section = one citable unit.
  When the explanation agent says "per HL-LTV-001", that citation resolves
  to exactly one chunk — verifiable and auditable.

• Per-product collections: "personal_loan" and "home_loan".
  The API uses metadata filtering (where={"version": "v2"}) to retrieve
  only the active policy version without mixing stale rules.

• YAML front-matter: each policy document declares policy_id, version,
  effective_from, effective_to, and product.  These become ChromaDB metadata
  on every chunk from that document.

• Idempotent: drops and rebuilds collections on every run.
  Stale chunks from a previous document version are a silent correctness bug.

Run from project root:
    python rag/ingest.py              # rebuild all collections
    python rag/ingest.py --dry-run    # count chunks, don't write
"""
import re
import sys
from pathlib import Path
from typing import Optional

import chromadb

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ── paths ─────────────────────────────────────────────────────────────────────
RAG_ROOT     = Path(__file__).resolve().parent
POLICIES_DIR = RAG_ROOT / "policies"
CHROMA_DIR   = RAG_ROOT / "chroma"

# Keep the generic collection for backward compatibility with the old /ask endpoint
GENERIC_COLLECTION = "eligibility"
PRODUCT_COLLECTIONS = ["personal_loan", "home_loan", "auto_loan"]

# ── parsing ───────────────────────────────────────────────────────────────────
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECTION_RE     = re.compile(
    r"^##\s+(?P<section>\d+)\.\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)
RULE_ID_RE     = re.compile(r"\*\*Rule ID:\*\*\s*(?P<id>[A-Z]+-[A-Z]+-\d+|[A-Z]+-\d+)")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML front-matter and return (metadata_dict, body_text).

    Falls back to empty dict when PyYAML is unavailable or front-matter
    is absent — ingest still runs, just without policy metadata.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end():]
    if not _HAS_YAML:
        print("  ⚠  PyYAML not installed — front-matter skipped (pip install pyyaml)")
        return {}, body
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        print(f"  ⚠  YAML parse error: {exc}")
        meta = {}
    return meta, body


def split_by_section(body: str, doc_name: str, doc_meta: dict) -> list[dict]:
    """Split a policy document body into one chunk per numbered section (## N. Title).

    Each chunk gets:
        doc          – source filename
        section      – section number string ("1", "2", …)
        title        – section title text
        rule_id      – Rule ID extracted from the section body, or None
        text         – flattened section text (whitespace-normalised)
        policy_id    – from front-matter
        version      – from front-matter
        effective_from  – from front-matter
        effective_to    – from front-matter (None = still active)
        product      – from front-matter
    """
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        print(f"  ⚠  No numbered sections found in {doc_name} — file skipped")
        return []

    chunks: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()

        rule_match = RULE_ID_RE.search(section_text)
        rule_id    = rule_match.group("id") if rule_match else None

        # Normalise whitespace so the embedding model sees clean text
        flat_text = " ".join(section_text.split())

        chunk: dict = {
            "doc":            doc_name,
            "section":        m.group("section"),
            "title":          m.group("title").strip(),
            "rule_id":        rule_id,
            "text":           flat_text,
            # front-matter fields (may be None/empty if doc has no front-matter)
            "policy_id":      str(doc_meta.get("policy_id", "")),
            "version":        str(doc_meta.get("version", "v1")),
            "effective_from": str(doc_meta.get("effective_from", "")),
            "effective_to":   str(doc_meta.get("effective_to", "") or ""),
            "product":        str(doc_meta.get("product", "")),
        }
        chunks.append(chunk)
    return chunks


# ── ingestion helpers ─────────────────────────────────────────────────────────

def _build_collection(
    client: chromadb.PersistentClient,
    name: str,
    chunks: list[dict],
    dry_run: bool,
) -> int:
    """Drop-and-rebuild a named ChromaDB collection."""
    if dry_run:
        print(f"  [dry-run] would write {len(chunks)} chunks → '{name}'")
        return len(chunks)

    try:
        client.delete_collection(name)
        print(f"  Deleted existing collection '{name}' (idempotent rebuild)")
    except Exception:
        pass  # first run

    collection = client.create_collection(name, metadata={"hnsw:space": "l2"})

    if not chunks:
        print(f"  ⚠  No chunks for '{name}' — empty collection created")
        return 0

    # ChromaDB requires unique IDs. Use "product:section:version" to avoid
    # collisions when multiple policy versions coexist in the same collection.
    ids = [
        f"{c['product']}:{c['section']}:{c['version']}" if c['product']
        else f"{c['doc']}:{c['section']}"
        for c in chunks
    ]

    # Build metadata dicts — ChromaDB only stores scalar values
    metadatas = []
    for c in chunks:
        meta: dict = {
            "doc":            c["doc"],
            "section":        c["section"],
            "title":          c["title"],
            "policy_id":      c["policy_id"],
            "version":        c["version"],
            "effective_from": c["effective_from"],
            "product":        c["product"],
        }
        if c["effective_to"]:
            meta["effective_to"] = c["effective_to"]
        if c["rule_id"]:
            meta["rule_id"] = c["rule_id"]
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
    )
    print(f"  [OK] '{name}' -- {collection.count()} chunks stored")
    return collection.count()


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    if not POLICIES_DIR.exists():
        sys.exit(
            f"Policy directory not found: {POLICIES_DIR}\n"
            "Run from the project root after creating rag/policies/*.md files."
        )

    policy_files = sorted(POLICIES_DIR.glob("*.md"))
    if not policy_files:
        sys.exit(f"No policy files found in {POLICIES_DIR}")

    print(f"\nLoanAssist RAG Ingestion {'[DRY RUN] ' if dry_run else ''}")
    print(f"Policies dir : {POLICIES_DIR}")
    print(f"Chroma dir   : {CHROMA_DIR}")
    print(f"Files found  : {len(policy_files)}\n")

    # Parse all documents
    product_chunks: dict[str, list[dict]] = {p: [] for p in PRODUCT_COLLECTIONS}
    all_chunks: list[dict] = []

    for path in policy_files:
        text      = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        product   = meta.get("product", "")
        version   = meta.get("version", "v1")
        chunks    = split_by_section(body, path.name, meta)

        print(f"  {path.name:40s}  product={product or '?':15s}  "
              f"version={version}  sections={len(chunks)}")

        for c in chunks:
            all_chunks.append(c)
            if product in product_chunks:
                product_chunks[product].append(c)

    print(f"\nTotal chunks: {len(all_chunks)}")
    for prod, chunks in product_chunks.items():
        print(f"  {prod}: {len(chunks)} chunks")

    if not dry_run:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Build per-product collections
    print(f"\nBuilding per-product collections ...")
    for product, chunks in product_chunks.items():
        _build_collection(client, product, chunks, dry_run)

    # Build generic collection (backward compatibility with /ask)
    print("\nBuilding generic 'eligibility' collection (backward compat) ...")
    _build_collection(client, GENERIC_COLLECTION, all_chunks, dry_run)

    if not dry_run:
        print(f"\nIngestion complete. Chroma index at: {CHROMA_DIR}/")
        print("\nSample chunks (doc . section . rule_id . first 80 chars):")
        for c in all_chunks[:6]:
            rid = c.get("rule_id") or "-"
            print(f"  {c['doc']:35s} . s{c['section']:3s} . {rid:15s} . {c['text'][:80]}...")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
