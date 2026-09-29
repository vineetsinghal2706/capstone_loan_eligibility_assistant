# LoanAssist v3

A **RAG-based loan pre-qualification advisor** that delivers consistent,
policy-grounded eligibility assessments for **Personal Loans, Home Loans, and Auto Loans** via a
professional streaming chat interface.

> **Preliminary assessment only.** This system does not constitute a loan offer,
> commitment, or final approval.

---

## What it does (v3 Highlights)

| Capability | Detail |
|:---|:---|
| **Semantic Loan Detection** | Inferred naturally from conversation without buttons or tags — "building loan" / "flat loan" → Home Loan; "car" / "scooter" / "bus" / "bike" / "EV" → Auto Loan; "cash loan" → Personal Loan |
| **Auto Loan End-to-End** | 11-section policy (`AL-AGE-001` to `AL-EXC-001`), on-road price LTV calculator (85% new, 70% used), vehicle age & tenure limits, 9 rule evaluators |
| **Conversational Memory & Profile Merge** | Retains all applicant data across conversation turns without re-asking for already provided fields |
| **On-Demand Document Checklist** | Clean conversational UX — checklist is provided only when the customer explicitly asks for documents |
| **Professional Chat UI** | Polished, single-column chat-first layout with dark-on-light bubbles, typing indicator, inline decision badges, metric strip, and expandable rule details |
| **Deterministic Rules Engine** | Pure-Python PASS / FAIL / MANUAL_REVIEW / INSUFFICIENT_INFORMATION — LLM never decides eligibility |
| **RAG over Versioned Policies** | Per-product ChromaDB collections (`personal_loan`, `home_loan`, `auto_loan`), metadata-filtered by active policy version |
| **Grounded Citations** | Every assessment cites exact Rule IDs and versioned policy documents |
| **Financial Calculators** | Pure-Python reducing-balance EMI, FOIR, LTV (property / on-road price), and max affordable loan |
| **Audit Logging** | Complete record per application: snapshot, policy version, rules applied, calculations, and decision |
| **GitHub CI/CD Evaluation Gates** | Fast unit tests on every push (rules + calculators) and Promptfoo regression gates on pull requests |
| **3-Layer Safety Defence** | Layer 1 Input Guard → Layer 2 Grounded Prompts (temp=0) → Layer 3 Per-Sentence Streaming Output Guard |

---

## Architecture

```
User (Browser :8501)
  └─► Streamlit Chat UI v3
       └─► POST /chat (SSE Stream)
             ├─► Layer 1: Input Guard (injection blocklist, PII detection, topic filter)
             ├─► Profile Extractor (LLM temp=0 + Python alias fallback + profile merge)
             │
             ├─ Incomplete profile ─► Conversation Agent (LLM, asks missing fields only)
             ├─ Document request  ─► Document Agent (on-demand checklist)
             │
             └─ Complete profile  ─► Policy RAG (ChromaDB, active version filtered)
                                      Deterministic Rules Engine (pure Python)
                                      Financial Calculators (pure Python)
                                      Explanation Agent (LLM, grounded explanation)
                                      Layer 3: Output Guard (per-sentence buffer)
                                      Audit Logger (JSONL + in-memory store)
```

### Services (Docker Compose)

| Service | URL | Purpose |
|:---|:---|:---|
| **API** | `http://localhost:8001` | FastAPI: `/chat`, `/ask`, `/audit/{id}`, `/scenario`, `/health`, `/metrics`, `/versions` |
| **LiteLLM Admin UI & Proxy** | `http://localhost:4000/ui` (Proxy: `http://localhost:4000`) | OpenAI-compatible proxy & dashboard (Master Key: `sk-master-key-loanassist`) |
| **Local model** | `http://localhost:8090` | Qwen 2.5-1.5B local inference server |
| **Prometheus** | `http://localhost:9090` | Metrics scraping target |
| **Grafana** | `http://localhost:3000` | Golden signals dashboard (`admin` / `admin`) |

The Streamlit UI runs at `http://localhost:8501`.  
ChromaDB is embedded and persists to `rag/chroma/`.

---

## Supported Loan Products

### 1. Personal Loan
- **Age:** 21 to 60 years
- **Income:** >= Rs.25,000 / month
- **Employment:** Salaried >= 12 months, Self-employed >= 24 months
- **Credit Score:** >= 700
- **FOIR:** <= 50%
- **Loan Amount:** Rs.50,000 to Rs.25,000,000 (capped at 30x monthly income)
- **Tenure:** 12 to 60 months
- **Indicative Interest:** 12.0% p.a.

### 2. Home Loan
- **Age:** >= 21 years; loan closure before age 70
- **Income:** >= Rs.40,000 / month
- **Employment:** Salaried >= 24 months, Self-employed >= 36 months (24-35 months triggers `MANUAL_REVIEW`)
- **Credit Score:** >= 700
- **FOIR:** <= 55%
- **LTV:** <= 85% for loans <= Rs.30L; <= 80% for loans > Rs.30L
- **Property:** Apartment, Villa, or Residential House
- **Loan Amount:** Rs.5,00,000 to Rs.5,00,00,000
- **Tenure:** 12 to 360 months
- **Indicative Interest:** 8.5% p.a.

### 3. Auto Loan (New in v3)
- **Age:** 21 to 65 years; loan closure before age 70
- **Income:** >= Rs.20,000 / month
- **Employment:** Salaried >= 12 months, Self-employed >= 24 months
- **Credit Score:** >= 680
- **FOIR:** <= 50%
- **LTV (on On-Road Price):** <= 85% for new vehicles; <= 70% for used vehicles
- **Vehicle Eligibility:** Two-wheelers, four-wheelers, commercial vehicles (new), electric vehicles
- **Used Vehicle Limits:** Max age 10 years at application, max tenure 60 months
- **Loan Amount:** Rs.50,00,000 max (Rs.50,000 min)
- **Tenure:** 12 to 84 months (new), up to 48 months (commercial)
- **Indicative Interest:** 9.0% p.a.

---

## Prerequisites

- **Python 3.11** or newer
- **Docker Engine and Docker Compose v2** (for containerised services)
- ~1 GB disk space for the local model weights

Verify installations:
```bash
python --version
docker --version
docker compose version
```

---

## Python Environment Setup

### 1. Clone & Navigate to Project
```bash
cd Loan_Eligibility_Assistant
```

### 2. Configure Environment File
Copy `.env.example` to `.env`:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Key configuration options in `.env`:
```dotenv
API_KEY=local-dev-key
LLM_BASE_URL=http://localhost:4000/v1
LLM_MODEL=openrouter-mini
PROMPT_VERSION=v1

# Active policy versions (ChromaDB filters by active version)
ACTIVE_PERSONAL_LOAN_VERSION=v2
ACTIVE_HOME_LOAN_VERSION=v1
ACTIVE_AUTO_LOAN_VERSION=v1

# Cloud fallback API Key (optional - e.g. OpenRouter)
OPENROUTER_API_KEY=
```

### 3. Create & Activate Python Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1
```

*(If execution policy blocks script activation, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

#### On Linux / macOS:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/Scripts/activate
```

### 4. Install Dependencies
Install requirements for both API and UI:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install API requirements (FastAPI, ChromaDB, PyYAML, Pytest, LiteLLM client)
pip install -r api/requirements.txt

# Install UI requirements (Streamlit, Requests, python-dotenv)
pip install -r ui/requirements.txt
```

### 5. Create Log & Data Directories
```bash
# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path logs, rag/chroma

# Linux / macOS
mkdir -p logs rag/chroma
```

*(On Linux/macOS, export your UID/GID if running docker volume mounts: `export HOST_UID=$(id -u); export HOST_GID=$(id -g)`)*

---

## Policy Ingestion (RAG Setup)

Run this step to chunk and index the markdown policies into ChromaDB before running the application:

```bash
python rag/ingest.py
```

Expected output:
```
LoanAssist RAG Ingestion
Policies dir : .../rag/policies
Chroma dir   : .../rag/chroma
Files found  : 4

  auto_loan_v1.md         product=auto_loan     version=v1  sections=11
  home_loan_v1.md         product=home_loan     version=v1  sections=11
  personal_loan_v1.md     product=personal_loan version=v1  sections=9
  personal_loan_v2.md     product=personal_loan version=v2  sections=9

Total chunks: 40
  personal_loan: 18 chunks
  home_loan: 11 chunks
  auto_loan: 11 chunks

Building per-product collections ...
  [OK] 'personal_loan' -- 18 chunks stored
  [OK] 'home_loan' -- 11 chunks stored
  [OK] 'auto_loan' -- 11 chunks stored
Building generic 'eligibility' collection (backward compat) ...
  [OK] 'eligibility' -- 40 chunks stored

Ingestion complete.
```

---

## Running the Application

### Option A: Full Stack with Docker Compose (Recommended)

1. **Start all backend services:**
   ```bash
   docker compose up -d --build
   ```

2. **Verify running containers:**
   ```bash
   docker compose ps
   ```

3. **Check API Health:**
   ```bash
   curl http://localhost:8001/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "version": "3.0.0",
     "rag_collections": {
       "personal_loan": 18,
       "home_loan": 11,
       "auto_loan": 11,
       "eligibility": 40
     },
     "active_policy_versions": {
       "personal_loan": "v2",
       "home_loan": "v1",
       "auto_loan": "v1"
     }
   }
   ```

4. **Launch Streamlit Chat UI:**
   ```bash
   streamlit run ui/chat_app.py
   ```
   Open `http://localhost:8501` in your browser.

---

### Option B: Local Python Execution (Development / Debugging)

If running without Docker:

1. **Start the API directly:**
   ```bash
   # From project root with .venv activated
   cd api
   python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Run Streamlit in a separate terminal:**
   ```bash
   # From project root with .venv activated
   streamlit run ui/chat_app.py
   ```

---

## Running Tests & Evaluation

### 1. Unit Tests (Rules Engine + Calculators)
Run the 80 unit tests covering calculations, boundary rules, policy criteria, and auto loan logic:

```bash
pytest api/tests/ -v
```

Expected output:
```
============================= 80 passed in 0.20s ==============================
```

### 2. CI/CD Promptfoo Evaluation Gate
To execute the automated regression and safety evaluation gate locally (requires running API):

```bash
# Windows (Git Bash or WSL) / Linux / macOS
bash scripts/eval-gate.sh
```

Evaluates:
- Personal Loan eligibility, credit score boundaries, and FOIR capping
- Home Loan LTV boundaries and commercial property rejection
- Auto Loan eligibility, new vs used LTV limits, and vehicle age limits
- Semantic loan intent detection ("house loan" $\rightarrow$ Home Loan; "car loan" $\rightarrow$ Auto Loan)
- Prompt injection resistance and refusal handling

### 3. Labelled Benchmark Dataset
`tests/labelled_set.json` contains 25 ground-truth test cases across Personal, Home, and Auto Loans for regression benchmarking.

---

## API Reference

### `POST /chat` — Streaming Eligibility Chat (Primary)
Server-Sent Events (SSE) streaming endpoint:

| Event Type | Description |
|:---|:---|
| `token` | Streamed text chunk (buffered by sentence through Layer 3 output guard) |
| `profile_update` | Extracted profile snapshot and missing required fields |
| `decision` | Structured decision object (`POTENTIALLY_ELIGIBLE`, `NOT_ELIGIBLE`, `MANUAL_REVIEW`, `INSUFFICIENT_INFORMATION`) |
| `calculations` | Financial outputs: reducing EMI, FOIR, LTV, and max affordable loan |
| `citations` | Retrieved policy clauses with document name, rule ID, and version |
| `audit_ref` | Unique application ID for tracking and compliance audit |
| `error` | Error or safety refusal notification |

Example request:
```bash
curl -N -X POST http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-dev-key' \
  -d '{"session_id":"sess-1","message":"I need a car loan. Age 30, income Rs.60,000, credit score 720."}'
```

### `POST /scenario` — What-If Simulator
Override profile fields to test alternative terms or financial adjustments:
```bash
curl -X POST http://localhost:8001/scenario \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-dev-key' \
  -d '{
    "session_id": "sess-1",
    "overrides": {
      "on_road_price": 700000,
      "requested_amount": 500000
    }
  }'
```

### `GET /audit/{application_id}` — Compliance Audit Record
Retrieve full underwriting logs including input snapshot, rule checks, calculations, and policy IDs.

### `GET /versions` — Policy Versions
Inspect active versions across all loan types and prompts.

---

## Project Layout

```
Loan_Eligibility_Assistant/
├── api/
│   ├── app.py                 # FastAPI application (/chat SSE, /scenario, /audit, /health)
│   ├── calculators.py         # Deterministic financial calculators (EMI, FOIR, LTV)
│   ├── rules_engine.py        # Pure-Python eligibility rules engine (PL, HL, AL)
│   ├── profile_extractor.py   # LLM extraction, semantic aliases, profile merge
│   ├── guardrails.py          # 3-layer security (injection blocklist, PII, topic gate)
│   ├── redact.py              # PII detection & masking
│   ├── requirements.txt       # Backend dependencies
│   └── tests/
│       ├── test_calculators.py   # Calculator unit tests
│       └── test_rules_engine.py  # Rules engine unit tests (PL, HL, AL)
├── prompts/
│   ├── registry.yaml          # Versioned prompt registry (v1 production, v2 canary)
│   └── loader.py              # YAML prompt loader
├── rag/
│   ├── policies/
│   │   ├── personal_loan_v1.md # Personal loan policy v1
│   │   ├── personal_loan_v2.md # Personal loan policy v2 (active)
│   │   ├── home_loan_v1.md     # Home loan policy v1 (active)
│   │   └── auto_loan_v1.md     # Auto loan policy v1 (active)
│   ├── ingest.py              # ChromaDB chunking and indexing script
│   └── chroma/                # ChromaDB vector index directory
├── ui/
│   ├── chat_app.py            # Streamlit v3 professional chat interface
│   └── requirements.txt       # Streamlit UI dependencies
├── tests/
│   └── labelled_set.json      # 25 labelled ground-truth test cases
├── .github/workflows/
│   └── eval-gate.yml          # GitHub Actions CI/CD eval gate pipeline
├── scripts/
│   └── eval-gate.sh           # Promptfoo runner script
├── docker-compose.yml         # Container orchestration
├── promptfooconfig.yaml       # Promptfoo automated test suites
├── .env.example               # Template environment configuration
└── README.md                  # Comprehensive system documentation
```

---

## Troubleshooting

| Problem | Root Cause | Solution |
|:---|:---|:---|
| **API cannot import `chromadb`** | Running inside a Python venv without ChromaDB | Run `pip install -r api/requirements.txt` in active `.venv` |
| **RAG returns 0 chunks** | ChromaDB index not built | Run `python rag/ingest.py` before starting the API |
| **Model service starting slowly** | Local Qwen weights downloading | Check `docker compose logs -f model`; wait for healthy check |
| **UI displays connection error** | API is not running on port 8001 | Verify with `curl http://localhost:8001/health` and check container status |
| **Script execution error on Windows** | PowerShell execution policy restricted | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| **Scenario returns 404** | Session has no profile stored | Start a `/chat` message first to establish session context before simulating |