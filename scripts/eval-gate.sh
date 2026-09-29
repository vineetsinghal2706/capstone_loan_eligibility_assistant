#!/usr/bin/env bash
# eval-gate.sh — the release gate. Non-zero exit = release BLOCKED.
# Used manually (Step 3) and by the git pre-push hook (Step 4).
set -e
cd "$(dirname "$0")/.."

export API_KEY=$(grep API_KEY .env | cut -d= -f2)

# Gate precondition: the stable server must be up
if ! curl -sf http://localhost:8081/health > /dev/null; then
  echo "GATE ERROR: v1 server not reachable on :8081 — run 'docker compose up -d' first."
  exit 1
fi

echo "=== Running eval gate (promptfoo) ==="
npx --yes promptfoo@latest eval -c promptfooconfig.yaml --no-cache

echo "=== GATE PASSED — release may proceed ==="
