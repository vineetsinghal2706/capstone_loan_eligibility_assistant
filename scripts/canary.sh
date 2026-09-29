#!/usr/bin/env bash
# canary.sh — progressive rollout of v2 with an automated eval gate at each stage.
# 10% -> 50% -> 100%. Any eval failure triggers automatic rollback to 100% v1.
set -e
cd "$(dirname "$0")/.."
export API_KEY=$(grep API_KEY .env | cut -d= -f2)

write_conf() {  # write_conf <v1_weight> <v2_weight>  (0 = omit that version)
  local v1=$1 v2=$2 servers=""
  [ "$v1" != "0" ] && servers="    server llm-v1:8080 weight=$v1;\n"
  [ "$v2" != "0" ] && servers="$servers    server llm-v2:8080 weight=$v2;\n"
  printf 'events {}\nhttp {\n  upstream llm {\n%b  }\n  server {\n    listen 80;\n    location / {\n      proxy_pass http://llm;\n      proxy_read_timeout 300s;\n    }\n  }\n}\n' "$servers" > nginx/nginx.conf
  docker compose exec lb nginx -s reload
}

run_gate() {
  npx --yes promptfoo@latest eval -c promptfooconfig.canary.yaml --no-cache
}

rollback() {
  echo ""
  echo "!!! CANARY FAILED THE EVAL GATE — ROLLING BACK TO 100% v1 !!!"
  write_conf 1 0
  echo "Rollback complete. All traffic on v1. Investigate v2 before retrying."
  exit 1
}

echo "=== Pre-flight: is v2 healthy? ==="
curl -sf http://localhost:8082/health > /dev/null || { echo "v2 not up on :8082"; exit 1; }

echo "=== STAGE 1: canary at 10% (v1:9 / v2:1) ==="
write_conf 9 1
run_gate || rollback
echo "--- Stage 1 evals passed ---"

echo "=== STAGE 2: canary at 50% (v1:1 / v2:1) ==="
write_conf 1 1
run_gate || rollback
echo "--- Stage 2 evals passed ---"

echo "=== STAGE 3: promote — 100% v2 ==="
write_conf 0 1
echo ""
echo "=== PROMOTION COMPLETE: v2 is serving all traffic ==="
echo "Verify:  for i in 1 2 3; do curl -s localhost:8080/version; echo; done"
