#!/usr/bin/env bash
# rollback.sh — the fire-drill command. Instantly returns 100% of traffic to v1.
# Rollback FIRST, debug SECOND.
set -e
cd "$(dirname "$0")/.."

printf 'events {}\nhttp {\n  upstream llm {\n    server llm-v1:8080;\n  }\n  server {\n    listen 80;\n    location / {\n      proxy_pass http://llm;\n      proxy_read_timeout 300s;\n    }\n  }\n}\n' > nginx/nginx.conf
docker compose exec lb nginx -s reload

echo "ROLLBACK COMPLETE — all traffic on v1."
echo "Verify:  curl -s localhost:8080/version"
