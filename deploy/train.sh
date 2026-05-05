#!/bin/bash
# Set up and run a MultiAgentTrainer training session against a local vLLM server.
#
# SCP this file to the EC2 host, then mount it into a training container:
#
#   scp deploy/train.sh ubuntu@HOST:/home/ubuntu/train.sh
#   docker run -d \
#     -v /home/ubuntu/train.sh:/train.sh \
#     -e MODEL_ENDPOINT=http://host.docker.internal:8000 \
#     -e MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct \
#     --add-host host.docker.internal:host-gateway \
#     python:3.12-slim bash /train.sh
#
# Required env vars:
#   MODEL_ENDPOINT   vLLM base URL, e.g. http://host.docker.internal:8000
#   MODEL_ID         HuggingFace model ID served at that endpoint
#
# Optional env vars:
#   MAT_REPO         Git URL or local path for MultiAgentTrainer (default: PyPI release)
#   SOURCE_REPO      Git URL to use as the training data source
#   TRAIN_TIME       Seconds per experiment (default: 300)
#   MAX_EXPERIMENTS  Number of experiments to run (default: 50)
#   OUTPUT_DIR       Where to write results (default: /output)
#   GITHUB_TOKEN     Passed through for private source repos

set -euo pipefail

MODEL_ENDPOINT="${MODEL_ENDPOINT:?MODEL_ENDPOINT is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
TRAIN_TIME="${TRAIN_TIME:-300}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
SOURCE_REPO="${SOURCE_REPO:-}"

# ── dependencies ───────────────────────────────────────────────────────────────

apt-get update -qq && apt-get install -y -qq git curl

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if [ -n "${MAT_REPO:-}" ]; then
    uv tool install "git+${MAT_REPO}"
else
    uv tool install multiagenttrainer
fi

export PATH="$HOME/.local/bin:$PATH"

# ── query helper ───────────────────────────────────────────────────────────────
# Installed to /usr/local/bin so it's on PATH inside the agent_command subprocess.

cat > /usr/local/bin/mat-query <<'PYEOF'
#!/usr/bin/env python3
"""Query a vLLM chat endpoint and print the reply. Usage: mat-query <endpoint> <model> <prompt>"""
import json, sys, urllib.request, urllib.error

endpoint, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 4096,
}).encode()
req = urllib.request.Request(
    f"{endpoint.rstrip('/')}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print(json.loads(r.read())["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()}")
except OSError as e:
    sys.exit(f"Connection error: {e}")
PYEOF
chmod +x /usr/local/bin/mat-query

# ── wait for vLLM to be ready ──────────────────────────────────────────────────

echo "Waiting for vLLM at ${MODEL_ENDPOINT}..."
until curl -sf "${MODEL_ENDPOINT}/health" &>/dev/null; do
    sleep 5
done
echo "vLLM ready."

# ── write config ───────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

CONFIG_FILE="$(mktemp /tmp/mat-XXXXXX.yaml)"

if [ -n "${SOURCE_REPO}" ]; then
    SOURCES_BLOCK="sources:
  - type: remote_repo
    url: \"${SOURCE_REPO}\""
else
    SOURCES_BLOCK=""
fi

cat > "${CONFIG_FILE}" <<EOF
autoresearch:
  train_time: ${TRAIN_TIME}

training:
  agent_command: "mat-query ${MODEL_ENDPOINT} ${MODEL_ID} {prompt}"
  max_experiments: ${MAX_EXPERIMENTS}
  output_dir: ${OUTPUT_DIR}

${SOURCES_BLOCK}
EOF

# ── run ────────────────────────────────────────────────────────────────────────

echo "Starting training — model: ${MODEL_ID}, experiments: ${MAX_EXPERIMENTS}"
mat --config "${CONFIG_FILE}"
