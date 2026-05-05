#!/bin/bash
# Entrypoint for the trainer container.
# Dependencies (mat, mat-query) are pre-installed in the image by Dockerfile.trainer.
#
# Required env vars:
#   MODEL_ENDPOINT   vLLM base URL, e.g. http://host.docker.internal:8000
#   MODEL_ID         HuggingFace model ID served at that endpoint
#
# Optional env vars:
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

# ── wait for vLLM ─────────────────────────────────────────────────────────────

echo "Waiting for vLLM at ${MODEL_ENDPOINT}..."
until curl -sf "${MODEL_ENDPOINT}/health" &>/dev/null; do
    sleep 5
done
echo "vLLM ready."

# ── write config ──────────────────────────────────────────────────────────────

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

# ── run ───────────────────────────────────────────────────────────────────────

echo "Starting training — model: ${MODEL_ID}, experiments: ${MAX_EXPERIMENTS}"
mat --config "${CONFIG_FILE}"
