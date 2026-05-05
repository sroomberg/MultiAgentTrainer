#!/bin/bash
set -euxo pipefail

# ── Docker ────────────────────────────────────────────────────────────────────
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# ── NVIDIA drivers ────────────────────────────────────────────────────────────
apt-get install -y linux-headers-$(uname -r)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -q
apt-get install -y -q nvidia-driver-535 nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ── vLLM inference server ─────────────────────────────────────────────────────
docker run -d \
    --gpus all \
    --restart unless-stopped \
    --name vllm \
    -p {port}:{port} \
    -e HUGGING_FACE_HUB_TOKEN={hf_token} \
    vllm/vllm-openai:latest \
    --model {model_id} \
    --port {port} \
    --gpu-memory-utilization {gpu_memory_utilization}

# ── trainer image ─────────────────────────────────────────────────────────────
mkdir -p /opt/mat-trainer
cat > /opt/mat-trainer/Dockerfile <<'DOCKERFILE'
{dockerfile_trainer}
DOCKERFILE
cat > /opt/mat-trainer/train.sh <<'TRAINSH'
{train_sh}
TRAINSH
cat > /opt/mat-trainer/mat-query <<'MATQUERY'
{mat_query}
MATQUERY

docker build -t mat-trainer /opt/mat-trainer

# ── trainer container ─────────────────────────────────────────────────────────
# Starts immediately; train.sh polls /health until vLLM is ready.
docker run -d \
    --restart unless-stopped \
    --name trainer \
    --add-host host.docker.internal:host-gateway \
    -e MODEL_ENDPOINT=http://host.docker.internal:{port} \
    -e MODEL_ID={model_id} \
    -e TRAIN_TIME={train_time} \
    -e MAX_EXPERIMENTS={max_experiments} \
    -e SOURCE_REPO={source_repo} \
    -v /home/ubuntu/training-output:/output \
    mat-trainer

echo "setup complete: {model_id}" > /var/log/mat-setup.log
