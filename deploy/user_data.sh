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

echo "vllm setup complete: {model_id}" > /var/log/mat-setup.log
