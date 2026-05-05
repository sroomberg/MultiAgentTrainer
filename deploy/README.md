# deploy — AWS inference infrastructure for MultiAgentTrainer

Pulumi project that provisions one dedicated EC2 GPU instance per model, each running a vLLM inference server with an OpenAI-compatible API.

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- AWS credentials configured (`aws configure` or env vars)
- An SSH key pair (default: `~/.ssh/id_ed25519`)
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to any gated models you want to serve

## Quick start

```bash
cd deploy
pulumi stack init dev

# Required: restrict access to your IP
pulumi config set --path 'allowed_ssh_cidrs[0]' "$(curl -s ifconfig.me)/32"
pulumi config set --path 'allowed_inference_cidrs[0]' "$(curl -s ifconfig.me)/32"

# Required: HuggingFace token (stored as a Pulumi secret)
pulumi config set --secret mat-model-infra:hf_token hf_xxx

pulumi up
```

## Configuration

```bash
# REQUIRED: CIDRs allowed to SSH into instances
pulumi config set --path 'allowed_ssh_cidrs[0]' "203.0.113.10/32"

# REQUIRED: CIDRs allowed to hit the inference port (machine running agenttester)
pulumi config set --path 'allowed_inference_cidrs[0]' "203.0.113.10/32"

# REQUIRED: HuggingFace token
pulumi config set --secret mat-model-infra:hf_token hf_xxx

# Default instance type for all models (default: g4dn.xlarge — 1x T4 16GB)
pulumi config set instance_type g5.xlarge

# SSH public key path (default: ~/.ssh/id_ed25519.pub)
pulumi config set ssh_public_key_path ~/.ssh/my_key.pub

# AWS region
pulumi config set aws:region us-west-2
```

To override the instance type or GPU utilization for a specific model, add fields to its entry in `Pulumi.dev.yaml`:

```yaml
mat-model-infra:models:
  - name: llama3-70b
    model_id: meta-llama/Meta-Llama-3-70B-Instruct
    instance_type: g4dn.12xlarge
    gpu_memory_utilization: 0.95
```

## Instance sizing guide

| Model size | VRAM needed | Recommended instance |
|---|---|---|
| 7B (fp16) | ~14 GB | `g4dn.xlarge` (1× T4 16 GB) |
| 13B (fp16) | ~26 GB | `g5.2xlarge` (1× A10G 24 GB) |
| 70B (fp16) | ~140 GB | `p3.8xlarge` (4× V100 64 GB) |

## Outputs

After `pulumi up`:

```bash
# All model endpoints
pulumi stack output model_endpoints

# Config snippet to paste into agenttester.yaml
pulumi stack output agenttester_yaml_snippet
```

The snippet looks like:

```yaml
agents:
  llama3:
    command: 'python scripts/query_model.py http://1.2.3.4:8000 meta-llama/Meta-Llama-3-8B-Instruct {prompt}'
    host: localhost
    commit_style: manual
    timeout: 120

  mistral:
    command: 'python scripts/query_model.py http://5.6.7.8:8000 mistralai/Mistral-7B-Instruct-v0.2 {prompt}'
    host: localhost
    commit_style: manual
    timeout: 120
```

## What gets provisioned per model

- **EC2 instance** (`g4dn.xlarge` by default) with 100 GB gp3 root volume
- **vLLM container** started on port 8000 with `--restart unless-stopped`
- **Shared security group** — SSH from `allowed_ssh_cidrs`, port 8000 from `allowed_inference_cidrs`
- **IAM role + instance profile** with SSM access for debugging via Session Manager

Model weights are pulled from HuggingFace on first boot. Expect 5–15 minutes before the endpoint is ready depending on model size.

## Running training

`train.sh` is a self-contained script that sets up MultiAgentTrainer inside a container and runs it against the local vLLM server. SCP it to each host after `pulumi up`, then launch a training container:

```bash
# Get the host IPs
pulumi stack output public_ips

# Copy the script to a host
scp deploy/train.sh ubuntu@HOST_IP:/home/ubuntu/train.sh

# Start a training container on that host (over SSH)
ssh ubuntu@HOST_IP docker run -d \
  --name trainer \
  --add-host host.docker.internal:host-gateway \
  -v /home/ubuntu/train.sh:/train.sh \
  -e MODEL_ENDPOINT=http://host.docker.internal:8000 \
  -e MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct \
  -e MAX_EXPERIMENTS=50 \
  -e TRAIN_TIME=300 \
  -e SOURCE_REPO=https://github.com/your-org/your-repo \
  -v /home/ubuntu/training-output:/output \
  python:3.12-slim bash /train.sh

# Follow logs
ssh ubuntu@HOST_IP docker logs -f trainer
```

The script waits for vLLM to finish loading the model before starting experiments, so it's safe to launch it immediately after the instance boots.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_ENDPOINT` | yes | — | vLLM base URL |
| `MODEL_ID` | yes | — | HuggingFace model ID |
| `TRAIN_TIME` | no | `300` | Seconds per experiment |
| `MAX_EXPERIMENTS` | no | `50` | Number of experiments |
| `OUTPUT_DIR` | no | `/output` | Results directory |
| `SOURCE_REPO` | no | — | Git repo to use as training data source |
| `MAT_REPO` | no | — | Install MultiAgentTrainer from this git URL instead of PyPI |
| `GITHUB_TOKEN` | no | — | For private source repos |

## Teardown

```bash
pulumi destroy
```
