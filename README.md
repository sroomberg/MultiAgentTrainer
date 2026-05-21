# MultiAgentTrainer

Collect data from multiple sources, run autonomous LLM training experiments using [autoresearch](https://github.com/karpathy/autoresearch), and fine-tune open-source or managed models on your corpus. Configure everything in a YAML file and let `mat` handle ingestion, corpus building, training, and fine-tuning.

MultiAgentTrainer can be used standalone, but is designed as a companion to [AgentTester](https://github.com/sroomberg/agenttester) — use AgentTester to evaluate and compare coding agents, then use MultiAgentTrainer to train models on the data those agents produce and consume.

## Install

```bash
uv pip install -e ".[dev]"

# For open-source fine-tuning (HuggingFace + PEFT/LoRA):
uv pip install -e ".[opensource]"
```

## Quick Start

```bash
# List configured data sources
mat sources

# Ingest data sources without training (inspect the corpus)
mat ingest

# Run the full pipeline: ingest → corpus → train
mat train

# Run with overrides
mat train --max-experiments 10 --output-dir ./my-runs

# Label a run so it shows up clearly in mat watch
mat train --name llama3

# Ingest extra repos on the fly (owner/repo or full URL; repeatable)
mat train --repo myorg/myrepo --repo https://github.com/user/other.git

# Run multiple models in parallel and watch progress live
mat train --config llama3.yaml --name llama3 &
mat train --config mistral.yaml --name mistral &
mat watch

# Check past training runs
mat status
```

## Data Sources

Configure data sources in `multiagenttrainer.yaml`:

```yaml
sources:
  # Local git repository
  - type: local_repo
    path: /home/user/my-project

  # Any git-cloneable URL
  - type: remote_repo
    url: "https://github.com/user/repo.git"
    branch: main

  # GitHub repository (web URL)
  - type: github_repo
    url: "https://github.com/user/repo"

  # All repos in a GitHub organisation
  - type: github_org
    url: "https://github.com/my-org"
    max_repos: 50
    visibility: all   # all | public | private

  # AWS Bedrock knowledge base
  - type: bedrock_knowledge_base
    knowledge_base_id: "ABCDEF1234"
    region: "us-east-1"
    query: "training data for code generation"
    max_results: 100

  # Explicit list of repositories (owner/repo shorthands or full URLs)
  - type: github_repo_list
    repos:
      - myorg/myrepo
      - https://github.com/user/other.git
    branch: main  # optional; applied to all repos
```

You can also pass repos on the command line without editing the config:

```bash
mat train --repo myorg/myrepo --repo user/other
mat ingest --repo myorg/myrepo
mat sources --repo myorg/myrepo
```

## Configuration

Copy `config.example.yaml` to `multiagenttrainer.yaml` in your working directory.

### Top-level sections

| Section | Description |
|---------|-------------|
| `autoresearch` | Autoresearch repo URL/path, branch, train time, optional `program.md` override |
| `sources` | List of data sources to ingest |
| `training` | Agent command, max experiments, output directory, execution target |
| `machines` | Named execution targets for distributing training across multiple hosts |
| `notifications` | Failure alert channels (currently AWS SES) |
| `finetuner` | Fine-tuning backend and job configuration |

### Execution targets

By default experiments run locally. Set `training.execution.type` to run on a remote host or inside a container instead.

**SSH** — rsync the workspace to a remote machine and run experiments over SSH:

```yaml
training:
  execution:
    type: ssh
    ssh_host: user@gpu-box.example.com   # required
    ssh_key: ~/.ssh/id_ed25519           # optional; uses SSH default otherwise
    remote_dir: /tmp/mat-runs            # base dir on the remote host
```

**Docker** — copy the workspace into a running container and exec commands inside it:

```yaml
training:
  execution:
    type: docker
    container: my-training-container     # required; must already be running
    container_dir: /tmp/mat-runs         # base dir inside the container
```

### Multi-Machine Training

Define named machines in the top-level `machines` block. `mat train` distributes experiments concurrently across all machines — each machine runs its own sequential batch via `asyncio.gather`. Omit this section to run everything locally.

```yaml
machines:
  - name: gpu-large
    execution:
      type: ssh
      ssh_host: trainer1.example.com
      ssh_key: ~/.ssh/id_ed25519
      remote_dir: /tmp/mat-runs
    # Use a larger model on the beefier host
    agent_command: >-
      claude -p {prompt}
      --allowedTools "Bash,Read,Edit"
      --permission-mode acceptEdits
      --model claude-opus-4-7

  - name: gpu-small
    execution:
      type: ssh
      ssh_host: trainer2.example.com
      remote_dir: /tmp/mat-runs
    agent_command: >-
      claude -p {prompt}
      --allowedTools "Bash,Read,Edit"
      --permission-mode acceptEdits
      --model claude-haiku-4-5
```

Each `MachineConfig` supports:

| Field | Description |
|-------|-------------|
| `name` | Unique identifier (referenced by `finetuner.targets`) |
| `execution` | Same `ExecutionConfig` as `training.execution` |
| `agent_command` | Overrides the global `training.agent_command` for this host |

### Notifications

Send failure alerts when a training experiment or fine-tuning job fails. Currently supports AWS SES. Requires `boto3` and a verified SES sender address.

```yaml
notifications:
  ses:
    from_email: alerts@example.com
    to_emails:
      - oncall@example.com
    region: us-east-1
    subject_prefix: "[MultiAgentTrainer]"  # prepended to the email subject
```

### Source Types

| Type | Required Fields | Optional Fields |
|------|----------------|-----------------|
| `local_repo` | `path` | `include`, `exclude`, `name` |
| `remote_repo` | `url` | `branch`, `name` |
| `github_repo` | `url` | `branch`, `name` |
| `github_org` | `url` | `max_repos`, `visibility`, `name` |
| `github_repo_list` | `repos` | `branch`, `name` |
| `bedrock_knowledge_base` | `knowledge_base_id` | `region`, `query`, `max_results`, `name` |

## Fine-Tuning

Fine-tune models directly on your ingested corpus using `mat finetune`. Two backends are supported today; more can be added by subclassing `FineTuner`.

### Open-source models (HuggingFace + LoRA/QLoRA)

Requires a local GPU and `pip install 'multiagenttrainer[opensource]'`.

```yaml
# multiagenttrainer.yaml
finetuner:
  backend: opensource
  jobs_dir: ./finetune-jobs
  opensource:
    model_id: meta-llama/Llama-3.2-1B
    output_dir: ./finetuned-models
    lora_r: 16
    lora_alpha: 32
    num_epochs: 3
    batch_size: 4
    use_4bit: true          # QLoRA — requires bitsandbytes + CUDA
```

```bash
# Ingest sources first (or skip if you already have a corpus)
mat ingest

# Fine-tune on the ingested corpus
mat finetune start

# Or point at an arbitrary corpus file
mat finetune start --corpus /path/to/corpus.txt --name my-run

# List all jobs
mat finetune list

# Check a job
mat finetune status <job-id>
```

Training runs in-process and blocks until complete. The LoRA adapter and tokenizer are saved to `output_dir/<job-id>/`.

### AWS Bedrock model customization

Uses your existing `boto3` credentials. Submits a Bedrock customization job and returns immediately — poll with `mat finetune status`.

```yaml
finetuner:
  backend: bedrock
  jobs_dir: ./finetune-jobs
  bedrock:
    base_model_id: amazon.titan-text-lite-v1
    region: us-east-1
    role_arn: arn:aws:iam::123456789012:role/BedrockFineTuningRole
    output_s3_uri: s3://my-bucket/finetuned-models/
    training_data_s3_uri: s3://my-bucket/training-data/
    customization_type: CONTINUED_PRE_TRAINING   # or FINE_TUNING
    epochs: 1
```

```bash
mat finetune start
mat finetune status <job-arn>
mat finetune cancel <job-arn>
```

### Multi-target fine-tuning

Define a `targets` list under `finetuner` to run multiple (model, machine) pairings concurrently. Each target inherits the backend defaults from `finetuner.opensource` or `finetuner.bedrock` and overrides only the fields it declares.

```yaml
finetuner:
  backend: opensource
  jobs_dir: ./finetune-jobs
  opensource:
    model_id: meta-llama/Llama-3.2-1B   # default — targets can override
    output_dir: ./finetuned-models
    num_epochs: 3

  targets:
    - name: llama-3b-large
      model_id: meta-llama/Llama-3.2-3B
      machine: gpu-large        # metadata — documents which host this is sized for
      backend: opensource
      num_epochs: 3
      batch_size: 2

    - name: llama-1b-small
      model_id: meta-llama/Llama-3.2-1B
      machine: gpu-small
      backend: opensource
      num_epochs: 5
      batch_size: 4

    - name: titan-bedrock
      model_id: amazon.titan-text-lite-v1
      backend: bedrock
      customization_type: FINE_TUNING
```

`mat finetune start` starts all targets concurrently via `ThreadPoolExecutor`. Each target gets its own subdirectory under `jobs_dir`.

| Target field | Description |
|--------------|-------------|
| `name` | Label for this target |
| `model_id` | Model to fine-tune (overrides backend default) |
| `machine` | Name of a `machines` entry (informational) |
| `backend` | `opensource` or `bedrock` (overrides `finetuner.backend`) |
| `num_epochs` / `batch_size` / `lora_r` | OpenSource overrides |
| `customization_type` | Bedrock override |

### Adding a new backend

Subclass `FineTuner`, implement the four abstract methods, add a config dataclass, and register it in `finetuner/registry.py`:

```python
# finetuner/finetuner.py
class AnthropicFineTuner(FineTuner):
    def prepare_dataset(self, corpus_path): ...
    def start_job(self, dataset, job_name): ...
    def get_status(self, job_id): ...
    def cancel_job(self, job_id): ...
    def describe(self): ...

# finetuner/registry.py
if cfg.backend == "anthropic":
    return AnthropicFineTuner(cfg.anthropic, jobs_dir, console)
```

## How It Works

1. **Ingest** — Fetch data from all configured sources (clone repos, query Bedrock KBs)
2. **Build corpus** — Walk fetched files, filter by include/exclude globs, concatenate into a single corpus
3. **Setup** — Clone autoresearch, inject the corpus, optionally override `program.md`
4. **Train** — Launch the agent command iteratively for up to `max_experiments` rounds
5. **Report** — Generate a markdown report with experiment results, best `val_bpb`, and stats
6. **Fine-tune** *(optional)* — Run `mat finetune start` to fine-tune a model on the same corpus

## Development

```bash
uv pip install -e ".[dev]"
ruff check src/ tests/
ruff format src/ tests/
pytest
```

## Docker

```bash
docker compose run --rm mat train
docker compose run --rm mat sources
```

## Library Usage

```python
import asyncio
from pathlib import Path
from multiagenttrainer import Ingester, Runner, load_config

async def main():
    cfg = load_config()
    ingester = Ingester(cfg.sources, Path(".staging"))
    ingester.fetch_all()
    ingester.build_corpus(Path("corpus.txt"))

    runner = Runner(cfg.autoresearch, cfg.training, name="my-run")
    workspace = runner.setup_workspace(Path("corpus.txt"))
    results = await runner.run_experiments(workspace)
    for r in results:
        print(f"experiment {r.experiment_id}: val_bpb={r.val_bpb}")

asyncio.run(main())
```
