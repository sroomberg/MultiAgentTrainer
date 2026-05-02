# MultiAgentTrainer

> **⚠️ Experimental** — This project is under active development. APIs, config format, and CLI flags may change without notice.

Collect data from multiple sources and run autonomous LLM training experiments using [autoresearch](https://github.com/karpathy/autoresearch). Configure your data sources in a YAML file, and `mat` handles ingestion, corpus building, and launching autonomous training runs.

MultiAgentTrainer can be used standalone, but is designed as a companion to [AgentTester](https://github.com/sroomberg/agenttester) — use AgentTester to evaluate and compare coding agents, then use MultiAgentTrainer to train models on the data those agents produce and consume.

## Install

```bash
uv pip install -e ".[dev]"
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
```

## Configuration

Copy `config.example.yaml` to `multiagenttrainer.yaml` in your working directory.

### Top-level sections

| Section | Description |
|---------|-------------|
| `autoresearch` | Autoresearch repo URL/path, branch, train time, optional `program.md` override |
| `sources` | List of data sources to ingest |
| `training` | Agent command, max experiments, output directory |

### Source Types

| Type | Required Fields | Optional Fields |
|------|----------------|-----------------|
| `local_repo` | `path` | `include`, `exclude`, `name` |
| `remote_repo` | `url` | `branch`, `name` |
| `github_repo` | `url` | `branch`, `name` |
| `github_org` | `url` | `max_repos`, `visibility`, `name` |
| `bedrock_knowledge_base` | `knowledge_base_id` | `region`, `query`, `max_results`, `name` |

## How It Works

1. **Ingest** — Fetch data from all configured sources (clone repos, query Bedrock KBs)
2. **Build corpus** — Walk fetched files, filter by include/exclude globs, concatenate into a single corpus
3. **Setup** — Clone autoresearch, inject the corpus, optionally override `program.md`
4. **Train** — Launch the agent command iteratively for up to `max_experiments` rounds
5. **Report** — Generate a markdown report with experiment results, best `val_bpb`, and stats

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

    runner = Runner(cfg.autoresearch, cfg.training)
    workspace = runner.setup_workspace(Path("corpus.txt"))
    results = await runner.run_experiments(workspace)
    for r in results:
        print(f"experiment {r.experiment_id}: val_bpb={r.val_bpb}")

asyncio.run(main())
```
