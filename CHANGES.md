# Changelog

## v1.1.2 — 2026-05-12
- Add `GitHubRepoListSource` source type for ingesting an explicit list of repositories; accepts `owner/repo` shorthands or full URLs
- Expose `--repo` / `-r` flag on `mat train`, `mat ingest`, and `mat sources` commands
- Deduplicate executor `run()` timeout/decode logic across `LocalExecutor`, `SSHExecutor`, and `DockerExecutor`
- Consolidate staging-dir preparation into `DataSource._prepare_dest()`
- Extract `FineTuneJob._safe_id()` to eliminate duplicated job-ID sanitisation
- Extract `DataSource._branch_suffix()` to deduplicate `describe()` formatting

## v1.1.0 — 2026-05-09
- Fix TRL 0.29+ API compatibility
- Improve training defaults
- Add HF token support and gradient checkpointing toggle

## v1.0.1 — 2026-05-08
- Add README

## v1.0.0 — 2026-05-08
- Initial release
