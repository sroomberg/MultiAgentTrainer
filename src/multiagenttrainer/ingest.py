"""Ingest data from configured sources into a training corpus."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

from rich.console import Console

from .sources import DataSource

log = logging.getLogger(__name__)

# File extensions treated as text for corpus building.
_TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".zsh",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".kt",
    ".rb",
    ".pl",
    ".lua",
    ".r",
    ".sql",
    ".html",
    ".css",
    ".xml",
    ".csv",
}


class Ingester:
    """Fetch all data sources and assemble a text corpus."""

    def __init__(
        self,
        sources: list[DataSource],
        staging_dir: Path,
        console: Console | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self.sources = sources
        self.staging_dir = staging_dir
        self.console = console or Console()
        self.include = include or ["**/*"]
        self.exclude = exclude or []

    def fetch_all(self) -> list[Path]:
        """Fetch every source into the staging directory.

        Returns the list of top-level paths produced.
        """
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        fetched: list[Path] = []

        for src in self.sources:
            self.console.print(f"  [dim]fetching:[/dim] {src.describe()}")
            try:
                path = src.fetch(self.staging_dir)
                fetched.append(path)
                self.console.print(f"  [green]✓[/green] {src.describe()}")
            except Exception as e:
                self.console.print(f"  [red]✗ {src.describe()}: {e}[/red]")
                log.warning("Failed to fetch %s", src.describe(), exc_info=True)

        return fetched

    def build_corpus(self, output_path: Path) -> int:
        """Walk the staging directory and concatenate text files into a corpus.

        Returns the number of files included.
        """
        files = self._collect_files()
        count = 0

        with open(output_path, "w") as out:
            for fp in sorted(files):
                try:
                    text = fp.read_text(errors="replace")
                except Exception:
                    log.debug("Skipping unreadable file: %s", fp)
                    continue

                # Write with a file-path header so the model sees provenance.
                rel = fp.relative_to(self.staging_dir)
                out.write(f"\n### FILE: {rel}\n\n")
                out.write(text)
                out.write("\n")
                count += 1

        self.console.print(f"  [bold]Corpus:[/bold] {count} files → {output_path}")
        return count

    # ------------------------------------------------------------------

    def _collect_files(self) -> list[Path]:
        """Collect text files from the staging dir, respecting globs."""
        result: list[Path] = []

        for path in self.staging_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _TEXT_EXTENSIONS:
                continue

            rel = str(path.relative_to(self.staging_dir))

            # Include filter
            if not any(fnmatch.fnmatch(rel, pat) for pat in self.include):
                continue

            # Exclude filter
            if any(fnmatch.fnmatch(rel, pat) for pat in self.exclude):
                continue

            result.append(path)

        return result
