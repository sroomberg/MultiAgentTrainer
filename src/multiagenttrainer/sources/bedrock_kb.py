"""Data source for AWS Bedrock knowledge bases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import DataSource

log = logging.getLogger(__name__)


class BedrockKnowledgeBaseSource(DataSource):
    """Retrieve content from an AWS Bedrock knowledge base.

    Uses the ``Retrieve`` API to pull relevant chunks, writing each
    result to a separate text file in the staging directory.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        region: str = "us-east-1",
        query: str = "training data",
        max_results: int = 100,
        name: str | None = None,
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.region = region
        self.query = query
        self.max_results = max_results
        self.name = name or f"bedrock-{knowledge_base_id}"

    def fetch(self, staging_dir: Path) -> Path:
        import boto3  # late import so boto3 is only required when used

        dest = self._prepare_dest(staging_dir)

        client = boto3.client("bedrock-agent-runtime", region_name=self.region)

        results: list[dict[str, object]] = []
        next_token: str | None = None

        while len(results) < self.max_results:
            kwargs: dict[str, object] = {
                "knowledgeBaseId": self.knowledge_base_id,
                "retrievalQuery": {"text": self.query},
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": min(self.max_results - len(results), 100),
                    }
                },
            }
            if next_token:
                kwargs["nextToken"] = next_token

            response = client.retrieve(**kwargs)
            chunks = response.get("retrievalResults", [])
            if not chunks:
                break

            results.extend(chunks)
            next_token = response.get("nextToken")
            if not next_token:
                break

        log.info(
            "Retrieved %d chunks from Bedrock KB %s",
            len(results),
            self.knowledge_base_id,
        )

        # Write each chunk to a file.
        for i, chunk in enumerate(results):
            content = chunk.get("content", {}).get("text", "")
            metadata = chunk.get("metadata", {})
            location = chunk.get("location", {})

            doc = {
                "content": content,
                "metadata": metadata,
                "location": location,
            }
            file_path = dest / f"chunk_{i:04d}.json"
            file_path.write_text(json.dumps(doc, indent=2))

            # Also write raw text for corpus building.
            if content:
                text_path = dest / f"chunk_{i:04d}.txt"
                text_path.write_text(content)

        return dest

    def describe(self) -> str:
        return (
            f"bedrock_kb: {self.knowledge_base_id} "
            f"(region={self.region}, max_results={self.max_results})"
        )
