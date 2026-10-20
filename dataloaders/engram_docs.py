"""Loader for ENGRAM repository QA datasets (`dataset.json` of the Repo Dataset Builder)."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EngramDocsRetrievalData:
    """Retrieval data built from one repository's questions and cited records."""

    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    relevant_docs: dict[str, dict[str, int]]


@dataclass
class EngramDocsDatasetLoader:
    """Build a retrieval dataset from an ENGRAM `dataset.json`.

    The corpus is every record cited by some item, keyed by its display label.
    Citations are listed most relevant first, so relevance is graded by rank:
    with five citations the first scores 5 and the last 1.
    """

    dataset_path: str | Path
    encoding: str = "utf-8"

    def load(self) -> EngramDocsRetrievalData:
        """Parse the items into queries, cited records and relevance judgements."""
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"ENGRAM dataset not found: {dataset_path}")

        dataset = json.loads(dataset_path.read_text(encoding=self.encoding))

        corpus: dict[str, str] = {}
        queries: dict[str, str] = {}
        relevant_docs: dict[str, dict[str, int]] = {}

        for item in dataset["items"]:
            query_id = str(item["id"])
            labels = [str(label) for label in item["citation_labels"]]
            citations = [str(citation).strip() for citation in item["citations"]]
            if len(labels) != len(citations):
                raise ValueError(
                    f"{query_id}: {len(labels)} citation_labels but "
                    f"{len(citations)} citations"
                )

            for label, text in zip(labels, citations):
                if corpus.setdefault(label, text) != text:
                    raise ValueError(f"{query_id}: {label} differs from its earlier citation")

            queries[query_id] = str(item["question"]).strip()
            relevant_docs[query_id] = {
                label: len(labels) - rank for rank, label in enumerate(labels)
            }

        return EngramDocsRetrievalData(
            name=str(dataset["repository"]["name"]),
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
        )
