"""Data schemas for Groundtruth retrieval evaluation.

Uses Pydantic V2 for strict validation, serialization, and data integrity.
"""

from pathlib import Path
from typing import Any
import json
from pydantic import BaseModel, Field


class Passage(BaseModel):
    """Represents a single document passage or chunk in the legal corpus."""

    id: str = Field(description="Unique identifier for the passage (e.g. 'doc_contract_001')")
    title: str = Field(description="Case title, filing name, or document heading")
    category: str = Field(description="Legal practice area / category (e.g. 'contract_dispute')")
    text: str = Field(description="Full text content of the passage")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata such as jurisdiction, court, year, docket",
    )


class GoldenQuery(BaseModel):
    """Represents a benchmark query with human-verified ground truth relevance."""

    id: str = Field(description="Unique query identifier (e.g. 'q_contract_001')")
    query: str = Field(description="Attorney research question")
    category: str = Field(description="Query category matching legal practice area")
    relevant_passages: dict[str, int] = Field(
        description="Mapping from passage_id -> relevance grade (1 = low, 2 = medium, 3 = critical/landmark)"
    )

    @property
    def relevant_ids(self) -> set[str]:
        """Returns set of all relevant passage IDs (for Recall@k and MRR)."""
        return set(self.relevant_passages.keys())

    @property
    def relevance_scores(self) -> dict[str, int]:
        """Returns dict of passage_id -> graded score (for nDCG@k)."""
        return self.relevant_passages


class EvaluationDataset(BaseModel):
    """Complete versioned evaluation dataset containing the corpus and golden queries."""

    version: str = Field(default="1.0", description="Dataset schema version")
    description: str = Field(
        default="Legal Research Assistant Golden Evaluation Dataset",
        description="Dataset description and provenance notes",
    )
    passages: list[Passage] = Field(description="Corpus of legal passages to be indexed")
    queries: list[GoldenQuery] = Field(description="Golden test queries with relevance annotations")

    def get_passage_by_id(self, passage_id: str) -> Passage | None:
        """Lookup a passage by its unique ID."""
        for p in self.passages:
            if p.id == passage_id:
                return p
        return None

    def get_queries_by_category(self, category: str) -> list[GoldenQuery]:
        """Filter queries by a specific legal category."""
        return [q for q in self.queries if q.category == category]

    def get_all_categories(self) -> list[str]:
        """Return unique sorted list of all query categories."""
        return sorted(list(set(q.category for q in self.queries)))

    def save_to_json(self, path: str | Path) -> None:
        """Serialize dataset to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path: str | Path) -> "EvaluationDataset":
        """Load and validate dataset from a JSON file."""
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        return cls.model_validate_json(content)
