from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Confidence = Literal["supported", "inferred", "unknown"]
ValidationStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    text: str
    section: str = "Lead"
    article_title: str = ""
    is_primary: bool = False
    relevance: float = 0.0


@dataclass(frozen=True)
class WikipediaEventMetadata:
    title: str = ""
    date_or_period: str = ""
    place: str = ""
    participants: list[str] = field(default_factory=list)
    summary: str = ""
    visual_reconstruction_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchResult:
    query: str
    sources: list[Source] = field(default_factory=list)
    primary_article: str = ""
    related_articles: list[str] = field(default_factory=list)
    metadata: WikipediaEventMetadata = field(default_factory=WikipediaEventMetadata)
    limitations: list[str] = field(default_factory=list)
    analysis: HistoricalAnalysis | None = None
    references: ReferenceReport | None = None

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"ARTICLE: {item.article_title or item.title}\nSECTION: {item.section}\n"
            f"URL: {item.url}\n{item.text}"
            for item in self.sources
        )


@dataclass(frozen=True)
class Fact:
    statement: str
    category: str
    confidence: Confidence
    article: str
    section: str


@dataclass(frozen=True)
class SceneConstraints:
    must_include: list[str] = field(default_factory=list)
    may_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    do_not_over_specify: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HistoricalAnalysis:
    identified_event: str = ""
    date_or_period: str = ""
    place: str = ""
    participants: list[str] = field(default_factory=list)
    event_type: str = "unknown"
    environment: list[str] = field(default_factory=list)
    architecture: list[str] = field(default_factory=list)
    clothing: list[str] = field(default_factory=list)
    weapons: list[str] = field(default_factory=list)
    transport: list[str] = field(default_factory=list)
    everyday_objects: list[str] = field(default_factory=list)
    natural_features: list[str] = field(default_factory=list)
    visual_actions: list[str] = field(default_factory=list)
    unknown_or_disputed: list[str] = field(default_factory=list)
    possible_anachronisms: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    constraints: SceneConstraints = field(default_factory=SceneConstraints)


@dataclass(frozen=True)
class ReferenceSpec:
    path: Path
    use_for: list[str] = field(default_factory=list)
    do_not_copy: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceInfo:
    path: str
    use_for: list[str] = field(default_factory=list)
    do_not_copy: list[str] = field(default_factory=list)
    valid: bool = False
    width: int | None = None
    height: int | None = None
    format: str = ""
    description: str = ""
    description_status: str = "not_requested"
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceReport:
    references: list[ReferenceInfo] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_prompt: str
    metadata: dict[str, object] = field(default_factory=dict)
    structured_description: dict[str, object] = field(default_factory=dict)
    used_facts: list[Fact] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    reference_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class ImageResult:
    path: Path
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    measured_value: object
    threshold: object
    status: ValidationStatus
    explanation: str


@dataclass(frozen=True)
class TechnicalValidationReport:
    status: Literal["passed", "failed", "disabled"]
    checks: list[ValidationCheck] = field(default_factory=list)
    explanation: str = ""


@dataclass(frozen=True)
class PerspectiveFrame:
    frame_number: int
    yaw: float
    pitch: float
    path: str


@dataclass(frozen=True)
class VisualIssue:
    error_type: str
    description: str
    frame_number: int | None
    yaw: float | None
    pitch: float | None
    severity: Literal["warning", "major", "critical"]
    scope: Literal["local", "global"]
    suggested_fix: str


@dataclass(frozen=True)
class VisualValidationReport:
    status: str
    issues: list[VisualIssue] = field(default_factory=list)
    explanation: str = ""


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    research: ResearchResult
    prompt: PromptResult
    image: ImageResult
    status: str = "accepted"
    attempts: int = 1
