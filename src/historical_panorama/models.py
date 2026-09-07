from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class ResearchResult:
    query: str
    sources: list[Source] = field(default_factory=list)

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"SOURCE: {item.title}\nURL: {item.url}\n{item.text}" for item in self.sources
        )


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    negative_prompt: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageResult:
    path: Path
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    run_dir: Path
    research: ResearchResult
    prompt: PromptResult
    image: ImageResult

