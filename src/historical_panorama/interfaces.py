from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    HistoricalAnalysis,
    ImageResult,
    PerspectiveFrame,
    PromptResult,
    ReferenceReport,
    ResearchResult,
    VisualValidationReport,
)


class InformationProvider(Protocol):
    def research(self, event: str) -> ResearchResult: ...


class HistoricalFactExtractor(Protocol):
    def extract(self, event: str, research: ResearchResult, run_id: str) -> HistoricalAnalysis: ...


class PromptBuilder(Protocol):
    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult: ...


class ImageGenerator(Protocol):
    supports_image_conditioning: bool

    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult: ...


class VisualValidator(Protocol):
    def validate(
        self,
        frames: list[PerspectiveFrame],
        analysis: HistoricalAnalysis,
        references: ReferenceReport,
        run_id: str,
    ) -> VisualValidationReport: ...


class ReferenceAnalyzer(Protocol):
    def analyze_references(self, report: ReferenceReport, run_id: str) -> ReferenceReport: ...
