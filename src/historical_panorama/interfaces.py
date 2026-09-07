from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import ImageResult, PromptResult, ResearchResult


class InformationProvider(Protocol):
    def research(self, event: str) -> ResearchResult: ...


class PromptBuilder(Protocol):
    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult: ...


class ImageGenerator(Protocol):
    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult: ...

