from pathlib import Path

from PIL import Image

from historical_panorama.models import ImageResult, PromptResult, ResearchResult, Source
from historical_panorama.pipeline import HistoricalPanoramaPipeline


class FakeResearch:
    def research(self, event: str) -> ResearchResult:
        return ResearchResult(event, [Source("Source", "https://example.test", "Historical facts")])


class FakePrompt:
    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult:
        assert research.context
        return PromptResult("equirectangular historical panorama", "modern objects")


class FakeImage:
    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult:
        path = output_dir / "panorama.png"
        Image.new("RGB", (64, 32), "navy").save(path)
        return ImageResult(path, {"run_id": run_id})


def test_pipeline_writes_a_complete_run(tmp_path: Path):
    pipeline = HistoricalPanoramaPipeline(FakeResearch(), FakePrompt(), FakeImage(), tmp_path)
    result = pipeline.run("Test event")

    assert result.image.path.is_file()
    assert (result.run_dir / "research.json").is_file()
    assert (result.run_dir / "prompt.json").is_file()
    assert (result.run_dir / "manifest.json").is_file()

