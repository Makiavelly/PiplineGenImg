import json
from pathlib import Path

from PIL import Image

from historical_panorama.models import (
    Fact,
    HistoricalAnalysis,
    ImageResult,
    PromptResult,
    ReferenceSpec,
    ResearchResult,
    SceneConstraints,
    Source,
    TechnicalValidationReport,
    ValidationCheck,
    VisualValidationReport,
)
from historical_panorama.pipeline import HistoricalPanoramaPipeline


class FakeResearch:
    def __init__(self):
        self.calls = 0

    def research(self, event: str) -> ResearchResult:
        self.calls += 1
        return ResearchResult(event, [Source("Source", "https://example.test", "Historical facts")])


class FakeFacts:
    def extract(self, event, research, run_id):
        fact = Fact("Timber walls were built.", "architecture", "supported", "Source", "Lead")
        return HistoricalAnalysis(
            identified_event=event,
            facts=[fact],
            constraints=SceneConstraints(must_include=[fact.statement]),
        )


class FakePrompt:
    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult:
        assert research.context
        return PromptResult("equirectangular historical panorama", "modern objects")


class FakeImage:
    supports_image_conditioning = True

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult:
        self.calls += 1
        path = output_dir / "panorama.png"
        Image.new("RGB", (64, 32), "navy").save(path)
        metadata = {
            "run_id": run_id,
            "attempt": prompt.metadata["attempt"],
            "prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
        }
        return ImageResult(path, metadata)


class FakeProjector:
    def generate_standard_views(self, image_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        return []


class PassingValidator:
    def validate(self, *args, **kwargs):
        return TechnicalValidationReport("passed", [])


class ExplodingValidator:
    def validate(self, *args, **kwargs):
        raise AssertionError("technical validator must not be called")


class FailingValidator:
    def validate(self, *args, **kwargs):
        return TechnicalValidationReport("failed", [
            ValidationCheck("aspect_ratio_2_to_1", 1.5, 2.0, "failed", "wrong ratio")
        ])


class UnreadableValidator:
    def validate(self, *args, **kwargs):
        return TechnicalValidationReport("failed", [
            ValidationCheck("file_exists", True, True, "passed", "exists"),
            ValidationCheck("image_readable", False, True, "failed", "unreadable"),
        ])


class CountingVisualValidator:
    def __init__(self):
        self.run_ids = []

    def validate(self, frames, analysis, references, run_id):
        self.run_ids.append(run_id)
        return VisualValidationReport("passed", [], "ok")


def test_pipeline_writes_a_complete_run(tmp_path: Path):
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path,
        fact_extractor=FakeFacts(), technical_validator=PassingValidator(),
        perspective_projector=FakeProjector(),
    )
    result = pipeline.run("Test event")

    assert result.status == "accepted"
    assert result.image.path.is_file()
    assert (result.run_dir / "research.json").is_file()
    assert (result.run_dir / "analysis.json").is_file()
    assert (result.run_dir / "prompt.json").is_file()
    assert (result.run_dir / "attempts/attempt_01/technical_validation.json").is_file()
    assert (result.run_dir / "manifest.json").is_file()


def test_pipeline_resume_reuses_research_prompt_and_generation(tmp_path: Path):
    research, image = FakeResearch(), FakeImage()
    pipeline = HistoricalPanoramaPipeline(
        research, FakePrompt(), image, tmp_path,
        fact_extractor=FakeFacts(), technical_validator=PassingValidator(),
        perspective_projector=FakeProjector(),
    )
    first = pipeline.run("Test event")
    second = pipeline.run("", resume_dir=first.run_dir)

    assert second.status == "accepted"
    assert research.calls == 1
    assert image.calls == 1


def test_pipeline_limits_full_regeneration_and_changes_retry_prompt(tmp_path: Path):
    image = FakeImage()
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), image, tmp_path,
        fact_extractor=FakeFacts(), technical_validator=FailingValidator(),
        perspective_projector=FakeProjector(), pipeline_config={"max_full_generations": 3},
    )
    result = pipeline.run("Test event")

    assert result.status == "manual_review_required"
    assert result.attempts == 3
    assert image.calls == 3
    prompts = [
        (result.run_dir / f"attempts/attempt_{attempt:02d}/prompt.json").read_text()
        for attempt in (1, 2, 3)
    ]
    assert prompts[0] != prompts[1] != prompts[2]


def test_pipeline_accepts_valid_reference_path_without_json_serialization_error(tmp_path: Path):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "red").save(reference)
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path / "runs",
        fact_extractor=FakeFacts(), technical_validator=PassingValidator(),
        perspective_projector=FakeProjector(),
    )

    result = pipeline.run(
        "Test event", [ReferenceSpec(reference, ["clothing"], ["background"])]
    )

    assert result.status == "accepted"
    assert str(reference) in (result.run_dir / "prompt.json").read_text(encoding="utf-8")


def test_pipeline_skips_projection_for_unreadable_generation(tmp_path: Path):
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path,
        fact_extractor=FakeFacts(), technical_validator=UnreadableValidator(),
        perspective_projector=FakeProjector(), pipeline_config={"max_full_generations": 1},
    )
    result = pipeline.run("Test event")
    report = (result.run_dir / "attempts/attempt_01/visual_validation.json").read_text()
    assert result.status == "manual_review_required"
    assert "visual_validation_unavailable" in report


def test_pipeline_can_disable_visual_validation(tmp_path: Path):
    visual = CountingVisualValidator()
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path,
        fact_extractor=FakeFacts(), technical_validator=PassingValidator(),
        perspective_projector=FakeProjector(), visual_validator=visual,
        pipeline_config={"visual_validation_runs": 0},
    )

    result = pipeline.run("Test event")

    assert result.status == "accepted"
    assert visual.run_ids == []
    report = json.loads(
        (result.run_dir / "attempts/attempt_01/visual_validation.json").read_text()
    )
    assert report["status"] == "visual_validation_unavailable"
    assert "disabled" in report["explanation"]


def test_pipeline_can_disable_technical_validation(tmp_path: Path):
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path,
        fact_extractor=FakeFacts(), technical_validator=ExplodingValidator(),
        perspective_projector=FakeProjector(),
        pipeline_config={
            "technical_validation_enabled": False,
            "visual_validation_runs": 0,
        },
    )

    result = pipeline.run("Test event")

    assert result.status == "accepted"
    report = json.loads(
        (result.run_dir / "attempts/attempt_01/technical_validation.json").read_text()
    )
    assert report["status"] == "disabled"
    assert report["checks"] == []
    assert "disabled" in report["explanation"]
    final = json.loads((result.run_dir / "final_report.json").read_text())
    assert final["technical_validation_status"] == "disabled"


def test_pipeline_runs_and_saves_requested_visual_validation_count(tmp_path: Path):
    visual = CountingVisualValidator()
    pipeline = HistoricalPanoramaPipeline(
        FakeResearch(), FakePrompt(), FakeImage(), tmp_path,
        fact_extractor=FakeFacts(), technical_validator=PassingValidator(),
        perspective_projector=FakeProjector(), visual_validator=visual,
        pipeline_config={"visual_validation_runs": 3},
    )

    result = pipeline.run("Test event")

    assert result.status == "accepted"
    assert len(visual.run_ids) == 3
    assert len(set(visual.run_ids)) == 3
    attempt = result.run_dir / "attempts/attempt_01"
    assert all(
        (attempt / f"visual_validation_{number:02d}.json").is_file()
        for number in (1, 2, 3)
    )
    aggregate = json.loads((attempt / "visual_validation.json").read_text())
    assert aggregate["status"] == "passed"
    assert "Aggregated 3" in aggregate["explanation"]
