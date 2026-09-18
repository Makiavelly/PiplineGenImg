from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controller import RetryController
from .interfaces import (
    HistoricalFactExtractor,
    ImageGenerator,
    InformationProvider,
    PromptBuilder,
    ReferenceAnalyzer,
    VisualValidator,
)
from .models import (
    Fact,
    HistoricalAnalysis,
    ImageResult,
    PipelineResult,
    PromptResult,
    ReferenceInfo,
    ReferenceReport,
    ReferenceSpec,
    ResearchResult,
    SceneConstraints,
    Source,
    VisualIssue,
    VisualValidationReport,
    WikipediaEventMetadata,
)
from .perspective import PerspectiveProjector
from .references import ReferenceProcessor
from .schemas import historical_analysis_from_dict
from .state import RunState
from .validation import TechnicalPanoramaValidator
from .visual_validation import UnavailableVisualValidator

LOG = logging.getLogger(__name__)


class HistoricalPanoramaPipeline:
    def __init__(
        self,
        information_provider: InformationProvider,
        prompt_builder: PromptBuilder,
        image_generator: ImageGenerator,
        output_root: Path,
        *,
        fact_extractor: HistoricalFactExtractor | None = None,
        reference_processor: ReferenceProcessor | None = None,
        technical_validator: TechnicalPanoramaValidator | None = None,
        perspective_projector: PerspectiveProjector | None = None,
        visual_validator: VisualValidator | None = None,
        reference_analyzer: ReferenceAnalyzer | None = None,
        retry_controller: RetryController | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ):
        config = pipeline_config or {}
        self.information_provider = information_provider
        self.fact_extractor = fact_extractor
        self.prompt_builder = prompt_builder
        self.image_generator = image_generator
        self.output_root = output_root
        self.reference_processor = reference_processor or ReferenceProcessor()
        self.technical_validator = technical_validator or TechnicalPanoramaValidator()
        self.perspective_projector = perspective_projector or PerspectiveProjector(
            frame_size=int(config.get("perspective_frame_size", 512))
        )
        self.visual_validator = visual_validator or UnavailableVisualValidator()
        self.reference_analyzer = reference_analyzer
        self.retry_controller = retry_controller or RetryController(
            max_full_generations=int(config.get("max_full_generations", 3))
        )

    def run(
        self,
        event: str,
        references: list[ReferenceSpec] | None = None,
        resume_dir: Path | None = None,
    ) -> PipelineResult:
        event = event.strip()
        if not event and resume_dir is None:
            raise ValueError("Event name cannot be empty")
        run_dir, run_id, event = self._start_or_resume(event, resume_dir)
        state = RunState(run_dir)

        research_path = run_dir / "research.json"
        if research_path.is_file():
            research = self._load_research(research_path)
            LOG.info("[resume] Reusing Wikipedia materials")
        else:
            LOG.info("[1/7] Researching Wikipedia: %s", event)
            research = self.information_provider.research(event)
            state.write_json(research_path, asdict(research))
            state.update(stage="wikipedia_complete")

        references_path = run_dir / "references.json"
        if references_path.is_file():
            reference_report = self._load_references(references_path)
            LOG.info("[resume] Reusing reference report")
        else:
            LOG.info("[2/7] Validating %d reference image(s)", len(references or []))
            reference_report = self.reference_processor.process(references)
            needs_analysis = any(
                item.valid and item.use_for for item in reference_report.references
            )
            if needs_analysis and self.reference_analyzer is not None:
                try:
                    reference_report = self.reference_analyzer.analyze_references(
                        reference_report, run_id
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    reference_report = replace(
                        reference_report,
                        limitations=reference_report.limitations + [
                            "Reference image analysis failed; continuing without descriptions: "
                            f"{type(exc).__name__}: {exc}"
                        ],
                    )
            elif needs_analysis:
                reference_report = replace(
                    reference_report,
                    limitations=reference_report.limitations + [
                        "Reference image analysis unavailable: no multimodal analyzer is configured"
                    ],
                )
            state.write_json(references_path, asdict(reference_report))
            state.update(stage="references_complete")

        analysis_path = run_dir / "analysis.json"
        if analysis_path.is_file():
            analysis = historical_analysis_from_dict(json.loads(analysis_path.read_text(encoding="utf-8")))
            LOG.info("[resume] Reusing structured historical analysis")
        else:
            LOG.info("[3/7] Extracting source-grounded historical facts")
            analysis = (
                self.fact_extractor.extract(event, research, run_id)
                if self.fact_extractor
                else self._legacy_analysis(event, research)
            )
            state.write_json(analysis_path, asdict(analysis))
            state.write_json(run_dir / "constraints.json", asdict(analysis.constraints))
            state.update(stage="analysis_complete")

        research = replace(research, analysis=analysis, references=reference_report)
        prompt_path = run_dir / "prompt.json"
        if prompt_path.is_file():
            prompt = self._load_prompt(prompt_path)
            LOG.info("[resume] Reusing generated prompt")
        else:
            LOG.info("[4/7] Building the grounded panorama prompt")
            prompt = self.prompt_builder.build(event, research, run_id)
            valid_reference_paths = [
                Path(item.path) for item in reference_report.references if item.valid
            ]
            prompt = replace(prompt, reference_paths=valid_reference_paths)
            state.write_json(prompt_path, asdict(prompt))
            state.write_json(run_dir / "structured_description.json", prompt.structured_description)
            state.write_json(run_dir / "used_facts.json", [asdict(fact) for fact in prompt.used_facts])
            state.write_json(run_dir / "sources.json", prompt.sources)
            state.update(stage="prompt_complete")

        final_status, final_image, attempts, rejection_reasons = self._run_attempts(
            run_id, run_dir, prompt, analysis, reference_report, state
        )
        manifest = {
            "run_id": run_id,
            "event": event,
            "status": final_status,
            "attempts": attempts,
            "image": str(final_image.path),
            "image_metadata": final_image.metadata,
            "rejection_reasons": rejection_reasons,
            "sources": [asdict(source) for source in research.sources],
            "limitations": research.limitations + reference_report.limitations,
        }
        state.write_json(run_dir / "manifest.json", manifest)
        final_visual_path = (
            run_dir / "attempts" / f"attempt_{attempts:02d}" / "visual_validation.json"
        )
        final_visual = (
            json.loads(final_visual_path.read_text(encoding="utf-8"))
            if final_visual_path.is_file()
            else {"status": "visual_validation_unavailable"}
        )
        state.write_json(
            run_dir / "final_report.json",
            {
                "status": final_status,
                "attempts": attempts,
                "rejection_reasons": rejection_reasons,
                "visual_validation_status": final_visual.get("status"),
                "visual_validation_available": final_visual.get("status")
                != "visual_validation_unavailable",
            },
        )
        state.update(stage="complete", status=final_status, attempts=attempts)
        LOG.info("Pipeline finished with status=%s: %s", final_status, final_image.path)
        return PipelineResult(
            run_id, run_dir, research, prompt, final_image, final_status, attempts
        )

    def _run_attempts(
        self,
        run_id: str,
        run_dir: Path,
        base_prompt: PromptResult,
        analysis: HistoricalAnalysis,
        references: ReferenceReport,
        state: RunState,
    ) -> tuple[str, ImageResult, int, list[str]]:
        correction = ""
        last_image: ImageResult | None = None
        last_reasons: list[str] = []
        maximum = self.retry_controller.max_full_generations
        for attempt in range(1, maximum + 1):
            attempt_dir = run_dir / "attempts" / f"attempt_{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            report_path = attempt_dir / "attempt_report.json"
            if report_path.is_file():
                previous = json.loads(report_path.read_text(encoding="utf-8"))
                if previous.get("decision", {}).get("status") == "accepted":
                    image = self._load_image(attempt_dir / "panorama.png", attempt_dir)
                    self._publish_final(image.path, run_dir)
                    return "accepted", replace(image, path=run_dir / "panorama.png"), attempt, []
                correction = str(previous.get("decision", {}).get("correction", correction))

            attempt_prompt = self._attempt_prompt(base_prompt, attempt, correction)
            state.write_json(attempt_dir / "prompt.json", asdict(attempt_prompt))
            panorama_path = attempt_dir / "panorama.png"
            if panorama_path.is_file():
                LOG.info("[resume] Reusing generated attempt %d", attempt)
                image = self._load_image(panorama_path, attempt_dir)
            else:
                LOG.info("[5/7] Generating panorama attempt %d/%d", attempt, maximum)
                image = self.image_generator.generate(attempt_prompt, attempt_dir, run_id)
            last_image = image
            state.update(stage="generation_complete", current_attempt=attempt)

            LOG.info("[6/7] Running deterministic technical validation")
            expected_width, expected_height = self._expected_output_size()
            technical = self.technical_validator.validate(
                image.path, expected_width=expected_width, expected_height=expected_height
            )
            state.write_json(attempt_dir / "technical_validation.json", asdict(technical))

            frames_dir = attempt_dir / "perspective_frames"
            readable = all(
                check.status == "passed"
                for check in technical.checks
                if check.name in {"file_exists", "image_readable"}
            )
            frames = []
            projection_error = ""
            if readable:
                try:
                    frames = self.perspective_projector.generate_standard_views(
                        image.path, frames_dir
                    )
                except (OSError, ValueError) as exc:
                    projection_error = f"Perspective conversion failed: {type(exc).__name__}: {exc}"
            else:
                projection_error = "Perspective conversion skipped because the image is unreadable"
            if projection_error:
                state.write_json(frames_dir / "frames.json", [])
                visual = VisualValidationReport(
                    "visual_validation_unavailable", explanation=projection_error
                )
            else:
                LOG.info("[7/7] Running visual/historical validation")
                try:
                    visual = self.visual_validator.validate(
                        frames, analysis, references, run_id
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    visual = VisualValidationReport(
                        "visual_validation_unavailable",
                        explanation=(
                            "Multimodal validation failed; deterministic checks remain valid: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            state.write_json(attempt_dir / "visual_validation.json", asdict(visual))
            supports_inpainting = bool(getattr(self.image_generator, "supports_inpainting", False))
            decision = self.retry_controller.decide(
                attempt, technical, visual, supports_inpainting=supports_inpainting
            )
            state.write_json(
                report_path,
                {
                    "attempt": attempt,
                    "generation": image.metadata,
                    "decision": asdict(decision),
                },
            )
            state.update(stage="attempt_validated", current_attempt=attempt, decision=decision.status)
            last_reasons = decision.reasons
            if decision.status == "accepted":
                self._publish_final(image.path, run_dir)
                return "accepted", replace(image, path=run_dir / "panorama.png"), attempt, []
            if not decision.retry:
                break
            correction = decision.correction
            if decision.use_inpainting and hasattr(self.image_generator, "inpaint"):
                # No built-in generator currently exposes this capability. External generators
                # may implement it; otherwise the controller safely uses a full regeneration.
                LOG.info("Inpainting requested by validator; external generator will handle it")
            LOG.warning("Attempt %d rejected; next prompt correction: %s", attempt, correction)

        if last_image is None:
            raise RuntimeError("No image generation attempt was completed")
        self._publish_final(last_image.path, run_dir)
        return (
            "manual_review_required",
            replace(last_image, path=run_dir / "panorama.png"),
            maximum,
            last_reasons,
        )

    def _start_or_resume(self, event: str, resume_dir: Path | None) -> tuple[Path, str, str]:
        if resume_dir is not None:
            run_dir = Path(resume_dir)
            state = RunState(run_dir).load()
            if not state:
                raise ValueError(f"Resume directory has no state.json: {run_dir}")
            return run_dir, str(state["run_id"]), str(state.get("event", event))
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        safe_event = re.sub(r"[^\w-]+", "-", event.lower(), flags=re.UNICODE).strip("-")[:60]
        run_dir = self.output_root / f"{run_id}-{safe_event or 'event'}"
        run_dir.mkdir(parents=True, exist_ok=False)
        RunState(run_dir).update(run_id=run_id, event=event, stage="created", status="running")
        return run_dir, run_id, event

    @staticmethod
    def _legacy_analysis(event: str, research: ResearchResult) -> HistoricalAnalysis:
        facts = [
            Fact(source.text[:500], "event", "supported", source.article_title or source.title, source.section)
            for source in research.sources[:3]
            if source.text
        ]
        return HistoricalAnalysis(
            identified_event=event,
            facts=facts,
            constraints=SceneConstraints(
                must_include=[fact.statement for fact in facts],
                do_not_over_specify=["Details not explicitly covered by available sources"],
            ),
        )

    @staticmethod
    def _attempt_prompt(base: PromptResult, attempt: int, correction: str) -> PromptResult:
        prompt = base.prompt
        negative = base.negative_prompt
        if attempt > 1:
            if not correction:
                raise ValueError("A retry must include a concrete prompt correction")
            prompt = f"{prompt} Regeneration correction: {correction}."
            negative = f"{negative}, unresolved prior defects"
        metadata = dict(base.metadata)
        metadata.update({"attempt": attempt, "seed": int(metadata.get("seed", 42)) + attempt - 1})
        return replace(base, prompt=prompt, negative_prompt=negative, metadata=metadata)

    def _expected_output_size(self) -> tuple[int | None, int | None]:
        options = getattr(self.image_generator, "options", {})
        if not isinstance(options, dict):
            return None, None
        width = options.get("output_width", options.get("width"))
        height = options.get("output_height", options.get("height"))
        return (int(width), int(height)) if width and height else (None, None)

    @staticmethod
    def _publish_final(source: Path, run_dir: Path) -> None:
        destination = run_dir / "panorama.png"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    @staticmethod
    def _load_research(path: Path) -> ResearchResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sources"] = [Source(**source) for source in data.get("sources", [])]
        data["metadata"] = WikipediaEventMetadata(**data.get("metadata", {}))
        data["analysis"] = None
        data["references"] = None
        return ResearchResult(**data)

    @staticmethod
    def _load_references(path: Path) -> ReferenceReport:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ReferenceReport(
            references=[ReferenceInfo(**item) for item in data.get("references", [])],
            limitations=data.get("limitations", []),
        )

    @staticmethod
    def _load_prompt(path: Path) -> PromptResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["used_facts"] = [Fact(**fact) for fact in data.get("used_facts", [])]
        data["reference_paths"] = [Path(value) for value in data.get("reference_paths", [])]
        return PromptResult(**data)

    @staticmethod
    def _load_image(path: Path, attempt_dir: Path) -> ImageResult:
        metadata_path = attempt_dir / "generation.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return ImageResult(path, metadata)
