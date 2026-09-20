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
    TechnicalValidationReport,
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
        self.visual_validation_runs = int(config.get("visual_validation_runs", 1))
        self.technical_validation_enabled = bool(
            config.get("technical_validation_enabled", True)
        )
        if not 0 <= self.visual_validation_runs <= 5:
            raise ValueError("visual_validation_runs must be between 0 and 5")

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
            generator_accepts_images = bool(
                getattr(self.image_generator, "supports_image_conditioning", False)
            )
            needs_analysis = not generator_accepts_images and any(
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
            if valid_reference_paths and bool(
                getattr(self.image_generator, "supports_image_conditioning", False)
            ):
                prompt = replace(
                    prompt,
                    prompt=self._add_direct_reference_guidance(prompt.prompt, reference_report),
                    reference_paths=valid_reference_paths,
                )
            else:
                prompt = replace(prompt, reference_paths=[])
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
        final_technical_path = (
            run_dir / "attempts" / f"attempt_{attempts:02d}" / "technical_validation.json"
        )
        final_technical = (
            json.loads(final_technical_path.read_text(encoding="utf-8"))
            if final_technical_path.is_file()
            else {"status": "unavailable"}
        )
        state.write_json(
            run_dir / "final_report.json",
            {
                "status": final_status,
                "attempts": attempts,
                "rejection_reasons": rejection_reasons,
                "technical_validation_status": final_technical.get("status"),
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

            if self.technical_validation_enabled:
                LOG.info("[6/7] Running deterministic technical validation")
                expected_width, expected_height = self._expected_output_size()
                technical = self.technical_validator.validate(
                    image.path, expected_width=expected_width, expected_height=expected_height
                )
            else:
                LOG.info("[6/7] Deterministic technical validation is disabled")
                technical = TechnicalValidationReport(
                    "disabled",
                    explanation="Technical validation was disabled by pipeline configuration",
                )
            state.write_json(attempt_dir / "technical_validation.json", asdict(technical))

            frames_dir = attempt_dir / "perspective_frames"
            readable = self.technical_validation_enabled and all(
                check.status == "passed"
                for check in technical.checks
                if check.name in {"file_exists", "image_readable"}
            )
            if not self.technical_validation_enabled:
                # The perspective projector performs its own image decoding. This is not
                # treated as a technical validation result.
                readable = True
            frames = []
            projection_error = ""
            if self.visual_validation_runs == 0:
                state.write_json(frames_dir / "frames.json", [])
                visual = VisualValidationReport(
                    "visual_validation_unavailable",
                    explanation="Visual validation was disabled by pipeline configuration",
                )
            elif readable:
                try:
                    frames = self.perspective_projector.generate_standard_views(
                        image.path, frames_dir
                    )
                except (OSError, ValueError) as exc:
                    projection_error = f"Perspective conversion failed: {type(exc).__name__}: {exc}"
            else:
                projection_error = "Perspective conversion skipped because the image is unreadable"
            if self.visual_validation_runs == 0:
                pass
            elif projection_error:
                state.write_json(frames_dir / "frames.json", [])
                visual = VisualValidationReport(
                    "visual_validation_unavailable", explanation=projection_error
                )
            else:
                LOG.info(
                    "[7/7] Running %d visual/historical validation check(s)",
                    self.visual_validation_runs,
                )
                visual_reports = []
                for check_number in range(1, self.visual_validation_runs + 1):
                    validation_run_id = (
                        f"{run_id}-attempt-{attempt}-visual-{check_number}"
                    )
                    try:
                        report = self.visual_validator.validate(
                            frames, analysis, references, validation_run_id
                        )
                    except (RuntimeError, OSError, ValueError) as exc:
                        report = VisualValidationReport(
                            "visual_validation_unavailable",
                            explanation=(
                                "Multimodal validation failed; deterministic checks remain valid: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    visual_reports.append(report)
                    state.write_json(
                        attempt_dir / f"visual_validation_{check_number:02d}.json",
                        asdict(report),
                    )
                visual = self._aggregate_visual_reports(visual_reports)
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
    def _add_direct_reference_guidance(
        prompt: str, references: ReferenceReport
    ) -> str:
        rules = []
        number = 0
        for item in references.references:
            if not item.valid:
                continue
            number += 1
            use_for = ", ".join(item.use_for) or "no unspecified details"
            do_not_copy = ", ".join(item.do_not_copy) or "nothing beyond use_for"
            rules.append(
                f"Reference image {number}: use only for [{use_for}]; "
                f"do not copy [{do_not_copy}]."
            )
        return (
            prompt
            + " Attached reference images are visual conditioning inputs. "
            + " ".join(rules)
            + " Never copy any unrequested property from a reference image."
        )

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
    def _aggregate_visual_reports(
        reports: list[VisualValidationReport],
    ) -> VisualValidationReport:
        available = [
            report
            for report in reports
            if report.status != "visual_validation_unavailable"
        ]
        if not available:
            explanations = [report.explanation for report in reports if report.explanation]
            return VisualValidationReport(
                "visual_validation_unavailable",
                explanation="; ".join(explanations)
                or "All requested visual validation checks were unavailable",
            )
        unique: dict[tuple[object, ...], VisualIssue] = {}
        for report in available:
            for issue in report.issues:
                key = (
                    issue.error_type,
                    issue.description,
                    issue.frame_number,
                    issue.yaw,
                    issue.pitch,
                    issue.severity,
                    issue.scope,
                    issue.suggested_fix,
                )
                unique[key] = issue
        failed = any(report.status == "failed" for report in available)
        unavailable_count = len(reports) - len(available)
        explanation = (
            f"Aggregated {len(available)} available visual validation check(s)"
        )
        if unavailable_count:
            explanation += f"; {unavailable_count} check(s) unavailable"
        return VisualValidationReport(
            "failed" if failed else "passed",
            list(unique.values()),
            explanation,
        )

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
