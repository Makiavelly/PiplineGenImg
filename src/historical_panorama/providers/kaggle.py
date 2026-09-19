from __future__ import annotations

import base64
import io
import json
import re
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from ..kaggle_runner import KaggleError, KaggleKernelRunner
from ..models import (
    HistoricalAnalysis,
    ImageResult,
    PerspectiveFrame,
    PromptResult,
    ReferenceReport,
    ResearchResult,
    VisualValidationReport,
)
from ..schemas import (
    HISTORICAL_ANALYSIS_SCHEMA,
    REFERENCE_DESCRIPTION_SCHEMA,
    VISUAL_VALIDATION_SCHEMA,
    historical_analysis_from_dict,
    validate_reference_descriptions,
    visual_validation_from_dict,
)

TEMPLATES = Path(__file__).resolve().parent.parent / "kaggle_templates"


class KaggleHistoricalFactExtractor:
    def __init__(
        self,
        runner: KaggleKernelRunner,
        kernel_slug: str,
        model_id: str,
        accelerator: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 1400,
    ):
        if max_new_tokens < 512:
            raise ValueError("fact extractor max_new_tokens must be at least 512")
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = accelerator
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens

    def extract(self, event: str, research: ResearchResult, run_id: str) -> HistoricalAnalysis:
        materials = self._compact_materials(research)
        output = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title="Historical panorama fact extractor",
            template_path=TEMPLATES / "facts_kernel.py.tpl",
            payload={
                "run_id": run_id,
                "event": event,
                "materials": materials,
                "limitations": research.limitations,
                "model_id": self.model_id,
                "load_in_4bit": self.load_in_4bit,
                "max_new_tokens": self.max_new_tokens,
                "schema": HISTORICAL_ANALYSIS_SCHEMA,
            },
            expected_files=["analysis.json"],
            accelerator=self.accelerator,
        )
        data = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        if data.pop("run_id", None) != run_id:
            raise KaggleError("Received stale historical analysis: run_id does not match")
        analysis = historical_analysis_from_dict(data)
        allowed_citations = {
            (str(item["article"]), str(item["section"])) for item in materials
        }
        invalid = [
            (fact.article, fact.section)
            for fact in analysis.facts
            if (fact.article, fact.section) not in allowed_citations
        ]
        if invalid:
            raise KaggleError(
                "Historical analysis contains citations absent from supplied Wikipedia "
                f"materials: {invalid[:5]}"
            )
        return analysis

    @staticmethod
    def _compact_materials(research: ResearchResult) -> list[dict[str, object]]:
        """Keep source identity and useful text without overwhelming a small Kaggle LLM."""
        ranked = sorted(
            research.sources,
            key=lambda source: (
                source.is_primary,
                source.relevance,
                source.section.lower() in {"lead", "infobox"},
                len(source.text),
            ),
            reverse=True,
        )
        materials: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        total_characters = 0
        for source in ranked:
            article = source.article_title or source.title
            identity = (article, source.section)
            if identity in seen or not source.text.strip():
                continue
            text = re.sub(r"\s+", " ", source.text).strip()[:900]
            if total_characters + len(text) > 14000:
                break
            materials.append(
                {
                    "source_id": len(materials),
                    "article": article,
                    "section": source.section,
                    "text": text,
                }
            )
            seen.add(identity)
            total_characters += len(text)
            if len(materials) >= 18:
                break
        return materials


class KagglePromptBuilder:
    def __init__(
        self,
        runner: KaggleKernelRunner,
        kernel_slug: str,
        model_id: str,
        accelerator: str | None = None,
        load_in_4bit: bool = False,
    ):
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = accelerator
        self.load_in_4bit = load_in_4bit

    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult:
        if research.analysis is None:
            raise KaggleError("Historical analysis is required before prompt construction")
        output = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title="Historical panorama prompt builder",
            template_path=TEMPLATES / "prompt_kernel.py.tpl",
            payload={
                "run_id": run_id,
                "event": event,
                "analysis": asdict(research.analysis),
                "references": asdict(research.references) if research.references else {},
                "model_id": self.model_id,
                "load_in_4bit": self.load_in_4bit,
            },
            expected_files=["prompt.json"],
            accelerator=self.accelerator,
        )
        data = json.loads((output / "prompt.json").read_text(encoding="utf-8"))
        if data.get("run_id") != run_id:
            raise KaggleError("Received stale prompt output: run_id does not match")
        self._validate_prompt(data.get("prompt", ""), data.get("negative_prompt", ""))
        used_indices = [
            index
            for index in data.get("used_fact_indices", [])
            if isinstance(index, int) and 0 <= index < len(research.analysis.facts)
        ]
        used_facts = [research.analysis.facts[index] for index in used_indices]
        source_lookup = {
            (source.article_title or source.title, source.section): source.url
            for source in research.sources
        }
        sources = list({
            (fact.article, fact.section): {
                "article": fact.article,
                "section": fact.section,
                "url": source_lookup.get((fact.article, fact.section), ""),
            }
            for fact in used_facts
        }.values())
        return PromptResult(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            metadata={
                "provider": "kaggle",
                "model_id": self.model_id,
                "load_in_4bit": self.load_in_4bit,
            },
            structured_description=asdict(research.analysis),
            used_facts=used_facts,
            sources=sources,
        )

    @staticmethod
    def _validate_prompt(prompt: str, negative_prompt: str = "") -> None:
        words = prompt.split()
        ascii_letters = sum(character.isascii() and character.isalpha() for character in prompt)
        all_letters = sum(character.isalpha() for character in prompt)
        english_ratio = ascii_letters / max(all_letters, 1)
        problems = []
        if len(words) < 55:
            problems.append(f"too short ({len(words)} words; minimum 55)")
        if len(words) > 190:
            problems.append(f"too long for SDXL ({len(words)} words; maximum 190)")
        if english_ratio < 0.95:
            problems.append(f"not predominantly English ({english_ratio:.0%})")
        required = (
            "seamless equirectangular 360-degree panorama",
            "2:1 aspect ratio",
            "360° × 180°",
            "human eye level",
            "continuous level horizon",
            "consistent lighting",
            "no visible seam",
            "no repeated objects",
            "no mirrored duplicates",
            "no excessive distortion near the poles",
        )
        missing = [value for value in required if value.lower() not in prompt.lower()]
        if missing:
            problems.append("missing panorama requirements: " + ", ".join(missing))
        if "sources" in prompt.lower():
            problems.append("contains an instruction/template fragment")
        if "**" in prompt or re.search(r"(?im)^(title|description|location|year):", prompt):
            problems.append("contains headings or Markdown instead of a plain image prompt")
        negative_required = (
            "modern objects", "visible dates", "city names", "country names",
            "geographic coordinates", "maps", "information signs", "interface elements",
            "watermark", "text",
        )
        missing_negative = [value for value in negative_required if value not in negative_prompt.lower()]
        if missing_negative:
            problems.append("negative prompt misses answer-hiding rules: " + ", ".join(missing_negative))
        if problems:
            raise KaggleError("Prompt quality gate rejected model output: " + "; ".join(problems))


class KaggleVisualValidator:
    """Runs Qwen2.5-VL in Kaggle for grounded visual and reference analysis."""

    def __init__(
        self,
        runner: KaggleKernelRunner,
        kernel_slug: str,
        model_id: str,
        accelerator: str | None = None,
        max_input_size: int = 384,
        max_new_tokens: int = 1600,
    ):
        if not 128 <= max_input_size <= 768:
            raise ValueError("visual validator max_input_size must be between 128 and 768")
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = accelerator
        self.max_input_size = max_input_size
        self.max_new_tokens = max_new_tokens

    def validate(
        self,
        frames: list[PerspectiveFrame],
        analysis: HistoricalAnalysis,
        references: ReferenceReport,
        run_id: str,
    ) -> VisualValidationReport:
        if not frames:
            return VisualValidationReport(
                "visual_validation_unavailable",
                explanation="No perspective frames were available for multimodal validation",
            )
        images = [
            {
                "frame_number": frame.frame_number,
                "yaw": frame.yaw,
                "pitch": frame.pitch,
                "image_base64": self._encode_image(Path(frame.path)),
            }
            for frame in frames
        ]
        output = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title="Historical panorama visual validator",
            template_path=TEMPLATES / "visual_validator_kernel.py.tpl",
            payload={
                "mode": "validate_panorama",
                "run_id": run_id,
                "model_id": self.model_id,
                "max_new_tokens": self.max_new_tokens,
                "images": images,
                "analysis": asdict(analysis),
                "references": asdict(references),
                "schema": VISUAL_VALIDATION_SCHEMA,
            },
            expected_files=["visual_validation.json"],
            accelerator=self.accelerator,
        )
        data = json.loads((output / "visual_validation.json").read_text(encoding="utf-8"))
        if data.pop("run_id", None) != run_id:
            raise KaggleError("Received stale visual validation: run_id does not match")
        report = visual_validation_from_dict(data)
        frame_lookup = {frame.frame_number: frame for frame in frames}
        for issue in report.issues:
            if issue.frame_number is None:
                continue
            frame = frame_lookup.get(issue.frame_number)
            if frame is None:
                raise KaggleError(
                    f"Visual validator referenced unknown frame #{issue.frame_number}"
                )
            if issue.yaw != frame.yaw or issue.pitch != frame.pitch:
                raise KaggleError(
                    f"Visual validator returned incorrect yaw/pitch for frame #{issue.frame_number}"
                )
        return report

    def analyze_references(self, report: ReferenceReport, run_id: str) -> ReferenceReport:
        candidates = [
            (index, item)
            for index, item in enumerate(report.references)
            if item.valid and item.use_for
        ]
        if not candidates:
            return report
        images = [
            {
                "index": index,
                "use_for": item.use_for,
                "do_not_copy": item.do_not_copy,
                "image_base64": self._encode_image(Path(item.path)),
            }
            for index, item in candidates
        ]
        output = self.runner.execute(
            kernel_slug=f"{self.kernel_slug}-references",
            title="Historical reference image analyzer",
            template_path=TEMPLATES / "visual_validator_kernel.py.tpl",
            payload={
                "mode": "describe_references",
                "run_id": run_id,
                "model_id": self.model_id,
                "max_new_tokens": min(self.max_new_tokens, 1000),
                "images": images,
                "schema": REFERENCE_DESCRIPTION_SCHEMA,
            },
            expected_files=["reference_descriptions.json"],
            accelerator=self.accelerator,
        )
        data = json.loads(
            (output / "reference_descriptions.json").read_text(encoding="utf-8")
        )
        if data.pop("run_id", None) != run_id:
            raise KaggleError("Received stale reference analysis: run_id does not match")
        validated = validate_reference_descriptions(data)
        descriptions = {
            item["index"]: item["description"] for item in validated["descriptions"]
        }
        updated = []
        limitations = list(report.limitations)
        for index, item in enumerate(report.references):
            if item.valid and item.use_for:
                description = str(descriptions.get(index, "")).strip()
                if description:
                    item = replace(
                        item, description=description, description_status="completed"
                    )
                else:
                    item = replace(item, description_status="analysis_failed")
                    limitations.append(
                        f"Reference analysis returned no description for {item.path}"
                    )
            updated.append(item)
        return ReferenceReport(updated, limitations)

    def _encode_image(self, path: Path) -> str:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((self.max_input_size, self.max_input_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


class KaggleImageGenerator:
    supports_image_conditioning = False
    supports_inpainting = False
    template_name = "sdxl_kernel.py.tpl"
    kernel_title = "Historical panorama SDXL generator"
    provider_name = "kaggle_sdxl"

    def __init__(self, runner: KaggleKernelRunner, kernel_slug: str, model_id: str, **options):
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = str(options.pop("accelerator", "NvidiaTeslaT4"))
        self.options = options

    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult:
        width = int(self.options.get("width", 1024))
        height = int(self.options.get("height", 512))
        output_width = int(self.options.get("output_width", width))
        output_height = int(self.options.get("output_height", height))
        if width != height * 2 or output_width != output_height * 2:
            raise ValueError("Generation and output dimensions must both have a 2:1 aspect ratio")
        started = time.monotonic()
        remote = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title=self.kernel_title,
            template_path=TEMPLATES / self.template_name,
            payload={
                "run_id": run_id,
                "prompt": prompt.prompt,
                "negative_prompt": prompt.negative_prompt,
                "model_id": self.model_id,
                **self.options,
                "seed": int(prompt.metadata.get("seed", self.options.get("seed", 42))),
            },
            expected_files=["panorama.png", "generation.json"],
            accelerator=self.accelerator,
        )
        metadata = json.loads((remote / "generation.json").read_text(encoding="utf-8"))
        if metadata.get("run_id") != run_id:
            raise KaggleError("Received stale image output: run_id does not match")
        destination = output_dir / "panorama.png"
        shutil.copy2(remote / "panorama.png", destination)
        shutil.copy2(remote / "generation.json", output_dir / "generation.json")
        with Image.open(destination) as image:
            if image.width != image.height * 2:
                raise KaggleError(
                    f"Generator returned {image.width}x{image.height}; equirectangular output must be 2:1"
                )
            metadata["width"], metadata["height"] = image.size
        metadata.update({"provider": self.provider_name, "model_id": self.model_id})
        metadata.update(
            {
                "attempt": int(prompt.metadata.get("attempt", 1)),
                "prompt": prompt.prompt,
                "negative_prompt": prompt.negative_prompt,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "result_path": str(destination),
            }
        )
        (output_dir / "generation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ImageResult(path=destination, metadata=metadata)


class KaggleSD35ImageGenerator(KaggleImageGenerator):
    """Stable Diffusion 3.5 Medium on Kaggle with T4-compatible CPU offload."""

    template_name = "sd35_kernel.py.tpl"
    # Kaggle requires the title-derived slug to match a new kernel id.
    kernel_title = "Historical panorama sd35 generator"
    provider_name = "kaggle_sd35"
