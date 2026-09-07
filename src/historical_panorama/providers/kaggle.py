from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from PIL import Image

from ..kaggle_runner import KaggleError, KaggleKernelRunner
from ..models import ImageResult, PromptResult, ResearchResult

TEMPLATES = Path(__file__).resolve().parent.parent / "kaggle_templates"


class KagglePromptBuilder:
    def __init__(
        self,
        runner: KaggleKernelRunner,
        kernel_slug: str,
        model_id: str,
        accelerator: str | None = None,
    ):
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = accelerator

    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult:
        output = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title="Historical panorama prompt builder",
            template_path=TEMPLATES / "prompt_kernel.py.tpl",
            payload={
                "run_id": run_id,
                "event": event,
                "context": research.context,
                "model_id": self.model_id,
            },
            expected_files=["prompt.json"],
            accelerator=self.accelerator,
        )
        data = json.loads((output / "prompt.json").read_text(encoding="utf-8"))
        if data.get("run_id") != run_id:
            raise KaggleError("Received stale prompt output: run_id does not match")
        self._validate_prompt(data.get("prompt", ""))
        return PromptResult(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            metadata={"provider": "kaggle", "model_id": self.model_id},
        )

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        words = prompt.split()
        ascii_letters = sum(character.isascii() and character.isalpha() for character in prompt)
        all_letters = sum(character.isalpha() for character in prompt)
        english_ratio = ascii_letters / max(all_letters, 1)
        problems = []
        if len(words) < 55:
            problems.append(f"too short ({len(words)} words; minimum 55)")
        if len(words) > 120:
            problems.append(f"too long for SDXL ({len(words)} words; maximum 120)")
        if english_ratio < 0.95:
            problems.append(f"not predominantly English ({english_ratio:.0%})")
        if "equirectangular" not in prompt.lower():
            problems.append("missing equirectangular geometry requirement")
        if "sources" in prompt.lower():
            problems.append("contains an instruction/template fragment")
        if "**" in prompt or re.search(r"(?im)^(title|description|location|year):", prompt):
            problems.append("contains headings or Markdown instead of a plain image prompt")
        if problems:
            raise KaggleError("Prompt quality gate rejected model output: " + "; ".join(problems))


class KaggleImageGenerator:
    def __init__(self, runner: KaggleKernelRunner, kernel_slug: str, model_id: str, **options):
        self.runner = runner
        self.kernel_slug = kernel_slug
        self.model_id = model_id
        self.accelerator = str(options.pop("accelerator", "NvidiaTeslaT4"))
        self.options = options

    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult:
        remote = self.runner.execute(
            kernel_slug=self.kernel_slug,
            title="Historical panorama SDXL generator",
            template_path=TEMPLATES / "sdxl_kernel.py.tpl",
            payload={
                "run_id": run_id,
                "prompt": prompt.prompt,
                "negative_prompt": prompt.negative_prompt,
                "model_id": self.model_id,
                **self.options,
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
        metadata.update({"provider": "kaggle", "model_id": self.model_id})
        return ImageResult(path=destination, metadata=metadata)
