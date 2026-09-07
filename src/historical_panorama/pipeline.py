from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .interfaces import ImageGenerator, InformationProvider, PromptBuilder
from .models import PipelineResult

LOG = logging.getLogger(__name__)


class HistoricalPanoramaPipeline:
    def __init__(
        self,
        information_provider: InformationProvider,
        prompt_builder: PromptBuilder,
        image_generator: ImageGenerator,
        output_root: Path,
    ):
        self.information_provider = information_provider
        self.prompt_builder = prompt_builder
        self.image_generator = image_generator
        self.output_root = output_root

    def run(self, event: str) -> PipelineResult:
        event = event.strip()
        if not event:
            raise ValueError("Event name cannot be empty")
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        safe_event = re.sub(r"[^\w-]+", "-", event.lower(), flags=re.UNICODE).strip("-")[:60]
        run_dir = self.output_root / f"{run_id}-{safe_event or 'event'}"
        run_dir.mkdir(parents=True, exist_ok=False)

        LOG.info("[1/3] Researching historical event: %s", event)
        research = self.information_provider.research(event)
        self._write_json(run_dir / "research.json", asdict(research))

        LOG.info("[2/3] Building an equirectangular panorama prompt")
        prompt = self.prompt_builder.build(event, research, run_id)
        self._write_json(run_dir / "prompt.json", asdict(prompt))

        LOG.info("[3/3] Generating the panorama")
        image = self.image_generator.generate(prompt, run_dir, run_id)
        manifest = {
            "run_id": run_id,
            "event": event,
            "image": str(image.path),
            "image_metadata": image.metadata,
            "sources": [asdict(source) for source in research.sources],
        }
        self._write_json(run_dir / "manifest.json", manifest)
        LOG.info("Pipeline completed: %s", image.path)
        return PipelineResult(run_id, run_dir, research, prompt, image)

    @staticmethod
    def _write_json(path: Path, data: object) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

