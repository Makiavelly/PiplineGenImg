from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import build_pipeline
from .references import reference_specs_from_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--event", required=True)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    pipeline, checks = build_pipeline(args.config)
    reference_items = (
        json.loads(args.references.read_text(encoding="utf-8")) if args.references else []
    )
    if reference_items and not bool(
        getattr(pipeline.image_generator, "supports_image_conditioning", False)
    ):
        raise ValueError(
            "Configured image generator does not support images in its input context"
        )
    checks.check_connection()
    result = pipeline.run(args.event, references=reference_specs_from_json(reference_items))
    if result.status not in {"accepted", "manual_review_required"}:
        raise RuntimeError(f"Unexpected pipeline status: {result.status}")


if __name__ == "__main__":
    main()
