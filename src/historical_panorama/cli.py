from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import requests

from .config import build_pipeline
from .kaggle_runner import KaggleError
from .references import reference_specs_from_json
from .schemas import StructuredOutputError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a historical 360-degree panorama")
    result.add_argument("event", nargs="?", help="Historical event, e.g. 'Бородинское сражение'")
    result.add_argument("--config", type=Path, default=Path("config.yaml"))
    result.add_argument(
        "--references",
        type=Path,
        help="JSON file containing 0-4 reference image specifications",
    )
    result.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Inline JSON reference specification; may be supplied up to four times",
    )
    result.add_argument("--resume", type=Path, help="Resume an existing run directory")
    result.add_argument(
        "--check-connections",
        "--check-kaggle",
        dest="check_connections",
        action="store_true",
        help="Only verify connections required by the configured providers",
    )
    result.add_argument(
        "--skip-connection-check",
        action="store_true",
        help="Skip the optional preflight and let the first provider request verify connectivity",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        pipeline, connection_checks = build_pipeline(args.config)
        if not args.skip_connection_check:
            connection_checks.check_connection()
        if args.check_connections:
            return
        if not args.event and args.resume is None:
            parser().error("event is required unless --check-connections is used")
        reference_items = []
        if args.references:
            reference_items.extend(json.loads(args.references.read_text(encoding="utf-8")))
        reference_items.extend(json.loads(value) for value in args.reference)
        references = reference_specs_from_json(reference_items)
        result = pipeline.run(args.event or "", references=references, resume_dir=args.resume)
        print(result.image.path)
        if result.status != "accepted":
            logging.error("Pipeline requires manual review; see %s", result.run_dir / "final_report.json")
            raise SystemExit(2)
    except (KaggleError, StructuredOutputError, requests.RequestException, LookupError, ValueError) as exc:
        logging.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
