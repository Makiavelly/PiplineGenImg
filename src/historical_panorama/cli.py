from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

from .config import build_pipeline
from .kaggle_runner import KaggleError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a historical 360-degree panorama")
    result.add_argument("event", nargs="?", help="Historical event, e.g. 'Бородинское сражение'")
    result.add_argument("--config", type=Path, default=Path("config.yaml"))
    result.add_argument(
        "--check-connections",
        "--check-kaggle",
        dest="check_connections",
        action="store_true",
        help="Only verify connections required by the configured providers",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        pipeline, connection_checks = build_pipeline(args.config)
        connection_checks.check_connection()
        if args.check_connections:
            return
        if not args.event:
            parser().error("event is required unless --check-connections is used")
        result = pipeline.run(args.event)
        print(result.image.path)
    except (KaggleError, requests.RequestException, LookupError, ValueError) as exc:
        logging.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
