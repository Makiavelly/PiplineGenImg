from __future__ import annotations

import importlib
import os
from pathlib import Path

import yaml

from .factories import (
    ConnectionChecks,
    FactoryContext,
    image_generator_factories,
    information_provider_factories,
    prompt_builder_factories,
)
from .pipeline import HistoricalPanoramaPipeline


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_pipeline(config_path: Path) -> tuple[HistoricalPanoramaPipeline, ConnectionChecks]:
    load_dotenv()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    for module_name in config.get("factory_modules", []):
        importlib.import_module(str(module_name))

    context = FactoryContext(config)
    information = information_provider_factories.create(config["search"], context)
    prompt_builder = prompt_builder_factories.create(config["prompt_builder"], context)
    image_generator = image_generator_factories.create(config["image_generator"], context)
    pipeline = HistoricalPanoramaPipeline(
        information, prompt_builder, image_generator, Path(config.get("output_dir", "runs"))
    )
    return pipeline, context.connection_checks
