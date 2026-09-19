from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .factories import (
    ConnectionChecks,
    FactoryContext,
    fact_extractor_factories,
    image_generator_factories,
    information_provider_factories,
    prompt_builder_factories,
    visual_validator_factories,
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


def build_pipeline(
    config_path: Path,
    *,
    visual_validation_runs_override: int | None = None,
    technical_validation_enabled_override: bool | None = None,
) -> tuple[HistoricalPanoramaPipeline, ConnectionChecks]:
    load_dotenv()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    return build_pipeline_from_mapping(
        config,
        visual_validation_runs_override=visual_validation_runs_override,
        technical_validation_enabled_override=technical_validation_enabled_override,
    )


def build_pipeline_from_mapping(
    config: Mapping[str, Any],
    *,
    visual_validation_runs_override: int | None = None,
    technical_validation_enabled_override: bool | None = None,
) -> tuple[HistoricalPanoramaPipeline, ConnectionChecks]:
    """Build a pipeline from an already parsed configuration.

    Keeping this entry point separate from YAML loading lets alternate frontends build
    provider-specific configurations without writing credentials to disk.
    """
    config = dict(config)
    for module_name in config.get("factory_modules", []):
        importlib.import_module(str(module_name))

    context = FactoryContext(config)
    information = information_provider_factories.create(config["search"], context)
    fact_config = config.get("fact_extractor")
    if fact_config is None:
        prompt_config = dict(config["prompt_builder"])
        prompt_config["kernel_slug"] = f"{prompt_config.get('kernel_slug', 'historical-panorama')}-facts"
        fact_config = prompt_config
    fact_extractor = fact_extractor_factories.create(fact_config, context)
    prompt_builder = prompt_builder_factories.create(config["prompt_builder"], context)
    image_generator = image_generator_factories.create(config["image_generator"], context)
    pipeline_config = dict(config.get("pipeline", {}))
    if visual_validation_runs_override is not None:
        if not 0 <= visual_validation_runs_override <= 5:
            raise ValueError("visual_validation_runs override must be between 0 and 5")
        pipeline_config["visual_validation_runs"] = visual_validation_runs_override
    if technical_validation_enabled_override is not None:
        pipeline_config["technical_validation_enabled"] = (
            technical_validation_enabled_override
        )
    visual_config = config.get("visual_validator", {"provider": "unavailable"})
    if int(pipeline_config.get("visual_validation_runs", 1)) == 0:
        visual_config = {"provider": "unavailable"}
    visual_validator = visual_validator_factories.create(visual_config, context)
    pipeline = HistoricalPanoramaPipeline(
        information,
        prompt_builder,
        image_generator,
        Path(config.get("output_dir", "runs")),
        fact_extractor=fact_extractor,
        visual_validator=visual_validator,
        reference_analyzer=visual_validator,
        pipeline_config=pipeline_config,
    )
    return pipeline, context.connection_checks
