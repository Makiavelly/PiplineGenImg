from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

from .interfaces import (
    HistoricalFactExtractor,
    ImageGenerator,
    InformationProvider,
    PromptBuilder,
    VisualValidator,
)
from .kaggle_runner import KaggleKernelRunner, KaggleSettings
from .providers import (
    KaggleHistoricalFactExtractor,
    KaggleImageGenerator,
    KaggleSD35ImageGenerator,
    KagglePromptBuilder,
    KaggleVisualValidator,
    OpenAICompatibleClient,
    OpenAICompatibleHistoricalFactExtractor,
    OpenAICompatibleImageGenerator,
    OpenAICompatiblePromptBuilder,
    WikipediaInformationProvider,
)
from .visual_validation import UnavailableVisualValidator

T = TypeVar("T")
ProviderConfig = Mapping[str, Any]


class ConnectionChecker(Protocol):
    def check_connection(self) -> None: ...


class ConnectionChecks:
    """Runs each external-service preflight once."""

    def __init__(self) -> None:
        self._checks: list[ConnectionChecker] = []

    def add(self, checker: ConnectionChecker) -> None:
        if all(existing is not checker for existing in self._checks):
            self._checks.append(checker)

    def check_connection(self) -> None:
        for checker in self._checks:
            checker.check_connection()


class FactoryContext:
    """Shared, lazily-created infrastructure available to provider factories."""

    def __init__(self, root_config: ProviderConfig):
        self.root_config = root_config
        self.connection_checks = ConnectionChecks()
        self._kaggle_runner: KaggleKernelRunner | None = None
        self._compatible_clients: dict[tuple[str, str, int], OpenAICompatibleClient] = {}

    def kaggle_runner(self) -> KaggleKernelRunner:
        if self._kaggle_runner is not None:
            return self._kaggle_runner
        username = os.getenv("KAGGLE_USERNAME", "").strip()
        if not username:
            raise ValueError("KAGGLE_USERNAME is not set; copy .env.example to .env")
        config = cast(ProviderConfig, self.root_config.get("kaggle", {}))
        self._kaggle_runner = KaggleKernelRunner(
            KaggleSettings(
                username=username,
                poll_interval_seconds=int(config.get("poll_interval_seconds", 15)),
                timeout_seconds=int(config.get("timeout_seconds", 1800)),
                work_dir=Path(config.get("work_dir", ".kaggle-work")),
                force_ipv4=bool(config.get("force_ipv4", False)),
            )
        )
        self.connection_checks.add(self._kaggle_runner)
        return self._kaggle_runner

    def compatible_client(self, provider_config: ProviderConfig) -> OpenAICompatibleClient:
        common = cast(ProviderConfig, self.root_config.get("openai_compatible", {}))
        base_url = str(
            provider_config.get(
                "base_url", common.get("base_url", "https://api.openai.com/v1")
            )
        )
        token_env = str(provider_config.get("token_env", common.get("token_env", "GPT_TOKEN")))
        timeout = int(
            provider_config.get("timeout_seconds", common.get("timeout_seconds", 300))
        )
        key = (base_url, token_env, timeout)
        if key not in self._compatible_clients:
            client = OpenAICompatibleClient(base_url, token_env, timeout)
            self._compatible_clients[key] = client
            self.connection_checks.add(client)
        return self._compatible_clients[key]


Factory = Callable[[ProviderConfig, FactoryContext], T]


class FactoryRegistry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Factory[T]] = {}

    def register(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        normalized = self._normalize(name)

        def decorator(factory: Factory[T]) -> Factory[T]:
            if normalized in self._factories:
                raise ValueError(f"Factory already registered for {self.kind}: {normalized}")
            self._factories[normalized] = factory
            return factory

        return decorator

    def create(self, config: ProviderConfig, context: FactoryContext) -> T:
        name = self._normalize(str(config.get("provider", "")))
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"Unknown {self.kind} provider {name!r}. Registered providers: {available}"
            ) from exc
        return factory(config, context)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("Provider name cannot be empty")
        return normalized


information_provider_factories = FactoryRegistry[InformationProvider]("information")
fact_extractor_factories = FactoryRegistry[HistoricalFactExtractor]("fact extractor")
prompt_builder_factories = FactoryRegistry[PromptBuilder]("prompt builder")
image_generator_factories = FactoryRegistry[ImageGenerator]("image generator")
visual_validator_factories = FactoryRegistry[VisualValidator]("visual validator")


@visual_validator_factories.register("unavailable")
def build_unavailable_visual_validator(
    config: ProviderConfig, context: FactoryContext
) -> VisualValidator:
    return UnavailableVisualValidator()


@information_provider_factories.register("wikipedia")
def build_wikipedia(config: ProviderConfig, _: FactoryContext) -> InformationProvider:
    return WikipediaInformationProvider(
        language=str(config.get("language", "ru")),
        max_results=int(config.get("max_results", 5)),
        timeout_seconds=int(config.get("timeout_seconds", 20)),
        max_related_articles=int(config.get("max_related_articles", 6)),
    )


@fact_extractor_factories.register("kaggle")
def build_kaggle_fact_extractor(
    config: ProviderConfig, context: FactoryContext
) -> HistoricalFactExtractor:
    return KaggleHistoricalFactExtractor(
        context.kaggle_runner(),
        str(config["kernel_slug"]),
        str(config["model_id"]),
        accelerator=cast(str | None, config.get("accelerator")),
        load_in_4bit=bool(config.get("load_in_4bit", False)),
        max_new_tokens=int(config.get("max_new_tokens", 1400)),
    )


@fact_extractor_factories.register("openai_compatible")
@fact_extractor_factories.register("tooken")
def build_compatible_fact_extractor(
    config: ProviderConfig, context: FactoryContext
) -> HistoricalFactExtractor:
    return OpenAICompatibleHistoricalFactExtractor(
        context.compatible_client(config),
        str(config["model_id"]),
        max_context_characters=int(config.get("max_context_characters", 120_000)),
        max_source_characters=int(config.get("max_source_characters", 12_000)),
        max_output_tokens=int(config.get("max_output_tokens", 5_000)),
    )


@prompt_builder_factories.register("kaggle")
def build_kaggle_prompt(config: ProviderConfig, context: FactoryContext) -> PromptBuilder:
    return KagglePromptBuilder(
        context.kaggle_runner(),
        str(config["kernel_slug"]),
        str(config["model_id"]),
        accelerator=cast(str | None, config.get("accelerator")),
        load_in_4bit=bool(config.get("load_in_4bit", False)),
    )


@prompt_builder_factories.register("openai_compatible")
@prompt_builder_factories.register("tooken")
def build_compatible_prompt_builder(
    config: ProviderConfig, context: FactoryContext
) -> PromptBuilder:
    return OpenAICompatiblePromptBuilder(
        context.compatible_client(config),
        str(config["model_id"]),
        max_output_tokens=int(config.get("max_output_tokens", 1_500)),
    )


@image_generator_factories.register("kaggle")
def build_kaggle_image(config: ProviderConfig, context: FactoryContext) -> ImageGenerator:
    options = {
        key: value
        for key, value in config.items()
        if key not in {"provider", "kernel_slug", "model_id"}
    }
    return KaggleImageGenerator(
        context.kaggle_runner(),
        str(config["kernel_slug"]),
        str(config["model_id"]),
        **options,
    )


@image_generator_factories.register("kaggle_sd35")
def build_kaggle_sd35_image(
    config: ProviderConfig, context: FactoryContext
) -> ImageGenerator:
    options = {
        key: value
        for key, value in config.items()
        if key not in {"provider", "kernel_slug", "model_id"}
    }
    return KaggleSD35ImageGenerator(
        context.kaggle_runner(),
        str(config["kernel_slug"]),
        str(config["model_id"]),
        **options,
    )


@image_generator_factories.register("openai_compatible")
@image_generator_factories.register("tooken")
def build_compatible_image(
    config: ProviderConfig, context: FactoryContext
) -> ImageGenerator:
    options = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "provider",
            "model_id",
            "base_url",
            "token_env",
            "timeout_seconds",
        }
    }
    return OpenAICompatibleImageGenerator(
        context.compatible_client(config), str(config["model_id"]), **options
    )


@visual_validator_factories.register("kaggle")
def build_kaggle_visual_validator(
    config: ProviderConfig, context: FactoryContext
) -> VisualValidator:
    return KaggleVisualValidator(
        context.kaggle_runner(),
        str(config["kernel_slug"]),
        str(config["model_id"]),
        accelerator=cast(str | None, config.get("accelerator")),
        max_input_size=int(config.get("max_input_size", 384)),
        max_new_tokens=int(config.get("max_new_tokens", 1600)),
    )
