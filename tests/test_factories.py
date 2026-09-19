import pytest

from historical_panorama.factories import (
    FactoryContext,
    FactoryRegistry,
    fact_extractor_factories,
    image_generator_factories,
    prompt_builder_factories,
    visual_validator_factories,
)


def test_registry_creates_registered_provider_and_normalizes_name():
    registry = FactoryRegistry[object]("test")
    expected = object()

    @registry.register("my-provider")
    def build(config, context):
        assert config["value"] == 7
        assert isinstance(context, FactoryContext)
        return expected

    created = registry.create({"provider": "MY_PROVIDER", "value": 7}, FactoryContext({}))
    assert created is expected
    assert registry.names == ("my_provider",)


def test_registry_rejects_duplicate_name():
    registry = FactoryRegistry[object]("test")
    registry.register("same")(lambda config, context: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("same")(lambda config, context: object())


def test_registry_reports_available_names():
    registry = FactoryRegistry[object]("test")
    registry.register("available")(lambda config, context: object())
    with pytest.raises(ValueError, match="Registered providers: available"):
        registry.create({"provider": "missing"}, FactoryContext({}))


def test_context_does_not_require_kaggle_until_requested(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    context = FactoryContext({})
    assert context.connection_checks._checks == []
    with pytest.raises(ValueError, match="KAGGLE_USERNAME"):
        context.kaggle_runner()


def test_visual_validator_registry_has_kaggle_and_fallback():
    assert "kaggle" in visual_validator_factories.names
    assert "unavailable" in visual_validator_factories.names


def test_image_generator_registry_has_sdxl_and_sd35():
    assert "kaggle" in image_generator_factories.names
    assert "kaggle_sd35" in image_generator_factories.names
    assert "tooken" in image_generator_factories.names
    assert "openai_compatible" in image_generator_factories.names
    assert "tooken" in fact_extractor_factories.names
    assert "tooken" in prompt_builder_factories.names
