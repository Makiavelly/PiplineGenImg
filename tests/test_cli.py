import pytest
import yaml

from historical_panorama.cli import parser
from historical_panorama.config import build_pipeline


def test_cli_accepts_visual_validation_run_override():
    args = parser().parse_args(["--visual-validation-runs", "0", "event"])
    assert args.visual_validation_runs == 0

    args = parser().parse_args(["--visual-validation-runs", "5", "event"])
    assert args.visual_validation_runs == 5


def test_cli_accepts_technical_validation_disable_flag():
    args = parser().parse_args(["--skip-technical-validation", "event"])
    assert args.skip_technical_validation is True


def test_cli_rejects_too_many_visual_validation_runs():
    with pytest.raises(SystemExit):
        parser().parse_args(["--visual-validation-runs", "6", "event"])


def test_zero_override_avoids_constructing_kaggle_visual_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config = {
        "search": {"provider": "wikipedia"},
        "fact_extractor": {"provider": "tooken", "model_id": "gpt"},
        "prompt_builder": {"provider": "tooken", "model_id": "gpt"},
        "image_generator": {"provider": "tooken", "model_id": "gpt-image-2"},
        "visual_validator": {
            "provider": "kaggle",
            "kernel_slug": "visual",
            "model_id": "qwen-vl",
        },
        "openai_compatible": {"base_url": "https://example.test/v1"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pipeline, checks = build_pipeline(path, visual_validation_runs_override=0)

    assert pipeline.visual_validation_runs == 0
    assert type(pipeline.visual_validator).__name__ == "UnavailableVisualValidator"
    assert len(checks._checks) == 1


def test_technical_validation_override_is_applied(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    config = {
        "search": {"provider": "wikipedia"},
        "fact_extractor": {"provider": "tooken", "model_id": "gpt"},
        "prompt_builder": {"provider": "tooken", "model_id": "gpt"},
        "image_generator": {"provider": "tooken", "model_id": "gpt-image-2"},
        "visual_validator": {"provider": "unavailable"},
        "openai_compatible": {"base_url": "https://example.test/v1"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pipeline, _ = build_pipeline(
        path, technical_validation_enabled_override=False
    )

    assert pipeline.technical_validation_enabled is False
