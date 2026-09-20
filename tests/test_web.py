import base64
import json
import errno
import io
from pathlib import Path

import pytest
from PIL import Image

from historical_panorama import web
from historical_panorama.web import (
    Job,
    JobManager,
    artifact_revision,
    collect_artifacts,
    configured_pipeline,
    reference_processing_mode,
    required_credentials,
    save_reference_uploads,
)


def base_config():
    return {
        "search": {"provider": "wikipedia", "language": "ru"},
        "fact_extractor": {"provider": "kaggle", "model_id": "facts"},
        "prompt_builder": {"provider": "kaggle", "model_id": "prompt"},
        "image_generator": {"provider": "kaggle_sd35", "model_id": "sd35"},
        "visual_validator": {"provider": "kaggle", "model_id": "vl"},
    }


def selections(**overrides):
    result = {
        "search": "wikipedia",
        "fact_extractor": "kaggle",
        "prompt_builder": "kaggle",
        "image_generator": "kaggle_sd35",
        "visual_validator": "kaggle",
    }
    result.update(overrides)
    return result


def test_web_configuration_changes_each_provider_independently(tmp_path: Path):
    config = configured_pipeline(
        base_config(), selections(image_generator="kaggle", visual_validator="unavailable"), tmp_path
    )

    assert config["search"]["language"] == "ru"
    assert config["image_generator"]["provider"] == "kaggle"
    assert config["image_generator"]["model_id"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config["visual_validator"] == {"provider": "unavailable"}
    assert config["output_dir"] == str(tmp_path)


def test_web_configuration_rejects_unknown_provider(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported provider"):
        configured_pipeline(base_config(), selections(prompt_builder="api_gpt"), tmp_path)


def test_web_configuration_can_select_tooken_for_text_and_images(tmp_path: Path):
    config = configured_pipeline(
        base_config(),
        selections(
            fact_extractor="tooken",
            prompt_builder="tooken",
            image_generator="tooken",
        ),
        tmp_path,
    )

    assert config["fact_extractor"]["model_id"] == "gpt-5.6-sol"
    assert config["fact_extractor"]["max_context_characters"] == 120000
    assert config["image_generator"]["model_id"] == "gpt-image-2"
    assert config["openai_compatible"]["base_url"] == "https://tooken.club/v1"


def test_web_configuration_sets_visual_validation_count(tmp_path: Path):
    config = configured_pipeline(base_config(), selections(), tmp_path, 4)

    assert config["pipeline"]["visual_validation_runs"] == 4


def test_web_configuration_can_disable_technical_validation(tmp_path: Path):
    config = configured_pipeline(
        base_config(), selections(), tmp_path,
        technical_validation_enabled=False,
    )

    assert config["pipeline"]["technical_validation_enabled"] is False


def test_required_credentials_follow_selected_providers():
    assert required_credentials(selections()) == ["kaggle_username", "kaggle_token", "hf_token"]
    assert required_credentials(selections(image_generator="kaggle")) == [
        "kaggle_username", "kaggle_token"
    ]
    assert required_credentials(
        selections(
            fact_extractor="tooken",
            prompt_builder="tooken",
            image_generator="tooken",
            visual_validator="unavailable",
        )
    ) == ["gpt_token"]


def test_job_snapshot_reads_pipeline_progress_and_result(tmp_path: Path):
    run_dir = tmp_path / "runs" / "one"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps({"stage": "attempt_validated", "current_attempt": 2}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "accepted", "attempts": 2}), encoding="utf-8"
    )
    job = Job("abc", "Event", tmp_path, status="complete", run_dir=run_dir)

    snapshot = job.snapshot()

    assert snapshot["progress"] == 88
    assert snapshot["attempt"] == 2
    assert snapshot["result"] == {"status": "accepted", "attempts": 2, "rejection_reasons": []}


def test_pipeline_artifacts_are_readable_and_hide_secrets(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "prompt.json").write_text(
        json.dumps(
            {
                "prompt": "Historical panorama",
                "negative_prompt": "modern objects",
                "metadata": {"api_key": "must-not-leak", "seed": 42},
            }
        ),
        encoding="utf-8",
    )

    artifacts = collect_artifacts(run_dir)

    assert artifact_revision(run_dir) != "0"
    assert artifacts[0]["title"] == "Промпт генерации"
    assert artifacts[0]["producer"] == "Модель промпта"
    assert artifacts[0]["content"]["prompt"] == "Historical panorama"
    assert artifacts[0]["content"]["metadata"]["api_key"] == "[скрыто]"


def test_job_manager_requires_credentials_when_provider_selection_is_omitted(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    manager = JobManager(config_path, tmp_path / "jobs")

    with pytest.raises(ValueError, match="kaggle_username"):
        manager.create({"event": "Test", "providers": {}, "credentials": {}})


def test_web_main_explains_port_conflict_without_traceback(
    monkeypatch, tmp_path: Path, capsys
):
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["web", "--config", str(config_path), "--jobs-dir", str(tmp_path / "jobs")],
    )

    def occupied(*args, **kwargs):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(web, "ThreadingHTTPServer", occupied)
    with pytest.raises(SystemExit) as exc:
        web.main()

    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "порт 8080 уже занят" in message
    assert "--port 8081" in message


def encoded_image(format: str = "PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 8), "navy").save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_web_saves_valid_reference_uploads(tmp_path: Path):
    references_path = save_reference_uploads(
        tmp_path,
        [{
            "name": "scene.png",
            "type": "image/png",
            "data": encoded_image(),
            "use_for": ["clothing", "weapon_shape"],
            "do_not_copy": ["background"],
        }],
    )

    assert references_path is not None
    references = json.loads(references_path.read_text(encoding="utf-8"))
    assert len(references) == 1
    assert Path(references[0]["path"]).is_file()
    assert Path(references[0]["path"]).suffix == ".png"
    assert references[0]["use_for"] == ["clothing", "weapon_shape"]
    assert references[0]["do_not_copy"] == ["background"]


def test_reference_route_uses_visual_model_for_text_only_generator():
    assert reference_processing_mode(selections()) == "visual_analyzer"
    assert reference_processing_mode(
        selections(visual_validator="unavailable")
    ) == "unavailable"


def test_web_rejects_upload_without_generator_or_visual_analyzer(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    manager = JobManager(config_path, tmp_path / "jobs")
    payload = {
        "event": "Test",
        "providers": selections(visual_validator="unavailable"),
        "credentials": {},
        "images": [{
            "name": "scene.png",
            "data": encoded_image(),
            "use_for": ["clothing"],
        }],
    }

    with pytest.raises(ValueError, match="не принимают"):
        manager.create(payload)


def test_web_rejects_unreadable_reference_upload(tmp_path: Path):
    invalid = base64.b64encode(b"not an image").decode("ascii")

    with pytest.raises(ValueError, match="не является читаемым изображением"):
        save_reference_uploads(
            tmp_path,
            [{"name": "fake.png", "data": invalid, "use_for": ["clothing"]}],
        )


def test_demo_run_needs_no_credentials_and_returns_configured_image(
    monkeypatch, tmp_path: Path
):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    demo_image = tmp_path / "fake.png"
    Image.new("RGB", (32, 16), "purple").save(demo_image)

    class ImmediateThread:
        def __init__(self, target, args, **kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(web.threading, "Thread", ImmediateThread)
    manager = JobManager(
        config_path, tmp_path / "jobs", demo_image=demo_image, demo_step_seconds=0
    )

    job = manager.create(
        {
            "event": "Постановочная историческая сцена",
            "providers": selections(),
            "credentials": {},
            "demo": True,
            "images": [
                {
                    "name": "context.png",
                    "type": "image/png",
                    "data": encoded_image(),
                }
            ],
        }
    )

    snapshot = job.snapshot()
    assert job.status == "complete"
    assert snapshot["demo"] is True
    assert snapshot["progress"] == 100
    assert snapshot["result"]["status"] == "accepted"
    assert (job.run_dir / "panorama.png").read_bytes() == demo_image.read_bytes()
    assert (job.run_dir / "references.json").is_file()
    demo_references = json.loads(
        (job.run_dir / "references.json").read_text(encoding="utf-8")
    )
    assert demo_references[0]["use_for"] == []
    assert demo_references[0]["do_not_copy"] == []
    artifacts = manager.artifacts(job.id)
    assert artifacts is not None
    artifact_paths = {item["path"] for item in artifacts["artifacts"]}
    assert "analysis.json" in artifact_paths
    assert "prompt.json" in artifact_paths
    assert "attempts/attempt_01/generation.json" in artifact_paths
    assert "attempts/attempt_01/technical_validation.json" in artifact_paths
    manifest = json.loads((job.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["context_images"] == 1
    assert "fake.png" == manager.options()["demo"]["image_name"]


def test_demo_run_reports_missing_fake_image(tmp_path: Path):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    manager = JobManager(
        config_path, tmp_path / "jobs", demo_image=tmp_path / "missing.png"
    )

    with pytest.raises(ValueError, match="не найдено"):
        manager.create(
            {
                "event": "Demo",
                "providers": selections(),
                "credentials": {},
                "demo": True,
            }
        )
