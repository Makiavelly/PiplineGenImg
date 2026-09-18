import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from historical_panorama.models import (
    HistoricalAnalysis,
    PerspectiveFrame,
    ReferenceInfo,
    ReferenceReport,
    SceneConstraints,
)
from historical_panorama.providers.kaggle import KaggleVisualValidator
from historical_panorama.schemas import StructuredOutputError


class FakeRunner:
    def __init__(self, root: Path, response: dict):
        self.root = root
        self.response = response
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        self.root.mkdir(parents=True, exist_ok=True)
        filename = kwargs["expected_files"][0]
        payload = dict(self.response)
        payload["run_id"] = kwargs["payload"]["run_id"]
        (self.root / filename).write_text(json.dumps(payload), encoding="utf-8")
        return self.root


def make_image(path: Path, size=(800, 400)) -> None:
    Image.new("RGB", size, "navy").save(path)


def test_kaggle_visual_validator_sends_frames_and_parses_strict_json(tmp_path: Path):
    image = tmp_path / "frame.png"
    make_image(image)
    runner = FakeRunner(tmp_path / "output", {
        "status": "failed",
        "issues": [{
            "error_type": "visible_text",
            "description": "Text is visible on a sign.",
            "frame_number": 1,
            "yaw": 0.0,
            "pitch": 0.0,
            "severity": "critical",
            "scope": "local",
            "suggested_fix": "remove the sign text",
        }],
        "explanation": "One critical issue was found.",
    })
    validator = KaggleVisualValidator(
        runner, "visual", "Qwen/Qwen2.5-VL-3B-Instruct", "NvidiaTeslaT4",
        max_input_size=384,
    )

    report = validator.validate(
        [PerspectiveFrame(1, 0.0, 0.0, str(image))],
        HistoricalAnalysis(constraints=SceneConstraints(must_not_include=["text"])),
        ReferenceReport(),
        "run-1",
    )

    assert report.issues[0].severity == "critical"
    call = runner.calls[0]
    assert call["payload"]["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert call["accelerator"] == "NvidiaTeslaT4"
    encoded = base64.b64decode(call["payload"]["images"][0]["image_base64"])
    encoded_path = tmp_path / "encoded.jpg"
    encoded_path.write_bytes(encoded)
    with Image.open(encoded_path) as resized:
        assert max(resized.size) <= 384


def test_visual_validator_rejects_invalid_model_json(tmp_path: Path):
    image = tmp_path / "frame.png"
    make_image(image)
    runner = FakeRunner(tmp_path / "output", {
        "status": "failed",
        "issues": [{"error_type": "broken"}],
        "explanation": "invalid issue",
    })
    validator = KaggleVisualValidator(runner, "visual", "model")
    with pytest.raises(StructuredOutputError, match="Visual validation JSON is invalid"):
        validator.validate(
            [PerspectiveFrame(1, 0, 0, str(image))],
            HistoricalAnalysis(),
            ReferenceReport(),
            "run-1",
        )


def test_multimodal_reference_analysis_uses_only_valid_requested_references(tmp_path: Path):
    image = tmp_path / "reference.png"
    make_image(image, (64, 64))
    runner = FakeRunner(tmp_path / "output", {
        "descriptions": [{"index": 0, "description": "A coarse undyed wool weave."}],
    })
    validator = KaggleVisualValidator(runner, "visual", "model")
    report = ReferenceReport([
        ReferenceInfo(
            str(image), ["clothing_material"], ["background"], True, 64, 64, "PNG"
        ),
        ReferenceInfo(str(tmp_path / "bad.png"), ["weapon"], [], False),
    ])

    result = validator.analyze_references(report, "run-2")

    assert result.references[0].description_status == "completed"
    assert result.references[0].description == "A coarse undyed wool weave."
    sent = runner.calls[0]["payload"]["images"]
    assert len(sent) == 1
    assert sent[0]["use_for"] == ["clothing_material"]
    assert sent[0]["do_not_copy"] == ["background"]


def test_visual_kernel_forbids_inventing_historical_requirements():
    template = (
        Path(__file__).parents[1]
        / "src/historical_panorama/kaggle_templates/visual_validator_kernel.py.tpl"
    ).read_text(encoding="utf-8")
    assert "Do not add\nnew historical requirements" in template
    assert "do_not_over_specify details as an error" in template
