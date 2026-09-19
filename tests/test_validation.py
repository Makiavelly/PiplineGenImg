from pathlib import Path

from PIL import Image

from historical_panorama.validation import TechnicalPanoramaValidator


def check(report, name):
    return next(item for item in report.checks if item.name == name)


def test_validator_checks_exact_two_to_one_ratio(tmp_path: Path):
    path = tmp_path / "wrong.png"
    Image.new("RGB", (100, 64), "black").save(path)
    report = TechnicalPanoramaValidator().validate(path)
    assert check(report, "aspect_ratio_2_to_1").status == "failed"


def test_validator_does_not_run_removed_pixel_content_heuristics(tmp_path: Path):
    path = tmp_path / "uniform.png"
    Image.new("RGB", (128, 64), "black").save(path)
    report = TechnicalPanoramaValidator().validate(path)

    assert report.status == "passed"
    assert {item.name for item in report.checks} == {
        "file_exists", "image_readable", "dimensions", "aspect_ratio_2_to_1"
    }
