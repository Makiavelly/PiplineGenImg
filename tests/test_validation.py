from pathlib import Path

import numpy as np
from PIL import Image

from historical_panorama.validation import TechnicalPanoramaValidator


def textured(width=128, height=64):
    yy, xx = np.indices((height, width))
    values = ((xx * 17 + yy * 31) % 180 + 30).astype(np.uint8)
    return np.stack((values, np.roll(values, 2, axis=0), np.roll(values, 3, axis=1)), axis=2)


def check(report, name):
    return next(item for item in report.checks if item.name == name)


def test_validator_checks_exact_two_to_one_ratio(tmp_path: Path):
    path = tmp_path / "wrong.png"
    Image.fromarray(textured(100, 64)).save(path)
    report = TechnicalPanoramaValidator().validate(path)
    assert check(report, "aspect_ratio_2_to_1").status == "failed"


def test_validator_detects_gross_seam_discontinuity(tmp_path: Path):
    array = textured()
    array[:, :4] = 0
    array[:, -4:] = 255
    path = tmp_path / "seam.png"
    Image.fromarray(array).save(path)
    report = TechnicalPanoramaValidator().validate(path)
    assert check(report, "seam_color").status == "failed"
    assert check(report, "seam_brightness").status == "failed"
