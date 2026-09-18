from pathlib import Path

from PIL import Image

from historical_panorama.models import ReferenceSpec
from historical_panorama.references import ReferenceProcessor


def test_reference_processor_keeps_empty_input_backward_compatible():
    assert ReferenceProcessor().process().references == []


def test_reference_processor_skips_corrupt_image(tmp_path: Path):
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_bytes(b"not an image")
    good = tmp_path / "good.png"
    Image.new("RGB", (10, 8), "red").save(good)

    report = ReferenceProcessor().process([
        ReferenceSpec(corrupt, ["clothing"], ["background"]),
        ReferenceSpec(good, ["weapon_shape"], ["composition"]),
    ])

    assert not report.references[0].valid
    assert report.references[1].valid
    assert report.references[1].use_for == ["weapon_shape"]
    assert report.references[1].do_not_copy == ["composition"]
    assert report.limitations
