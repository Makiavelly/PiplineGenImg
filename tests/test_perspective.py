from pathlib import Path

import numpy as np
from PIL import Image

from historical_panorama.perspective import PerspectiveProjector


def test_perspective_projector_creates_required_views(tmp_path: Path):
    x = np.linspace(0, 255, 128, dtype=np.uint8)
    image = np.tile(x, (64, 1))
    rgb = np.stack((image, image, image), axis=2)
    panorama = tmp_path / "pano.png"
    Image.fromarray(rgb).save(panorama)

    frames = PerspectiveProjector(frame_size=24).generate_standard_views(
        panorama, tmp_path / "frames"
    )

    assert len(frames) == 10
    assert {(frame.yaw, frame.pitch) for frame in frames if frame.pitch == 0} == {
        (float(yaw), 0.0) for yaw in range(0, 360, 45)
    }
    assert {(frame.yaw, frame.pitch) for frame in frames if frame.pitch != 0} == {
        (0.0, 60.0), (0.0, -60.0)
    }
    assert all(Path(frame.path).is_file() for frame in frames)
