from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .models import PerspectiveFrame


class PerspectiveProjector:
    def __init__(self, frame_size: int = 512, fov_degrees: float = 90.0):
        self.frame_size = frame_size
        self.fov_degrees = fov_degrees

    def generate_standard_views(self, panorama_path: Path, output_dir: Path) -> list[PerspectiveFrame]:
        directions = [(float(yaw), 0.0) for yaw in range(0, 360, 45)]
        directions.extend([(0.0, 60.0), (0.0, -60.0)])
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(panorama_path) as image:
            panorama = np.asarray(image.convert("RGB"), dtype=np.float32)
        frames = []
        for number, (yaw, pitch) in enumerate(directions, start=1):
            rendered = self.render(panorama, yaw=yaw, pitch=pitch)
            name = f"frame_{number:02d}_yaw_{int(yaw):03d}_pitch_{int(pitch):+03d}.png"
            path = output_dir / name
            Image.fromarray(rendered).save(path)
            frames.append(PerspectiveFrame(number, yaw, pitch, str(path)))
        (output_dir / "frames.json").write_text(
            json.dumps([asdict(frame) for frame in frames], indent=2), encoding="utf-8"
        )
        return frames

    def render(self, panorama: np.ndarray, *, yaw: float, pitch: float) -> np.ndarray:
        size = self.frame_size
        tangent = math.tan(math.radians(self.fov_degrees) / 2)
        coordinates = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
        grid_x, grid_y = np.meshgrid(coordinates, -coordinates)
        x = grid_x * tangent
        y = grid_y * tangent
        z = np.ones_like(x)
        norm = np.sqrt(x * x + y * y + z * z)
        x, y, z = x / norm, y / norm, z / norm

        pitch_rad = math.radians(pitch)
        y, z = y * math.cos(pitch_rad) + z * math.sin(pitch_rad), -y * math.sin(pitch_rad) + z * math.cos(pitch_rad)
        yaw_rad = math.radians(yaw)
        x, z = x * math.cos(yaw_rad) + z * math.sin(yaw_rad), -x * math.sin(yaw_rad) + z * math.cos(yaw_rad)

        longitude = np.arctan2(x, z)
        latitude = np.arcsin(np.clip(y, -1, 1))
        height, width = panorama.shape[:2]
        source_x = (longitude / (2 * math.pi) + 0.5) * width
        source_y = (0.5 - latitude / math.pi) * height
        return self._bilinear(panorama, source_x, source_y)

    @staticmethod
    def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        x0 = np.floor(x).astype(np.int64) % width
        x1 = (x0 + 1) % width
        y0 = np.clip(np.floor(y).astype(np.int64), 0, height - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        dx = (x - np.floor(x))[..., None]
        dy = (y - np.floor(y))[..., None]
        top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
        bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
        return np.clip(top * (1 - dy) + bottom * dy, 0, 255).astype(np.uint8)

