from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .models import TechnicalValidationReport, ValidationCheck


@dataclass(frozen=True)
class TechnicalThresholds:
    min_blur_variance: float = 35.0
    max_dark_tile_fraction: float = 0.20
    max_empty_tile_fraction: float = 0.25
    max_corrupt_tile_fraction: float = 0.20
    max_seam_color_mae: float = 45.0
    max_seam_brightness_delta: float = 30.0
    max_seam_structure_delta: float = 35.0
    max_horizon_delta: float = 0.12
    seam_strip_fraction: float = 0.02


class TechnicalPanoramaValidator:
    def __init__(self, thresholds: TechnicalThresholds | None = None):
        self.thresholds = thresholds or TechnicalThresholds()

    def validate(
        self, path: Path, *, expected_width: int | None = None, expected_height: int | None = None
    ) -> TechnicalValidationReport:
        checks: list[ValidationCheck] = []
        if not path.is_file():
            checks.append(self._check("file_exists", False, True, False, "Image file is missing"))
            return TechnicalValidationReport("failed", checks)
        checks.append(self._check("file_exists", True, True, True, "Image file exists"))
        try:
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32)
                width, height = rgb.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            checks.append(
                self._check(
                    "image_readable", type(exc).__name__, "readable image", False,
                    f"Image cannot be decoded: {exc}",
                )
            )
            return TechnicalValidationReport("failed", checks)
        checks.append(self._check("image_readable", True, True, True, "Image decoded successfully"))

        size_ok = width > 0 and height > 0
        if expected_width is not None:
            size_ok = size_ok and width == expected_width
        if expected_height is not None:
            size_ok = size_ok and height == expected_height
        expected = (
            f"{expected_width}x{expected_height}"
            if expected_width is not None and expected_height is not None
            else "positive dimensions"
        )
        checks.append(self._check("dimensions", f"{width}x{height}", expected, size_ok, "Image dimensions checked"))
        ratio = width / max(height, 1)
        checks.append(
            self._check(
                "aspect_ratio_2_to_1", round(ratio, 5), 2.0, abs(ratio - 2.0) <= 0.001,
                "Equirectangular panorama must have an exact 2:1 aspect ratio",
            )
        )

        gray = array.mean(axis=2)
        tile_means, tile_stds = self._tile_statistics(gray)
        dark_fraction = float(np.mean(tile_means < 3.0))
        checks.append(
            self._check(
                "black_regions", round(dark_fraction, 5), self.thresholds.max_dark_tile_fraction,
                dark_fraction <= self.thresholds.max_dark_tile_fraction,
                "Fraction of nearly black analysis tiles",
            )
        )
        empty_fraction = float(np.mean(tile_stds < 1.0))
        checks.append(
            self._check(
                "empty_regions", round(empty_fraction, 5), self.thresholds.max_empty_tile_fraction,
                empty_fraction <= self.thresholds.max_empty_tile_fraction,
                "Fraction of almost uniform analysis tiles",
            )
        )
        corrupt_fraction = float(np.mean(((tile_means < 1) | (tile_means > 254)) & (tile_stds < 0.5)))
        checks.append(
            self._check(
                "corrupted_regions", round(corrupt_fraction, 5),
                self.thresholds.max_corrupt_tile_fraction,
                corrupt_fraction <= self.thresholds.max_corrupt_tile_fraction,
                "Fraction of clipped uniform tiles that may indicate damaged output",
            )
        )

        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        blur_variance = float(np.var(laplacian)) if laplacian.size else 0.0
        checks.append(
            self._check(
                "sharpness", round(blur_variance, 4), self.thresholds.min_blur_variance,
                blur_variance >= self.thresholds.min_blur_variance,
                "Variance of a discrete Laplacian; low values indicate excessive blur",
            )
        )

        checks.extend(self._seam_checks(array))
        status = "passed" if all(check.status == "passed" for check in checks) else "failed"
        return TechnicalValidationReport(status, checks)

    def _seam_checks(self, array: np.ndarray) -> list[ValidationCheck]:
        height, width, _ = array.shape
        strip = max(2, int(width * self.thresholds.seam_strip_fraction))
        left = array[:, :strip]
        right = array[:, -strip:]
        color_mae = float(np.mean(np.abs(left - right)))
        brightness_delta = float(abs(left.mean() - right.mean()))
        left_gray, right_gray = left.mean(axis=2), right.mean(axis=2)
        left_structure = np.abs(np.diff(left_gray, axis=0)).mean(axis=1)
        right_structure = np.abs(np.diff(right_gray, axis=0)).mean(axis=1)
        structure_delta = float(np.mean(np.abs(left_structure - right_structure)))
        left_horizon = int(np.argmax(left_structure)) if left_structure.size else 0
        right_horizon = int(np.argmax(right_structure)) if right_structure.size else 0
        horizon_delta = abs(left_horizon - right_horizon) / max(height, 1)
        values = (
            ("seam_color", color_mae, self.thresholds.max_seam_color_mae, "Mean RGB difference"),
            (
                "seam_brightness", brightness_delta, self.thresholds.max_seam_brightness_delta,
                "Mean brightness change",
            ),
            (
                "seam_structure", structure_delta, self.thresholds.max_seam_structure_delta,
                "Vertical edge-profile difference",
            ),
            (
                "seam_horizon", horizon_delta, self.thresholds.max_horizon_delta,
                "Relative height difference between strongest horizontal boundaries",
            ),
        )
        return [
            self._check(name, round(value, 5), threshold, value <= threshold, explanation)
            for name, value, threshold, explanation in values
        ]

    @staticmethod
    def _tile_statistics(gray: np.ndarray, rows: int = 8, columns: int = 16) -> tuple[np.ndarray, np.ndarray]:
        means, stds = [], []
        for y_part in np.array_split(gray, rows, axis=0):
            for tile in np.array_split(y_part, columns, axis=1):
                means.append(float(tile.mean()) if tile.size else 0.0)
                stds.append(float(tile.std()) if tile.size else 0.0)
        return np.asarray(means), np.asarray(stds)

    @staticmethod
    def _check(name: str, value: object, threshold: object, passed: bool, explanation: str) -> ValidationCheck:
        return ValidationCheck(name, value, threshold, "passed" if passed else "failed", explanation)


def technical_report_to_dict(report: TechnicalValidationReport) -> dict:
    return asdict(report)

