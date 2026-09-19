from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import TechnicalValidationReport, ValidationCheck


class TechnicalPanoramaValidator:
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
                width, height = image.size
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

        status = "passed" if all(check.status == "passed" for check in checks) else "failed"
        return TechnicalValidationReport(status, checks)

    @staticmethod
    def _check(name: str, value: object, threshold: object, passed: bool, explanation: str) -> ValidationCheck:
        return ValidationCheck(name, value, threshold, "passed" if passed else "failed", explanation)


def technical_report_to_dict(report: TechnicalValidationReport) -> dict:
    return asdict(report)
