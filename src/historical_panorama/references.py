from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import ReferenceInfo, ReferenceReport, ReferenceSpec

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_REFERENCES = 4


class ReferenceProcessor:
    """Validates references locally; it never invents visual descriptions."""

    def process(self, references: list[ReferenceSpec] | None = None) -> ReferenceReport:
        specs = references or []
        limitations: list[str] = []
        if len(specs) > MAX_REFERENCES:
            limitations.append(
                f"Only the first {MAX_REFERENCES} references were processed; {len(specs)} supplied"
            )
            specs = specs[:MAX_REFERENCES]
        results = [self._inspect(spec) for spec in specs]
        limitations.extend(issue for item in results for issue in item.issues)
        return ReferenceReport(results, limitations)

    def _inspect(self, spec: ReferenceSpec) -> ReferenceInfo:
        path = Path(spec.path)
        base = {
            "path": str(path),
            "use_for": list(spec.use_for),
            "do_not_copy": list(spec.do_not_copy),
        }
        if not path.is_file():
            return ReferenceInfo(**base, issues=[f"Reference does not exist: {path}"])
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                image_format = (image.format or path.suffix.lstrip(".")).upper()
                if image_format not in SUPPORTED_FORMATS:
                    return ReferenceInfo(
                        **base,
                        width=image.width,
                        height=image.height,
                        format=image_format,
                        issues=[f"Unsupported reference format {image_format}: {path}"],
                    )
                return ReferenceInfo(
                    **base,
                    valid=True,
                    width=image.width,
                    height=image.height,
                    format=image_format,
                    description_status="visual_analysis_unavailable" if spec.use_for else "not_requested",
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ReferenceInfo(
                **base,
                issues=[f"Unreadable reference {path}: {type(exc).__name__}: {exc}"],
            )


def reference_specs_from_json(items: object) -> list[ReferenceSpec]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("References JSON must be an array")
    result = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"Reference #{index + 1} must be an object containing path")
        result.append(
            ReferenceSpec(
                path=Path(str(item["path"])),
                use_for=[str(value) for value in item.get("use_for", [])],
                do_not_copy=[str(value) for value in item.get("do_not_copy", [])],
            )
        )
    return result
