from __future__ import annotations

from dataclasses import dataclass

from .models import TechnicalValidationReport, VisualValidationReport


@dataclass(frozen=True)
class AttemptDecision:
    status: str
    retry: bool
    use_inpainting: bool
    correction: str
    reasons: list[str]


class RetryController:
    TECHNICAL_FIXES = {
        "aspect_ratio_2_to_1": "preserve an exact 2:1 equirectangular canvas",
        "black_regions": "fill the complete spherical view; remove black or missing regions",
        "empty_regions": "add coherent scene detail to empty regions without inventing new objects",
        "corrupted_regions": "remove clipped or corrupted image regions",
        "sharpness": "increase coherent detail and reduce excessive blur",
        "seam_color": "make colors continuous across the left-right seam",
        "seam_brightness": "keep lighting and brightness continuous across the seam",
        "seam_structure": "continue structures naturally across the seam without duplication",
        "seam_horizon": "preserve one continuous level horizon across the seam",
        "dimensions": "use the configured output dimensions",
        "image_readable": "produce a valid readable PNG image",
        "file_exists": "produce the requested image file",
    }

    def __init__(self, max_full_generations: int = 3):
        if not 1 <= max_full_generations <= 3:
            raise ValueError("max_full_generations must be between 1 and 3")
        self.max_full_generations = max_full_generations

    def decide(
        self,
        attempt: int,
        technical: TechnicalValidationReport,
        visual: VisualValidationReport,
        *,
        supports_inpainting: bool,
    ) -> AttemptDecision:
        failed = [check for check in technical.checks if check.status == "failed"]
        critical = [issue for issue in visual.issues if issue.severity == "critical"]
        if not failed and not critical:
            return AttemptDecision("accepted", False, False, "", [])

        reasons = [f"{check.name}: {check.explanation}" for check in failed]
        reasons.extend(f"{issue.error_type}: {issue.description}" for issue in critical)
        local = bool(critical) and all(issue.scope == "local" for issue in critical) and not failed
        corrections = [self.TECHNICAL_FIXES.get(check.name, check.explanation) for check in failed]
        corrections.extend(issue.suggested_fix for issue in critical if issue.suggested_fix)
        correction = "; ".join(dict.fromkeys(value for value in corrections if value))
        if attempt >= self.max_full_generations:
            return AttemptDecision("manual_review_required", False, False, correction, reasons)
        if local and supports_inpainting:
            return AttemptDecision("retry_inpainting", True, True, correction, reasons)
        if not correction:
            correction = "correct the reported global consistency problems and do not repeat them"
        return AttemptDecision("retry_full_generation", True, False, correction, reasons)

