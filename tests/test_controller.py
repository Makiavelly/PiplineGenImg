from historical_panorama.controller import RetryController
from historical_panorama.models import (
    TechnicalValidationReport,
    ValidationCheck,
    VisualIssue,
    VisualValidationReport,
)


def failed_technical():
    return TechnicalValidationReport("failed", [
        ValidationCheck("seam_color", 100, 45, "failed", "visible color discontinuity")
    ])


def test_controller_adds_concrete_correction_for_regeneration():
    decision = RetryController(3).decide(
        1, failed_technical(), VisualValidationReport("visual_validation_unavailable"),
        supports_inpainting=False,
    )
    assert decision.status == "retry_full_generation"
    assert "seam" in decision.correction


def test_controller_uses_inpainting_only_for_supported_local_issue():
    issue = VisualIssue("deformed_face", "face is malformed", 1, 0, 0, "critical", "local", "repair the face")
    decision = RetryController(3).decide(
        1, TechnicalValidationReport("passed", []), VisualValidationReport("failed", [issue]),
        supports_inpainting=True,
    )
    assert decision.use_inpainting


def test_controller_stops_after_configured_limit():
    decision = RetryController(3).decide(
        3, failed_technical(), VisualValidationReport("visual_validation_unavailable"),
        supports_inpainting=False,
    )
    assert decision.status == "manual_review_required"
    assert not decision.retry
