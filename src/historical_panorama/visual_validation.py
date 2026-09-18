from __future__ import annotations

from .models import (
    HistoricalAnalysis,
    PerspectiveFrame,
    ReferenceReport,
    VisualValidationReport,
)


class UnavailableVisualValidator:
    """Truthful fallback used when no multimodal model is configured."""

    def validate(
        self,
        frames: list[PerspectiveFrame],
        analysis: HistoricalAnalysis,
        references: ReferenceReport,
        run_id: str,
    ) -> VisualValidationReport:
        return VisualValidationReport(
            status="visual_validation_unavailable",
            explanation="No multimodal validation model is configured; no visual claims were made",
        )

