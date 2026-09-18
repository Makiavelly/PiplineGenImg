from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from .models import (
    Fact,
    HistoricalAnalysis,
    SceneConstraints,
    VisualIssue,
    VisualValidationReport,
)

FACT_CATEGORIES = [
    "event",
    "date_or_period",
    "place",
    "participants",
    "event_type",
    "environment",
    "architecture",
    "clothing",
    "weapons",
    "transport",
    "everyday_objects",
    "natural_features",
    "visual_actions",
    "unknown_or_disputed",
    "possible_anachronisms",
]

STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

HISTORICAL_ANALYSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "identified_event",
        "date_or_period",
        "place",
        "participants",
        "event_type",
        "environment",
        "architecture",
        "clothing",
        "weapons",
        "transport",
        "everyday_objects",
        "natural_features",
        "visual_actions",
        "unknown_or_disputed",
        "possible_anachronisms",
        "facts",
        "constraints",
    ],
    "properties": {
        "identified_event": {"type": "string"},
        "date_or_period": {"type": "string"},
        "place": {"type": "string"},
        "participants": STRING_ARRAY,
        "event_type": {"type": "string"},
        "environment": STRING_ARRAY,
        "architecture": STRING_ARRAY,
        "clothing": STRING_ARRAY,
        "weapons": STRING_ARRAY,
        "transport": STRING_ARRAY,
        "everyday_objects": STRING_ARRAY,
        "natural_features": STRING_ARRAY,
        "visual_actions": STRING_ARRAY,
        "unknown_or_disputed": STRING_ARRAY,
        "possible_anachronisms": STRING_ARRAY,
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "category", "confidence", "article", "section"],
                "properties": {
                    "statement": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": FACT_CATEGORIES},
                    "confidence": {
                        "type": "string",
                        "enum": ["supported", "inferred", "unknown"],
                    },
                    "article": {"type": "string"},
                    "section": {"type": "string"},
                },
            },
        },
        "constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "must_include",
                "may_include",
                "must_not_include",
                "do_not_over_specify",
            ],
            "properties": {
                "must_include": STRING_ARRAY,
                "may_include": STRING_ARRAY,
                "must_not_include": STRING_ARRAY,
                "do_not_over_specify": STRING_ARRAY,
            },
        },
    },
}

VISUAL_ISSUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "error_type", "description", "frame_number", "yaw", "pitch",
        "severity", "scope", "suggested_fix",
    ],
    "properties": {
        "error_type": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "frame_number": {"type": ["integer", "null"], "minimum": 1},
        "yaw": {"type": ["number", "null"]},
        "pitch": {"type": ["number", "null"]},
        "severity": {"enum": ["warning", "major", "critical"]},
        "scope": {"enum": ["local", "global"]},
        "suggested_fix": {"type": "string"},
    },
}

VISUAL_VALIDATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "issues", "explanation"],
    "properties": {
        "status": {"enum": ["passed", "failed"]},
        "issues": {"type": "array", "items": VISUAL_ISSUE_SCHEMA},
        "explanation": {"type": "string"},
    },
}

REFERENCE_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["descriptions"],
    "properties": {
        "descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "description"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "description": {"type": "string"},
                },
            },
        }
    },
}


class StructuredOutputError(ValueError):
    pass


def validate_historical_analysis(data: object) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(HISTORICAL_ANALYSIS_SCHEMA).iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise StructuredOutputError(f"Historical analysis JSON is invalid: {details}")
    return data  # type: ignore[return-value]


def historical_analysis_from_dict(data: object) -> HistoricalAnalysis:
    validated = validate_historical_analysis(data)
    constraints = SceneConstraints(**validated["constraints"])
    facts = [Fact(**fact) for fact in validated["facts"]]
    values = dict(validated)
    values["constraints"] = constraints
    values["facts"] = facts
    return HistoricalAnalysis(**values)


def visual_validation_from_dict(data: object) -> VisualValidationReport:
    errors = sorted(
        Draft202012Validator(VISUAL_VALIDATION_SCHEMA).iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise StructuredOutputError(f"Visual validation JSON is invalid: {details}")
    values = dict(data)  # type: ignore[arg-type]
    values["issues"] = [VisualIssue(**issue) for issue in values["issues"]]
    return VisualValidationReport(**values)


def validate_reference_descriptions(data: object) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(REFERENCE_DESCRIPTION_SCHEMA).iter_errors(data),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise StructuredOutputError(f"Reference description JSON is invalid: {details}")
    return data  # type: ignore[return-value]
