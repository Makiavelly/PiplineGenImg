import pytest

from historical_panorama.schemas import StructuredOutputError, historical_analysis_from_dict


def valid_analysis():
    return {
        "identified_event": "A historical event",
        "date_or_period": "sixteenth century",
        "place": "a river hill",
        "participants": ["builders"],
        "event_type": "construction",
        "environment": ["summer"],
        "architecture": ["timber walls"],
        "clothing": ["linen shirts"],
        "weapons": [],
        "transport": ["carts"],
        "everyday_objects": ["axes"],
        "natural_features": ["rivers"],
        "visual_actions": ["raising walls"],
        "unknown_or_disputed": ["exact tower shape"],
        "possible_anachronisms": ["modern machinery"],
        "facts": [{
            "statement": "Workers raised timber walls.",
            "category": "architecture",
            "confidence": "supported",
            "article": "Event",
            "section": "Construction",
        }],
        "constraints": {
            "must_include": ["timber construction"],
            "may_include": ["carts"],
            "must_not_include": ["modern machinery"],
            "do_not_over_specify": ["exact tower shape"],
        },
    }


def test_structured_analysis_accepts_cited_facts():
    result = historical_analysis_from_dict(valid_analysis())
    assert result.facts[0].confidence == "supported"
    assert result.facts[0].section == "Construction"


def test_structured_analysis_rejects_invalid_confidence_and_extra_fields():
    data = valid_analysis()
    data["facts"][0]["confidence"] = "certain"
    data["invented"] = True
    with pytest.raises(StructuredOutputError):
        historical_analysis_from_dict(data)
