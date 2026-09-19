import json
from pathlib import Path

import pytest

from historical_panorama.models import ResearchResult, Source
from historical_panorama.kaggle_runner import KaggleError
from historical_panorama.providers.kaggle import KaggleHistoricalFactExtractor
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


def test_fact_extractor_compacts_materials_without_urls_or_service_fields():
    sources = [
        Source(
            title=f"Article {index}",
            url=f"https://example.test/wiki/{index}",
            text=(" useful historical material " * 100),
            section="Lead" if index == 0 else f"Section {index}",
            article_title=f"Article {index}",
            is_primary=index == 0,
            relevance=1.0 - index / 100,
        )
        for index in range(30)
    ]

    materials = KaggleHistoricalFactExtractor._compact_materials(
        ResearchResult("event", sources)
    )

    assert materials[0]["article"] == "Article 0"
    assert len(materials) <= 18
    assert sum(len(item["text"]) for item in materials) <= 14000
    assert all(set(item) == {"source_id", "article", "section", "text"} for item in materials)


class AnalysisRunner:
    def __init__(self, output_dir, analysis):
        self.output_dir = output_dir
        self.analysis = analysis
        self.call = None

    def execute(self, **kwargs):
        self.call = kwargs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(self.analysis)
        payload["run_id"] = kwargs["payload"]["run_id"]
        (self.output_dir / "analysis.json").write_text(json.dumps(payload), encoding="utf-8")
        return self.output_dir


def test_fact_extractor_sends_four_bit_option_and_accepts_exact_citation(tmp_path):
    runner = AnalysisRunner(tmp_path / "output", valid_analysis())
    extractor = KaggleHistoricalFactExtractor(
        runner, "facts", "Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True
    )
    research = ResearchResult(
        "event",
        [Source("Event", "https://example.test", "Workers raised timber walls.",
                section="Construction", article_title="Event", is_primary=True)],
    )

    result = extractor.extract("event", research, "run-1")

    assert result.facts[0].article == "Event"
    assert runner.call["payload"]["load_in_4bit"] is True
    assert runner.call["payload"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert runner.call["payload"]["max_new_tokens"] == 1400


def test_fact_extractor_rejects_hallucinated_citation(tmp_path):
    data = valid_analysis()
    data["facts"][0]["article"] = "Invented article"
    runner = AnalysisRunner(tmp_path / "output", data)
    extractor = KaggleHistoricalFactExtractor(runner, "facts", "model")
    research = ResearchResult(
        "event",
        [Source("Event", "", "Historical material long enough.",
                section="Construction", article_title="Event", is_primary=True)],
    )
    with pytest.raises(KaggleError, match="citations absent"):
        extractor.extract("event", research, "run-1")


def test_fact_kernel_uses_schema_constrained_decoding_and_source_ids():
    template = (
        Path(__file__).parents[1]
        / "src/historical_panorama/kaggle_templates/facts_kernel.py.tpl"
    ).read_text(encoding="utf-8")
    assert "build_transformers_prefix_allowed_tokens_fn" in template
    assert '"source_id"' in template
    assert 'fact["article"] = source["article"]' in template
    assert 'REQUEST.get("max_new_tokens", 1400)' in template
    assert 'model_schema["properties"]["facts"]["maxItems"] = 8' in template
    assert 'fact["statement"] = sentence[:600].strip()' in template
    assert "eos_token_id=-1" not in template
    assert template.index("subprocess.check_call") < template.index("from transformers import")
