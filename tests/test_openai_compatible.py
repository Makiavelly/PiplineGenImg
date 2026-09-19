import base64
import io
import json
from pathlib import Path

from PIL import Image

from historical_panorama.models import (
    Fact,
    HistoricalAnalysis,
    PromptResult,
    ResearchResult,
    SceneConstraints,
    Source,
)
from historical_panorama.providers.openai_compatible import (
    OpenAICompatibleHistoricalFactExtractor,
    OpenAICompatibleImageGenerator,
    OpenAICompatiblePromptBuilder,
)


def analysis_dict():
    return {
        "identified_event": "Fortress construction",
        "date_or_period": "sixteenth century",
        "place": "a river hill",
        "participants": ["builders"],
        "event_type": "construction",
        "environment": ["summer riverside"],
        "architecture": ["unfinished timber walls"],
        "clothing": ["linen work clothes"],
        "weapons": [],
        "transport": ["river boats"],
        "everyday_objects": ["axes and ropes"],
        "natural_features": ["two rivers"],
        "visual_actions": ["raising wall sections"],
        "unknown_or_disputed": ["exact tower shape"],
        "possible_anachronisms": ["modern machinery"],
        "facts": [
            {
                "statement": "Builders raised timber walls.",
                "category": "architecture",
                "confidence": "supported",
                "article": "Event",
                "section": "Construction",
            }
        ],
        "constraints": {
            "must_include": ["timber construction"],
            "may_include": ["river boats"],
            "must_not_include": ["modern machinery"],
            "do_not_over_specify": ["exact tower shape"],
        },
    }


class FakeClient:
    def __init__(self, responses=None, image_response=None):
        self.responses = list(responses or [])
        self.image_response = image_response
        self.calls = []

    def response_text(self, model, instruction, max_output_tokens):
        self.calls.append((model, instruction, max_output_tokens))
        return self.responses.pop(0)

    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return self.image_response


def test_gpt_fact_extractor_uses_large_configurable_context():
    client = FakeClient([json.dumps(analysis_dict())])
    extractor = OpenAICompatibleHistoricalFactExtractor(
        client, "gpt-test", max_context_characters=50_000, max_source_characters=30_000
    )
    text = "historical material " * 1_500
    research = ResearchResult(
        "event",
        [Source("Event", "https://example.test", text, "Construction", "Event", True, 1.0)],
    )

    result = extractor.extract("event", research, "run-1")

    assert result.facts[0].article == "Event"
    assert len(client.calls[0][1]) > 14_000
    assert client.calls[0][2] == 5_000


def test_gpt_fact_extractor_retries_invalid_json_once():
    client = FakeClient(["not json", json.dumps(analysis_dict())])
    extractor = OpenAICompatibleHistoricalFactExtractor(client, "gpt-test")
    research = ResearchResult(
        "event",
        [Source("Event", "", "Builders raised timber walls.", "Construction", "Event")],
    )

    result = extractor.extract("event", research, "run-1")

    assert result.identified_event == "Fortress construction"
    assert len(client.calls) == 2
    assert "previous JSON was invalid" in client.calls[1][1]


def test_gpt_prompt_builder_preserves_panorama_contract():
    scene = " ".join(["historical"] * 70)
    client = FakeClient([json.dumps({"scene_description": scene, "used_fact_indices": [0]})])
    analysis = HistoricalAnalysis(
        identified_event="event",
        facts=[Fact("Builders raised timber walls.", "architecture", "supported", "Event", "Construction")],
        constraints=SceneConstraints(must_include=["timber walls"]),
    )
    research = ResearchResult(
        "event",
        [Source("Event", "https://example.test", "text", "Construction", "Event")],
        analysis=analysis,
    )

    result = OpenAICompatiblePromptBuilder(client, "gpt-test").build(
        "event", research, "run-1"
    )

    assert "seamless equirectangular 360-degree panorama" in result.prompt.lower()
    assert "strict 2:1 aspect ratio" in result.prompt.lower()
    assert "360° × 180°" in result.prompt
    assert result.used_facts == analysis.facts


def test_gpt_image_generator_decodes_and_converts_to_exact_two_to_one(tmp_path: Path):
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "red").save(buffer, format="PNG")
    client = FakeClient(
        image_response={
            "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii")}],
            "usage": {"images": 1},
        }
    )
    generator = OpenAICompatibleImageGenerator(
        client,
        "gpt-image-2",
        size="1536x1024",
        quality="high",
        output_width=200,
        output_height=100,
    )

    result = generator.generate(
        PromptResult(
            "Seamless equirectangular 360-degree panorama, strict 2:1 aspect ratio",
            "text, watermark",
            {"attempt": 1, "seed": 42},
        ),
        tmp_path,
        "run-1",
    )

    with Image.open(result.path) as image:
        assert image.size == (200, 100)
    payload = client.calls[0][2]["json"]
    assert payload["model"] == "gpt-image-2"
    assert "true seamless equirectangular 360-degree panorama" in payload["prompt"]
    assert "b64_json" not in (tmp_path / "generation.json").read_text(encoding="utf-8")
