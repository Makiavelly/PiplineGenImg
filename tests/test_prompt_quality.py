import pytest

from historical_panorama.kaggle_runner import KaggleError
from historical_panorama.providers.kaggle import KagglePromptBuilder


def test_prompt_quality_rejects_failed_model_fragment():
    with pytest.raises(KaggleError, match="quality gate"):
        KagglePromptBuilder._validate_prompt("1551 оду SOURCES, equirectangular panorama")


def test_prompt_quality_rejects_markdown_hallucination():
    prompt = "**Title:** Cretaceous Fortress " + "English historical detail " * 60 + " equirectangular"
    with pytest.raises(KaggleError, match="headings or Markdown"):
        KagglePromptBuilder._validate_prompt(prompt)


def test_prompt_quality_accepts_detailed_english_prompt():
    prompt = (
        "Sviyazhsk fortress construction in 1551 on a high wooded hill above the Volga and "
        "Sviyaga rivers. Russian carpenters in linen shirts, wool coats, leather boots and "
        "caps raise enormous prefabricated oak walls using ropes, axes and timber scaffolds. "
        "Log towers, palisades, unfinished gates, wood chips, carts and stacked beams surround "
        "the viewer under natural summer daylight. Show a busy populated outdoor worksite with "
        "physically plausible actions and documentary realism. Full 360-degree equirectangular "
        "panorama, spherical 360x180 field of view, seamless left and right edges, level horizon "
        "centered vertically, viewer at human eye height inside the scene, consistent scale, "
        "photorealistic historical reconstruction, no modern objects and no anachronisms."
    )
    KagglePromptBuilder._validate_prompt(prompt)
