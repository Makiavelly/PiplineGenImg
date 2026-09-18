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
        "Seamless equirectangular 360-degree panorama, strict 2:1 aspect ratio, full 360° × 180° "
        "spherical view, camera at human eye level, continuous level horizon, consistent lighting "
        "around the entire circumference, no visible seam, no repeated objects, no mirrored "
        "duplicates, no excessive distortion near the poles. A timber fortress construction on a "
        "high wooded hill above two rivers. Carpenters in linen shirts, wool coats, leather boots and "
        "caps raise enormous prefabricated oak walls using ropes, axes and timber scaffolds. "
        "Log towers, palisades, unfinished gates, wood chips, carts and stacked beams surround "
        "the viewer under natural summer daylight. Show a busy populated outdoor worksite with "
        "physically plausible actions and documentary realism with consistent scale."
    )
    negative = (
        "modern objects, visible dates, city names, country names, geographic coordinates, maps, "
        "information signs, interface elements, watermark, text"
    )
    KagglePromptBuilder._validate_prompt(prompt, negative)
