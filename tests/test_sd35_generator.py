from pathlib import Path
from unittest.mock import Mock

from historical_panorama.kaggle_runner import KaggleKernelRunner
from historical_panorama.providers.kaggle import KaggleSD35ImageGenerator


def test_sd35_kernel_uses_sd3_pipeline_and_memory_saving():
    template = (
        Path(__file__).parents[1]
        / "src/historical_panorama/kaggle_templates/sd35_kernel.py.tpl"
    ).read_text(encoding="utf-8")

    assert "StableDiffusion3Pipeline" in template
    assert "StableDiffusionXLPipeline" not in template
    assert "enable_model_cpu_offload" in template
    assert "pipe.vae.enable_slicing()" in template
    assert "pipe.vae.enable_tiling()" in template
    assert 'get_secret("HF_TOKEN")' in template
    assert 'REQUEST.get("hf_token"' in template
    assert "Add HF_TOKEN to the local .env" in template
    assert "HF_TOKEN was loaded" in template
    assert '"max_sequence_length"' in template
    assert '"skip_guidance_layers": [7, 8, 9]' in template


def test_sd35_kernel_template_is_valid_python():
    path = (
        Path(__file__).parents[1]
        / "src/historical_panorama/kaggle_templates/sd35_kernel.py.tpl"
    )
    source = path.read_text(encoding="utf-8").replace("__PAYLOAD_BASE64__", "e30=")
    compile(source, str(path), "exec")


def test_sd35_kernel_title_resolves_to_configured_slug():
    assert (
        KaggleKernelRunner._safe_slug(KaggleSD35ImageGenerator.kernel_title)
        == "historical-panorama-sd35-generator"
    )


def test_sd35_generator_adds_token_only_to_private_payload(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    generator = KaggleSD35ImageGenerator(Mock(), "kernel", "model")

    assert generator._private_payload() == {"hf_token": "hf_test_token"}
    assert "hf_token" not in generator.options
