import base64
import json
import os
import subprocess
import sys
import time

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "diffusers>=0.35,<1",
        "transformers>=4.46,<5",
        "accelerate>=1,<2",
        "sentencepiece>=0.2,<1",
        "protobuf>=4,<7",
    ]
)

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

STARTED = time.monotonic()

if not torch.cuda.is_available():
    raise RuntimeError("Stable Diffusion 3.5 Medium requires a Kaggle GPU")


def huggingface_token():
    """Read gated-model credentials from Kaggle Secrets without embedding them in the kernel."""
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not available to this Kaggle notebook. Open the "
                "historical-panorama-sd35-generator notebook, add or enable the HF_TOKEN "
                f"secret for it, and rerun. Kaggle Secrets error: {type(exc).__name__}"
            ) from exc
    if not token or not token.startswith("hf_"):
        raise RuntimeError(
            "HF_TOKEN is available but does not look like a Hugging Face user token"
        )
    return token


width = int(REQUEST.get("width", 1024))
height = int(REQUEST.get("height", 512))
output_width = int(REQUEST.get("output_width", width))
output_height = int(REQUEST.get("output_height", height))
if width != height * 2 or output_width != output_height * 2:
    raise ValueError("Both generation and output dimensions must have a 2:1 aspect ratio")
if width % 16 or height % 16:
    raise ValueError("Stable Diffusion 3.5 dimensions must be divisible by 16")

token = huggingface_token()
try:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        REQUEST["model_id"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        token=token,
    )
except OSError as exc:
    message = str(exc)
    if "gated" in message.lower() or "401" in message or "403" in message:
        raise RuntimeError(
            "HF_TOKEN was loaded, but its Hugging Face account cannot access the gated "
            "Stable Diffusion 3.5 model. Accept the model license with the same account "
            "that owns this token and ensure the token has read access to gated models."
        ) from exc
    raise

# A T4 cannot keep the transformer and all three text encoders in VRAM together.
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

seed = int(REQUEST.get("seed", 42))
steps = int(REQUEST.get("steps", 28))
guidance_scale = float(REQUEST.get("guidance_scale", 4.5))
max_sequence_length = int(REQUEST.get("max_sequence_length", 512))
generator = torch.Generator(device="cuda").manual_seed(seed)

generation_options = {
    "prompt": REQUEST["prompt"],
    "prompt_3": REQUEST["prompt"],
    "negative_prompt": REQUEST["negative_prompt"],
    "negative_prompt_3": REQUEST["negative_prompt"],
    "width": width,
    "height": height,
    "num_inference_steps": steps,
    "guidance_scale": guidance_scale,
    "max_sequence_length": max_sequence_length,
    "generator": generator,
}
if REQUEST.get("skip_layer_guidance", True):
    generation_options.update(
        {
            "skip_guidance_layers": [7, 8, 9],
            "skip_layer_guidance_scale": float(
                REQUEST.get("skip_layer_guidance_scale", 2.8)
            ),
            "skip_layer_guidance_start": float(
                REQUEST.get("skip_layer_guidance_start", 0.01)
            ),
            "skip_layer_guidance_stop": float(
                REQUEST.get("skip_layer_guidance_stop", 0.2)
            ),
        }
    )

with torch.inference_mode():
    image = pipe(**generation_options).images[0]

if image.size != (output_width, output_height):
    image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
image.save("/kaggle/working/panorama.png", format="PNG", optimize=True)

with open("/kaggle/working/generation.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "run_id": REQUEST["run_id"],
            "attempt": int(REQUEST.get("attempt", 1)),
            "seed": seed,
            "generation_width": width,
            "generation_height": height,
            "output_width": output_width,
            "output_height": output_height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "prompt": REQUEST["prompt"],
            "negative_prompt": REQUEST["negative_prompt"],
            "model_id": REQUEST["model_id"],
            "model_version": getattr(pipe.config, "_commit_hash", None),
            "scheduler": pipe.scheduler.__class__.__name__,
            "pipeline": pipe.__class__.__name__,
            "elapsed_seconds": round(time.monotonic() - STARTED, 3),
            "result_path": "/kaggle/working/panorama.png",
        },
        file,
        indent=2,
    )

print(f"IMAGE_RESPONSE_READY run_id={REQUEST['run_id']} size={output_width}x{output_height}")
