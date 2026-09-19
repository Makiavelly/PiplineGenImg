import base64
import json
import time

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
STARTED = time.monotonic()

if not torch.cuda.is_available():
    raise RuntimeError("A Kaggle GPU accelerator is required but CUDA is unavailable")

width = int(REQUEST.get("width", 1024))
height = int(REQUEST.get("height", 512))
output_width = int(REQUEST.get("output_width", width))
output_height = int(REQUEST.get("output_height", height))
if width != height * 2 or output_width != output_height * 2:
    raise ValueError("Both generation and output dimensions must have a 2:1 aspect ratio")
if width % 8 or height % 8:
    raise ValueError("SDXL dimensions must be divisible by 8")

pipe = StableDiffusionXLPipeline.from_pretrained(
    REQUEST["model_id"],
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

seed = int(REQUEST.get("seed", 42))
generator = torch.Generator(device="cuda").manual_seed(seed)
image = pipe(
    prompt=REQUEST["prompt"],
    negative_prompt=REQUEST["negative_prompt"],
    width=width,
    height=height,
    num_inference_steps=int(REQUEST.get("steps", 30)),
    guidance_scale=float(REQUEST.get("guidance_scale", 7.0)),
    generator=generator,
).images[0]

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
            "steps": int(REQUEST.get("steps", 30)),
            "guidance_scale": float(REQUEST.get("guidance_scale", 7.0)),
            "prompt": REQUEST["prompt"],
            "negative_prompt": REQUEST["negative_prompt"],
            "model_id": REQUEST["model_id"],
            "model_version": getattr(pipe.config, "_commit_hash", None),
            "scheduler": pipe.scheduler.__class__.__name__,
            "elapsed_seconds": round(time.monotonic() - STARTED, 3),
            "result_path": "/kaggle/working/panorama.png",
        },
        file,
        indent=2,
    )

print(f"IMAGE_RESPONSE_READY run_id={REQUEST['run_id']} size={output_width}x{output_height}")
