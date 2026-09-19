import base64
import io
import json
import re
import subprocess
import sys

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "transformers>=4.49,<5",
        "accelerate>=0.34,<2",
        "lm-format-enforcer>=0.10,<1",
    ]
)

import torch
from jsonschema import Draft202012Validator
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

if not torch.cuda.is_available():
    raise RuntimeError("The multimodal validator requires a Kaggle GPU, but CUDA is unavailable")

processor = AutoProcessor.from_pretrained(REQUEST["model_id"])
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    REQUEST["model_id"],
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

image_records = REQUEST["images"]
images = [
    Image.open(io.BytesIO(base64.b64decode(item["image_base64"]))).convert("RGB")
    for item in image_records
]
schema = REQUEST["schema"]
validator = Draft202012Validator(schema)


def build_instruction():
    if REQUEST["mode"] == "describe_references":
        metadata = [
            {
                "index": item["index"],
                "use_for": item["use_for"],
                "do_not_copy": item["do_not_copy"],
            }
            for item in image_records
        ]
        return f"""
Analyze the reference images in their listed order. For each image, describe ONLY visual
details explicitly requested by use_for. Do not mention, describe, infer, or turn into a
requirement anything listed in do_not_copy. Do not identify a date, country, city, event, or
person unless that category is explicitly present in use_for. Keep every description factual,
compact, and usable as visual guidance. Return ONLY JSON matching this schema:
{json.dumps(schema)}
Reference metadata: {json.dumps(metadata, ensure_ascii=False)}
""".strip()

    frame_metadata = [
        {
            "frame_number": item["frame_number"],
            "yaw": item["yaw"],
            "pitch": item["pitch"],
        }
        for item in image_records
    ]
    analysis = REQUEST["analysis"]
    constraints = analysis["constraints"]
    return f"""
Inspect all perspective frames of one generated 360-degree historical panorama. Images are
provided in the same order as frame metadata. Return ONLY JSON matching this schema:
{json.dumps(schema)}

Check for deformed faces, extra or missing limbs, merged people, impossible building geometry,
floating objects, repeated objects, mirrored duplicates, disagreement between adjacent views,
text, watermarks, modern objects, anachronisms, missing must_include elements, present
must_not_include elements, and contradictions with the supplied structured Wikipedia analysis.

Do not treat absence of unknown details or do_not_over_specify details as an error. Do not add
new historical requirements from your own knowledge. Judge historical consistency only against
the supplied analysis. Use critical only when the panorama must be rejected; use major for a
clear but non-critical defect and warning for uncertainty. Use scope=local when a specific frame
region can be repaired, otherwise global. For global issues frame_number/yaw/pitch may be null.
For a frame-specific issue copy its exact number, yaw and pitch from metadata.

Frame metadata: {json.dumps(frame_metadata)}
Structured analysis: {json.dumps(analysis, ensure_ascii=False)}
must_include: {json.dumps(constraints['must_include'], ensure_ascii=False)}
must_not_include: {json.dumps(constraints['must_not_include'], ensure_ascii=False)}
do_not_over_specify: {json.dumps(constraints['do_not_over_specify'], ensure_ascii=False)}
Reference information: {json.dumps(REQUEST.get('references', {}), ensure_ascii=False)}
""".strip()


instruction = build_instruction()


def generate(text):
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=images, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        processor.tokenizer, JsonSchemaParser(schema)
    )
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(REQUEST.get("max_new_tokens", 1600)),
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    output = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def parse_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


raw = generate(instruction)
errors = []
for attempt in range(2):
    try:
        result = parse_json(raw)
        errors = list(validator.iter_errors(result))
        if not errors:
            break
    except (json.JSONDecodeError, ValueError) as exc:
        errors = [exc]
    if attempt == 1:
        raise RuntimeError(
            "Multimodal JSON validation failed: " + "; ".join(str(error) for error in errors[:8])
        )
    raw = generate(
        instruction
        + "\nCorrect the previous invalid response. Return the complete JSON object only. Errors: "
        + "; ".join(str(error) for error in errors[:8])
        + "\nPrevious response: "
        + raw
    )

result["run_id"] = REQUEST["run_id"]
filename = (
    "visual_validation.json"
    if REQUEST["mode"] == "validate_panorama"
    else "reference_descriptions.json"
)
with open(f"/kaggle/working/{filename}", "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
print(f"MULTIMODAL_RESPONSE_READY mode={REQUEST['mode']} run_id={REQUEST['run_id']}")
