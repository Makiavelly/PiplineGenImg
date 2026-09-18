import base64
import json
import re

import torch
from jsonschema import Draft202012Validator
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
if not torch.cuda.is_available():
    raise RuntimeError("The prompt model requires a Kaggle GPU, but CUDA is unavailable")

tokenizer = AutoTokenizer.from_pretrained(REQUEST["model_id"])
model = AutoModelForCausalLM.from_pretrained(
    REQUEST["model_id"], torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda")

response_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_description", "used_fact_indices"],
    "properties": {
        "scene_description": {"type": "string", "minLength": 180, "maxLength": 1600},
        "used_fact_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        },
    },
}
analysis = REQUEST["analysis"]
references = REQUEST.get("references", {})
instruction = f"""
Create a compact English visual scene description for image generation using ONLY the
structured historical analysis below. Do not use raw source material and do not introduce
new historical requirements.

Describe the event, period, physical place, participants, eye-level observation point,
human actions, architecture, materials, clothing, weapons, environment, and relative layout
when supported. Apply must_include and must_not_include exactly. Treat may_include cautiously.
Do not give a precise appearance to anything in do_not_over_specify or marked unknown.
Never combine historical periods or substitute the modern appearance of the place.

Reference descriptions may be used only for their use_for fields. Never turn do_not_copy
elements into scene requirements. Empty descriptions mean that no visual analysis occurred.

Return ONLY JSON matching this schema: {json.dumps(response_schema)}
used_fact_indices are zero-based indices into analysis.facts actually used in the description.
The scene_description must be 65-110 English words, plain prose, without headings or Markdown.

Structured analysis: {json.dumps(analysis, ensure_ascii=False)}
Reference report: {json.dumps(references, ensure_ascii=False)}
""".strip()


def generate(text):
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You create grounded image prompts from structured facts."},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=7168).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=700, do_sample=False, repetition_penalty=1.05)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


validator = Draft202012Validator(response_schema)
raw = generate(instruction)
for attempt in range(2):
    try:
        result = parse_json(raw)
        errors = list(validator.iter_errors(result))
        word_count = len(result.get("scene_description", "").split())
        if not 65 <= word_count <= 110:
            errors.append(ValueError(f"scene_description has {word_count} words; expected 65-110"))
        if not errors:
            break
    except (json.JSONDecodeError, ValueError) as exc:
        errors = [exc]
    if attempt == 1:
        raise RuntimeError("Prompt JSON validation failed: " + "; ".join(str(e) for e in errors[:8]))
    raw = generate(
        instruction + "\nCorrect the previous invalid response. Errors: "
        + "; ".join(str(error) for error in errors[:8]) + "\nPrevious response: " + raw
    )

geometry = (
    "Seamless equirectangular 360-degree panorama, strict 2:1 aspect ratio, full 360° × 180° "
    "spherical view, camera at human eye level, continuous level horizon, consistent lighting "
    "around the entire circumference, no visible seam, no repeated objects, no mirrored "
    "duplicates, no excessive distortion near the poles."
)
prompt = f"{geometry} {result['scene_description']}"
negative = (
    "modern objects, anachronisms, text, letters, numbers, visible dates, captions, labels, "
    "city names, country names, geographic coordinates, maps, information signs, interface "
    "elements, answer-revealing text, watermark, logo, frame, border, split screen, collage, "
    "fisheye circle, little planet, cubemap, visible seam, discontinuous horizon, inconsistent "
    "lighting, repeated objects, duplicated people, mirrored duplicates, excessive polar "
    "distortion, deformed faces, extra limbs, missing limbs, floating objects"
)

with open("/kaggle/working/prompt.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "run_id": REQUEST["run_id"],
            "prompt": prompt,
            "negative_prompt": negative,
            "scene_description": result["scene_description"],
            "used_fact_indices": result["used_fact_indices"],
        },
        file,
        ensure_ascii=False,
        indent=2,
    )
print(f"PROMPT_RESPONSE_READY run_id={REQUEST['run_id']} chars={len(prompt)}")
