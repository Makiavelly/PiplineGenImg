import base64
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))

event = REQUEST["event"]
context = REQUEST["context"][:12000]
instruction = f"""
Historical event requested by the user: {event}

Reference material (it may contain irrelevant search results; use only facts directly
connected to the requested event):
{context}

Write one plain ENGLISH visual scene description of 55-75 words for a historically accurate
reconstruction. Begin immediately with the event, exact year, place and the most important
visible action. Then describe terrain, architecture, construction material, people, clothing,
tools, daylight and camera surroundings. Prefer explicit facts from the most relevant source.
Never change a stated material, date, location, purpose or opposing force. If a detail is not
supported, omit it rather than inventing it. Show a populated outdoor event, not an empty or
completed structure. Do not include projection/camera-format instructions; those are added
separately. No headings, Markdown, analysis, citations, quotation marks, labels or lists.
Return only the compact English scene description.
""".strip()

if not torch.cuda.is_available():
    raise RuntimeError("The configured prompt model requires a Kaggle GPU, but CUDA is unavailable")
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(REQUEST["model_id"])
model = AutoModelForCausalLM.from_pretrained(
    REQUEST["model_id"],
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
messages = [
    {
        "role": "system",
        "content": "You are an expert historical reconstruction art director. Follow output language and format exactly.",
    },
    {"role": "user", "content": instruction},
]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(
    chat,
    return_tensors="pt",
    truncation=True,
    max_length=4096,
).to(device)
with torch.inference_mode():
    tokens = model.generate(
        **inputs,
        max_new_tokens=140,
        do_sample=False,
        repetition_penalty=1.1,
    )
generated = tokens[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated, skip_special_tokens=True).strip()

geometry = (
    "Full 360-degree equirectangular panorama, spherical 360x180 view, seamless wraparound "
    "environment, level horizon centered vertically, human eye height, photorealistic documentary reconstruction"
)
response = f"{geometry}. {response}"

negative = (
    "text, captions, watermark, logo, frame, border, split screen, collage, fisheye circle, "
    "little planet, cubemap, duplicated people, repeated objects, discontinuous seam, tilted horizon, "
    "modern clothing, modern buildings, cars, electrical wires, anachronisms, fantasy, illustration"
)

with open("/kaggle/working/prompt.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "run_id": REQUEST["run_id"],
            "prompt": response,
            "negative_prompt": negative,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

print(f"PROMPT_RESPONSE_READY run_id={REQUEST['run_id']} chars={len(response)}")
