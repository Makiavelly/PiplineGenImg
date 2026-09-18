import base64
import json
import re

import torch
from jsonschema import Draft202012Validator
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
if not torch.cuda.is_available():
    raise RuntimeError("The fact extraction model requires a Kaggle GPU, but CUDA is unavailable")

tokenizer = AutoTokenizer.from_pretrained(REQUEST["model_id"])
model = AutoModelForCausalLM.from_pretrained(
    REQUEST["model_id"], torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda")

materials = json.dumps(REQUEST["materials"], ensure_ascii=False)
schema_text = json.dumps(REQUEST["schema"], ensure_ascii=False)
instruction = f"""
Extract a source-grounded historical scene analysis from the Wikipedia material below.
Return ONLY one JSON object conforming exactly to the supplied JSON Schema.

Rules:
- Every fact must cite the exact Wikipedia article and section present in the material.
- Use confidence=supported only when the source states it directly.
- Use inferred only for a cautious conclusion and unknown when evidence is insufficient.
- Never present an assumption as established fact.
- Never invent a concrete appearance for an unknown building, costume, weapon or vehicle.
- Never add objects merely for spectacle, mix historical periods, or substitute the modern
  state of a place for its historical state.
- must_include contains only supported visually important elements.
- may_include contains cautious plausible elements.
- must_not_include contains anachronisms and source contradictions.
- do_not_over_specify identifies visual details whose exact appearance is unknown.
- Use empty strings/arrays when data is absent. Do not omit required properties.

Requested event: {REQUEST['event']}
Known collection limitations: {json.dumps(REQUEST['limitations'], ensure_ascii=False)}
JSON Schema: {schema_text}
Wikipedia materials: {materials[:28000]}
""".strip()


def generate(user_text):
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You extract auditable historical facts into strict JSON."},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=7168).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=2600,
            do_sample=False,
            repetition_penalty=1.05,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


validator = Draft202012Validator(REQUEST["schema"])
errors = []
raw = generate(instruction)
for attempt in range(2):
    try:
        result = parse_json(raw)
        errors = list(validator.iter_errors(result))
        if not errors:
            result["run_id"] = REQUEST["run_id"]
            with open("/kaggle/working/analysis.json", "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            print(f"FACTS_RESPONSE_READY run_id={REQUEST['run_id']} facts={len(result['facts'])}")
            break
    except (json.JSONDecodeError, ValueError) as exc:
        errors = [exc]
    if attempt == 1:
        raise RuntimeError("Model failed strict JSON validation: " + "; ".join(str(e) for e in errors[:8]))
    correction = (
        "Your previous response was invalid. Return the complete corrected JSON object only. "
        "Validation errors: " + "; ".join(str(error) for error in errors[:8]) +
        "\nPrevious response:\n" + raw
    )
    raw = generate(instruction + "\n\n" + correction)

