import base64
import copy
import json
import re
import subprocess
import sys

REQUEST = json.loads(base64.b64decode("__PAYLOAD_BASE64__").decode("utf-8"))
runtime_packages = [
    "transformers>=4.49,<5",
    "accelerate>=0.34,<2",
    "lm-format-enforcer>=0.10,<1",
]
if REQUEST.get("load_in_4bit", False):
    runtime_packages.append("bitsandbytes>=0.45,<1")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", *runtime_packages]
)

import torch
from jsonschema import Draft202012Validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

if not torch.cuda.is_available():
    raise RuntimeError("The fact extraction model requires a Kaggle GPU, but CUDA is unavailable")

tokenizer = AutoTokenizer.from_pretrained(REQUEST["model_id"])
model_options = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
if REQUEST.get("load_in_4bit", False):
    model_options.update(
        {
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        }
    )
model = AutoModelForCausalLM.from_pretrained(REQUEST["model_id"], **model_options)
if not REQUEST.get("load_in_4bit", False):
    model = model.to("cuda")
model.eval()

materials = json.dumps(REQUEST["materials"], ensure_ascii=False)
model_schema = copy.deepcopy(REQUEST["schema"])
for property_name in (
    "participants", "environment", "architecture", "clothing", "weapons",
    "transport", "everyday_objects", "natural_features", "visual_actions",
    "unknown_or_disputed", "possible_anachronisms",
):
    model_schema["properties"][property_name]["maxItems"] = 4
model_schema["properties"]["facts"]["maxItems"] = 8
for constraint_schema in model_schema["properties"]["constraints"]["properties"].values():
    constraint_schema["maxItems"] = 4
model_schema["properties"]["facts"]["items"] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category", "confidence", "source_id"],
    "properties": {
        "category": REQUEST["schema"]["properties"]["facts"]["items"]["properties"]["category"],
        # A fact statement is copied from the cited Wikipedia fragment below, so facts
        # produced by this compact selection step are direct, supported assertions.
        "confidence": {"type": "string", "enum": ["supported"]},
        "source_id": {
            "type": "integer",
            "minimum": 0,
            "maximum": max(len(REQUEST["materials"]) - 1, 0),
        },
    },
}
instruction = f"""
Extract a source-grounded historical scene analysis from the Wikipedia material below.
Return ONLY one compact JSON object. Do not copy source text wholesale.

Rules:
- Every fact must cite one supplied material by its integer source_id.
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
- Write generated scene descriptions in concise English when possible. Article names,
  section names, and fact statements are restored from Wikipedia by the program.
- Return at most 8 facts and at most 4 concise entries in each descriptive array.
- Keep every entry to one short sentence or phrase. Prefer omission to repetition.
- Never repeat an assertion or generate a sequence of centuries, dates, or hypothetical events.
- For each fact, select category, confidence=supported and source_id only. Never generate
  statement, article or section: the program restores all three from the cited material.

Requested event: {REQUEST['event']}
Known collection limitations: {json.dumps(REQUEST['limitations'], ensure_ascii=False)}
Wikipedia materials: {materials}

Use exactly these keys and value types (the example strings are placeholders, not facts):
{{
  "identified_event": "string", "date_or_period": "string", "place": "string",
  "participants": ["string"], "event_type": "string", "environment": ["string"],
  "architecture": ["string"], "clothing": ["string"], "weapons": ["string"],
  "transport": ["string"], "everyday_objects": ["string"],
  "natural_features": ["string"], "visual_actions": ["string"],
  "unknown_or_disputed": ["string"], "possible_anachronisms": ["string"],
  "facts": [{{"category": "event|date_or_period|place|participants|event_type|environment|architecture|clothing|weapons|transport|everyday_objects|natural_features|visual_actions|unknown_or_disputed|possible_anachronisms", "confidence": "supported", "source_id": 0}}],
  "constraints": {{"must_include": ["string"], "may_include": ["string"],
    "must_not_include": ["string"], "do_not_over_specify": ["string"]}}
}}
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
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=6144).to("cuda")
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        tokenizer, JsonSchemaParser(model_schema)
    )
    max_new_tokens = int(REQUEST.get("max_new_tokens", 1400))
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    generated = output[0][prompt_length:]
    return (
        tokenizer.decode(generated, skip_special_tokens=True).strip(),
        int(generated.shape[0]),
        int(generated.shape[0]) >= max_new_tokens,
    )


def parse_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"):]
    if not candidate.startswith("{"):
        raise ValueError("Model response does not contain a JSON object")
    result, _end = json.JSONDecoder().raw_decode(candidate)
    return result


validator = Draft202012Validator(REQUEST["schema"])
errors = []
raw, generated_tokens, hit_token_limit = generate(instruction)
for attempt in range(2):
    try:
        result = parse_json(raw)
        model_errors = list(Draft202012Validator(model_schema).iter_errors(result))
        if model_errors:
            errors = model_errors
            raise ValueError("Constrained response did not match the internal schema")
        if result.get("facts") and not REQUEST["materials"]:
            raise ValueError("Facts were returned although no source materials were supplied")
        for fact in result.get("facts", []):
            source = REQUEST["materials"][fact.pop("source_id")]
            source_text = re.sub(r"\s+", " ", str(source["text"])).strip()
            sentence = re.split(r"(?<=[.!?])\s+", source_text, maxsplit=1)[0]
            fact["statement"] = sentence[:600].strip()
            fact["article"] = source["article"]
            fact["section"] = source["section"]
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
        diagnostic = raw[:2000].replace("\n", " ")
        raise RuntimeError(
            "Model failed strict JSON validation: "
            + "; ".join(str(e) for e in errors[:8])
            + f". Generated tokens: {generated_tokens}; token limit reached: {hit_token_limit}"
            + f". Raw response prefix: {diagnostic!r}"
        )
    correction = (
        "Your previous response was invalid or incomplete. Return a shorter, complete corrected "
        "JSON object only. Use no more than 6 facts and 3 entries per descriptive array. "
        "Validation errors: " + "; ".join(str(error) for error in errors[:8])
    )
    raw, generated_tokens, hit_token_limit = generate(instruction + "\n\n" + correction)
