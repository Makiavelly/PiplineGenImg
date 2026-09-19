from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from jsonschema import Draft202012Validator

from ..models import HistoricalAnalysis, ImageResult, PromptResult, ResearchResult
from ..schemas import (
    HISTORICAL_ANALYSIS_SCHEMA,
    StructuredOutputError,
    historical_analysis_from_dict,
)
from .kaggle import KagglePromptBuilder


class OpenAICompatibleClient:
    """Small Responses/Images client usable with OpenAI-compatible gateways."""

    def __init__(
        self,
        base_url: str,
        token_env: str = "GPT_TOKEN",
        timeout_seconds: int = 300,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _token(self) -> str:
        token = os.getenv(self.token_env, "").strip()
        if not token:
            raise ValueError(f"{self.token_env} is not set")
        return token

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        headers.update(
            {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}
        )
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        response = self.session.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            headers=headers,
            timeout=self.timeout_seconds,
            **kwargs,
        )
        if response.status_code >= 400:
            try:
                body = response.json()
                error = body.get("error", body) if isinstance(body, dict) else body
                message = error.get("message", error) if isinstance(error, dict) else error
            except (ValueError, AttributeError):
                message = response.text[:500]
            raise requests.HTTPError(
                f"Compatible API returned HTTP {response.status_code}: {message}",
                response=response,
            )
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Compatible API response must be a JSON object")
        if data.get("error"):
            error = data["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise requests.HTTPError(f"Compatible API returned an error: {message}")
        return data

    def check_connection(self) -> None:
        self.request("GET", "models")

    def response_text(self, model: str, instruction: str, max_output_tokens: int) -> str:
        data = self.request(
            "POST",
            "responses",
            json={
                "model": model,
                "input": instruction,
                "max_output_tokens": max_output_tokens,
            },
        )
        if isinstance(data.get("output_text"), str):
            return data["output_text"].strip()
        parts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        if not parts:
            raise ValueError("Responses API returned no text output")
        return "\n".join(parts).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") :]
    if not candidate.startswith("{"):
        raise ValueError("Model response does not contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(candidate)
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


class OpenAICompatibleHistoricalFactExtractor:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model_id: str,
        max_context_characters: int = 120_000,
        max_source_characters: int = 12_000,
        max_output_tokens: int = 5_000,
    ):
        if max_context_characters < 14_000:
            raise ValueError("max_context_characters must be at least 14000")
        self.client = client
        self.model_id = model_id
        self.max_context_characters = max_context_characters
        self.max_source_characters = max_source_characters
        self.max_output_tokens = max_output_tokens

    def extract(self, event: str, research: ResearchResult, run_id: str) -> HistoricalAnalysis:
        materials = self._materials(research)
        instruction = self._instruction(event, research, materials)
        errors: list[str] = []
        raw = ""
        for attempt in range(2):
            suffix = ""
            if attempt:
                suffix = (
                    "\nYour previous JSON was invalid. Return a complete corrected JSON object "
                    "only. Validation errors: " + "; ".join(errors[:8])
                )
            raw = self.client.response_text(
                self.model_id, instruction + suffix, self.max_output_tokens
            )
            try:
                data = _parse_json_object(raw)
                analysis = historical_analysis_from_dict(data)
                self._validate_citations(analysis, materials)
                return analysis
            except (ValueError, json.JSONDecodeError, StructuredOutputError) as exc:
                errors = [str(exc)]
        raise StructuredOutputError(
            "Historical analysis remained invalid after correction: "
            + "; ".join(errors[:8])
            + f". Response prefix: {raw[:600]!r}"
        )

    def _materials(self, research: ResearchResult) -> list[dict[str, str]]:
        ranked = sorted(
            research.sources,
            key=lambda source: (source.is_primary, source.relevance, len(source.text)),
            reverse=True,
        )
        result: list[dict[str, str]] = []
        total = 0
        seen: set[tuple[str, str]] = set()
        for source in ranked:
            article = source.article_title or source.title
            identity = (article, source.section)
            text = re.sub(r"\s+", " ", source.text).strip()[: self.max_source_characters]
            if identity in seen or not text:
                continue
            remaining = self.max_context_characters - total
            if remaining <= 0:
                break
            text = text[:remaining]
            result.append({"article": article, "section": source.section, "text": text})
            seen.add(identity)
            total += len(text)
        return result

    @staticmethod
    def _validate_citations(
        analysis: HistoricalAnalysis, materials: list[dict[str, str]]
    ) -> None:
        allowed = {(item["article"], item["section"]) for item in materials}
        invalid = [
            (fact.article, fact.section)
            for fact in analysis.facts
            if (fact.article, fact.section) not in allowed
        ]
        if invalid:
            raise StructuredOutputError(
                f"Analysis cites Wikipedia sections absent from the supplied context: {invalid[:5]}"
            )

    @staticmethod
    def _instruction(
        event: str, research: ResearchResult, materials: list[dict[str, str]]
    ) -> str:
        return f"""
Extract an auditable historical scene analysis from the supplied cleaned Wikipedia sections.
Return ONLY one JSON object matching the JSON Schema below, with no Markdown.

Every factual assertion must include its statement, category, confidence and the exact article
and section names from the supplied material. Use supported only for direct evidence, inferred
for cautious contextual conclusions and unknown when evidence is insufficient. Never present an
assumption as fact, invent the appearance of unknown buildings, add spectacle, mix historical
periods, or replace the historical place with its modern state. Populate must_include,
may_include, must_not_include and do_not_over_specify conservatively. Use empty values when data
is absent and never omit required fields.

Requested event: {event}
Collection limitations: {json.dumps(research.limitations, ensure_ascii=False)}
JSON Schema: {json.dumps(HISTORICAL_ANALYSIS_SCHEMA, ensure_ascii=False)}
Wikipedia materials: {json.dumps(materials, ensure_ascii=False)}
""".strip()


class OpenAICompatiblePromptBuilder:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model_id: str,
        max_output_tokens: int = 1_500,
    ):
        self.client = client
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens

    def build(self, event: str, research: ResearchResult, run_id: str) -> PromptResult:
        if research.analysis is None:
            raise ValueError("Historical analysis is required before prompt construction")
        instruction = self._instruction(research)
        errors: list[str] = []
        raw = ""
        result: dict[str, Any] | None = None
        for attempt in range(2):
            suffix = ""
            if attempt:
                suffix = "\nCorrect the JSON. Errors: " + "; ".join(errors[:8])
            raw = self.client.response_text(
                self.model_id, instruction + suffix, self.max_output_tokens
            )
            try:
                candidate = _parse_json_object(raw)
                self._validate_scene(candidate, len(research.analysis.facts))
                result = candidate
                break
            except (ValueError, json.JSONDecodeError) as exc:
                errors = [str(exc)]
        if result is None:
            raise StructuredOutputError(
                "Prompt builder remained invalid after correction: "
                + "; ".join(errors[:8])
                + f". Response prefix: {raw[:600]!r}"
            )

        geometry = (
            "Seamless equirectangular 360-degree panorama, strict 2:1 aspect ratio, full "
            "360° × 180° spherical view, camera at human eye level, continuous level horizon, "
            "consistent lighting around the entire circumference, no visible seam, no repeated "
            "objects, no mirrored duplicates, no excessive distortion near the poles."
        )
        negative = (
            "modern objects, anachronisms, text, letters, numbers, visible dates, captions, "
            "labels, city names, country names, geographic coordinates, maps, information signs, "
            "interface elements, answer-revealing text, watermark, logo, frame, border, split "
            "screen, collage, fisheye circle, little planet, cubemap, visible seam, discontinuous "
            "horizon, inconsistent lighting, repeated objects, duplicated people, mirrored "
            "duplicates, excessive polar distortion, deformed faces, extra limbs, missing limbs, "
            "floating objects"
        )
        prompt = f"{geometry} {result['scene_description']}"
        KagglePromptBuilder._validate_prompt(prompt, negative)
        indices = list(dict.fromkeys(result["used_fact_indices"]))
        used_facts = [research.analysis.facts[index] for index in indices]
        source_urls = {
            (source.article_title or source.title, source.section): source.url
            for source in research.sources
        }
        sources = [
            {
                "article": fact.article,
                "section": fact.section,
                "url": source_urls.get((fact.article, fact.section), ""),
            }
            for fact in used_facts
        ]
        return PromptResult(
            prompt=prompt,
            negative_prompt=negative,
            metadata={"provider": "openai_compatible", "model_id": self.model_id},
            structured_description=asdict(research.analysis),
            used_facts=used_facts,
            sources=list({(x["article"], x["section"]): x for x in sources}.values()),
        )

    @staticmethod
    def _validate_scene(data: dict[str, Any], fact_count: int) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["scene_description", "used_fact_indices"],
            "properties": {
                "scene_description": {"type": "string", "minLength": 180, "maxLength": 1100},
                "used_fact_indices": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "integer", "minimum": 0},
                },
            },
        }
        errors = list(Draft202012Validator(schema).iter_errors(data))
        indices = data.get("used_fact_indices", [])
        if any(index >= fact_count for index in indices if isinstance(index, int)):
            errors.append(ValueError("used_fact_indices contains an out-of-range index"))
        words = str(data.get("scene_description", "")).split()
        if not 65 <= len(words) <= 110:
            errors.append(ValueError(f"scene_description has {len(words)} words; expected 65-110"))
        if errors:
            raise ValueError("; ".join(str(error) for error in errors[:8]))

    @staticmethod
    def _instruction(research: ResearchResult) -> str:
        assert research.analysis is not None
        return f"""
Create a concise English visual scene description using ONLY the structured historical analysis.
Return ONLY JSON with keys scene_description (65-110 English words) and used_fact_indices (unique
zero-based indices). Describe the event, period, place, participants, human-eye observation point,
actions, architecture, materials, clothing, weapons, environment and relative layout only where
supported. Apply must_include and must_not_include exactly, treat may_include cautiously, and do
not specify details marked unknown or do_not_over_specify. Never introduce later consequences,
modern appearances, dates/signage visible inside the image, or facts absent from the analysis.
Reference descriptions may be used only for use_for; never copy do_not_copy elements.

Structured analysis: {json.dumps(asdict(research.analysis), ensure_ascii=False)}
Reference report: {json.dumps(asdict(research.references) if research.references else {}, ensure_ascii=False)}
""".strip()


class OpenAICompatibleImageGenerator:
    supports_image_conditioning = False
    supports_inpainting = False

    def __init__(self, client: OpenAICompatibleClient, model_id: str, **options: Any):
        self.client = client
        self.model_id = model_id
        self.options = options

    def generate(self, prompt: PromptResult, output_dir: Path, run_id: str) -> ImageResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_width = int(self.options.get("output_width", 2048))
        output_height = int(self.options.get("output_height", 1024))
        if output_width != output_height * 2:
            raise ValueError("GPT image output dimensions must have a 2:1 aspect ratio")
        size = str(self.options.get("size", "1536x1024"))
        quality = str(self.options.get("quality", "high"))
        complete_prompt = (
            f"{prompt.prompt}\n\nThe delivered image must be a true seamless equirectangular "
            "360-degree panorama covering the full 360° × 180° sphere, composed for an exact "
            "2:1 canvas with matching left and right edges and one continuous level horizon. "
            f"Strict exclusions: {prompt.negative_prompt}. "
            "Do not render any excluded element."
        )
        started = time.monotonic()
        response = self.client.request(
            "POST",
            "images/generations",
            json={
                "model": self.model_id,
                "prompt": complete_prompt,
                "size": size,
                "quality": quality,
                "n": 1,
            },
        )
        items = response.get("data", [])
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise ValueError("Images API returned no image")
        item = items[0]
        if isinstance(item.get("b64_json"), str):
            content = base64.b64decode(item["b64_json"], validate=True)
        elif isinstance(item.get("url"), str):
            download = requests.get(item["url"], timeout=self.client.timeout_seconds)
            download.raise_for_status()
            content = download.content
        else:
            raise ValueError("Images API returned neither b64_json nor url")

        with Image.open(io.BytesIO(content)) as source:
            image = source.convert("RGB")
            original_size = image.size
            original_path = output_dir / "original.png"
            image.save(original_path, format="PNG")
            image = self._fit_two_to_one(image)
            if image.size != (output_width, output_height):
                image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
            destination = output_dir / "panorama.png"
            image.save(destination, format="PNG", optimize=True)

        metadata = {
            "run_id": run_id,
            "attempt": int(prompt.metadata.get("attempt", 1)),
            "seed": int(prompt.metadata.get("seed", 42)),
            "seed_applied": False,
            "prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
            "provider": "openai_compatible",
            "model_id": self.model_id,
            "model_version": response.get("model", self.model_id),
            "requested_size": size,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "width": output_width,
            "height": output_height,
            "quality": quality,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "result_path": str(destination),
            "postprocess": "center_crop_to_2:1_then_resize"
            if original_size[0] != original_size[1] * 2
            else "resize_only",
            "usage": response.get("usage", {}),
        }
        (output_dir / "generation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ImageResult(destination, metadata)

    @staticmethod
    def _fit_two_to_one(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height * 2:
            return image
        if width > height * 2:
            target_width = height * 2
            left = (width - target_width) // 2
            return image.crop((left, 0, left + target_width, height))
        target_height = width // 2
        top = (height - target_height) // 2
        return image.crop((0, top, width, top + target_height))
