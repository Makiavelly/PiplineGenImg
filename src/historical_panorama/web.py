from __future__ import annotations

import argparse
import base64
import binascii
import errno
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from PIL import Image, UnidentifiedImageError


PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "web_static"
STAGE_KEYS = ("search", "fact_extractor", "prompt_builder", "image_generator", "visual_validator")
STAGE_LABELS = {
    "search": "Исторические источники",
    "fact_extractor": "Извлечение фактов",
    "prompt_builder": "Подготовка промпта",
    "image_generator": "Генерация панорамы",
    "visual_validator": "Визуальная проверка",
}
PROVIDER_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "search": {
        "wikipedia": {"label": "Wikipedia", "description": "MediaWiki API · без модели"},
    },
    "fact_extractor": {
        "kaggle": {"label": "Kaggle · Qwen 2.5", "description": "GPU notebook · 4-bit NF4"},
        "tooken": {"label": "Tooken · GPT 5.6", "description": "Большой контекст · Responses API"},
    },
    "prompt_builder": {
        "kaggle": {"label": "Kaggle · Qwen 2.5", "description": "Структурированный image prompt"},
        "tooken": {"label": "Tooken · GPT 5.6", "description": "Структурированный image prompt"},
    },
    "image_generator": {
        "kaggle_sd35": {
            "label": "Kaggle · SD 3.5",
            "description": "Stable Diffusion 3.5 Medium · текущий kernel только text-to-image",
            "supports_images": False,
        },
        "kaggle": {
            "label": "Kaggle · SDXL",
            "description": "SDXL text-to-image · текущий kernel не передаёт референсы",
            "supports_images": False,
        },
        "tooken": {
            "label": "Tooken · GPT Image 2",
            "description": "gpt-image-2 · текущий Tooken endpoint не передаёт референсы напрямую",
            "supports_images": False,
        },
    },
    "visual_validator": {
        "kaggle": {"label": "Kaggle · Qwen 2.5 VL", "description": "Проверка perspective-кадров"},
        "unavailable": {"label": "Пропустить", "description": "Только локальные технические проверки"},
    },
}
MAX_UPLOADS = 4
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
UPLOAD_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
PROVIDER_CONFIGS: dict[tuple[str, str], dict[str, Any]] = {
    ("fact_extractor", "tooken"): {
        "provider": "tooken",
        "model_id": "gpt-5.6-sol",
        "max_context_characters": 120000,
        "max_source_characters": 12000,
        "max_output_tokens": 5000,
    },
    ("prompt_builder", "tooken"): {
        "provider": "tooken",
        "model_id": "gpt-5.6-sol",
        "max_output_tokens": 1500,
    },
    ("image_generator", "tooken"): {
        "provider": "tooken",
        "model_id": "gpt-image-2",
        "size": "1536x1024",
        "quality": "high",
        "output_width": 2048,
        "output_height": 1024,
        "seed": 42,
    },
    ("image_generator", "kaggle"): {
        "provider": "kaggle",
        "kernel_slug": "historical-panorama-sdxl-generator",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "accelerator": "NvidiaTeslaT4",
        "width": 1024,
        "height": 512,
        "output_width": 2048,
        "output_height": 1024,
        "steps": 30,
        "guidance_scale": 6.5,
        "seed": 42,
    },
    ("visual_validator", "unavailable"): {"provider": "unavailable"},
}
STAGE_PROGRESS = {
    "created": (4, "Подготовка запуска"),
    "wikipedia_complete": (16, "Источники собраны"),
    "references_complete": (24, "Референсы проверены"),
    "analysis_complete": (39, "Исторические факты извлечены"),
    "prompt_complete": (51, "Промпт подготовлен"),
    "generation_complete": (72, "Панорама сгенерирована"),
    "attempt_validated": (88, "Результат проверен"),
    "complete": (100, "Готово"),
}
DEMO_STAGES = (
    ("created", "Конфигурация запуска сохранена"),
    ("wikipedia_complete", "Исторические источники найдены и ранжированы"),
    ("references_complete", "Входные материалы и ограничения проверены"),
    ("analysis_complete", "Исторические факты извлечены и структурированы"),
    ("prompt_complete", "Промпт для генератора подготовлен"),
    ("generation_complete", "Панорама сгенерирована"),
    ("attempt_validated", "Техническая и визуальная проверки пройдены"),
)
ARTIFACT_INFO = {
    "research.json": ("Материалы источников", "Wikipedia", "Извлечение фактов"),
    "references.json": ("Контекст изображений", "Анализ референсов", "Построение промпта"),
    "reference_analysis.json": ("Анализ изображений контекста", "Анализ референсов", "Построение промпта"),
    "analysis.json": ("Структурированные исторические факты", "Модель фактов", "Построение промпта"),
    "constraints.json": ("Ограничения сцены", "Модель фактов", "Построение промпта"),
    "prompt.json": ("Промпт генерации", "Модель промпта", "Генератор изображения"),
    "structured_description.json": ("Структурированное описание", "Модель промпта", "Генератор изображения"),
    "used_facts.json": ("Факты, использованные в промпте", "Модель промпта", "Отчёт запуска"),
    "sources.json": ("Источники промпта", "Модель промпта", "Отчёт запуска"),
    "generation.json": ("Параметры генерации", "Генератор изображения", "Валидаторы"),
    "technical_validation.json": ("Техническая проверка", "Технический валидатор", "Retry controller"),
    "visual_validation.json": ("Визуальная проверка", "Визуальная модель", "Retry controller"),
    "attempt_report.json": ("Решение по попытке", "Retry controller", "Следующая попытка / результат"),
    "manifest.json": ("Manifest результата", "Pipeline", "Администратор"),
    "final_report.json": ("Итоговый отчёт", "Pipeline", "Администратор"),
}
SECRET_KEY_MARKERS = ("token", "secret", "password", "api_key", "authorization")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_artifact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[скрыто]"
                if any(marker in str(key).lower() for marker in SECRET_KEY_MARKERS)
                else _safe_artifact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        limited = [_safe_artifact_value(item) for item in value[:100]]
        if len(value) > 100:
            limited.append(f"… ещё элементов: {len(value) - 100}")
        return limited
    if isinstance(value, str) and len(value) > 20_000:
        return value[:20_000] + f"\n… [сокращено символов: {len(value) - 20_000}]"
    return value


def _artifact_paths(run_dir: Path) -> list[Path]:
    root_names = (
        "research.json",
        "references.json",
        "reference_analysis.json",
        "analysis.json",
        "constraints.json",
        "prompt.json",
        "structured_description.json",
        "used_facts.json",
        "sources.json",
    )
    result = [run_dir / name for name in root_names if (run_dir / name).is_file()]
    attempts_dir = run_dir / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.glob("attempt_*")):
            for name in (
                "prompt.json",
                "generation.json",
                "technical_validation.json",
                "visual_validation.json",
                "attempt_report.json",
            ):
                path = attempt_dir / name
                if path.is_file():
                    result.append(path)
    for name in ("manifest.json", "final_report.json"):
        path = run_dir / name
        if path.is_file():
            result.append(path)
    return result


def artifact_revision(run_dir: Path | None) -> str:
    if run_dir is None:
        return "0"
    paths = _artifact_paths(run_dir)
    if not paths:
        return "0"
    return f"{len(paths)}:" + ":".join(
        f"{path.stat().st_mtime_ns}-{path.stat().st_size}" for path in paths
    )


def collect_artifacts(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    artifacts: list[dict[str, Any]] = []
    for path in _artifact_paths(run_dir):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        relative = path.relative_to(run_dir)
        title, producer, consumer = ARTIFACT_INFO.get(
            path.name, (path.stem.replace("_", " ").title(), "Pipeline", "Следующий этап")
        )
        if len(relative.parts) > 1:
            attempt = relative.parts[-2].replace("attempt_", "попытка ")
            title = f"{title} · {attempt}"
        artifacts.append(
            {
                "path": str(relative),
                "title": title,
                "producer": producer,
                "consumer": consumer,
                "content": _safe_artifact_value(content),
            }
        )
    return artifacts


def load_base_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    return config


def configured_pipeline(
    base: dict[str, Any],
    selections: dict[str, str],
    output_dir: Path,
    visual_validation_runs: int | None = None,
    technical_validation_enabled: bool | None = None,
) -> dict[str, Any]:
    """Apply frontend selections without allowing arbitrary config injection."""
    config = deepcopy(base)
    for stage in STAGE_KEYS:
        selected = selections.get(stage, str(config.get(stage, {}).get("provider", "")))
        if selected not in PROVIDER_PRESETS[stage]:
            raise ValueError(f"Unsupported provider for {stage}: {selected!r}")
        existing_provider = str(config.get(stage, {}).get("provider", ""))
        if selected == existing_provider:
            current = dict(config.get(stage, {}))
            current["provider"] = selected
            config[stage] = current
        else:
            config[stage] = deepcopy(
                PROVIDER_CONFIGS.get((stage, selected), {"provider": selected})
            )
    if "tooken" in selections.values():
        config.setdefault(
            "openai_compatible",
            {
                "base_url": "https://tooken.club/v1",
                "token_env": "GPT_TOKEN",
                "timeout_seconds": 300,
            },
        )
    if visual_validation_runs is not None:
        pipeline = dict(config.get("pipeline", {}))
        pipeline["visual_validation_runs"] = visual_validation_runs
        config["pipeline"] = pipeline
    if technical_validation_enabled is not None:
        pipeline = dict(config.get("pipeline", {}))
        pipeline["technical_validation_enabled"] = technical_validation_enabled
        config["pipeline"] = pipeline
    config["output_dir"] = str(output_dir)
    return config


def required_credentials(selections: dict[str, str]) -> list[str]:
    providers = set(selections.values())
    required: list[str] = []
    if any(value.startswith("kaggle") for value in providers):
        required.extend(["kaggle_username", "kaggle_token"])
    if selections.get("image_generator") == "kaggle_sd35":
        required.append("hf_token")
    if "tooken" in providers or "openai_compatible" in providers:
        required.append("gpt_token")
    return required


def image_provider_supports_uploads(provider: str) -> bool:
    metadata = PROVIDER_PRESETS["image_generator"].get(provider, {})
    return bool(metadata.get("supports_images", False))


def reference_processing_mode(selections: dict[str, str]) -> str:
    """Return the real route used for reference images by the selected providers."""
    if image_provider_supports_uploads(selections.get("image_generator", "")):
        return "generator"
    if selections.get("visual_validator") == "kaggle":
        return "visual_analyzer"
    return "unavailable"


def save_reference_uploads(
    job_root: Path,
    items: object,
    *,
    require_use_for: bool = True,
) -> Path | None:
    """Decode validated browser uploads and return a ReferenceSpec JSON file."""
    if items in (None, []):
        return None
    if not isinstance(items, list):
        raise ValueError("Фотографии должны быть переданы списком")
    if len(items) > MAX_UPLOADS:
        raise ValueError(f"Можно прикрепить не более {MAX_UPLOADS} фотографий")

    decoded: list[tuple[bytes, str, list[str], list[str]]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("data"), str):
            raise ValueError(f"Фотография #{index} передана в неверном формате")
        try:
            content = base64.b64decode(item["data"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"Не удалось декодировать фотографию #{index}") from exc
        if not content:
            raise ValueError(f"Фотография #{index} пустая")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Фотография #{index} превышает лимит 8 МБ")
        categories: dict[str, list[str]] = {}
        for key in ("use_for", "do_not_copy"):
            raw = item.get(key, [])
            if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
                raise ValueError(f"Поле {key} фотографии #{index} должно быть списком строк")
            categories[key] = list(
                dict.fromkeys(value.strip() for value in raw if value.strip())
            )
        if require_use_for and not categories["use_for"]:
            raise ValueError(
                f"Для фотографии #{index} укажите хотя бы одну категорию use_for"
            )
        decoded.append(
            (
                content,
                str(item.get("name", f"reference-{index}")),
                categories["use_for"],
                categories["do_not_copy"],
            )
        )

    uploads_dir = job_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    references: list[dict[str, Any]] = []
    for index, (content, original_name, use_for, do_not_copy) in enumerate(
        decoded, start=1
    ):
        temporary = uploads_dir / f"reference-{index}.upload"
        temporary.write_bytes(content)
        try:
            with Image.open(temporary) as image:
                image.verify()
                image_format = str(image.format or "").upper()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Файл «{original_name}» не является читаемым изображением") from exc
        suffix = UPLOAD_FORMATS.get(image_format)
        if suffix is None:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Формат файла «{original_name}» не поддерживается")
        destination = uploads_dir / f"reference-{index}{suffix}"
        temporary.replace(destination)
        references.append(
            {
                "path": str(destination.resolve()),
                "use_for": use_for,
                "do_not_copy": do_not_copy,
            }
        )

    references_path = job_root / "references.json"
    references_path.write_text(
        json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return references_path


@dataclass
class Job:
    id: str
    event: str
    root: Path
    created_at: str = field(default_factory=_utc_now)
    status: str = "queued"
    error: str = ""
    logs: list[str] = field(default_factory=list)
    run_dir: Path | None = None
    demo: bool = False
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.run_dir and (self.run_dir / "state.json").is_file():
            try:
                state = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        stage = str(state.get("stage", "created" if self.status == "running" else self.status))
        progress, stage_label = STAGE_PROGRESS.get(stage, (1, "Ожидание запуска"))
        if self.status == "failed":
            stage_label = "Ошибка выполнения"
        result = {
            "id": self.id,
            "event": self.event,
            "created_at": self.created_at,
            "status": self.status,
            "stage": stage,
            "stage_label": stage_label,
            "progress": progress,
            "attempt": state.get("current_attempt"),
            "error": self.error,
            "logs": self.logs[-80:],
            "has_image": bool(self.run_dir and (self.run_dir / "panorama.png").is_file()),
            "demo": self.demo,
            "artifacts_revision": artifact_revision(self.run_dir),
        }
        manifest_path = self.run_dir / "manifest.json" if self.run_dir else None
        if manifest_path and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result["result"] = {
                    "status": manifest.get("status"),
                    "attempts": manifest.get("attempts"),
                    "rejection_reasons": manifest.get("rejection_reasons", []),
                }
            except (OSError, ValueError):
                pass
        return result


class JobManager:
    def __init__(
        self,
        base_config: Path,
        jobs_root: Path,
        demo_image: Path | None = None,
        demo_step_seconds: float = 1.25,
    ):
        self.base_config_path = base_config
        self.base_config = load_base_config(base_config)
        self.jobs_root = jobs_root
        self.demo_image = (
            demo_image if demo_image is not None else base_config.parent / "fake.png"
        ).resolve()
        self.demo_step_seconds = max(0.0, demo_step_seconds)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def options(self) -> dict[str, Any]:
        stages = []
        for key in STAGE_KEYS:
            selected = str(self.base_config.get(key, {}).get("provider", ""))
            providers = [dict(id=provider, **metadata) for provider, metadata in PROVIDER_PRESETS[key].items()]
            stages.append({"id": key, "label": STAGE_LABELS[key], "selected": selected, "providers": providers})
        runs = int(self.base_config.get("pipeline", {}).get("visual_validation_runs", 1))
        return {
            "stages": stages,
            "technical_validation": {
                "enabled": bool(
                    self.base_config.get("pipeline", {}).get(
                        "technical_validation_enabled", True
                    )
                ),
            },
            "visual_validation": {
                "enabled": runs > 0
                and str(self.base_config.get("visual_validator", {}).get("provider", ""))
                != "unavailable",
                "runs": max(1, runs),
                "max_runs": 5,
            },
            "demo": {
                "available": self.demo_image.is_file(),
                "image_name": self.demo_image.name,
            },
        }

    def create(self, payload: dict[str, Any]) -> Job:
        event = str(payload.get("event", "")).strip()
        if not event:
            raise ValueError("Введите историческое событие или сцену")
        selections = payload.get("providers", {})
        credentials = payload.get("credentials", {})
        if not isinstance(selections, dict) or not isinstance(credentials, dict):
            raise ValueError("Invalid request payload")
        selections = {
            stage: str(
                selections.get(stage, self.base_config.get(stage, {}).get("provider", ""))
            )
            for stage in STAGE_KEYS
        }
        demo = bool(payload.get("demo", False))
        if demo and not self.demo_image.is_file():
            raise ValueError(f"Демонстрационное изображение не найдено: {self.demo_image}")
        visual_settings = payload.get("visual_validation", {})
        if not isinstance(visual_settings, dict):
            raise ValueError("Invalid visual validation settings")
        visual_enabled = bool(
            visual_settings.get(
                "enabled", selections.get("visual_validator") != "unavailable"
            )
        )
        try:
            visual_runs = int(visual_settings.get("runs", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Количество визуальных проверок должно быть целым числом") from exc
        if visual_enabled and not 1 <= visual_runs <= 5:
            raise ValueError("Количество визуальных проверок должно быть от 1 до 5")
        if not visual_enabled or selections.get("visual_validator") == "unavailable":
            selections["visual_validator"] = "unavailable"
            visual_runs = 0
        technical_settings = payload.get("technical_validation", {})
        if not isinstance(technical_settings, dict):
            raise ValueError("Invalid technical validation settings")
        technical_enabled = bool(technical_settings.get("enabled", True))
        uploads = payload.get("images", [])
        if uploads and not demo and reference_processing_mode(selections) == "unavailable":
            raise ValueError(
                "Ни генератор, ни выбранный визуальный анализатор не принимают "
                "референсные изображения."
            )
        missing = [] if demo else [
            name
            for name in required_credentials(selections)
            if not str(credentials.get(name, "")).strip()
        ]
        if missing:
            raise ValueError("Не заполнены данные доступа: " + ", ".join(missing))

        job_id = uuid.uuid4().hex[:12]
        job_root = self.jobs_root / job_id
        output_dir = job_root / "runs"
        job_root.mkdir(parents=True, exist_ok=False)
        references_path = save_reference_uploads(
            job_root, uploads, require_use_for=not demo
        )
        config = configured_pipeline(
            self.base_config,
            selections,
            output_dir,
            visual_validation_runs=visual_runs,
            technical_validation_enabled=technical_enabled,
        )
        config_path = job_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        job = Job(job_id, event, job_root, demo=demo)
        with self.lock:
            self.jobs[job_id] = job
        target = self._run_demo if demo else self._run
        args = (
            (job, selections, technical_enabled, visual_runs, references_path)
            if demo
            else (job, config_path, credentials, references_path)
        )
        thread = threading.Thread(
            target=target,
            args=args,
            name=f"panorama-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    @staticmethod
    def _write_demo_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _write_demo_artifacts(
        self,
        job: Job,
        stage: str,
        selections: dict[str, str],
        references_path: Path | None,
        technical_enabled: bool,
        visual_runs: int,
    ) -> None:
        assert job.run_dir is not None
        run_dir = job.run_dir
        analysis = {
            "identified_event": job.event,
            "date_or_period": "Исторический период определён по материалам источников",
            "place": "Место события уточнено моделью",
            "participants": ["Исторические участники события"],
            "event_type": "historical_reconstruction",
            "environment": ["Исторически достоверное окружение", "Естественное освещение"],
            "architecture": ["Архитектура соответствующего периода"],
            "clothing": ["Одежда соответствует эпохе и социальному положению"],
            "facts": [
                {
                    "statement": f"Сцена реконструирует событие: {job.event}",
                    "category": "event",
                    "confidence": "supported",
                    "article": "Исторический источник",
                    "section": "Основные сведения",
                }
            ],
            "constraints": {
                "must_include": ["Главное действие события", "Исторически достоверная среда"],
                "may_include": ["Участники на среднем и дальнем плане"],
                "must_not_include": ["Современные предметы", "Текст и водяные знаки"],
                "do_not_over_specify": ["Детали, не подтверждённые источниками"],
            },
        }
        prompt_text = (
            "Seamless equirectangular 360-degree panorama, 2:1 aspect ratio, "
            f"historically accurate cinematic reconstruction of {job.event}, "
            "human eye level, continuous level horizon, authentic period architecture, "
            "clothing and everyday objects, natural spatial composition, consistent lighting, "
            "360° × 180°, no visible seam, no repeated objects, no mirrored duplicates, "
            "no excessive distortion near the poles, highly detailed documentary realism."
        )
        negative_prompt = (
            "modern objects, visible dates, city names, country names, geographic coordinates, "
            "maps, information signs, interface elements, watermark, text, anachronisms, "
            "deformed people, duplicated characters, broken horizon"
        )

        if stage == "wikipedia_complete":
            self._write_demo_json(
                run_dir / "research.json",
                {
                    "query": job.event,
                    "primary_article": job.event,
                    "related_articles": ["Исторический контекст", "Участники события"],
                    "metadata": {
                        "title": job.event,
                        "summary": "Очищенные и ранжированные материалы для следующего этапа.",
                        "visual_reconstruction_notes": [
                            "Соблюдать материальную культуру периода",
                            "Не добавлять современные объекты",
                        ],
                    },
                    "sources": [
                        {
                            "title": job.event,
                            "url": "https://ru.wikipedia.org/",
                            "section": "Основные сведения",
                            "text": "Исторический материал очищен и подготовлен для модели фактов.",
                            "relevance": 1.0,
                            "is_primary": True,
                        }
                    ],
                    "limitations": [],
                },
            )
        elif stage == "references_complete":
            input_references = []
            if references_path is not None:
                loaded = json.loads(references_path.read_text(encoding="utf-8"))
                input_references = loaded if isinstance(loaded, list) else []
            report = {
                "references": [
                    {
                        "path": Path(str(item.get("path", "reference"))).name,
                        "use_for": item.get("use_for", []),
                        "do_not_copy": item.get("do_not_copy", []),
                        "valid": True,
                        "description_status": "ready_for_context",
                    }
                    for item in input_references
                    if isinstance(item, dict)
                ],
                "limitations": [],
            }
            self._write_demo_json(run_dir / "references.json", input_references)
            self._write_demo_json(run_dir / "reference_analysis.json", report)
        elif stage == "analysis_complete":
            self._write_demo_json(run_dir / "analysis.json", analysis)
            self._write_demo_json(run_dir / "constraints.json", analysis["constraints"])
        elif stage == "prompt_complete":
            prompt = {
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
                "metadata": {
                    "provider": selections["prompt_builder"],
                    "model_output": True,
                    "seed": 42,
                },
                "structured_description": analysis,
                "used_facts": analysis["facts"],
                "sources": [{"article": job.event, "section": "Основные сведения"}],
                "reference_paths": [],
            }
            self._write_demo_json(run_dir / "prompt.json", prompt)
            self._write_demo_json(run_dir / "structured_description.json", analysis)
            self._write_demo_json(run_dir / "used_facts.json", analysis["facts"])
            self._write_demo_json(run_dir / "sources.json", {"sources": prompt["sources"]})
        elif stage == "generation_complete":
            attempt_dir = run_dir / "attempts" / "attempt_01"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            self._write_demo_json(
                attempt_dir / "prompt.json",
                {"prompt": prompt_text, "negative_prompt": negative_prompt, "metadata": {"attempt": 1, "seed": 42}},
            )
            shutil.copy2(self.demo_image, attempt_dir / "panorama.png")
            shutil.copy2(self.demo_image, run_dir / "panorama.png")
            with Image.open(self.demo_image) as image:
                width, height = image.size
            self._write_demo_json(
                attempt_dir / "generation.json",
                {
                    "provider": selections["image_generator"],
                    "model_id": "configured-image-model",
                    "attempt": 1,
                    "seed": 42,
                    "width": width,
                    "height": height,
                    "prompt": prompt_text,
                    "negative_prompt": negative_prompt,
                },
            )
        elif stage == "attempt_validated":
            attempt_dir = run_dir / "attempts" / "attempt_01"
            technical = {
                "status": "passed" if technical_enabled else "disabled",
                "checks": [
                    {"name": "image_readable", "status": "passed", "explanation": "PNG успешно декодирован"},
                    {"name": "aspect_ratio", "status": "passed", "measured_value": "2:1"},
                    {"name": "seam", "status": "passed", "explanation": "Заметный шов не обнаружен"},
                ] if technical_enabled else [],
            }
            visual = {
                "status": "passed" if visual_runs else "visual_validation_unavailable",
                "runs": visual_runs,
                "issues": [],
                "explanation": "Исторических несоответствий и критических дефектов не обнаружено.",
            }
            self._write_demo_json(attempt_dir / "technical_validation.json", technical)
            self._write_demo_json(attempt_dir / "visual_validation.json", visual)
            self._write_demo_json(
                attempt_dir / "attempt_report.json",
                {
                    "attempt": 1,
                    "generation": {"provider": selections["image_generator"], "seed": 42},
                    "decision": {"status": "accepted", "retry": False, "reasons": []},
                },
            )

    def _run_demo(
        self,
        job: Job,
        selections: dict[str, str],
        technical_enabled: bool,
        visual_runs: int,
        references_path: Path | None,
    ) -> None:
        job.status = "running"
        job.run_dir = job.root / "runs" / f"demo-{job.id}"
        job.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = job.run_dir / "state.json"
        provider_summary = ", ".join(
            f"{stage}={provider}" for stage, provider in selections.items()
        )
        job.logs.append(f"Запуск {job.id} создан")
        job.logs.append(f"Провайдеры: {provider_summary}")
        try:
            for index, (stage, message) in enumerate(DEMO_STAGES, start=1):
                state: dict[str, Any] = {
                    "run_id": job.id,
                    "event": job.event,
                    "stage": stage,
                    "status": "running",
                    "demo": True,
                }
                if stage in {"generation_complete", "attempt_validated"}:
                    state["current_attempt"] = 1
                self._write_demo_json(state_path, state)
                self._write_demo_artifacts(
                    job,
                    stage,
                    selections,
                    references_path,
                    technical_enabled,
                    visual_runs,
                )
                job.logs.append(f"[{index}/7] {message}")
                time.sleep(self.demo_step_seconds)

            shutil.copy2(self.demo_image, job.run_dir / "panorama.png")
            reference_count = 0
            if references_path is not None:
                references = json.loads(references_path.read_text(encoding="utf-8"))
                reference_count = len(references) if isinstance(references, list) else 0
            manifest = {
                "run_id": job.id,
                "event": job.event,
                "status": "accepted",
                "attempts": 1,
                "image": str(job.run_dir / "panorama.png"),
                "rejection_reasons": [],
                "demo": True,
                "technical_validation_enabled": technical_enabled,
                "visual_validation_runs": visual_runs,
                "providers": selections,
                "context_images": reference_count,
            }
            self._write_demo_json(job.run_dir / "manifest.json", manifest)
            self._write_demo_json(
                job.run_dir / "final_report.json",
                {
                    "status": "accepted",
                    "attempts": 1,
                    "rejection_reasons": [],
                    "demo": True,
                },
            )
            self._write_demo_json(
                state_path,
                {
                    "run_id": job.id,
                    "event": job.event,
                    "stage": "complete",
                    "status": "accepted",
                    "attempts": 1,
                    "current_attempt": 1,
                    "demo": True,
                },
            )
            job.logs.append("DONE | Pipeline завершён, результат принят")
            job.status = "complete"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def artifacts(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        return {
            "revision": artifact_revision(job.run_dir),
            "artifacts": collect_artifacts(job.run_dir),
        }

    def _run(
        self,
        job: Job,
        config_path: Path,
        credentials: dict[str, Any],
        references_path: Path | None,
    ) -> None:
        job.status = "running"
        env = os.environ.copy()
        env.update(
            {
                "KAGGLE_USERNAME": str(credentials.get("kaggle_username", "")),
                "KAGGLE_API_TOKEN": str(credentials.get("kaggle_token", "")),
                "KAGGLE_KEY": str(credentials.get("kaggle_token", "")),
                "HF_TOKEN": str(credentials.get("hf_token", "")),
                "GPT_TOKEN": str(credentials.get("gpt_token", "")),
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "historical_panorama.web_worker",
            "--config",
            str(config_path),
            "--event",
            job.event,
        ]
        if references_path is not None:
            command.extend(["--references", str(references_path)])
        try:
            job.process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert job.process.stdout is not None
            for raw_line in job.process.stdout:
                line = raw_line.rstrip()
                if line:
                    job.logs.append(line)
                runs = sorted((job.root / "runs").glob("*/state.json"))
                if runs:
                    job.run_dir = runs[-1].parent
            code = job.process.wait()
            runs = sorted((job.root / "runs").glob("*/state.json"))
            if runs:
                job.run_dir = runs[-1].parent
            if code == 0:
                job.status = "complete"
            else:
                job.status = "failed"
                job.error = job.logs[-1] if job.logs else f"Pipeline exited with code {code}"
        except Exception as exc:  # subprocess boundary: surface every failure to the admin
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"


class AdminHandler(BaseHTTPRequestHandler):
    server_version = "HistoricalPanoramaAdmin/1.0"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/options":
            self._json(self.manager.options())
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job = self.manager.get(parts[2]) if len(parts) >= 3 else None
            if not job:
                self._json({"error": "Запуск не найден"}, HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 4 and parts[3] in {"image", "download"}:
                image = job.run_dir / "panorama.png" if job.run_dir else None
                if not image or not image.is_file():
                    self._json({"error": "Изображение ещё не готово"}, HTTPStatus.NOT_FOUND)
                    return
                self._file(
                    image,
                    "image/png",
                    download_name=("fake.png" if job.demo else "panorama.png")
                    if parts[3] == "download"
                    else None,
                )
                return
            if len(parts) == 4 and parts[3] == "artifacts":
                payload = self.manager.artifacts(job.id)
                self._json(payload or {"revision": "0", "artifacts": []})
                return
            self._json(job.snapshot())
            return
        if path in {"/", "/index.html"}:
            self._file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.css":
            self._file(STATIC_ROOT / "app.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._file(STATIC_ROOT / "app.js", "text/javascript; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 48 * 1024 * 1024:
                raise ValueError("Request is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON object expected")
            job = self.manager.create(payload)
            self._json(job.snapshot(), HTTPStatus.CREATED)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _file(
        self, path: Path, content_type: str, download_name: str | None = None
    ) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download_name:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download_name}"'
            )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data: blob:",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Historical Panorama admin web interface")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8080)
    result.add_argument("--config", type=Path, default=Path("config.yaml"))
    result.add_argument("--jobs-dir", type=Path, default=Path(".web-runs"))
    result.add_argument(
        "--demo-image",
        type=Path,
        help="Image returned by demonstration runs (defaults to fake.png near config)",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    manager = JobManager(args.config, args.jobs_dir, demo_image=args.demo_image)
    try:
        server = ThreadingHTTPServer((args.host, args.port), AdminHandler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        alternative = args.port + 1
        print(
            f"Не удалось запустить Admin UI: порт {args.port} уже занят.\n"
            f"Попробуйте другой порт:\n"
            f"  {Path(sys.executable).name} -m historical_panorama.web "
            f"--config {args.config} --port {alternative}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    server.manager = manager  # type: ignore[attr-defined]
    print(f"Admin UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
