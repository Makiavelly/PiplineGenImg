from __future__ import annotations

import argparse
import base64
import binascii
import errno
import json
import os
import subprocess
import sys
import threading
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
    },
    "prompt_builder": {
        "kaggle": {"label": "Kaggle · Qwen 2.5", "description": "Структурированный image prompt"},
    },
    "image_generator": {
        "kaggle_sd35": {
            "label": "Kaggle · SD 3.5",
            "description": "Stable Diffusion 3.5 Medium · только текст",
            "supports_images": False,
        },
        "kaggle": {
            "label": "Kaggle · SDXL",
            "description": "Совместимый Kaggle image kernel · только текст",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_base_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    return config


def configured_pipeline(base: dict[str, Any], selections: dict[str, str], output_dir: Path) -> dict[str, Any]:
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
    config["output_dir"] = str(output_dir)
    return config


def required_credentials(selections: dict[str, str]) -> list[str]:
    providers = set(selections.values())
    required: list[str] = []
    if any(value.startswith("kaggle") for value in providers):
        required.extend(["kaggle_username", "kaggle_token"])
    if selections.get("image_generator") == "kaggle_sd35":
        required.append("hf_token")
    return required


def image_provider_supports_uploads(provider: str) -> bool:
    metadata = PROVIDER_PRESETS["image_generator"].get(provider, {})
    return bool(metadata.get("supports_images", False))


def save_reference_uploads(job_root: Path, items: object) -> Path | None:
    """Decode validated browser uploads and return a ReferenceSpec JSON file."""
    if items in (None, []):
        return None
    if not isinstance(items, list):
        raise ValueError("Фотографии должны быть переданы списком")
    if len(items) > MAX_UPLOADS:
        raise ValueError(f"Можно прикрепить не более {MAX_UPLOADS} фотографий")

    decoded: list[tuple[bytes, str]] = []
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
        decoded.append((content, str(item.get("name", f"reference-{index}"))))

    uploads_dir = job_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    references: list[dict[str, Any]] = []
    for index, (content, original_name) in enumerate(decoded, start=1):
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
                "use_for": [],
                "do_not_copy": [],
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
    def __init__(self, base_config: Path, jobs_root: Path):
        self.base_config_path = base_config
        self.base_config = load_base_config(base_config)
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def options(self) -> dict[str, Any]:
        stages = []
        for key in STAGE_KEYS:
            selected = str(self.base_config.get(key, {}).get("provider", ""))
            providers = [dict(id=provider, **metadata) for provider, metadata in PROVIDER_PRESETS[key].items()]
            stages.append({"id": key, "label": STAGE_LABELS[key], "selected": selected, "providers": providers})
        return {"stages": stages}

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
        uploads = payload.get("images", [])
        if uploads and not image_provider_supports_uploads(selections["image_generator"]):
            raise ValueError(
                "Выбранный генератор не поддерживает изображения в контексте. "
                "Удалите фотографии или выберите совместимую модель."
            )
        missing = [name for name in required_credentials(selections) if not str(credentials.get(name, "")).strip()]
        if missing:
            raise ValueError("Не заполнены данные доступа: " + ", ".join(missing))

        job_id = uuid.uuid4().hex[:12]
        job_root = self.jobs_root / job_id
        output_dir = job_root / "runs"
        job_root.mkdir(parents=True, exist_ok=False)
        references_path = save_reference_uploads(job_root, uploads)
        config = configured_pipeline(self.base_config, selections, output_dir)
        config_path = job_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        job = Job(job_id, event, job_root)
        with self.lock:
            self.jobs[job_id] = job
        thread = threading.Thread(
            target=self._run,
            args=(job, config_path, credentials, references_path),
            name=f"panorama-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

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
            if len(parts) == 4 and parts[3] == "image":
                image = job.run_dir / "panorama.png" if job.run_dir else None
                if not image or not image.is_file():
                    self._json({"error": "Изображение ещё не готово"}, HTTPStatus.NOT_FOUND)
                    return
                self._file(image, "image/png")
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

    def _file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if path.suffix == ".png" else "public, max-age=300")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:")
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
    return result


def main() -> None:
    args = parser().parse_args()
    manager = JobManager(args.config, args.jobs_dir)
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
