from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)


class KaggleError(RuntimeError):
    pass


@dataclass(frozen=True)
class KaggleSettings:
    username: str
    poll_interval_seconds: int = 15
    timeout_seconds: int = 1800
    work_dir: Path = Path(".kaggle-work")
    force_ipv4: bool = False


class KaggleKernelRunner:
    """Pushes a script kernel, waits for it, and downloads validated outputs."""

    TERMINAL_SUCCESS = {"complete"}
    TERMINAL_FAILURE = {
        "error",
        "cancelerror",
        "cancelled",
        "cancel_requested",
        "cancel_acknowledged",
    }

    def __init__(self, settings: KaggleSettings):
        self.settings = settings
        config_dir = settings.work_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(config_dir.resolve()))
        self.cli = shutil.which("kaggle")
        if not self.cli:
            venv_cli = Path(sys.executable).with_name("kaggle")
            if venv_cli.is_file():
                self.cli = str(venv_cli)

    def _cli_command(self) -> list[str]:
        if self.settings.force_ipv4:
            return [sys.executable, "-m", "historical_panorama.kaggle_cli_ipv4"]
        if not self.cli:
            raise KaggleError("Kaggle CLI is not installed. Run: pip install kaggle")
        return [self.cli]

    def check_connection(self) -> None:
        if not self.cli:
            raise KaggleError("Kaggle CLI is not installed. Run: pip install kaggle")
        if not (os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY")):
            raise KaggleError(
                "Kaggle credentials are missing. Set KAGGLE_API_TOKEN (preferred) or KAGGLE_KEY."
            )
        LOG.info("Kaggle: checking authentication for user %s ...", self.settings.username)
        try:
            self._run([*self._cli_command(), "kernels", "list", "--page-size", "1"], timeout=60)
        except KaggleError as exc:
            raise KaggleError(f"Kaggle connection/authentication check failed: {exc}") from exc
        LOG.info("Kaggle: connection and authentication succeeded.")

    def execute(
        self,
        *,
        kernel_slug: str,
        title: str,
        template_path: Path,
        payload: dict[str, object],
        expected_files: list[str],
        accelerator: str | None = None,
    ) -> Path:
        safe_slug = self._safe_slug(kernel_slug)
        kernel_ref = f"{self.settings.username}/{safe_slug}"
        stage_dir = self.settings.work_dir / safe_slug
        output_dir = stage_dir / "output"
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
        template = template_path.read_text(encoding="utf-8")
        script = template.replace("__PAYLOAD_BASE64__", encoded)
        if script == template:
            raise KaggleError(f"Payload marker is absent in template {template_path}")
        script_path = stage_dir / "kernel.py"
        same_request = (
            script_path.is_file()
            and script_path.read_text(encoding="utf-8") == script
        )
        script_path.write_text(script, encoding="utf-8")
        metadata = {
            "id": kernel_ref,
            "title": title,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": bool(accelerator),
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (stage_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        reuse_completed = same_request and self._remote_kernel_is_complete(kernel_ref)
        if reuse_completed:
            LOG.info(
                "Kaggle [%s]: identical request already completed; downloading existing response",
                kernel_ref,
            )
        else:
            LOG.info("Kaggle [%s]: uploading and starting kernel ...", kernel_ref)
            push_command = [*self._cli_command(), "kernels", "push", "-p", str(stage_dir)]
            if accelerator:
                push_command.extend(["--accelerator", accelerator])
                LOG.info("Kaggle [%s]: requested accelerator=%s", kernel_ref, accelerator)
            else:
                LOG.info("Kaggle [%s]: using CPU runtime", kernel_ref)
            push = self._run(push_command, timeout=180)
            LOG.info("Kaggle [%s]: upload accepted: %s", kernel_ref, self._one_line(push.stdout))
            self._wait(kernel_ref)

        LOG.info("Kaggle [%s]: downloading response ...", kernel_ref)
        self._download_outputs(kernel_ref, output_dir, expected_files)
        LOG.info("Kaggle [%s]: response received successfully (%s).", kernel_ref, expected_files)
        return output_dir

    def _remote_kernel_is_complete(self, kernel_ref: str) -> bool:
        try:
            result = self._run(
                [*self._cli_command(), "kernels", "status", kernel_ref], timeout=60
            )
        except KaggleError as exc:
            LOG.warning(
                "Kaggle [%s]: could not check completed request for recovery: %s",
                kernel_ref,
                exc,
            )
            return False
        return self._parse_status(f"{result.stdout}\n{result.stderr}".lower()) in self.TERMINAL_SUCCESS

    def _download_outputs(
        self, kernel_ref: str, output_dir: Path, expected_files: list[str]
    ) -> None:
        # Never let files from an older kernel version satisfy the output contract.
        for name in expected_files:
            path = output_dir / name
            if path.is_file():
                path.unlink()

        last_error: KaggleError | None = None
        for attempt in range(1, 6):
            try:
                self._run(
                    [
                        *self._cli_command(), "kernels", "output", kernel_ref,
                        "-p", str(output_dir), "-o",
                    ],
                    timeout=300,
                )
                missing = [
                    name for name in expected_files if not (output_dir / name).is_file()
                ]
                if not missing:
                    return
                last_error = KaggleError(
                    f"Kaggle response arrived, but files are missing: {missing}"
                )
            except KaggleError as exc:
                last_error = exc
            if attempt < 5:
                LOG.warning(
                    "Kaggle [%s]: response download failed (%d/5): %s",
                    kernel_ref,
                    attempt,
                    last_error,
                )
                time.sleep(self.settings.poll_interval_seconds)
        assert last_error is not None
        raise last_error

    def _wait(self, kernel_ref: str) -> None:
        deadline = time.monotonic() + self.settings.timeout_seconds
        last_status = None
        consecutive_errors = 0
        while time.monotonic() < deadline:
            try:
                result = self._run(
                    [*self._cli_command(), "kernels", "status", kernel_ref], timeout=60
                )
                consecutive_errors = 0
            except KaggleError as exc:
                consecutive_errors += 1
                LOG.warning(
                    "Kaggle [%s]: status request failed (%d/5): %s",
                    kernel_ref,
                    consecutive_errors,
                    exc,
                )
                if consecutive_errors >= 5:
                    raise KaggleError(
                        f"Kaggle status polling failed 5 times for {kernel_ref}: {exc}"
                    ) from exc
                time.sleep(self.settings.poll_interval_seconds)
                continue
            text = f"{result.stdout}\n{result.stderr}".lower()
            status = self._parse_status(text)
            if status != last_status:
                LOG.info("Kaggle [%s]: status=%s", kernel_ref, status)
                last_status = status
            if status in self.TERMINAL_SUCCESS:
                return
            if status in self.TERMINAL_FAILURE:
                log_tail = self._get_log_tail(kernel_ref)
                details = f" Log tail: {log_tail}" if log_tail else ""
                raise KaggleError(f"Kaggle kernel failed with status={status}.{details}")
            time.sleep(self.settings.poll_interval_seconds)
        raise KaggleError(f"Kaggle kernel timed out after {self.settings.timeout_seconds}s: {kernel_ref}")

    @staticmethod
    def _parse_status(text: str) -> str:
        quoted = re.search(r'\b(?:has\s+)?status\b[^a-z0-9]+["\']([^"\']+)["\']', text)
        if quoted:
            value = quoted.group(1)
        else:
            match = re.search(r"\b(?:has\s+)?status\b[^a-z0-9]+([a-z0-9_. -]+)", text)
            if not match:
                return "unknown"
            value = match.group(1).splitlines()[0]
        # Kaggle CLI 2.x may print enum values such as KernelWorkerStatus.COMPLETE.
        value = value.rsplit(".", 1)[-1]
        return value.strip().lower().replace(" ", "_")

    def _get_log_tail(self, kernel_ref: str) -> str:
        try:
            result = self._run([*self._cli_command(), "kernels", "logs", kernel_ref], timeout=60)
        except KaggleError as exc:
            LOG.warning("Kaggle [%s]: could not download failure log: %s", kernel_ref, exc)
            return ""
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return " | ".join(lines[-12:])[-3000:]

    @staticmethod
    def _safe_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        if not slug:
            raise ValueError("Kaggle kernel slug is empty")
        return slug[:50]

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())[:500]

    @staticmethod
    def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise KaggleError(f"Command timed out: {' '.join(command[:3])}") from exc
        except subprocess.CalledProcessError as exc:
            details = " ".join((exc.stderr or exc.stdout or "no details").split())
            raise KaggleError(f"Kaggle command failed ({exc.returncode}): {details}") from exc
