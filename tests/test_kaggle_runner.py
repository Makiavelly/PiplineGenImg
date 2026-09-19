import sys
from unittest.mock import Mock

from historical_panorama.kaggle_runner import KaggleError
from historical_panorama.kaggle_runner import KaggleKernelRunner, KaggleSettings


def test_status_parser():
    assert KaggleKernelRunner._parse_status('kernel x/y has status "complete"') == "complete"
    assert (
        KaggleKernelRunner._parse_status('kernel x/y has status "KernelWorkerStatus.ERROR"')
        == "error"
    )
    assert (
        KaggleKernelRunner._parse_status('kernel x/y has status "KernelWorkerStatus.COMPLETE"')
        == "complete"
    )
    assert KaggleKernelRunner._parse_status('status: running') == "running"
    assert KaggleKernelRunner._parse_status('status = RUNNING\nmore output'.lower()) == "running"
    assert KaggleKernelRunner._parse_status('unrelated output') == "unknown"


def test_slug_is_sanitized():
    assert KaggleKernelRunner._safe_slug("My Prompt_builder!") == "my-prompt-builder"


def test_ipv4_setting_uses_project_cli_wrapper(tmp_path):
    runner = KaggleKernelRunner(
        KaggleSettings("user", work_dir=tmp_path, force_ipv4=True)
    )
    assert runner._cli_command() == [
        sys.executable, "-m", "historical_panorama.kaggle_cli_ipv4"
    ]


def test_status_polling_retries_transient_network_error(tmp_path, monkeypatch):
    runner = KaggleKernelRunner(
        KaggleSettings("user", poll_interval_seconds=0, timeout_seconds=10, work_dir=tmp_path)
    )
    responses = [RuntimeError("network"), Mock(stdout='status "complete"', stderr="")]

    def run(*_args, **_kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            from historical_panorama.kaggle_runner import KaggleError
            raise KaggleError(str(value))
        return value

    monkeypatch.setattr(runner, "_run", run)
    runner._wait("user/kernel")
    assert responses == []


def test_output_download_retries_and_removes_stale_files(tmp_path, monkeypatch):
    runner = KaggleKernelRunner(
        KaggleSettings("user", poll_interval_seconds=0, work_dir=tmp_path)
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "generation.json").write_text("stale", encoding="utf-8")
    calls = 0

    def run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KaggleError("temporary DNS failure")
        (output / "generation.json").write_text("fresh", encoding="utf-8")
        (output / "panorama.png").write_bytes(b"png")
        return Mock(stdout="", stderr="")

    monkeypatch.setattr(runner, "_run", run)
    runner._download_outputs(
        "user/kernel", output, ["panorama.png", "generation.json"]
    )

    assert calls == 2
    assert (output / "generation.json").read_text(encoding="utf-8") == "fresh"


def test_completed_remote_kernel_can_be_reused(tmp_path, monkeypatch):
    runner = KaggleKernelRunner(KaggleSettings("user", work_dir=tmp_path))
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: Mock(stdout='status "complete"', stderr=""),
    )
    assert runner._remote_kernel_is_complete("user/kernel") is True
