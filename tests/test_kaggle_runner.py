from historical_panorama.kaggle_runner import KaggleKernelRunner


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
