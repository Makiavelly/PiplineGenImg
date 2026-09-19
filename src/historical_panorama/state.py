from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, **values: object) -> dict[str, Any]:
        state = self.load()
        state.update(values)
        self.write_json(self.path, state)
        return state

    @staticmethod
    def write_json(path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                default=lambda value: str(value) if isinstance(value, Path) else _raise_type(value),
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


def _raise_type(value: object) -> object:
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
