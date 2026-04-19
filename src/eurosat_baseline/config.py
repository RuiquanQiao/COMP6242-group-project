from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "auto"))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "outputs/default"))


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
