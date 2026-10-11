"""Per-dataset source configs (``configs/sources/<name>.yaml``) and class mapping."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fpvdd.labels import CLASS_NAMES


@dataclass(frozen=True)
class SourceConfig:
    """Parsed source config.

    ``class_map`` maps a source class name to one of our class ids, or to None, meaning the
    box is dropped but the frame is kept (it then serves as a negative example).
    """

    name: str
    class_map: dict[str, int | None]
    options: dict[str, Any] = field(default_factory=dict)

    def map_class(self, source_class: str) -> int | None:
        """Our class id for ``source_class``; KeyError if the config does not cover it."""
        key = source_class.lower()
        if key not in self.class_map:
            raise KeyError(f"{self.name}: source class {source_class!r} missing in class_map")
        return self.class_map[key]


def parse_class_map(raw: dict[str, str | None]) -> dict[str, int | None]:
    """Validate ``{source_class: our_class_name | null}`` and resolve names to ids."""
    mapping: dict[str, int | None] = {}
    for source, target in raw.items():
        if target is None:
            mapping[str(source).lower()] = None
        elif target in CLASS_NAMES:
            mapping[str(source).lower()] = CLASS_NAMES.index(target)
        else:
            raise ValueError(f"unknown target class {target!r} for {source!r}; use {CLASS_NAMES}")
    return mapping


def load_source_config(path: Path) -> SourceConfig:
    """Load and validate a source config YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("name", "class_map"):
        if key not in data:
            raise ValueError(f"{path}: missing key {key!r}")
    return SourceConfig(
        name=str(data["name"]),
        class_map=parse_class_map(data["class_map"]),
        options=dict(data.get("options") or {}),
    )
