from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--override", action="append", default=[])
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--log-level", default="INFO")
    return result


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def configuration_digest(path: Path, overrides: list[str]) -> str:
    configuration = OmegaConf.load(path)
    configuration = OmegaConf.merge(configuration, OmegaConf.from_dotlist(overrides))
    content = OmegaConf.to_yaml(configuration, resolve=True, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_metadata(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")

