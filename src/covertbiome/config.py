from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    cancer: str
    modalities: tuple[str, ...]
    latent_dim: int
    random_seed: int


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return ExperimentConfig(
        cancer=str(raw["cancer"]),
        modalities=tuple(raw["modalities"]),
        latent_dim=int(raw.get("latent_dim", 32)),
        random_seed=int(raw.get("random_seed", 42)),
    )
