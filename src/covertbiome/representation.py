from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def build_latent_space(
    modalities: Mapping[str, pd.DataFrame], latent_dim: int = 32, random_seed: int = 42
) -> pd.DataFrame:
    """Public baseline: concatenate aligned modalities and project with PCA."""
    if not modalities:
        raise ValueError("At least one modality is required")

    frames = []
    for name, frame in modalities.items():
        renamed = frame.copy()
        renamed.columns = [f"{name}::{column}" for column in renamed.columns]
        frames.append(renamed)

    matrix = pd.concat(frames, axis=1, join="inner")
    if matrix.empty:
        raise ValueError("No aligned samples available")

    n_components = min(latent_dim, matrix.shape[0], matrix.shape[1])
    values = PCA(n_components=n_components, random_state=random_seed).fit_transform(matrix)
    columns = [f"z{i:03d}" for i in range(n_components)]
    return pd.DataFrame(np.asarray(values), index=matrix.index, columns=columns)
