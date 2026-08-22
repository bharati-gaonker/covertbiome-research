from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.preprocessing import StandardScaler


def harmonize_modalities(modalities: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align modalities on shared patient IDs and z-score numeric features."""
    if not modalities:
        raise ValueError("At least one modality is required")

    shared = set.intersection(*(set(frame.index) for frame in modalities.values()))
    if not shared:
        raise ValueError("No shared patient IDs across modalities")

    ordered_ids = sorted(shared)
    harmonized: dict[str, pd.DataFrame] = {}
    for name, frame in modalities.items():
        aligned = frame.loc[ordered_ids].select_dtypes(include="number")
        scaled = StandardScaler().fit_transform(aligned)
        harmonized[name] = pd.DataFrame(scaled, index=ordered_ids, columns=aligned.columns)
    return harmonized
