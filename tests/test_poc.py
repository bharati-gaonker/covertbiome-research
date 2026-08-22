import pandas as pd

from covertbiome.data import harmonize_modalities
from covertbiome.representation import build_latent_space


def test_harmonize_and_latent_space():
    index = ["P1", "P2", "P3"]
    rna = pd.DataFrame({"g1": [1.0, 2.0, 3.0], "g2": [4.0, 3.0, 2.0]}, index=index)
    prot = pd.DataFrame({"p1": [2.0, 3.0, 4.0]}, index=index)

    harmonized = harmonize_modalities({"rna": rna, "proteomics": prot})
    latent = build_latent_space(harmonized, latent_dim=2)

    assert list(latent.index) == index
    assert latent.shape == (3, 2)
