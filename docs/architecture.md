# Architecture

The public POC uses a deliberately simple, reproducible architecture:

1. Load modality-specific tabular features indexed by patient ID.
2. Intersect shared patients across modalities.
3. Standardize numeric features within each modality.
4. Concatenate harmonized representations.
5. Learn a low-dimensional latent representation with a transparent baseline.
6. Evaluate whether latent structure associates with biologically meaningful endpoints.

This repository does not include proprietary causal reasoning, mechanistic graph construction, private training objectives, unpublished targets, or production model weights.
