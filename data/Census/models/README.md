# Models

This directory contains serialized model artifacts generated during analysis. Most models are stored directly as `.joblib` files. Large model artifacts that exceed GitHub's file size limits are stored as multiple file segments.

To reconstruct a segmented model artifact, concatenate the parts in order:

```bash
cat <model>.part.* > <model>.joblib
```

For example:

```bash
cat segmentation-2026_06_22-03_06_42.joblib.part.* > segmentation-2026_06_22-03_06_42.joblib
```

All model artifacts were generated using the package versions specified in the project's `requirements.txt`. Loading models with different package versions may produce warnings or unexpected behavior.

