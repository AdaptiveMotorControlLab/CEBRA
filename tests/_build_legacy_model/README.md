# Helper script to build CEBRA checkpoints

This script builds CEBRA checkpoints for different versions of scikit-learn and CEBRA.
To build all models, run:

```bash
./generate.sh
```

The models are currently also stored in git directly due to their small size.

Related issue: https://github.com/AdaptiveMotorControlLab/CEBRA/issues/207
Related test: tests/test_sklearn_legacy.py
