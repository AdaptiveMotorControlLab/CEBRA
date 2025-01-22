import pathlib
import urllib.request

import numpy as np
import pytest

from cebra.integrations.sklearn.cebra import CEBRA

MODEL_VARIANTS = [
    "cebra-0.4.0-scikit-learn-1.4", "cebra-0.4.0-scikit-learn-1.6",
    "cebra-rc-scikit-learn-1.4", "cebra-rc-scikit-learn-1.6"
]


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS)
def test_load_legacy_model(model_variant):
    """Test loading a legacy CEBRA model."""

    X = np.random.normal(0, 1, (1000, 30))

    model_path = pathlib.Path(
        __file__
    ).parent / "_build_legacy_model" / f"cebra_model_{model_variant}.pt"

    if not model_path.exists():
        url = f"https://cebra.fra1.digitaloceanspaces.com/cebra_model_{model_variant}.pt"
        urllib.request.urlretrieve(url, model_path)

    loaded_model = CEBRA.load(model_path)

    assert loaded_model.model_architecture == "offset10-model"
    assert loaded_model.output_dimension == 8
    assert loaded_model.num_hidden_units == 16
    assert loaded_model.time_offsets == 10

    output = loaded_model.transform(X)
    assert isinstance(output, np.ndarray)
    assert output.shape[1] == loaded_model.output_dimension

    assert hasattr(loaded_model, "state_dict_")
    assert hasattr(loaded_model, "n_features_")
