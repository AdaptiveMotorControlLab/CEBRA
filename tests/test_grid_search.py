#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np

import cebra
import cebra.grid_search


def test_grid_search():
    X = np.random.uniform(0, 1, (1000, 50))
    X2 = np.random.uniform(0, 1, (800, 40))
    y_c = np.random.uniform(0, 1, (1000, 10))
    y_d = np.random.randint(0, 10, (1000))

    params_grid = dict(
        model_architecture=["offset10-model"],
        output_dimension=[3, 16],
        learning_rate=[0.001],
        time_offsets=10,
        max_iterations=5,
        batch_size=512,
        verbose=False,
    )

    datasets = {
        "dataset1": X,  # time contrastive learning
        "dataset2": (X, y_c),  # behavioral contrastive learning
        "dataset3": (X, y_d),  # behavioral contrastive learning
        "dataset4": (X, y_c, y_d),  # behavioral contrastive learning - hybrid
        "dataset5": (X2),  # time contrastive learning
    }

    grid_search = cebra.grid_search.GridSearch()
    grid_search.fit_models(params=params_grid,
                           datasets=datasets,
                           models_dir="saved_models")

    models, parameters = grid_search.load("saved_models")
    assert len(models) == len(parameters)

    best_model, best_model_name = grid_search.get_best_model(
        dataset_name="dataset1")
    assert "dataset1" in best_model_name
    assert best_model.__dict__["model_architecture"] == "offset10-model"
    assert best_model.__dict__["time_offsets"] == 10
    embedding = best_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    best_model, best_model_name = grid_search.get_best_model(
        dataset_name="dataset2")
    assert "dataset2" in best_model_name
    embedding = best_model.transform(X)
    assert isinstance(embedding, np.ndarray)

    best_model, best_model_name = grid_search.get_best_model(
        dataset_name="dataset5")
    assert "dataset5" in best_model_name
    embedding = best_model.transform(X2)
    assert isinstance(embedding, np.ndarray)

    df_results = grid_search.get_df_results()
    assert df_results.shape[0] == len(parameters)
    assert (df_results.shape[1] == len(parameters[0]) + 2
           )  # parameters to tune + dataset_name + loss
