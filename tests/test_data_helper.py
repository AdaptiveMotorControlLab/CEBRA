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
import pytest
import scipy.stats
import sklearn.model_selection

import cebra.data.helper as cebra_data_helper
import cebra.datasets


def _initialize_orthogonal_alignment_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))
    embedding_100_1d = np.random.uniform(0, 1, (100, 1))
    embedding_100_none = np.random.uniform(0, 1, (100,))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))
    labels_100_3d = np.random.uniform(0, 1, (100, 3))
    labels_150_3d = np.random.uniform(0, 1, (150, 3))
    labels_100_none = np.random.uniform(0, 1, (100,))
    labels_1_1d = np.random.uniform(0, 1, (1, 1))

    orthogonal_alignment_data = []

    # different sample sizes
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_1d, labels_150_1d))
    # different sample sizes and more than one feature in the labels
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_3d, labels_150_3d))
    # embedding to align has a single sample
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_1_4d, labels_100_1d, labels_1_1d))
    # embeddings with one feature dimension only
    orthogonal_alignment_data.append(
        (embedding_100_1d, embedding_100_1d, labels_100_1d, labels_100_1d))
    # embeddings with one feature dimension only, as None
    orthogonal_alignment_data.append(
        (embedding_100_none, embedding_100_none, labels_100_1d, labels_100_1d))
    # labels with one feature dimension only, as None
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_100_4d, labels_100_none, labels_100_none))
    # same sample size, no labels
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_100_4d, None, None))

    return orthogonal_alignment_data


def _initialize_invalid_orthogonal_alignment_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_110_6d = np.random.uniform(0, 1, (110, 6))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))
    labels_110_1d = np.random.uniform(0, 1, (110, 1))
    labels_1_1d = np.random.uniform(0, 1, (1, 1))

    orthogonal_alignment_data = []

    # Embeddings with different number of features
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_110_6d, labels_100_1d, labels_110_1d,
         "Invalid.*data"))
    # Reference embeddings and labels are not the same sample size
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_150_1d, labels_150_1d,
         "Mismatched.*data.*labels"))
    # Embeddings and labels to align are not the same sample size
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, labels_100_1d, labels_100_1d,
         "Mismatched.*data.*labels"))
    # Reference embedding contain only one sample
    orthogonal_alignment_data.append(
        (embedding_1_4d, embedding_100_4d, labels_1_1d, labels_100_1d,
         "Invalid.*reference.*data"))
    # Missing reference label
    orthogonal_alignment_data.append((embedding_100_4d, embedding_150_4d, None,
                                      labels_100_1d, "Missing.*labels"))
    # Missing alignment label
    orthogonal_alignment_data.append((embedding_100_4d, embedding_150_4d,
                                      labels_100_1d, None, "Missing.*labels"))
    # No labels for embeddings that are not already aligned on time (not the same number of samples)
    orthogonal_alignment_data.append(
        (embedding_100_4d, embedding_150_4d, None, None, "Missing.*labels"))

    return orthogonal_alignment_data


def _does_shape_match(data, aligned_data):
    if len(data.shape) == 2:
        is_matching = aligned_data.shape == data.shape
    else:
        is_matching = aligned_data.shape == (data.shape[0], 1)
    return is_matching


@pytest.mark.parametrize("ref_data,data,ref_labels,labels",
                         _initialize_orthogonal_alignment_data())
def test_orthogonal_alignment_shapes(ref_data, data, ref_labels, labels):
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    alignment_model.fit(ref_data, data, ref_labels, labels)
    aligned_embedding = alignment_model.transform(data)
    assert _does_shape_match(data, aligned_embedding)

    aligned_embedding = alignment_model.fit_transform(ref_data, data,
                                                      ref_labels, labels)
    assert _does_shape_match(data, aligned_embedding)

    # Test with non-default parameters
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=10)

    aligned_embedding = alignment_model.fit_transform(ref_data, data,
                                                      ref_labels, labels)
    assert _does_shape_match(data, aligned_embedding), (data.shape,
                                                        aligned_embedding.shape)


@pytest.mark.parametrize("ref_data,data,ref_labels,labels,match",
                         _initialize_invalid_orthogonal_alignment_data())
def test_invalid_orthogonal_alignment(ref_data, data, ref_labels, labels,
                                      match):
    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    with pytest.raises(ValueError, match=match):
        alignment_model.fit(ref_data, data, ref_labels, labels)


def test_orthogonal_alignment_without_labels():
    random_seed = 2160
    np.random.seed(random_seed)
    embedding_100_4d = np.random.uniform(0, 1, (1000, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (1000, 4))

    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()

    alignment_model.fit(embedding_100_4d, embedding_100_4d_2, None, None)
    aligned_embedding = alignment_model.transform(embedding_100_4d_2)

    alignment_model.fit(embedding_100_4d, embedding_100_4d_2)
    aligned_embedding_without_labels = alignment_model.transform(
        embedding_100_4d_2)

    assert np.allclose(aligned_embedding, aligned_embedding_without_labels)


@pytest.mark.parametrize("seed", [483, 425, 166, 672, 123])
def test_orthogonal_alignment(seed):
    np.random.seed(seed)
    embedding_100_4d = np.random.uniform(0, 1, (1000, 4))
    orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=4, random_state=seed)
    labels_100_1d = np.random.uniform(0, 1, (1000, 1))

    alignment_model = cebra_data_helper.OrthogonalProcrustesAlignment()
    aligned_embedding = alignment_model.fit_transform(ref_data=embedding_100_4d,
                                                      data=np.dot(
                                                          embedding_100_4d,
                                                          orthogonal_matrix),
                                                      ref_label=labels_100_1d,
                                                      label=labels_100_1d)
    assert np.allclose(aligned_embedding, embedding_100_4d, atol=0.03)

    # and without labels
    aligned_embedding = alignment_model.fit_transform(ref_data=embedding_100_4d,
                                                      data=np.dot(
                                                          embedding_100_4d,
                                                          orthogonal_matrix))
    assert np.allclose(aligned_embedding, embedding_100_4d, atol=0.03)


def _initialize_embedding_ensembling_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_3 = np.random.uniform(0, 1, (100, 4))
    embedding_110_none = np.random.uniform(0, 1, (110,))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_2 = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_3 = np.random.uniform(0, 1, (100, 1))

    embedding_ensembling_data = []

    # Embeddings of same shape, no labels
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, 0))
    # Embeddings with a single feature dimension (as None)
    embedding_ensembling_data.append(([embedding_110_none,
                                       embedding_110_none], None, 0))
    # Embeddings and corresponding labels to use for alignment (optional)
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3],
         [labels_100_1d, labels_100_1d_2, labels_100_1d_3], 0))
    # Parallel processing, with 2 jobs
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, 2))
    # Parallel processing, with all jobs available
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3], None, -1))

    return embedding_ensembling_data


def _initialize_invalid_embedding_ensembling_data():
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_3 = np.random.uniform(0, 1, (100, 4))
    embedding_150_4d = np.random.uniform(0, 1, (150, 4))
    embedding_100_6d = np.random.uniform(0, 1, (100, 6))
    embedding_1_4d = np.random.uniform(0, 1, (1, 4))

    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    labels_100_1d_2 = np.random.uniform(0, 1, (100, 1))
    labels_150_1d = np.random.uniform(0, 1, (150, 1))

    embedding_ensembling_data = []

    # Number of jobs set to None
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_4d_3], None, None, "Invalid.*n_jobs"))
    # Embeddings with different sample sizes
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_150_4d], None, 0, "Inconsistent.*embeddings"))
    # Embeddings with different number of features
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_6d], None, 0, "Inconsistent.*embeddings"))
    # Embeddings and labels with labels with a different sample size as their corresponding embedding
    embedding_ensembling_data.append(
        ([embedding_100_4d, embedding_100_4d_2,
          embedding_100_4d_3], [labels_100_1d, labels_100_1d_2,
                                labels_150_1d], 0, "Mismatched.*data.*labels"))
    # Embedding with a single sample (need at least {k_top})
    embedding_ensembling_data.append(
        ([embedding_1_4d, embedding_1_4d], None, 0, "Invalid.*data"))

    return embedding_ensembling_data


@pytest.mark.parametrize("embeddings,labels,n_jobs",
                         _initialize_embedding_ensembling_data())
def test_embedding_ensembling_shapes(embeddings, labels, n_jobs):

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=embeddings, labels=labels, n_jobs=n_jobs)
    assert _does_shape_match(embeddings[0], joint_embedding)


def test_embeddings_ensembling_without_labels():
    random_seed = 2160
    np.random.seed(random_seed)

    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    embedding_100_4d_2 = np.random.uniform(0, 1, (100, 4))

    # same sample size, no labels
    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2], labels=[None, None])
    joint_embedding_without_labels = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2])
    assert np.allclose(joint_embedding, joint_embedding_without_labels)


@pytest.mark.parametrize("embeddings,labels,n_jobs,match",
                         _initialize_invalid_embedding_ensembling_data())
def test_invalid_embedding_ensembling(embeddings, labels, n_jobs, match):
    with pytest.raises(ValueError, match=match):
        _ = cebra_data_helper.ensemble_embeddings(
            embeddings=embeddings,
            labels=labels,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize("seed", [483, 426, 166, 674, 123])
def test_embedding_ensembling(seed):
    np.random.seed(seed)
    embedding_100_4d = np.random.uniform(0, 1, (100, 4))
    labels_100_1d = np.random.uniform(0, 1, (100, 1))
    orthogonal_matrix = scipy.stats.ortho_group.rvs(dim=4, random_state=seed)
    orthogonal_matrix_2 = scipy.stats.ortho_group.rvs(dim=4,
                                                      random_state=seed + 1)

    embedding_100_4d_2 = np.dot(embedding_100_4d, orthogonal_matrix)
    embedding_100_4d_3 = np.dot(embedding_100_4d, orthogonal_matrix_2)

    labels = [labels_100_1d, labels_100_1d, labels_100_1d]

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3],
        labels=labels)
    assert np.allclose(joint_embedding, embedding_100_4d, atol=0.05)

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=[embedding_100_4d, embedding_100_4d_2, embedding_100_4d_3])
    assert np.allclose(joint_embedding, embedding_100_4d, atol=0.05)


@pytest.mark.benchmark
@pytest.mark.requires_dataset
def test_ensembling_performances(n_models=3):
    dataset = cebra.datasets.init("rat-hippocampus-single-achilles")

    scores, embeddings = [], []
    for i in range(n_models):
        cebra_model = cebra.CEBRA(model_architecture="offset10-model",
                                  max_iterations=100,
                                  batch_size=512,
                                  verbose=False)
        cebra_model.fit(dataset.neural)
        embedding = cebra_model.transform(dataset.neural)

        (train_embedding, valid_embedding, train_label,
         valid_label) = sklearn.model_selection.train_test_split(
             embedding, dataset.continuous_index[:, 0], test_size=0.3)

        decoder = cebra.KNNDecoder()
        decoder.fit(train_embedding, train_label)
        score = decoder.score(valid_embedding, valid_label)

        scores.append(score)
        embeddings.append(embedding)

    joint_embedding = cebra_data_helper.ensemble_embeddings(
        embeddings=embeddings)

    (train_embedding, valid_embedding, train_label,
     valid_label) = sklearn.model_selection.train_test_split(
         joint_embedding, dataset.continuous_index[:, 0], test_size=0.3)

    decoder = cebra.KNNDecoder()
    decoder.fit(train_embedding, train_label)
    score = decoder.score(valid_embedding, valid_label)

    assert all(score_i < score for score_i in scores)


def _original_orthogonal_procrustes_fit(model, ref_data, data, ref_label,
                                        label):
    """Pre-#309 ``OrthogonalProcrustesAlignment.fit``, copied verbatim.

    Every line below is copied from the implementation in commit
    ``d1842cc``, with ``self`` renamed to ``model`` and the transform
    returned instead of stored. The input validation and ``subsample``
    branches of the original are omitted: #309 leaves them untouched and
    the tests here never exercise them (labels always given,
    ``subsample=None``).
    """
    import scipy.linalg

    if len(ref_label.shape) == 1:
        ref_label = np.expand_dims(ref_label, axis=1)
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=1)

    distance = model._distance(label, ref_label)

    # keep indexes of the {self.top_k} labels the closest to the reference labels
    target_idx = np.argsort(distance, axis=1)[:, :model.top_k]

    # Get the whole data to align and only the selected closest samples
    # from the reference data.
    X = data[:, None].repeat(model.top_k, axis=1).reshape(-1, data.shape[1])
    Y = ref_data[target_idx].reshape(-1, ref_data.shape[1])

    # Compute orthogonal matrix that most closely maps X to Y using the orthogonal Procrustes problem.
    transform, _ = scipy.linalg.orthogonal_procrustes(X, Y)

    return transform


@pytest.mark.parametrize(
    "n_label, n_ref, dim, top_k",
    [
        (50, 40, 4, 5),
        (37, 60, 3, 1),
        (100, 20, 2, 10),
        (23, 7, 3, 7),
        # edge cases: single label row, single reference, tiny shapes
        (1, 8, 3, 2),
        (6, 1, 2, 1),
        (2, 2, 2, 2),
    ])
@pytest.mark.parametrize("label_dim", [1, 2])
def test_orthogonal_alignment_chunking_matches_full(n_label, n_ref, dim, top_k,
                                                    label_dim, monkeypatch):
    """Chunked top_k search reproduces the pre-#309 implementation."""
    rng = np.random.RandomState(0)
    ref_data = rng.uniform(0, 1, (n_ref, dim))
    data = rng.uniform(0, 1, (n_label, dim))
    # distinct (non-tie) labels so the reference ordering is unambiguous
    ref_label = (rng.permutation(n_ref).astype(float) +
                 rng.uniform(0, 1e-3, n_ref))[:, None]
    label = (rng.permutation(n_label).astype(float) +
             rng.uniform(0, 1e-3, n_label))[:, None]
    if label_dim == 1:
        ref_label, label = ref_label.ravel(), label.ravel()

    # oracle: the verbatim pre-PR (full-matrix) implementation
    oracle = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    ref_transform = _original_orthogonal_procrustes_fit(oracle, ref_data, data,
                                                        ref_label, label)

    # force many small chunks: 3 rows of `label` at a time
    monkeypatch.setattr(cebra_data_helper, "_PROCRUSTES_MAX_DISTANCE_ELEMENTS",
                        3 * n_ref)
    chunked = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    chunked.fit(ref_data, data, ref_label, label)
    np.testing.assert_array_equal(chunked._transform, ref_transform)

    # force a single chunk: must also be bit-identical
    monkeypatch.setattr(cebra_data_helper, "_PROCRUSTES_MAX_DISTANCE_ELEMENTS",
                        n_label * n_ref + 1)
    single = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    single.fit(ref_data, data, ref_label, label)
    np.testing.assert_array_equal(single._transform, ref_transform)
    np.testing.assert_array_equal(single.transform(data),
                                  chunked.transform(data))


def test_orthogonal_alignment_chunking_bit_identical_with_ties(monkeypatch):
    """Row chunking is bit-identical, even with tied distances (#307)."""
    rng = np.random.RandomState(1)
    n_label, n_ref, dim, top_k = 60, 30, 3, 5
    ref_data = rng.uniform(0, 1, (n_ref, dim))
    data = rng.uniform(0, 1, (n_label, dim))
    # integer labels with many duplicates -> tied distances
    ref_label = rng.randint(0, 5, size=(n_ref, 1)).astype(float)
    label = rng.randint(0, 5, size=(n_label, 1)).astype(float)

    monkeypatch.setattr(cebra_data_helper, "_PROCRUSTES_MAX_DISTANCE_ELEMENTS",
                        n_label * n_ref + 1)
    single = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    single.fit(ref_data, data, ref_label, label)

    monkeypatch.setattr(cebra_data_helper, "_PROCRUSTES_MAX_DISTANCE_ELEMENTS",
                        4 * n_ref)
    chunked = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    chunked.fit(ref_data, data, ref_label, label)
    np.testing.assert_array_equal(single._transform, chunked._transform)


def test_orthogonal_alignment_chunking_bounds_block_size(monkeypatch):
    """The top_k search runs in bounded blocks, not one full matrix (#307)."""
    rng = np.random.RandomState(2)
    n_label, n_ref, dim, top_k = 50, 10, 3, 4
    ref_data = rng.uniform(0, 1, (n_ref, dim))
    data = rng.uniform(0, 1, (n_label, dim))
    ref_label = rng.uniform(0, 1, (n_ref, 1))
    label = rng.uniform(0, 1, (n_label, 1))

    rows_per_call = []
    original = cebra_data_helper.OrthogonalProcrustesAlignment._distance

    def spy(self, label_i, label_j):
        rows_per_call.append(label_i.shape[0])
        return original(self, label_i, label_j)

    # 3 rows of `label` per block
    monkeypatch.setattr(cebra_data_helper, "_PROCRUSTES_MAX_DISTANCE_ELEMENTS",
                        3 * n_ref)
    monkeypatch.setattr(cebra_data_helper.OrthogonalProcrustesAlignment,
                        "_distance", spy)

    model = cebra_data_helper.OrthogonalProcrustesAlignment(top_k=top_k)
    model.fit(ref_data, data, ref_label, label)

    # the full (n_label, n_ref) distance matrix is never materialized
    assert max(rows_per_call) <= 3
    assert len(rows_per_call) == (n_label + 2) // 3
    assert sum(rows_per_call) == n_label
