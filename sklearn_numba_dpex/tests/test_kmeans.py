import inspect

import dpctl
import numpy as np
import pytest
import warnings
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils._testing import assert_allclose


_DEVICE = dpctl.SyclDevice()
_DEVICE_NAME = _DEVICE.name
_SUPPORTED_DTYPE = [np.float32]

if _DEVICE.has_aspect_fp64:
    _SUPPORTED_DTYPE.append(np.float64)


def _fail_if_no_dtype_support(xfail_fn, dtype):
    if dtype not in _SUPPORTED_DTYPE:
        xfail_fn(
            f"The default device {_DEVICE_NAME} does not have support for "
            "float64 operations."
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_same_results(dtype):
    _fail_if_no_dtype_support(pytest.xfail, dtype)

    random_seed = 42
    X, _ = make_blobs(random_state=random_seed)
    X = X.astype(dtype)

    kmeans_vanilla = KMeans(random_state=random_seed, algorithm="lloyd", max_iter=1)
    kmeans_engine = clone(kmeans_vanilla)

    # Fit a reference model with the default scikit-learn engine:
    kmeans_vanilla.fit(X)

    with config_context(engine_provider="sklearn_numba_dpex"):
        kmeans_engine.fit(X)

    # ensure same results
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_predict
    y_labels = kmeans_vanilla.fit_predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.fit_predict(X)
    assert_array_equal(y_labels, y_labels_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # test fit_transform
    y_transform = kmeans_vanilla.fit_transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.fit_transform(X)
    assert_allclose(y_transform, y_transform_engine)
    assert_array_equal(kmeans_vanilla.labels_, kmeans_engine.labels_)
    assert_allclose(kmeans_vanilla.cluster_centers_, kmeans_engine.cluster_centers_)
    assert_allclose(kmeans_vanilla.inertia_, kmeans_engine.inertia_)

    # # test predict method (returns labels)
    y_labels = kmeans_vanilla.predict(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_labels_engine = kmeans_engine.predict(X)
    assert_array_equal(y_labels, y_labels_engine)

    # test score method (returns negative inertia for each sample)
    y_scores = kmeans_vanilla.score(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_scores_engine = kmeans_engine.score(X)
    assert_allclose(y_scores, y_scores_engine)

    # test transform method (returns euclidean distances)
    y_transform = kmeans_vanilla.transform(X)
    with config_context(engine_provider="sklearn_numba_dpex"):
        y_transform_engine = kmeans_engine.transform(X)
    assert_allclose(y_transform, y_transform_engine)
