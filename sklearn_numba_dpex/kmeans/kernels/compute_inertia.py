import math
from functools import lru_cache

import numba_dpex.experimental as dpex_exp
import numpy as np
from numba_dpex.kernel_api import NdItem, NdRange


@lru_cache
def make_compute_inertia_kernel(n_samples, n_features, work_group_size, dtype):

    zero_idx = np.int64(0)
    zero_init = dtype(0.0)

    @dpex_exp.kernel
    # fmt: off
    def compute_inertia(
        nd_item: NdItem,
        X_t,                          # IN READ-ONLY   (n_features, n_samples)
        sample_weight,                # IN READ-ONLY   (n_features,)
        centroids_t,                  # IN READ-ONLY   (n_features, n_clusters)
        assignments_idx,              # IN READ-ONLY   (n_samples,)
        per_sample_inertia,           # OUT            (n_samples,)
    ):
        # fmt: on

        sample_idx = nd_item.get_global_id(zero_idx)

        if sample_idx >= n_samples:
            return

        inertia = zero_init

        centroid_idx = assignments_idx[sample_idx]

        for feature_idx in range(n_features):

            diff = X_t[feature_idx, sample_idx] - centroids_t[feature_idx, centroid_idx]
            inertia += diff * diff

        per_sample_inertia[sample_idx] = inertia * sample_weight[sample_idx]

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)

    def kernel_call(*args):
        dpex_exp.call_kernel(
            compute_inertia, NdRange((global_size,), (work_group_size,)), *args
        )

    return kernel_call
