import math
from functools import lru_cache

import numpy as np
import numba_dpex as dpex

from ._base_kmeans_kernel_funcs import (
    _make_initialize_window_kernel_funcs,
    _make_accumulate_sum_of_ops_kernel_func,
)
from sklearn_numba_dpex.common.random import make_rand_uniform_kernel_func

# NB: refer to the definition of the main lloyd function for a more comprehensive
# inline commenting of the kernel.


@lru_cache
def make_init_kmeansplusplus_kernel(
    n_samples,
    n_features,
    preferred_work_group_size_multiple,
    work_group_size,
    dtype,
):

    zero_idx = np.int64(0)
    zero_init = dtype(0.0)

    @dpex.kernel
    # fmt: off
    def init_kmeansplusplus(
        X_t,                      # IN READ-ONLY   (n_features, n_samples)
        sample_weight,            # IN READ-ONLY   (n_samples,)
        centers_t,                # OUT            (n_features, n_clusters)
        center_indices,           # OUT            (n_clusters,)
        closest_dist_sq,          # OUT            (n_samples,)
    ):
    # fmt: on
        sample_idx = dpex.get_global_id(zero_idx)
        if sample_idx >= n_samples:
            return

        starting_center_id_ = center_indices[zero_idx]

        sq_distance = zero_init
        for feature_idx in range(n_features):
            diff = X_t[feature_idx, sample_idx] - X_t[feature_idx, starting_center_id_]
            sq_distance += diff * diff

        sq_distance *= sample_weight[sample_idx]
        closest_dist_sq[sample_idx] = sq_distance

        if sample_idx > zero_idx:
            return

        for feature_idx in range(n_features):
            centers_t[feature_idx, zero_idx] = X_t[feature_idx, starting_center_id_]

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return init_kmeansplusplus[global_size, work_group_size]


@lru_cache
def make_sample_center_candidates_kernel(
    n_samples,
    n_local_trials,
    preferred_work_group_size_multiple,
    work_group_size,
    dtype,
):

    rand_uniform_kernel_func = make_rand_uniform_kernel_func(np.dtype(dtype))

    zero_idx = np.int64(0)
    one_incr = np.int32(1)
    one_decr = np.int32(-1)
    zero_init = dtype(0.0)
    max_candidate_id = np.int32(n_samples - 1)

    @dpex.kernel
    # fmt: off
    def sample_center_candidates(
        closest_dist_sq,          # IN             (n_features, n_samples)
        total_potential,          # IN             (1,)
        random_state,             # INOUT          (n_local_trials, 2)
        candidates_id,            # OUT            (n_local_trials,)
        ):
    # fmt: on
        local_trial_idx = dpex.get_global_id(zero_idx)
        if local_trial_idx >= n_local_trials:
            return
        random_value = (rand_uniform_kernel_func(random_state, local_trial_idx)
                        * total_potential[zero_idx])

        cumulative_potential = zero_init
        candidate_id = one_decr
        while (random_value > cumulative_potential) and (candidate_id < max_candidate_id):
            candidate_id += one_incr
            cumulative_potential += closest_dist_sq[candidate_id]
        candidates_id[local_trial_idx] = candidate_id

    global_size = (math.ceil(n_local_trials / work_group_size)) * (work_group_size)
    return sample_center_candidates[global_size, work_group_size]


@lru_cache
def make_kmeansplusplus_single_step_fixed_window_kernel(
    n_samples,
    n_features,
    n_candidates,
    preferred_work_group_size_multiple,
    centroids_window_width_multiplier,
    centroids_window_height,
    work_group_size,
    dtype,
):

    window_n_centroids = (
        preferred_work_group_size_multiple * centroids_window_width_multiplier
    )

    (
        _initialize_window_of_centroids,
        _load_window_of_centroids_and_features,
    ) = _make_initialize_window_kernel_funcs(
        n_samples,
        n_features,
        work_group_size,
        window_n_centroids,
        centroids_window_height,
        dtype,
    )

    _accumulate_sq_distances = _make_accumulate_sum_of_ops_kernel_func(
        n_samples,
        n_features,
        centroids_window_height,
        window_n_centroids,
        ops="squared_diff",
        dtype=dtype,
    )

    n_windows_per_feature = math.ceil(n_candidates / window_n_centroids)
    n_windows_per_centroid = math.ceil(n_features / centroids_window_height)

    centroids_window_shape = (centroids_window_height, (window_n_centroids + 1))

    zero_idx = np.int64(0)

    @dpex.kernel
    # fmt: off
    def kmeansplusplus_single_step(
        X_t,                               # IN READ-ONLY   (n_features, n_samples)
        sample_weight,                     # IN READ-ONLY   (n_samples,)
        candidates_ids,                    # IN             (n_candidates,)
        closest_dist_sq,                   # IN             (n_samples,)
        euclidean_distances_t,             # OUT            (n_candidates, n_samples)
    ):
    # fmt: on
        sample_idx = dpex.get_global_id(zero_idx)
        local_work_id = dpex.get_local_id(zero_idx)

        centroids_window = dpex.local.array(shape=centroids_window_shape, dtype=dtype)

        sq_distances = dpex.private.array(shape=window_n_centroids, dtype=dtype)

        first_centroid_idx = zero_idx

        window_loading_centroid_idx = local_work_id % window_n_centroids
        window_loading_feature_offset = local_work_id // window_n_centroids

        for _0 in range(n_windows_per_feature):
            _initialize_window_of_centroids(sq_distances)

            loading_centroid_idx = first_centroid_idx + window_loading_centroid_idx
            if loading_centroid_idx < n_candidates:
                loading_centroid_idx = candidates_ids[loading_centroid_idx]
            else:
                loading_centroid_idx = n_samples

            first_feature_idx = zero_idx

            for _1 in range(n_windows_per_centroid):

                _load_window_of_centroids_and_features(
                    first_feature_idx,
                    loading_centroid_idx,
                    window_loading_centroid_idx,
                    window_loading_feature_offset,
                    X_t,
                    centroids_window,
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                _accumulate_sq_distances(
                    sample_idx,
                    first_feature_idx,
                    X_t,
                    centroids_window,
                    sq_distances,
                )

                dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

                first_feature_idx += centroids_window_height

            if sample_idx < n_samples:
                sample_weight_ = sample_weight[sample_idx]
                closest_dist_sq_ = closest_dist_sq[sample_idx]
                for i in range(window_n_centroids):
                    centroid_idx = first_centroid_idx + i
                    if centroid_idx < n_candidates:
                        sq_distance_i = min(
                            sq_distances[i] * sample_weight_,
                            closest_dist_sq_)
                        euclidean_distances_t[first_centroid_idx + i, sample_idx] = sq_distance_i

            dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

            first_centroid_idx += window_n_centroids

    global_size = (math.ceil(n_samples / work_group_size)) * (work_group_size)
    return kmeansplusplus_single_step[global_size, work_group_size]
