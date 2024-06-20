import dpctl.tensor as dpt
import numpy as np

from sklearn_numba_dpex.common._utils import (
    _divide_by,
    _get_sequential_processing_device,
    _minus,
    _plus,
    _square,
)
from sklearn_numba_dpex.common.kernels import (
    make_apply_elementwise_func,
    make_broadcast_division_1d_2d_axis0_kernel,
    make_broadcast_ops_1d_2d_axis1_kernel,
    make_fill_kernel,
    make_half_l2_norm_2d_axis0_kernel,
)
from sklearn_numba_dpex.common.random import (
    create_xoroshiro128pp_states,
    get_random_raw,
)
from sklearn_numba_dpex.common.reductions import (
    make_argmin_reduction_1d_kernel,
    make_sum_reduction_2d_kernel,
)
from sklearn_numba_dpex.common.topk import topk_idx
from sklearn_numba_dpex.kmeans.kernels import (
    make_centroid_shifts_kernel,
    make_compute_euclidean_distances_fixed_window_kernel,
    make_compute_inertia_kernel,
    make_get_nb_distinct_clusters_kernel,
    make_is_same_clustering_kernel,
    make_kmeansplusplus_init_kernel,
    make_kmeansplusplus_single_step_fixed_window_kernel,
    make_label_assignment_fixed_window_kernel,
    make_lloyd_single_step_fixed_window_kernel,
    make_reduce_centroid_data_kernel,
    make_relocate_empty_clusters_kernel,
    make_sample_center_candidates_kernel,
)


def lloyd(
    X_t,
    sample_weight,
    centroids_t,
    use_uniform_weights,
    max_iter=300,
    verbose=False,
    tol=1e-4,
):
    n_features, n_samples = X_t.shape
    n_clusters = centroids_t.shape[1]
    compute_dtype = X_t.dtype.type

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = 8

    # Create a set of kernels
    (
        n_centroids_private_copies,
        fused_lloyd_fixed_window_single_step_kernel,
    ) = make_lloyd_single_step_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        # NB: the assignments are needed if verbose=True, and for strict convergence
        # checking. If systematic strict convergence checking is disabled in the future,
        # if could be set to False when verbose=False (thus marginally improving
        # performance).
        return_assignments=True,
        check_strict_convergence=True,
        sub_group_size=sub_group_size,
        work_group_size="max",
        dtype=compute_dtype,
        device=device,
    )

    assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        sub_group_size=sub_group_size,
        work_group_size="max",
        dtype=compute_dtype,
        device=device,
    )

    compute_inertia_kernel = make_compute_inertia_kernel(
        n_samples, n_features, max_work_group_size, compute_dtype
    )

    reset_cluster_sizes_private_copies_kernel = make_fill_kernel(
        fill_value=0,
        shape=(n_centroids_private_copies, n_clusters),
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    reset_centroids_private_copies_kernel = make_fill_kernel(
        fill_value=0,
        shape=(n_centroids_private_copies, n_features, n_clusters),
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    broadcast_division_kernel = make_broadcast_division_1d_2d_axis0_kernel(
        shape=(n_features, n_clusters),
        work_group_size=max_work_group_size,
    )

    compute_centroid_shifts_kernel = make_centroid_shifts_kernel(
        n_clusters=n_clusters,
        n_features=n_features,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
        (n_features, n_clusters),
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    reduce_inertia_kernel = make_sum_reduction_2d_kernel(
        shape=(n_samples,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    reduce_centroid_shifts_kernel = make_sum_reduction_2d_kernel(
        shape=(n_clusters,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    reduce_centroid_data_kernel = make_reduce_centroid_data_kernel(
        n_centroids_private_copies=n_centroids_private_copies,
        n_features=n_features,
        n_clusters=n_clusters,
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    # Allocate the necessary memory in the device global memory
    new_centroids_t = dpt.empty_like(centroids_t, device=device)
    centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    cluster_sizes = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    centroid_shifts = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    # NB: the same buffer is used for those two arrays because it is never needed
    # to store those simultaneously in memory.
    sq_dist_to_nearest_centroid = per_sample_inertia = dpt.empty(
        n_samples, dtype=compute_dtype, device=device
    )

    new_assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)
    assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)

    new_centroids_t_private_copies = dpt.empty(
        (n_centroids_private_copies, n_features, n_clusters),
        dtype=compute_dtype,
        device=device,
    )
    cluster_sizes_private_copies = dpt.empty(
        (n_centroids_private_copies, n_clusters),
        dtype=compute_dtype,
        device=device,
    )
    empty_clusters_list = dpt.empty(n_clusters, dtype=np.uint32, device=device)

    # n_empty_clusters_ is a scalar handled in kernels via a one-element array.
    n_empty_clusters = dpt.empty(1, dtype=np.int32, device=device)

    # allocation of one scalar where we store the result of strict convergence check
    strict_convergence_status = dpt.empty(1, dtype=np.uint32, device=device)

    verbose = bool(verbose)

    # The loop
    n_iteration = 0
    # See the main loop for a more elaborate note about checking "strict convergence"
    strict_convergence = False
    centroid_shifts_sum = np.inf

    # TODO: Investigate possible speedup with a custom dpctl queue with a custom
    # DAG of events and a final single "wait"
    while (n_iteration < max_iter) and (centroid_shifts_sum > tol):
        half_l2_norm_kernel(
            centroids_t,
            # OUT:
            centroids_half_l2_norm,
        )

        reset_cluster_sizes_private_copies_kernel(cluster_sizes_private_copies)
        reset_centroids_private_copies_kernel(new_centroids_t_private_copies)
        n_empty_clusters[0] = np.int32(0)

        # TODO: implement special case where only one copy is needed
        fused_lloyd_fixed_window_single_step_kernel(
            X_t,
            sample_weight,
            centroids_t,
            centroids_half_l2_norm,
            assignments_idx,
            # OUT:
            new_assignments_idx,
            strict_convergence_status,
            new_centroids_t_private_copies,
            cluster_sizes_private_copies,
        )

        reduce_centroid_data_kernel(
            cluster_sizes_private_copies,
            new_centroids_t_private_copies,
            # OUT:
            cluster_sizes,
            new_centroids_t,
            empty_clusters_list,
            n_empty_clusters,
        )

        if verbose:
            # ???: verbosity comes at the cost of performance since it triggers
            # computing exact inertia at each iteration. Shouldn't this be
            # documented ?
            compute_inertia_kernel(
                X_t,
                sample_weight,
                new_centroids_t,
                new_assignments_idx,
                # OUT:
                per_sample_inertia,
            )
            inertia, *_ = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
            print(f"Iteration {n_iteration}, inertia {inertia:5.3e}")

        n_empty_clusters_ = int(n_empty_clusters[0])
        if n_empty_clusters_ > 0:
            # NB: empty cluster very rarely occurs, and it's more efficient to
            # compute inertia and labels only after occurrences have been detected
            # at the cost of an additional pass on data, rather than computing
            # inertia by default during the first pass on data in case there's an
            # empty cluster.

            # if verbose is True and if sample_weight is uniform, distances to
            # closest centroids already have been computed in the main kernel
            if not verbose or not use_uniform_weights:
                # Note that we intentionally pass unit weights instead of
                # sample_weight so that per_sample_inertia will be updated to the
                # (unweighted) squared distance to the nearest centroid.
                compute_inertia_kernel(
                    X_t,
                    dpt.ones_like(sample_weight),
                    new_centroids_t,
                    new_assignments_idx,
                    # OUT:
                    sq_dist_to_nearest_centroid,
                )

            _relocate_empty_clusters(
                n_empty_clusters_,
                X_t,
                sample_weight,
                new_centroids_t,
                cluster_sizes,
                new_assignments_idx,
                empty_clusters_list,
                sq_dist_to_nearest_centroid,
                per_sample_inertia,
                max_work_group_size,
            )

        # Change `new_centroids_t` inplace
        broadcast_division_kernel(new_centroids_t, cluster_sizes)

        # ???: unlike sklearn, sklearn_intelex checks that pseudo_inertia decreases
        # and keep an additional copy of centroids that is updated only if the
        # value of the pseudo_inertia is smaller than all the past values.
        #
        # (the check should not be needed because we have theoritical guarantee
        # that inertia decreases, if it doesn't it should only be because of
        # rounding errors ?)
        #
        # To this purpose this code could be inserted here:
        #
        # if pseudo_inertia < best_pseudo_inertia:
        #     best_pseudo_inertia = pseudo_inertia
        #     copyto_kernel(centroids_t, best_centroids_t)
        #
        # Note that what that is saved as "best centroid" is the array before the
        # update, to which the pseudo inertia that is computed at this iteration
        # refers. For this reason, this strategy is not compatible with sklearn
        # unit tests, that consider new_centroids_t (the array after the update)
        # to be the best centroids at each iteration.
        centroids_t, new_centroids_t = (new_centroids_t, centroids_t)

        # ???: if two successive assignations have been computed equal, it's called
        # "strict convergence" and means that the algorithm has converged and can't get
        # better (since same assignments will produce the same centroid updates, which
        # will produce the same assignments, and so on..). scikit-learn decides to
        # check for strict convergence at each iteration, but that sounds expensive
        # since for each iteration it requires writing assignments in memory and
        # comparing it to the previous assignments.

        # Providing the user chooses a sensible value for `tol`, wouldn't the cost of
        # this check be in general greater than what the benefits ?

        # When `tol == 0` it is easy to see that lloyd can indeed fail to stop at the
        # right time due to numerical errors, and that strict convergence checking is
        # good. For the general case, it seems detrimental to performance, for little
        # gain except in case of, maybe, extremely imbalanced input data distribution.
        # Thus, shouldnt strict convergence checking be enabled only if `tol == 0` ?
        # (which is, moreover, the only case where strict convergence really is tested
        # in scikit learn)

        # For now the exact same behavior than scikit-learn's is mimicked.

        # See: https://github.com/scikit-learn/scikit-learn/issues/25716

        assignments_idx, new_assignments_idx = (
            new_assignments_idx,
            assignments_idx,
        )

        n_iteration += 1

        if n_iteration > 1:
            strict_convergence, *_ = strict_convergence_status
            if strict_convergence:
                break
            strict_convergence_status[0] = np.uint32(1)

        compute_centroid_shifts_kernel(
            centroids_t,
            new_centroids_t,
            # OUT:
            centroid_shifts,
        )

        centroid_shifts_sum, *_ = reduce_centroid_shifts_kernel(centroid_shifts)
        # Use numpy type to work around https://github.com/IntelPython/dpnp/issues/1238
        centroid_shifts_sum = compute_dtype(centroid_shifts_sum)

    if verbose:
        converged_at = n_iteration - 1
        if strict_convergence or (centroid_shifts_sum == 0):  # NB: possible if tol = 0
            print(f"Converged at iteration {converged_at}: strict convergence.")

        elif centroid_shifts_sum <= tol:
            print(
                f"Converged at iteration {converged_at}: center shift "
                f"{centroid_shifts_sum} within tolerance {tol}."
            )

    # Finally, run an assignment kernel to compute the assignments to the best
    # centroids found, along with the exact inertia.
    half_l2_norm_kernel(
        centroids_t,
        # OUT:
        centroids_half_l2_norm,
    )

    # NB: inertia and labels could be computed in a single fused kernel, however,
    # re-using the quantity Q = ((1/2)c^2 - <x.c>) that is computed in the
    # assignment kernel to compute distances to closest centroids to evaluate
    # |x|^2 - 2 * Q leads to numerical instability, we prefer evaluating the
    # expression |x-c|^2 which is stable but requires an additional pass on the
    # data.
    # See https://github.com/soda-inria/sklearn-numba-dpex/issues/28
    assignment_fixed_window_kernel(
        X_t,
        centroids_t,
        centroids_half_l2_norm,
        # OUT:
        assignments_idx,
    )

    compute_inertia_kernel(
        X_t,
        sample_weight,
        centroids_t,
        assignments_idx,
        # OUT:
        per_sample_inertia,
    )

    # inertia = per_sample_inertia.sum()
    inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
    # inertia is now a 1-sized numpy array, we transform it into a scalar:
    inertia = inertia[0]

    return assignments_idx, inertia, centroids_t, n_iteration


def _relocate_empty_clusters(
    n_empty_clusters,
    X_t,
    sample_weight,
    centroids_t,
    cluster_sizes,
    assignments_idx,
    empty_clusters_list,
    sq_dist_to_nearest_centroid,
    per_sample_inertia,
    work_group_size,
):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape

    samples_far_from_center = topk_idx(sq_dist_to_nearest_centroid, n_empty_clusters)

    # Centroids of empty clusters are relocated to samples in X that are the
    # farthest from their respective centroids. new_centroids_t is updated
    # accordingly.
    relocate_empty_clusters_kernel = make_relocate_empty_clusters_kernel(
        n_empty_clusters,
        n_features,
        work_group_size,
        compute_dtype,
    )

    relocate_empty_clusters_kernel(
        X_t,
        sample_weight,
        assignments_idx,
        samples_far_from_center,
        empty_clusters_list,
        # OUT
        per_sample_inertia,
        centroids_t,
        cluster_sizes,
    )


def prepare_data_for_lloyd(X_t, init, tol, sample_weight, copy_x):
    """It can be more numerically accurate to center the data first. If copy_x is True,
    then the original data is not modified. If False, the original data is modified,
    and put back later on (see `restore_data_after_lloyd`), but small numerical
    differences may be introduced by subtracting and then adding the data mean. Note
    that if the original data is not C-contiguous, a copy will be made even if copy_x
    is False."""

    n_features, n_samples = X_t.shape
    compute_dtype = X_t.dtype.type

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size

    sum_axis1_kernel = make_sum_reduction_2d_kernel(
        X_t.shape,
        axis=1,
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    elementwise_divide_by_n_samples_fn = _divide_by(compute_dtype(n_samples))

    divide_by_n_samples_kernel = make_apply_elementwise_func(
        (n_features,),
        elementwise_divide_by_n_samples_fn,
        max_work_group_size,
    )

    # At the time of writing this code, dpnp does not support functions (like `==`
    # operator) that would help computing `X_mean_is_zeroed` in a simpler
    # manner.
    # TODO: if dpnp support extends to relevant features, use it instead ?
    sum_of_squares_kernel = make_sum_reduction_2d_kernel(
        shape=(n_features,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
        fused_elementwise_func=_square,
    )

    X_mean = sum_axis1_kernel(X_t)[:, 0]

    # Change `X_mean` inplace
    divide_by_n_samples_kernel(X_mean)

    X_sum_squared = sum_of_squares_kernel(X_mean)[0]
    X_mean_is_zeroed = float(X_sum_squared) == 0.0

    if X_mean_is_zeroed:
        # If the data is already centered, there's no need to perform shift/unshift
        # steps. In this case, X_mean is set to None, thus carrying the information
        # that the data was already centered, and the shift/unshift steps will be
        # skipped.
        X_mean = None
    else:
        # subtract the mean of x for more accurate distance computations
        X_t = dpt.asarray(X_t, copy=copy_x)
        broadcast_X_minus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
            (n_features, n_samples),
            ops=_minus,
            work_group_size=max_work_group_size,
        )

        # Change `X_t` inplace
        broadcast_X_minus_X_mean(X_t, X_mean)

        if isinstance(init, dpt.usm_ndarray):
            n_clusters = init.shape[1]
            broadcast_init_minus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
                (n_features, n_clusters),
                ops=_minus,
                work_group_size=max_work_group_size,
            )
            # Change `init` inplace
            broadcast_init_minus_X_mean(init, X_mean)

    n_items = n_features * n_samples

    variance_kernel = make_sum_reduction_2d_kernel(
        shape=(n_items,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
        fused_elementwise_func=_square,
    )

    variance = variance_kernel(dpt.reshape(X_t, -1))
    # Use numpy type to work around https://github.com/IntelPython/dpnp/issues/1238
    tol = (dpt.asnumpy(variance)[0] / (n_features * n_samples)) * tol

    # check if sample_weight is uniform
    # At the time of writing this code, dpnp does not support functions (like `==`
    # operator) that would help computing `sample_weight_is_uniform` in a simpler
    # manner.
    # TODO: if dpnp support extends to relevant features, use it instead ?
    sum_sample_weight_kernel = make_sum_reduction_2d_kernel(
        shape=(n_samples,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    sample_weight_sum = compute_dtype(sum_sample_weight_kernel(sample_weight)[0])
    sample_weight_is_uniform = sample_weight_sum == (
        compute_dtype(sample_weight[0]) * n_samples
    )

    return X_t, X_mean, init, tol, sample_weight_is_uniform


def restore_data_after_lloyd(X_t, best_centers_t, X_mean, copy_x):
    """X_mean, the feature wise mean of X prior to the centerings of X and centers in
    `prepare_data_for_lloyd`, is re-added to X and centers.
    """
    if X_mean is None:
        # X and best_centers_t aren't translated back.
        return

    n_features, n_samples = X_t.shape
    n_clusters = best_centers_t.shape[1]

    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size

    best_centers_t = dpt.asarray(best_centers_t, copy=False)
    broadcast_init_plus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
        (n_features, n_clusters),
        ops=_plus,
        work_group_size=max_work_group_size,
    )
    # Change `best_centers_t` inplace
    broadcast_init_plus_X_mean(best_centers_t, X_mean)

    # NB: copy_x being set to False does not mean that no copy actually happened, only
    # that no copy was forced if it was not necessary with respect to what device,
    # dtype and order that are required at compute time. Nevertheless, there's no
    # simple way to check if a copy happened without assumptions on the type of the raw
    # input submitted by the user, but at the moment it is unknown what those
    # assumptions could be. As a result, the following instructions are ran every time,
    # even if it isn't useful when a copy has been made.
    # TODO: is there a set of assumptions that exhaustively describes the set of
    # accepted inputs, and also enables checking if a copy happened or not in a simple
    # way ?
    if not copy_x:
        X_t = dpt.asarray(X_t, copy=False)
        broadcast_X_plus_X_mean = make_broadcast_ops_1d_2d_axis1_kernel(
            (n_features, n_samples),
            ops=_plus,
            work_group_size=max_work_group_size,
        )
        # Change X_t inplace
        broadcast_X_plus_X_mean(X_t, X_mean)


def is_same_clustering(labels1, labels2, n_clusters):
    """Check if two arrays of labels are the same up to a permutation of the labels"""
    device = labels1.device.sycl_device

    is_same_clustering_kernel = make_is_same_clustering_kernel(
        n_samples=labels1.shape[0],
        n_clusters=n_clusters,
        work_group_size=device.max_work_group_size,
        device=device,
    )
    return is_same_clustering_kernel(labels1, labels2)


def get_nb_distinct_clusters(labels, n_clusters):
    device = labels.device.sycl_device

    get_nb_distinct_clusters_kernel = make_get_nb_distinct_clusters_kernel(
        n_samples=labels.shape[0],
        n_clusters=n_clusters,
        work_group_size=device.max_work_group_size,
        device=device,
    )

    clusters_seen = dpt.zeros((n_clusters,), dtype=np.int32, device=device)

    nb_distinct_clusters = dpt.zeros((1,), dtype=np.int32, device=device)

    get_nb_distinct_clusters_kernel(
        labels,
        clusters_seen,
        # OUT
        nb_distinct_clusters,
    )

    # Use numpy type to work around https://github.com/IntelPython/dpnp/issues/1238
    return dpt.asnumpy(nb_distinct_clusters[0])


def get_labels_inertia(X_t, centroids_t, sample_weight, with_inertia):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    n_clusters = centroids_t.shape[1]
    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = 8

    label_assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
        n_samples,
        n_features,
        n_clusters,
        sub_group_size=sub_group_size,
        work_group_size="max",
        dtype=compute_dtype,
        device=device,
    )

    half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
        (n_features, n_clusters),
        work_group_size=max_work_group_size,
        dtype=compute_dtype,
    )

    centroids_half_l2_norm = dpt.empty(n_clusters, dtype=compute_dtype, device=device)
    assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=device)

    half_l2_norm_kernel(
        centroids_t,
        # OUT
        centroids_half_l2_norm,
    )

    label_assignment_fixed_window_kernel(
        X_t,
        centroids_t,
        centroids_half_l2_norm,
        # OUT
        assignments_idx,
    )

    if not with_inertia:
        return assignments_idx, None

    compute_inertia_kernel = make_compute_inertia_kernel(
        n_samples, n_features, max_work_group_size, compute_dtype
    )

    reduce_inertia_kernel = make_sum_reduction_2d_kernel(
        shape=(n_samples,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    per_sample_inertia = dpt.empty(n_samples, dtype=compute_dtype, device=device)

    compute_inertia_kernel(
        X_t,
        sample_weight,
        centroids_t,
        assignments_idx,
        # OUT
        per_sample_inertia,
    )

    # inertia = per_sample_inertia.sum()
    inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))

    return assignments_idx, inertia


def get_euclidean_distances(X_t, Y_t):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    n_clusters = Y_t.shape[1]
    device = X_t.device.sycl_device
    sub_group_size = 8

    euclidean_distances_fixed_window_kernel = (
        make_compute_euclidean_distances_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            sub_group_size=sub_group_size,
            work_group_size="max",
            dtype=compute_dtype,
            device=device,
        )
    )

    euclidean_distances_t = dpt.empty(
        (n_clusters, n_samples), dtype=compute_dtype, device=device
    )

    euclidean_distances_fixed_window_kernel(
        X_t,
        Y_t,
        # OUT
        euclidean_distances_t,
    )

    return euclidean_distances_t.T


def kmeans_plusplus(
    X_t,
    sample_weight,
    n_clusters,
    random_state,
):
    compute_dtype = X_t.dtype.type
    n_features, n_samples = X_t.shape
    device = X_t.device.sycl_device
    max_work_group_size = device.max_work_group_size
    sub_group_size = 8

    # TODO: check that this implementation is correct when samples weights aren't
    # uniform.

    # Same retrial heuristic as scikit-learn (at least until <1.2)
    n_local_trials = 2 + int(np.log(n_clusters))

    (
        sequential_processing_device,
        sequential_processing_on_different_device,
    ) = _get_sequential_processing_device(device)

    kmeansplusplus_init_kernel = make_kmeansplusplus_init_kernel(
        n_samples,
        n_features,
        max_work_group_size,
        compute_dtype,
    )

    (
        kmeansplusplus_single_step_fixed_window_kernel
    ) = make_kmeansplusplus_single_step_fixed_window_kernel(
        n_samples,
        n_features,
        n_local_trials,
        sub_group_size,
        work_group_size="max",
        dtype=compute_dtype,
        device=device,
    )

    select_best_candidate_kernel = make_argmin_reduction_1d_kernel(
        n_local_trials,
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    reduce_potential_1d_kernel = make_sum_reduction_2d_kernel(
        shape=(n_samples,),
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    reduce_potential_2d_kernel = make_sum_reduction_2d_kernel(
        shape=(n_local_trials, n_samples),
        axis=1,
        work_group_size="max",
        device=device,
        dtype=compute_dtype,
    )

    random_state = create_xoroshiro128pp_states(
        n_local_trials,
        seed=random_state,
        device=sequential_processing_device,
    )
    if sequential_processing_on_different_device:
        sampling_work_group_size = sequential_processing_device.max_work_group_size
    else:
        sampling_work_group_size = max_work_group_size

    sample_center_candidates_kernel = make_sample_center_candidates_kernel(
        n_samples,
        n_local_trials,
        sampling_work_group_size,
        compute_dtype,
    )

    centers_t = dpt.empty((n_features, n_clusters), dtype=compute_dtype, device=device)

    center_indices = dpt.full((n_clusters,), -1, dtype=np.int32, device=device)

    sq_distances_t = dpt.empty(
        (n_local_trials, n_samples), dtype=compute_dtype, device=device
    )

    closest_dist_sq = dpt.empty((n_samples,), dtype=compute_dtype, device=device)

    candidate_ids = dpt.empty((n_local_trials,), dtype=np.int32, device=device)

    # Pick first center randomly
    # Use numpy type to work around https://github.com/IntelPython/dpnp/issues/1238
    random_uint64 = dpt.asnumpy(get_random_raw(random_state))[0]
    starting_center_id = random_uint64 % np.uint64(n_samples)
    center_indices[0] = np.int32(starting_center_id)

    # track index of point, initialize list of closest distances and calculate
    # current potential
    kmeansplusplus_init_kernel(
        X_t,
        sample_weight,
        # OUT
        centers_t,
        center_indices,
        closest_dist_sq,
    )
    total_potential = reduce_potential_1d_kernel(closest_dist_sq)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # First, let's sample indices of candidates using a empirical cumulative
        # density function built using the potential of the samples and squared
        # distances to each sample's closest centroids.

        # NB: this step consists in highly sequential instructions that can only
        # be parallelized in n_local_trials threads, it's up to 50x faster to run it
        # on CPU, and becomes the bottleneck if ran on GPU. Depending on hardware
        # and weight of data transfers, are there cases where it would be more
        # efficient to keep it on GPU ?
        # ???: would it be bad to sample the sample weight once outside the loop and
        # reuse it at each iteration ?
        if sequential_processing_on_different_device:
            candidate_ids = candidate_ids.to_device(sequential_processing_device)

            sample_center_candidates_kernel(
                closest_dist_sq.to_device(sequential_processing_device),
                total_potential.to_device(sequential_processing_device),
                # OUT
                random_state,
                candidate_ids,
            )
            candidate_ids = candidate_ids.to_device(device)
        else:
            sample_center_candidates_kernel(
                closest_dist_sq,
                total_potential,
                # OUT
                random_state,
                candidate_ids,
            )

        # Now, for each (sample, candidate)-pair, compute the minimum between
        # their distance and the previous minimum.

        # XXX: at the cost of one additional pass on data, we could avoid storing
        # entirely distance_to_candidates_t in memory, and save
        # `dtype.nbytes * n_local_trials * n_sample` bytes in memory.
        # Which is better ?
        kmeansplusplus_single_step_fixed_window_kernel(
            X_t,
            sample_weight,
            candidate_ids,
            closest_dist_sq,
            # OUT
            sq_distances_t,
        )

        candidate_potentials = reduce_potential_2d_kernel(sq_distances_t)[:, 0]
        # Use numpy type to work around https://github.com/IntelPython/dpnp/issues/1238
        best_candidate = dpt.asnumpy(
            select_best_candidate_kernel(candidate_potentials)
        )[0]

        total_potential = candidate_potentials[best_candidate : (best_candidate + 1)]

        # Pick the c-th centroid and update the distance
        # to the closest centroid for each sample.
        closest_dist_sq = sq_distances_t[best_candidate, :]
        center_index = candidate_ids[best_candidate]
        centers_t[:, c] = X_t[:, center_index]
        center_indices[c] = center_index

    return centers_t, center_indices
