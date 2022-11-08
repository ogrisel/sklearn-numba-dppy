import math
import warnings
from functools import lru_cache

import numpy as np
import dpctl.tensor as dpt
import dpctl
import dpnp
from sklearn.exceptions import DataConversionWarning

from sklearn_numba_dpex.device import DeviceParams

from sklearn_numba_dpex.common.kernels import (
    make_initialize_to_zeros_2d_kernel,
    make_initialize_to_zeros_3d_kernel,
    make_broadcast_division_1d_2d_kernel,
    make_half_l2_norm_2d_axis0_kernel,
    make_sum_reduction_1d_kernel,
)

from sklearn_numba_dpex.kmeans.kernels import (
    make_lloyd_single_step_fixed_window_kernel,
    make_compute_euclidean_distances_fixed_window_kernel,
    make_label_assignment_fixed_window_kernel,
    make_compute_inertia_kernel,
    make_relocate_empty_clusters_kernel,
    make_select_samples_far_from_centroid_kernel,
    make_centroid_shifts_kernel,
    make_reduce_centroid_data_kernel,
)


def _check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


class _IgnoreSampleWeight:
    pass


@lru_cache
class KMeansDriver:
    """GPU optimized implementation of Lloyd's k-means.

    The current implementation is called "fused fixed", it consists in a sliding window
    of fixed size on the value of the centroids that work items use to accumulate the
    distances of a given sample to each centroids. It is followed, in a same kernel, by
    an update of new centroids in a context of memory privatization to avoid a too high
    cost of atomic operations.

    This class instantiates into a callable that mimics the interface of sklearn's
    private function `_kmeans_single_lloyd` .

    Parameters
    ----------
    preferred_work_group_size_multiple : int
        The kernels will use this value to optimize the distribution of work items. If
        None, it is automatically fetched with pyopencl if possible, else a default of
        64 is applied. It is required to be a power of two.

    work_group_size_multiplier : int
        The size of groups of work items used to execute the kernels is defined as
        `work_group_size_multiplier * preferred_work_group_size_multiple` and, if None,
        is chosen to be equal to the value of max_work_group_size which is fetched
        with dpctl. It is required to be a power of two.

    centroids_window_width_multiplier : int
        The width of the window over the centroids is defined as
        `centroids_window_width_multiplier x preferred_work_group_size_multiple`. The
        higher the value, the higher the cost in shared memory. If None, will default
        to 1. It is required to be a power of two.

    centroids_window_height : int
        The height of the window, counted as a number of features. The higher the
        value, the higher the cost in shared memory. If None, will default to 16. It is
        required to be a power of two.

    global_mem_cache_size : int
        Size in bytes of the size of the global memory cache size. If None, the value
        will be automatically fetched with dpctl. It is used to estimate the maximum
        number of copies of the array of centroids that can be used for privatization.

    centroids_private_copies_max_cache_occupancy : float
        A maximum fraction of global_mem_cache_size that is allowed to be expected for
        use when estimating the maximum number of copies that can be used for
        privatization. If None, will default to 0.7.

    device: str
        A valid sycl device filter.

    X_layout: str
        'F' or 'C'. If None, will default to 'F'.

    dtype: np.float32 or np.float64
        The floating point precision that the kernels should use. If None, will adapt
        to the dtype of the data, else, will cast the data to the appropriate dtype.

    Notes
    -----
    The implementation has been extensively inspired by the "Fused Fixed" strategy
    exposed in [1]_, along with its reference implementatino by the same authors [2]_,
    and the reader can also refer to the complementary slide deck [3]_  with schemas
    that intuitively explain the main computation.

    .. [1] Kruliš, M., & Kratochvíl, M. (2020, August). Detailed analysis and
        optimization of CUDA K-means algorithm. In 49th International
        Conference on Parallel Processing-ICPP (pp. 1-11).

    .. [2] https://github.com/krulis-martin/cuda-kmeans

    .. [3] https://jnamaral.github.io/icpp20/slides/Krulis_Detailed.pdf

    """

    def __init__(
        self,
        preferred_work_group_size_multiple=None,
        work_group_size_multiplier=None,
        centroids_window_width_multiplier=None,
        centroids_window_height=None,
        global_mem_cache_size=None,
        centroids_private_copies_max_cache_occupancy=None,
        device=None,
        X_layout=None,
        dtype=None,
    ):
        dpctl_device = dpctl.SyclDevice(device)
        device_params = DeviceParams(dpctl_device)

        # TODO: set the best possible defaults for all the parameters based on an
        # exhaustive grid search.

        self.global_mem_cache_size = (
            global_mem_cache_size or device_params.global_mem_cache_size
        )

        self.preferred_work_group_size_multiple = _check_power_of_2(
            preferred_work_group_size_multiple
            or device_params.preferred_work_group_size_multiple
        )

        # So far the best default value has been empirically found to be 2 so we use
        # this default.
        # TODO: when it's available in dpctl, use the `max_group_size` attribute
        # exposed by the kernel instead ?
        work_group_size_multiplier = _check_power_of_2(work_group_size_multiplier or 2)

        self.work_group_size = (
            work_group_size_multiplier * self.preferred_work_group_size_multiple
        )

        self.centroids_window_width_multiplier = _check_power_of_2(
            centroids_window_width_multiplier or 1
        )

        self.centroids_window_height = _check_power_of_2(centroids_window_height or 16)

        self.centroids_private_copies_max_cache_occupancy = (
            centroids_private_copies_max_cache_occupancy or 0.7
        )

        self.device = dpctl_device

        # FIXME: "C" is not available at the time (raises a ValueError).
        self.X_layout = X_layout or "F"

        self.has_aspect_fp64 = device_params.has_aspect_fp64

    def lloyd(
        self, X, sample_weight, centers_init, max_iter=300, verbose=False, tol=1e-4
    ):
        """This call is expected to accept the same inputs than sklearn's private
        _kmeans_single_lloyd and produce the same outputs.
        """
        (X, sample_weight, cluster_centers, output_dtype) = self._check_inputs(
            X, sample_weight, centers_init
        )

        use_uniform_weights = (sample_weight == sample_weight[0]).all()

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X, sample_weight, cluster_centers
        )

        (assignments_idx, inertia, best_centroids, n_iteration,) = self._lloyd(
            X_t,
            sample_weight,
            centroids_t,
            use_uniform_weights,
            max_iter,
            verbose,
            tol,
        )

        # TODO: explore leveraging dpnp to benefit from USM to avoid moving
        # centroids back and forth between device and host memory in case
        # a subsequent `.predict` call is requested on the same GPU later.
        return (
            dpt.asnumpy(assignments_idx).astype(np.int32, copy=False),
            inertia,
            # XXX: having a C-contiguous centroid array is expected in sklearn in some
            # unit test and by the cython engine.
            np.ascontiguousarray(
                dpt.asnumpy(best_centroids).astype(output_dtype, copy=False)
            ),
            n_iteration,
        )

    def _lloyd(
        self,
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

        verbose = bool(verbose)

        # Create a set of kernels
        (
            n_centroids_private_copies,
            fused_lloyd_fixed_window_single_step_kernel,
        ) = make_lloyd_single_step_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            return_assignments=bool(verbose),
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            global_mem_cache_size=self.global_mem_cache_size,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            centroids_private_copies_max_cache_occupancy=self.centroids_private_copies_max_cache_occupancy,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        compute_inertia_kernel = make_compute_inertia_kernel(
            n_samples, n_features, self.work_group_size, compute_dtype
        )

        reset_cluster_sizes_private_copies_kernel = make_initialize_to_zeros_2d_kernel(
            size0=n_centroids_private_copies,
            size1=n_clusters,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        reset_centroids_private_copies_kernel = make_initialize_to_zeros_3d_kernel(
            size0=n_centroids_private_copies,
            size1=n_features,
            size2=n_clusters,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        broadcast_division_kernel = make_broadcast_division_1d_2d_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=self.work_group_size,
        )

        compute_centroid_shifts_kernel = make_centroid_shifts_kernel(
            n_clusters=n_clusters,
            n_features=n_features,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        reduce_inertia_kernel = make_sum_reduction_1d_kernel(
            size=n_samples,
            work_group_size=self.work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        reduce_centroid_shifts_kernel = make_sum_reduction_1d_kernel(
            size=n_clusters,
            work_group_size=self.work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        reduce_centroid_data_kernel = make_reduce_centroid_data_kernel(
            n_centroids_private_copies=n_centroids_private_copies,
            n_features=n_features,
            n_clusters=n_clusters,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        # Allocate the necessary memory in the device global memory
        new_centroids_t = dpt.empty_like(centroids_t, device=self.device)
        centroids_half_l2_norm = dpt.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        cluster_sizes = dpt.empty(n_clusters, dtype=compute_dtype, device=self.device)
        centroid_shifts = dpt.empty(n_clusters, dtype=compute_dtype, device=self.device)
        # NB: the same buffer is used for those two arrays because it is never needed
        # to store those simultaneously in memory.
        sq_dist_to_nearest_centroid = per_sample_inertia = dpt.empty(
            n_samples, dtype=compute_dtype, device=self.device
        )
        assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=self.device)
        new_centroids_t_private_copies = dpt.empty(
            (n_centroids_private_copies, n_features, n_clusters),
            dtype=compute_dtype,
            device=self.device,
        )
        cluster_sizes_private_copies = dpt.empty(
            (n_centroids_private_copies, n_clusters),
            dtype=compute_dtype,
            device=self.device,
        )
        empty_clusters_list = dpt.empty(n_clusters, dtype=np.uint32, device=self.device)

        # n_empty_clusters_ is a scalar handled in kernels via a one-element array.
        n_empty_clusters = dpt.empty(1, dtype=np.int32, device=self.device)

        # The loop
        n_iteration = 0
        centroid_shifts_sum = np.inf

        # TODO: Investigate possible speedup with a custom dpctl queue with a custom
        # DAG of events and a final single "wait"
        while (n_iteration < max_iter) and (centroid_shifts_sum > tol):
            half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

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
                new_centroids_t_private_copies,
                cluster_sizes_private_copies,
            )

            if verbose:
                # ???: verbosity comes at the cost of performance since it triggers
                # computing exact inertia at each iteration. Shouldn't this be
                # documented ?
                compute_inertia_kernel(
                    X_t,
                    sample_weight,
                    new_centroids_t,
                    assignments_idx,
                    per_sample_inertia,
                )
                inertia, *_ = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
                print(f"Iteration {n_iteration}, inertia {inertia:5.3e}")

            reduce_centroid_data_kernel(
                cluster_sizes_private_copies,
                new_centroids_t_private_copies,
                cluster_sizes,
                new_centroids_t,
                empty_clusters_list,
                n_empty_clusters,
            )

            n_empty_clusters_ = int(n_empty_clusters[0])
            if n_empty_clusters_ > 0:
                # NB: empty cluster very rarely occurs, and it's more efficient to
                # compute inertia and labels only after occurrences have been detected
                # at the cost of an additional pass on data, rather than computing
                # inertia by default during the first pass on data in case there's an
                # empty cluster.

                # if verbose is True, then assignments to closest centroids already have
                # been computed in the main kernel
                if not verbose:
                    assignment_fixed_window_kernel(
                        X_t, centroids_t, centroids_half_l2_norm, assignments_idx
                    )

                # if verbose is True and if sample_weight is uniform, distances to
                # closest centroids already have been computed in the main kernel
                if not verbose or not use_uniform_weights:
                    # Note that we intentionally we pass unit weights instead of
                    # sample_weight so that per_sample_inertia will be updated to the
                    # (unweighted) squared distance to the nearest centroid.
                    compute_inertia_kernel(
                        X_t,
                        dpt.ones_like(sample_weight),
                        centroids_t,
                        assignments_idx,
                        sq_dist_to_nearest_centroid,
                    )

                self._relocate_empty_clusters(
                    n_empty_clusters_,
                    X_t,
                    sample_weight,
                    new_centroids_t,
                    cluster_sizes,
                    assignments_idx,
                    empty_clusters_list,
                    sq_dist_to_nearest_centroid,
                    per_sample_inertia,
                )

            broadcast_division_kernel(new_centroids_t, cluster_sizes)

            compute_centroid_shifts_kernel(
                centroids_t, new_centroids_t, centroid_shifts
            )

            centroid_shifts_sum, *_ = reduce_centroid_shifts_kernel(centroid_shifts)

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

            n_iteration += 1

        if verbose:
            converged_at = n_iteration - 1
            if centroid_shifts_sum == 0:  # NB: possible if tol = 0
                print(f"Converged at iteration {converged_at}: strict convergence.")

            elif centroid_shifts_sum <= tol:
                print(
                    f"Converged at iteration {converged_at}: center shift "
                    f"{centroid_shifts_sum} within tolerance {tol}."
                )

        # Finally, run an assignment kernel to compute the assignments to the best
        # centroids found, along with the exact inertia.
        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        # NB: inertia and labels could be computed in a single fused kernel, however,
        # re-using the quantity Q = ((1/2)c^2 - <x.c>) that is computed in the
        # assignment kernel to compute distances to closest centroids to evaluate
        # |x|^2 - 2 * Q leads to numerical instability, we prefer evaluating the
        # expression |x-c|^2 which is stable but requires an additional pass on the
        # data.
        # See https://github.com/soda-inria/sklearn-numba-dpex/issues/28
        assignment_fixed_window_kernel(
            X_t, centroids_t, centroids_half_l2_norm, assignments_idx
        )

        compute_inertia_kernel(
            X_t, sample_weight, centroids_t, assignments_idx, per_sample_inertia
        )

        # inertia = per_sample_inertia.sum()
        inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))
        # inertia is now a 1-sized numpy array, we transform it into a scalar:
        inertia = inertia[0]

        return assignments_idx, inertia, centroids_t.T, n_iteration

    def _relocate_empty_clusters(
        self,
        n_empty_clusters,
        X_t,
        sample_weight,
        centroids_t,
        cluster_sizes,
        assignments_idx,
        empty_clusters_list,
        sq_dist_to_nearest_centroid,
        per_sample_inertia,
    ):
        compute_dtype = X_t.dtype.type
        n_features, n_samples = X_t.shape

        select_samples_far_from_centroid_kernel = (
            make_select_samples_far_from_centroid_kernel(
                n_empty_clusters, n_samples, self.work_group_size
            )
        )

        # NB: partition/argpartition kernels are hard to implement right, we use dpnp
        # implementation of `partition` and process to an additional pass on the data
        # to finish the argpartition.
        # ???: how does the dpnp GPU implementation of partition compare with
        # np.partition ?
        # TODO: if the performance compares well, we could also remove some of the
        # kernels in .kernels.utils and replace it with dpnp functions.
        kth = n_samples - n_empty_clusters
        threshold = dpnp.partition(
            dpnp.ndarray(
                shape=sq_dist_to_nearest_centroid.shape,
                buffer=sq_dist_to_nearest_centroid,
            ),
            kth=kth,
        ).get_array()[kth : (kth + 1)]

        samples_far_from_center = dpt.empty(
            n_samples, dtype=np.uint32, device=self.device
        )
        n_selected_gt_threshold = dpt.zeros(1, dtype=np.int32, device=self.device)
        n_selected_eq_threshold = dpt.ones(1, dtype=np.int32, device=self.device)
        select_samples_far_from_centroid_kernel(
            sq_dist_to_nearest_centroid,
            threshold,
            samples_far_from_center,
            n_selected_gt_threshold,
            n_selected_eq_threshold,
        )

        n_selected_gt_threshold_ = int(n_selected_gt_threshold[0])

        # Centroids of empty clusters are relocated to samples in X that are the
        # farthest from their respective centroids. new_centroids_t is updated
        # accordingly.
        relocate_empty_clusters_kernel = make_relocate_empty_clusters_kernel(
            n_empty_clusters,
            n_features,
            n_selected_gt_threshold_,
            self.work_group_size,
            compute_dtype,
        )

        relocate_empty_clusters_kernel(
            X_t,
            sample_weight,
            assignments_idx,
            samples_far_from_center,
            empty_clusters_list,
            per_sample_inertia,
            centroids_t,
            cluster_sizes,
        )

    def get_labels(self, X, centers):
        labels, _ = self._get_labels_inertia(X, centers, with_inertia=False)
        return dpt.asnumpy(labels).astype(np.int32, copy=False)

    def get_inertia(self, X, sample_weight, centers):
        _, inertia = self._get_labels_inertia(
            X, centers, sample_weight, with_inertia=True
        )
        return inertia

    def _get_labels_inertia(
        self, X, centers, sample_weight=_IgnoreSampleWeight, with_inertia=True
    ):
        X, sample_weight, centers, output_dtype = self._check_inputs(
            X, sample_weight=sample_weight, cluster_centers=centers
        )

        if sample_weight is _IgnoreSampleWeight:
            sample_weight = None

        X_t, sample_weight, centroids_t = self._load_transposed_data_to_device(
            X, sample_weight, centers
        )

        assignments_idx, inertia = self._driver_get_labels_inertia(
            X_t, centroids_t, sample_weight, with_inertia
        )

        if with_inertia:
            # inertia is a 1-sized numpy array, we transform it into a scalar:
            inertia = inertia.astype(output_dtype)[0]

        return assignments_idx, inertia

    def _driver_get_labels_inertia(self, X_t, centroids_t, sample_weight, with_inertia):
        compute_dtype = X_t.dtype.type
        n_features, n_samples = X_t.shape
        n_clusters = centroids_t.shape[1]

        label_assignment_fixed_window_kernel = make_label_assignment_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        half_l2_norm_kernel = make_half_l2_norm_2d_axis0_kernel(
            size0=n_features,
            size1=n_clusters,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        centroids_half_l2_norm = dpt.empty(
            n_clusters, dtype=compute_dtype, device=self.device
        )
        assignments_idx = dpt.empty(n_samples, dtype=np.uint32, device=self.device)

        half_l2_norm_kernel(centroids_t, centroids_half_l2_norm)

        label_assignment_fixed_window_kernel(
            X_t, centroids_t, centroids_half_l2_norm, assignments_idx
        )

        if not with_inertia:
            return assignments_idx, None

        compute_inertia_kernel = make_compute_inertia_kernel(
            n_samples, n_features, self.work_group_size, compute_dtype
        )

        reduce_inertia_kernel = make_sum_reduction_1d_kernel(
            size=n_samples,
            work_group_size=self.work_group_size,
            device=self.device,
            dtype=compute_dtype,
        )

        per_sample_inertia = dpt.empty(
            n_samples, dtype=compute_dtype, device=self.device
        )

        compute_inertia_kernel(
            X_t, sample_weight, centroids_t, assignments_idx, per_sample_inertia
        )

        # inertia = per_sample_inertia.sum()
        inertia = dpt.asnumpy(reduce_inertia_kernel(per_sample_inertia))

        return assignments_idx, inertia

    def get_euclidean_distances(self, X, Y):

        (X, _, Y, output_dtype) = self._check_inputs(
            X,
            sample_weight=_IgnoreSampleWeight,
            cluster_centers=Y,
        )

        X_t, _, Y_t = self._load_transposed_data_to_device(X, None, Y)

        euclidean_distances = self._get_euclidean_distances(X_t, Y_t)

        return dpt.asnumpy(euclidean_distances).astype(output_dtype, copy=False)

    def _get_euclidean_distances(self, X_t, Y_t):
        compute_dtype = X_t.dtype.type
        n_features, n_samples = X_t.shape
        n_clusters = Y_t.shape[1]

        euclidean_distances_fixed_window_kernel = make_compute_euclidean_distances_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=self.work_group_size,
            dtype=compute_dtype,
        )

        euclidean_distances_t = dpt.empty(
            (n_clusters, n_samples), dtype=compute_dtype, device=self.device
        )

        euclidean_distances_fixed_window_kernel(X_t, Y_t, euclidean_distances_t)
        return euclidean_distances_t.T

    def _set_dtype(self, X, sample_weight, centers_init):
        output_dtype = compute_dtype = np.dtype(X.dtype).type
        copy = True
        if (compute_dtype != np.float32) and (compute_dtype != np.float64):
            text = (
                f"KMeans has been set to compute with type {compute_dtype} but only "
                f"the types float32 and float64 are supported. The computations and "
                f"outputs will default back to float32 type."
            )
            output_dtype = compute_dtype = np.float32
        elif (compute_dtype == np.float64) and not self.has_aspect_fp64:
            text = (
                f"KMeans is set to compute with type {compute_dtype} but this type is "
                f"not supported by the device {self.device.name}. The computations "
                f"will default back to float32 type."
            )
            compute_dtype = np.float32

        else:
            copy = False

        if copy:
            text += (
                f" A copy of the data casted to type {compute_dtype} will be created. "
                f"To save memory and suppress this warning, ensure that the dtype of "
                f"the input data matches the dtype required for computations."
            )
            warnings.warn(text, DataConversionWarning)
            # TODO: instead of triggering a copy on the host side, we could use the
            # dtype to allocate a shared USM buffer and fill it with casted values from
            # X. In this case we should only warn when:
            #     (dtype == np.float64) and not self.has_aspect_fp64
            # The other cases would not trigger any additional memory copies.
            X = X.astype(compute_dtype)

        centers_init_dtype = centers_init.dtype
        if centers_init.dtype != compute_dtype:
            warnings.warn(
                f"The centers have been passed with type {centers_init_dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that the centers are passed "
                f"with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            centers_init = centers_init.astype(compute_dtype)

        if (sample_weight is not _IgnoreSampleWeight) and (
            sample_weight.dtype != compute_dtype
        ):
            warnings.warn(
                f"sample_weight has been passed with type {sample_weight.dtype} but "
                f"type {compute_dtype} is expected. A copy will be created with the "
                f"correct type {compute_dtype}. Ensure that sample_weight is passed "
                f"with the correct dtype to save memory and suppress this warning.",
                DataConversionWarning,
            )
            sample_weight = sample_weight.astype(compute_dtype)

        return X, sample_weight, centers_init, output_dtype

    def _check_inputs(self, X, sample_weight, cluster_centers):

        if sample_weight is None:
            sample_weight = np.ones(len(X), dtype=X.dtype)

        X, sample_weight, cluster_centers, output_dtype = self._set_dtype(
            X, sample_weight, cluster_centers
        )

        return X, sample_weight, cluster_centers, output_dtype

    def _load_transposed_data_to_device(self, X, sample_weight, cluster_centers):
        # Transfer the input data to device memory,
        # TODO: let the user pass directly dpt or dpnp arrays to avoid copies.
        if self.X_layout == "C":
            # TODO: support the C layout and benchmark it and default to it if
            # performances are better
            raise ValueError("C layout is currently not supported.")
            X_t = dpt.from_numpy(X, device=self.device).T
            assert (
                X_t.strides[0] == 1
            )  # Fortran memory layout, equivalent to C layout on transposed
        elif self.X_layout == "F":
            X_t = dpt.from_numpy(X.T, device=self.device)
            assert (
                X_t.strides[1] == 1
            )  # C memory layout, equivalent to Fortran layout on transposed
        else:
            raise ValueError(
                f"Expected X_layout to be equal to 'C' or 'F', but got {self.X_layout} ."
            )
        if sample_weight is not None:
            sample_weight = dpt.from_numpy(sample_weight, device=self.device)
        cluster_centers = dpt.from_numpy(cluster_centers.T, device=self.device)

        return X_t, sample_weight, cluster_centers
