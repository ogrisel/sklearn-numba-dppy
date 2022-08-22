import math
import warnings

import numpy as np
import dpctl

from sklearn_numba_dpex.utils._device import _DeviceParams

from sklearn_numba_dpex.kmeans.kernels import (
    make_initialize_to_zeros_1dim_kernel,
    make_initialize_to_zeros_2dim_kernel,
    make_initialize_to_zeros_3dim_kernel,
    make_copyto_kernel,
    make_broadcast_division_kernel,
    make_center_shift_kernel,
    make_half_l2_norm_dim0_kernel,
    make_sum_reduction_1dim_kernel,
    make_sum_reduction_2dim_kernel,
    make_sum_reduction_3dim_kernel,
    make_fused_fixed_window_kernel,
    make_assignment_fixed_window_kernel,
)


def _check_power_of_2(e):
    if e != 2 ** (math.log2(e)):
        raise ValueError(f"Expected a power of 2, got {e}")
    return e


class LLoydKMeansDriver:
    def __init__(
        self,
        global_mem_cache_size=None,
        preferred_work_group_size_multiple=None,
        work_group_size_multiplier=None,
        centroids_window_width_multiplier=None,
        centroids_window_height=None,
        centroids_private_copies_max_cache_occupancy=None,
        device=None,
        X_layout=None,
        dtype=None,
    ):
        dpctl_device = dpctl.SyclDevice(device)
        device_params = _DeviceParams(dpctl_device)

        self.global_mem_cache_size = (
            global_mem_cache_size or device_params.global_mem_cache_size
        )

        self.preferred_work_group_size_multiple = _check_power_of_2(
            preferred_work_group_size_multiple
            or device_params.preferred_work_group_size_multiple
        )

        self.work_group_size_multiplier = _check_power_of_2(
            work_group_size_multiplier
            or (
                device_params.max_work_group_size
                // self.preferred_work_group_size_multiple
            )
        )

        self.centroids_window_width_multiplier = _check_power_of_2(
            centroids_window_width_multiplier or 1
        )

        self.centroids_window_height = _check_power_of_2(centroids_window_height or 16)

        self.centroids_private_copies_max_cache_occupancy = (
            centroids_private_copies_max_cache_occupancy or 0.7
        )

        self.device = dpctl_device

        self.X_layout = X_layout or "F"

        self.has_aspect_fp64 = device_params.has_aspect_fp64
        self.has_aspect_fp16 = device_params.has_aspect_fp16

        self.dtype = dtype
        if dtype is not None:
            dtype = np.dtype(dtype).type
            if (
                (dtype != np.float16)
                and (dtype != np.float32)
                and (dtype != np.float64)
            ):
                raise ValueError(
                    f"Valid types are float64, float32 or float16, but got f{dtype}"
                )
            self.dtype = dtype
            if ((self.dtype == np.float64) and not self.has_aspect_fp64) or (
                (self.dtype == np.float16) and not self.has_aspect_fp16
            ):
                raise RuntimeError(
                    f"Computations with precision f{self.dtype} has been explicitly "
                    f"requested to the KMeans driver but the device {dpctl_device.name} does not "
                    f"support it."
                )

    def __call__(
        self,
        X,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        x_squared_norms=None,
        tol=1e-4,
        n_threads=1,
    ):
        """GPU optimized implementation of Lloyd's k-means.

        Inspired by the "Fused Fixed" strategy exposed in:
        Kruliš, M., & Kratochvíl, M. (2020, August). Detailed analysis and
        optimization of CUDA K-means algorithm. In 49th International
        Conference on Parallel Processing-ICPP (pp. 1-11).
        """
        work_group_size = (
            self.work_group_size_multiplier * self.preferred_work_group_size_multiple
        )

        X, centers_init, dtype = self._set_dtype(X, centers_init)

        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_clusters = centers_init.shape[0]

        reset_inertia = make_initialize_to_zeros_1dim_kernel(
            n_samples=n_samples, work_group_size=work_group_size, dtype=dtype
        )

        copyto = make_copyto_kernel(
            n_clusters, n_features, work_group_size=work_group_size
        )

        broadcast_division = make_broadcast_division_kernel(
            n_samples=n_clusters,
            n_features=n_features,
            work_group_size=work_group_size,
        )

        get_center_shifts = make_center_shift_kernel(
            n_clusters, n_features, work_group_size=work_group_size, dtype=dtype
        )

        half_l2_norm = make_half_l2_norm_dim0_kernel(
            n_clusters, n_features, work_group_size=work_group_size, dtype=dtype
        )

        # NB: assumes work_group_size is a power of two
        reduce_inertia = make_sum_reduction_1dim_kernel(
            n_samples, work_group_size=work_group_size, device=self.device, dtype=dtype
        )

        reduce_center_shifts = make_sum_reduction_1dim_kernel(
            n_clusters, work_group_size=work_group_size, device=self.device, dtype=dtype
        )

        (
            nb_centroids_private_copies,
            fixed_window_fused_kernel,
        ) = make_fused_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            global_mem_cache_size=self.global_mem_cache_size,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            centroids_private_copies_max_cache_occupancy=self.centroids_private_copies_max_cache_occupancy,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        reset_centroid_counts_private_copies = make_initialize_to_zeros_2dim_kernel(
            n_samples=n_clusters,
            n_features=nb_centroids_private_copies,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        reset_centroids_private_copies = make_initialize_to_zeros_3dim_kernel(
            dim0=nb_centroids_private_copies,
            dim1=n_features,
            dim2=n_clusters,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        reduce_centroid_counts_private_copies = make_sum_reduction_2dim_kernel(
            n_samples=n_clusters,
            n_features=nb_centroids_private_copies,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        reduce_centroids_private_copies = make_sum_reduction_3dim_kernel(
            dim0=nb_centroids_private_copies,
            dim1=n_features,
            dim2=n_clusters,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        fixed_window_assignment_kernel = make_assignment_fixed_window_kernel(
            n_samples,
            n_features,
            n_clusters,
            preferred_work_group_size_multiple=self.preferred_work_group_size_multiple,
            centroids_window_width_multiplier=self.centroids_window_width_multiplier,
            centroids_window_height=self.centroids_window_height,
            work_group_size=work_group_size,
            dtype=dtype,
        )

        # TODO: let the user pass directly dpctl.tensor arrays to avoid copies.
        if self.X_layout == "C":
            # TODO: support the C layout and benchmark it and default to it if
            # performances are better
            raise ValueError("C layout is currently not supported.")
            X_t = dpctl.tensor.from_numpy(X.astype(np.float32), device=self.device).T
            assert (
                X_t.strides[0] == 1
            )  # Fortran memory layout, equvalent to C layout on tranposed
        elif self.X_layout == "F":
            X_t = dpctl.tensor.from_numpy(X.astype(np.float32).T, device=self.device)
            assert (
                X_t.strides[1] == 1
            )  # C memory layout, equvalent to Fortran layout on transposed
        else:
            raise ValueError(
                f"Expected X_layout to be equal to 'C' or 'F', but got {self.X_layout} ."
            )

        centroids_t = dpctl.tensor.from_numpy(
            centers_init.astype(np.float32).T, device=self.device
        )
        centroids_t_copy_array = dpctl.tensor.empty_like(
            centroids_t, device=self.device
        )
        best_centroids_t = dpctl.tensor.empty_like(
            centroids_t_copy_array, device=self.device
        )
        centroids_half_l2_norm = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        centroid_counts = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        center_shifts = dpctl.tensor.empty(
            n_clusters, dtype=np.float32, device=self.device
        )
        inertia = dpctl.tensor.empty(n_samples, dtype=np.float32, device=self.device)
        assignments_idx = dpctl.tensor.empty(
            n_samples, dtype=np.uint32, device=self.device
        )

        centroids_t_private_copies = dpctl.tensor.empty(
            (nb_centroids_private_copies, n_features, n_clusters),
            dtype=np.float32,
            device=self.device,
        )
        centroid_counts_private_copies = dpctl.tensor.empty(
            (nb_centroids_private_copies, n_clusters),
            dtype=np.float32,
            device=self.device,
        )

        n_iteration = 0
        center_shifts_sum = np.inf
        best_inertia = np.inf
        copyto(centroids_t, best_centroids_t)

        while (n_iteration < max_iter) and (center_shifts_sum >= tol):
            half_l2_norm(centroids_t, centroids_half_l2_norm)

            reset_inertia(inertia)
            reset_centroid_counts_private_copies(centroid_counts_private_copies)
            reset_centroids_private_copies(centroids_t_private_copies)

            # TODO: implement special case where only one copy is needed
            fixed_window_fused_kernel(
                X_t,
                centroids_t,
                centroids_half_l2_norm,
                inertia,
                centroids_t_private_copies,
                centroid_counts_private_copies,
            )

            reduce_centroid_counts_private_copies(
                centroid_counts_private_copies, centroid_counts
            )
            reduce_centroids_private_copies(
                centroids_t_private_copies, centroids_t_copy_array
            )

            broadcast_division(centroids_t_copy_array, centroid_counts)

            get_center_shifts(centroids_t, centroids_t_copy_array, center_shifts)

            center_shifts_sum = dpctl.tensor.asnumpy(
                reduce_center_shifts(center_shifts)
            )[0]

            inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

            if inertia_sum < best_inertia:
                best_inertia = inertia_sum
                copyto(centroids_t, best_centroids_t)

            centroids_t, centroids_t_copy_array = (
                centroids_t_copy_array,
                centroids_t,
            )

            n_iteration += 1

        half_l2_norm(best_centroids_t, centroids_half_l2_norm)
        reset_inertia(inertia)

        fixed_window_assignment_kernel(
            X_t,
            best_centroids_t,
            centroids_half_l2_norm,
            inertia,
            assignments_idx,
        )

        inertia_sum = dpctl.tensor.asnumpy(reduce_inertia(inertia))[0]

        return (
            dpctl.tensor.asnumpy(assignments_idx).astype(np.int32),
            inertia_sum,
            np.ascontiguousarray(dpctl.tensor.asnumpy(best_centroids_t.T)),
            n_iteration,
        )

    def _set_dtype(self, X, centers_init):
        dtype = self.dtype or X.dtype
        dtype = np.dtype(dtype).type
        copy = True
        if (dtype != np.float16) and (dtype != np.float32) and (dtype != np.float64):
            text = (
                f"The data has been submitted with type {dtype} but only the types "
                f"float16, float32 or float64 are supported. The computations will "
                f"default back to float32 type."
            )
            dtype = np.float32
        elif ((dtype == np.float16) and not self.has_aspect_fp16) or (
            (dtype == np.float64) and not self.has_aspect_fp64
        ):
            text = (
                f"The data has been submitted with type {dtype} but this type is not "
                f"supported by the device {self.device.name}. The computations will "
                f"default back to float32 type."
            )
            dtype = np.float32
        elif dtype != X.dtype:
            text = (
                f"Kmeans is set to compute with dtype {dtype} but the data has "
                f"been submitted with type {X.dtype}."
            )

        else:
            copy = False

        if copy:
            text += (
                f" A copy of the data casted to type {dtype} will be created. To "
                f"save memory and prevent this warning, ensure that the dtype of "
                f"the input data matches the dtype required for computations."
            )
            warnings.warn(text, RuntimeWarning)
            X = X.astype(dtype)

        centers_init_dtype = centers_init.dtype
        if centers_init.dtype != dtype:
            warnings.warn(
                f"The centers have been initialized with type {centers_init_dtype} but "
                f"type {dtype} is expected. A copy will be created with the correct "
                f"type {dtype}. Ensure that the centers are initialized with the "
                f"correct dtype to save memory and disable this warning."
            )
            centers_init = centers_init.astype(dtype)

        if (self.dtype is None) and dtype is np.float64:
            warnings.warn(
                f"The computations will use type float64, but float32 computations are "
                f"generally regarded as faster and accurate enough. To disable this "
                f"warning, please explicitly set dtype to float64 when instanciating "
                f"the driver, or pass an array of float16 or float32 data."
            )

        return X, centers_init, dtype