# This implementation takes inspiration from other open-source implementations, such as
# [1] https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k_radix_selection.cuh  # noqa
# or
# [2] https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell  # noqa

# TODO: apply Dr. TopK optimizations
# (https://dl.acm.org/doi/pdf/10.1145/3458817.3476141)

import math
from functools import lru_cache

import dpctl.tensor as dpt
import numba_dpex as dpex
import numpy as np

from sklearn_numba_dpex.common._utils import (
    _enforce_matmul_like_work_group_geometry,
    _get_global_mem_cache_size,
    _get_sequential_processing_device,
    check_power_of_2,
)
from sklearn_numba_dpex.common.kernels import make_fill_kernel, make_range_kernel
from sklearn_numba_dpex.common.reductions import make_sum_reduction_2d_kernel

zero_idx = np.int64(0)
one_idx = np.int64(1)
two_idx = np.int64(2)
count_one_as_a_long = np.int64(1)
count_one_as_an_int = np.int32(1)


# The Top-K implemented in this file uses radix sorting. Radix sorting consists in
# using lexicographical order on bits to sort the items. It assumes that the
# lexicographical order on the bits of the items that are being sorted matches their
# natural ordering.

# However, this is not true for sets that mix positive and negative items, given that
# the dtypes that are supported set the first bit to 1 to encode the negative sign, and
# 0 to encode the positive sign, in such a way that negative floats are lesser than
# positive floats using the natural order while the opposite is true with
# lexicograpbical order on bits.

# Still, it is possible to enable radix sorting for dtypes that don't follow the
# lexicographical order, providing a bijection can be found with an unsigned type, such
# as the lexicographical order on the unsigned dtype matches the natural order on the
# input dtype. The target dtype must also support bitwise operations. Thus, float32 is
# mapped to uint32, float64 is mapped to uint64,...

# In practice such bijections can be found for all dtypes and are cheap to compute. The
# code that follows defines bijections to unsigned integers for float32 and float64
# floats, with the following bitwise transformations:
#    - reinterpret the float as an unsigned integer (i.e the same buffer used to encode
#      a float is used to encode an unsigned integer. Said differently, first transform
#      the float to the integer such as the bits used to encode the integer are the
#      same than the bits used to encode the float)
#    - if the float item is positive, set the first bit of the unsigned integer to 1
#    - if the float item is negative, flip all of the bits of the unsigned integer
#      (i.e set to 1 if bit is 0, else 0)
#
# See also:
# https://stackoverflow.com/questions/4640906/radix-sort-sorting-a-float-data/4641059#4641059  # noqa
#
# Note that our current usecases of the use of radix (selecting TopK distances, that
# are by definition positive floats)  only involve positive floats.
# TODO: for arrays of positive floats the mapping trick could be skipped ?

# dict that maps dtypes to the dtype of the space in which the radix sorting will be
# actually done
uint_type_mapping = {np.float32: np.uint32, np.float64: np.uint64}


# The following closure define a device function that transforms items to their
# counterpart in the sorting space. For a given dtype, it returns a function that takes
# as input an item from the input dtype, that has prealably been reinterpreted as an
# item of the target dtype (see `uint_type_mapping`). It returns another item from the
# target dtype such as the lexicographical order on transformed items matches the
# natural order on the the input dtype.

# The second closure define the inverse device function.


def _make_lexicographical_mapping_kernel_func(dtype):
    n_bits_per_item = _get_n_bits_per_item(dtype)
    sign_bit_idx = np.int32(n_bits_per_item - 1)

    uint_type = uint_type_mapping[dtype]
    sign_mask = uint_type(2 ** (sign_bit_idx))

    @dpex.func
    def lexicographical_mapping(item):
        mask = (-(item >> sign_bit_idx)) | sign_mask
        return item ^ mask

    return lexicographical_mapping


def _make_lexicographical_unmapping_kernel_func(dtype):
    n_bits_per_item = _get_n_bits_per_item(dtype)
    sign_bit_idx = np.int32(n_bits_per_item - 1)
    uint_type = uint_type_mapping[dtype]
    one_as_uint_dtype = uint_type(1)
    sign_mask = uint_type(2 ** (sign_bit_idx))

    @dpex.func
    def lexicographical_unmapping(item):
        mask = ((item >> sign_bit_idx) - one_as_uint_dtype) | sign_mask
        return item ^ mask

    return lexicographical_unmapping


def topk(array_in, k, group_sizes=None):
    """Compute the k greatest values found in each row of `array_in`.

    Parameters
    ----------
    array_in : dpctl.tensor array
        Input array in which looking for the top k values. `array_in` is expected to be
        one or two-dimensional. If two-dimensional, the `top-k` search is ran row-wise.
        For best performance, it is recommended to submit C-contiguous arrays.

    k: int
        Number of values to search for.

    group_sizes: tuple of int
        Can be optionnally used to configure `(work_group_size, sub_group_size)`
        parameters for the kernels.


    Returns
    -------
    result : dpctl.tensor array
        An array containing the k greatest valus found in each row of `array_in`.

    Notes
    -----
    The output is not deterministic: the order of the output is undefined. Successive
    calls can return the same items in different order.
    """
    _get_topk_kernel = _make_get_topk_kernel(
        k,
        array_in.shape,
        array_in.dtype.type,
        array_in.device.sycl_device,
        group_sizes,
        output="values",
    )

    return _get_topk_kernel(array_in)


def topk_idx(array_in, k, group_sizes=None):
    """Compute the indices of the k greatest values found in each row of `array_in`.

    Parameters
    ----------
    array_in : dpctl.tensor array
        Input array in which looking for the top k values. `array_in` is expected to be
        one or two-dimensional. If two-dimensional, the `top-k` search is ran row-wise.
        For best performance, it is recommended to submit C-contiguous arrays.

    k: int
        Number of values to search for.

    group_sizes: tuple of int
        Can be optionnally used to configure `(work_group_size, sub_group_size)`
        parameters for the kernels.


    Returns
    -------
    result : dpctl.tensor array
        An array with dtype int64 containing the indices of the k greatest values
        found in  each row of `array_in`.

    Notes
    -----
    The output is not deterministic:
        - the order of the output is undefined. Successive calls can return the same
        items in different order.

        - If there are more indices for the smallest top k value than the number of
        time this value occurs among the top k, then the indices that are returned
        for this value can be different between two successive calls.

    """
    _get_topk_kernel = _make_get_topk_kernel(
        k,
        array_in.shape,
        array_in.dtype.type,
        array_in.device.sycl_device,
        group_sizes,
        output="idx",
    )

    return _get_topk_kernel(array_in)


def _make_get_topk_kernel(
    k, shape, dtype, device, group_sizes, output, reuse_result_buffer=False
):
    """Returns a `_get_topk_kernel` closure.

    The closure can be passed an array with attributes `shape`, `dtype` and `device`
    and will perform a TopK search, returning requested top-k items.

    As long as a closure is referenced, it keeps in cache pre-allocated buffers and
    pre-defined kernel functions. Thus, it is more efficient to perform sequential
    calls to the same closure, since subsequent calls will not have the overhead of
    re-defining kernels and re-allocating buffers.

    For isolated calls, top-level user-exposed `topk` and `topk_idx` can be used
    instead. They include definition of kernels, allocation of buffers, and
    cleaning of said allocations afterwards.

    By default, the memory allocation for the result array is not reused. This is to
    avoid a previously computed result to be erased by a subsequent call to the same
    closure without the user noticing. Reusing the same buffer can still be enforced by
    setting `reuse_result_buffer=True`.
    """
    # TODO: it seems a kernel specialized for 1d arrays would show 10-20% better
    # performance. If this case becomes specifically relevant, consider implementing
    # this case separately rather than using the generic multirow top k for 1d arrays.
    is_1d = len(shape) == 1
    if is_1d:
        n_rows = 1
        n_cols = shape[0]
    else:
        n_rows, n_cols = shape

    work_group_size, get_topk_threshold = _make_get_topk_threshold_kernel(
        n_rows, n_cols, k, dtype, device, group_sizes
    )

    (
        _initialize_result,
        _initialize_result_col_idx,
        gather_results_kernel,
    ) = _get_gather_results_kernels(
        n_rows, n_cols, k, work_group_size, dtype, device, output, reuse_result_buffer
    )

    def _get_topk(array_in):
        if is_1d:
            array_in = dpt.reshape(array_in, (1, -1))

        (
            threshold,
            n_threshold_occurences_in_topk,
            n_threshold_occurences_in_data,
        ) = get_topk_threshold(array_in)

        result_col_idx = _initialize_result_col_idx()
        result = _initialize_result(array_in.dtype.type)

        gather_results_kernel(
            array_in,
            threshold,
            n_threshold_occurences_in_topk,
            n_threshold_occurences_in_data,
            result_col_idx,
            # OUT
            result,
        )

        if is_1d:
            return dpt.reshape(result, (-1,))

        return result

    return _get_topk


@lru_cache
def _get_gather_results_kernels(
    n_rows, n_cols, k, work_group_size, dtype, device, output, reuse_result_buffer
):
    if output == "values":
        gather_results_kernel = _make_gather_topk_kernel(
            n_rows,
            n_cols,
            k,
            work_group_size,
        )
        if reuse_result_buffer:
            result = dpt.empty((n_rows, k), dtype=dtype, device=device)

            def _initialize_result(dtype):
                return result

        else:

            def _initialize_result(dtype):
                return dpt.empty((n_rows, k), dtype=dtype, device=device)

    elif output == "idx":
        gather_results_kernel = _make_gather_topk_idx_kernel(
            n_rows,
            n_cols,
            k,
            work_group_size,
        )
        if reuse_result_buffer:
            result = dpt.empty((n_rows, k), dtype=np.int64, device=device)

            def _initialize_result(dtype):
                return result

        else:

            def _initialize_result(dtype):
                return dpt.empty((n_rows, k), dtype=np.int64, device=device)

    elif output == "values+idx":
        raise NotImplementedError

    else:
        raise ValueError(
            'Expected output parameter value to be equal to "values", "idx" or '
            f'"values+idx", but got {output} instead.'
        )

    # `result_col_idx` is used to maintain an atomically incremented index of the next
    # result value to be stored:.
    # Note that the ordering of the topk is non-deterministic and depends on the
    # concurrency of the parallel work items.
    result_col_idx = dpt.empty((n_rows,), dtype=np.int32, device=device)
    initialize_result_col_idx_kernel = make_fill_kernel(
        fill_value=0,
        shape=(n_rows,),
        work_group_size=work_group_size,
        dtype=np.int32,
    )

    def _initialize_result_col_idx():
        initialize_result_col_idx_kernel(result_col_idx)
        return result_col_idx

    return _initialize_result, _initialize_result_col_idx, gather_results_kernel


@lru_cache
def _make_get_topk_threshold_kernel(n_rows, n_cols, k, dtype, device, group_sizes):
    if n_cols < k:
        raise ValueError(
            "Expected k to be greater than or equal to the number of items in the "
            f"search space, but got k={k} and {n_cols} items in the search space."
        )

    if dtype not in uint_type_mapping:
        raise ValueError(
            f"topk currently only supports dtypes in {uint_type_mapping.keys()}, but "
            f"got dtype={dtype} ."
        )
    uint_type = uint_type_mapping[dtype]
    n_bits_per_item = _get_n_bits_per_item(dtype)

    if group_sizes is not None:
        work_group_size, sub_group_size = group_sizes
    else:
        work_group_size = device.max_work_group_size
        sub_group_size = min(device.sub_group_sizes)

    global_mem_cache_size = _get_global_mem_cache_size(device)
    counts_private_copies_max_cache_occupancy = 0.7

    (
        radix_size,
        radix_bits,
        n_counts_private_copies,
        create_radix_histogram_kernel,
    ) = _make_create_radix_histogram_kernel(
        n_rows,
        n_cols,
        64 if group_sizes is None else work_group_size,
        16 if group_sizes is None else sub_group_size,
        global_mem_cache_size,
        counts_private_copies_max_cache_occupancy,
        dtype,
        device,
    )

    # The kernel `check_radix_histogram` seems to be more adapted to cpu or gpu
    # depending on if `n_rows` is large enough to leverage enough of the
    # parallelization capabilities of the gpu.
    # TODO: benchmark the improvement

    # Some other steps in the main loop are more fitted for cpu than gpu.

    # To this purpose the following variables check availability of a cpu and wether
    # a data transfer is required when checking the radix histogramn and when updating
    # the radix filtering variables.
    (check_radix_histogram_device, check_radix_histogram_on_sequential_device,) = (
        sequential_processing_device,
        sequential_processing_on_different_device,
    ) = _get_sequential_processing_device(device)

    if n_rows >= device.max_compute_units:
        check_radix_histogram_on_sequential_device = False
        check_radix_histogram_device = device
        check_radix_histogram_work_group_size = work_group_size
    else:
        check_radix_histogram_work_group_size = (
            check_radix_histogram_device.max_work_group_size
        )

    change_device_for_radix_update = (
        check_radix_histogram_device.filter_string
        != sequential_processing_device.filter_string
    )

    update_radix_position, check_radix_histogram = _make_check_radix_histogram_kernel(
        radix_size, dtype, check_radix_histogram_work_group_size
    )

    # In each iteration of the main loop, a lesser, decreasing amount of top values are
    # searched for in a decreasing subset of data. The following variable records the
    # amount of top values to search for at the given iteration (starting at k).
    k_in_subset_ = dpt.empty(
        n_rows, dtype=np.int32, device=check_radix_histogram_device
    )
    initialize_k_in_subset_kernel = make_fill_kernel(
        fill_value=k, shape=(n_rows,), work_group_size=work_group_size, dtype=np.int32
    )

    # Depending on the data, it's possible that the search early stops before having to
    # scan all the bits of data (in the case where exactly top-k values can be
    # identified with a partial sort on some given prefix length).

    # If `n_rows > 1`, the search might terminate sooner in some rows than others.
    # Let's call "active rows" the rows for which the search is still ongoing at the
    # current iteration. Rows that are not "active rows" are finished searching and are
    # waiting for the search in other rows to complete.

    # Memory allocation for the number of currently active rows
    n_active_rows_ = dpt.empty(1, dtype=np.int64, device=check_radix_histogram_device)
    # Memory allocation for the number of currently active rows in the next iteration
    new_n_active_rows_ = dpt.empty(
        1, dtype=np.int64, device=check_radix_histogram_device
    )

    # NB: The following parameters might be transfered back and forth in memories of
    # different devices. In this case, pre-allocation is detrimental. Depending on
    # wether two different devices are used for different steps or not, their
    # respective memory buffers are cached, or re-allocated at every call.

    # TODO: it would be useful if `dpctl` supports copying a buffer from a given device
    # to a pre-allocated array of another device. It is currently not supported.

    # `active_rows_mapping` and `new_active_rows_mapping` are arrays that hold a list
    # of the indexes of currently active rows, at the current iteration, and at the
    # next iteration
    # NB: at a given iteration, only slots from 0 to `n_active_rows` are used

    # `desired_mask_value` is an array that defines the mask value to search for when
    # filtering data.
    if check_radix_histogram_on_sequential_device:  # no caching of buffers

        def initialize_active_rows_mapping():
            active_rows_mapping = dpt.arange(n_rows, dtype=np.int64, device=device)
            new_active_rows_mapping = dpt.zeros(
                n_rows, dtype=np.int64, device=check_radix_histogram_device
            )
            return active_rows_mapping, new_active_rows_mapping

        def initialize_desired_masked_value():
            return dpt.zeros((n_rows,), dtype=uint_type, device=device)

    else:  # use caching
        active_rows_mapping_ = dpt.empty(
            n_rows, dtype=np.int64, device=check_radix_histogram_device
        )
        new_active_rows_mapping_ = dpt.empty(
            n_rows, dtype=np.int64, device=check_radix_histogram_device
        )
        initialize_active_rows_mapping_kernel = make_range_kernel(
            n_rows, work_group_size
        )

        def initialize_active_rows_mapping():
            initialize_active_rows_mapping_kernel(active_rows_mapping_)
            return active_rows_mapping_, new_active_rows_mapping_

        desired_masked_value_ = dpt.empty((n_rows,), dtype=uint_type, device=device)
        initialize_desired_masked_value_kernel = make_fill_kernel(
            fill_value=0,
            shape=(n_rows,),
            work_group_size=work_group_size,
            dtype=uint_type,
        )

        def initialize_desired_masked_value():
            initialize_desired_masked_value_kernel(desired_masked_value_)
            return desired_masked_value_

    # `radix_position` holds the position of the radix that is currently used for
    # sorting data at a given iteration.
    # `mask_for_desired_value` defines, along with `desired_mask_value`, the condition
    # that is applied at each iteration to skip already data that has already been
    # filtered out by sorting on previous radixes. The remaining data is the data that
    # will be scanned at the current iteration.
    if sequential_processing_on_different_device:  # no caching

        def initialize_radix_mask():
            radix_position_ = dpt.asarray(
                [n_bits_per_item - radix_bits], dtype=uint_type, device=device
            )
            mask_for_desired_value_ = dpt.zeros((1,), dtype=uint_type, device=device)
            return radix_position_, mask_for_desired_value_

    else:  # use caching
        radix_position_ = dpt.empty(1, dtype=uint_type, device=device)
        mask_for_desired_value_ = dpt.empty((1,), dtype=uint_type, device=device)

        def initialize_radix_mask():
            radix_position_[0] = n_bits_per_item - radix_bits
            mask_for_desired_value_[0] = 0
            return radix_position_, mask_for_desired_value_

    # `privatized_counts` storess the counts of occurences of the values at the current
    # radix position. Several copies of it are created, to avoid conflicts on atomic
    # during the count update step. The copies are reduced by the reduction kernel
    # defined thereafter.
    privatized_counts = dpt.empty(
        (n_counts_private_copies, n_rows, radix_size), dtype=np.int64, device=device
    )
    # Our sum reduction kernel can only reduce 1d or 2d matrices but will be used to
    # reduce the 3d matrix of private counts over axis 0. It is made possible by
    # adequatly reshaping the 3d matrix before and after the kernel call to a 2d matrix.
    n_rows_x_radix_size = n_rows * radix_size
    reduce_privatized_counts = make_sum_reduction_2d_kernel(
        shape=(n_counts_private_copies, n_rows_x_radix_size),
        device=device,
        dtype=np.int64,
        work_group_size="max",
        axis=0,
        sub_group_size=sub_group_size,
    )
    initialize_privatized_counts = make_fill_kernel(
        fill_value=0,
        shape=(n_counts_private_copies, n_rows, radix_size),
        work_group_size=work_group_size,
        dtype=dtype,
    )

    # Will store the number of occurences of the top-k threshold value in the data
    threshold_count_ = dpt.empty(
        (n_rows,), dtype=np.int64, device=check_radix_histogram_device
    )
    initialize_threshold_count_kernel = make_fill_kernel(
        fill_value=0, shape=(n_rows,), work_group_size=work_group_size, dtype=np.int64
    )

    def _get_topk_threshold(array_in):
        # Use variables that are local to the closure, so it can be manipulated more
        # easily in the main loop
        k_in_subset, n_active_rows, new_n_active_rows, threshold_count = (
            k_in_subset_,
            n_active_rows_,
            new_n_active_rows_,
            threshold_count_,
        )

        # Initialize all buffers
        initialize_k_in_subset_kernel(k_in_subset)
        n_active_rows[0] = n_active_rows_scalar = n_rows
        initialize_threshold_count_kernel(threshold_count)
        active_rows_mapping, new_active_rows_mapping = initialize_active_rows_mapping()
        desired_masked_value = initialize_desired_masked_value()
        radix_position, mask_for_desired_value = initialize_radix_mask()

        # Reinterpret input as uint so we can use bitwise compute
        array_in_uint = dpt.usm_ndarray(
            shape=(n_rows, n_cols),
            dtype=uint_type,
            buffer=array_in,
        )

        # The main loop: each iteration consists in sorting partially the data on the
        # values of a given radix of size `radix_size`, then discarding values that are
        # below the top k values.
        while True:
            initialize_privatized_counts(privatized_counts)

            create_radix_histogram_kernel(
                array_in_uint,
                n_active_rows_scalar,
                active_rows_mapping,
                mask_for_desired_value,
                desired_masked_value,
                radix_position,
                # OUT
                privatized_counts,
            )

            privatized_counts_ = dpt.reshape(
                privatized_counts, (n_counts_private_copies, n_rows_x_radix_size)
            )

            counts = dpt.reshape(
                reduce_privatized_counts(privatized_counts_), (n_rows, radix_size)
            )

            if check_radix_histogram_on_sequential_device:
                counts = counts.to_device(check_radix_histogram_device)
                desired_masked_value = desired_masked_value.to_device(
                    check_radix_histogram_device
                )
                radix_position = radix_position.to_device(check_radix_histogram_device)
                active_rows_mapping = active_rows_mapping.to_device(
                    check_radix_histogram_device
                )

            new_n_active_rows[0] = 0

            check_radix_histogram(
                counts,
                radix_position,
                n_active_rows,
                active_rows_mapping,
                # INOUT
                k_in_subset,
                desired_masked_value,
                # OUT
                threshold_count,
                new_n_active_rows,
                new_active_rows_mapping,
            )

            # If the top k values have been found in all rows, can exit early.
            if (n_active_rows_scalar := int(new_n_active_rows[0])) == 0:
                break

            # Else, update `radix_position` continue searching using the next radix
            if sequential_processing_on_different_device:
                mask_for_desired_value = mask_for_desired_value.to_device(
                    sequential_processing_device
                )

            if change_device_for_radix_update:
                radix_position = radix_position.to_device(sequential_processing_device)

            update_radix_position(radix_position, mask_for_desired_value)

            if sequential_processing_on_different_device:
                radix_position = radix_position.to_device(device)
                mask_for_desired_value = mask_for_desired_value.to_device(device)

            # Prepare next iteration
            n_active_rows, new_n_active_rows = new_n_active_rows, n_active_rows
            new_n_active_rows[:] = 0
            active_rows_mapping, new_active_rows_mapping = (
                new_active_rows_mapping,
                active_rows_mapping,
            )
            if check_radix_histogram_on_sequential_device:
                desired_masked_value = desired_masked_value.to_device(device)
                active_rows_mapping = active_rows_mapping.to_device(device)

        # Ensure data is located on the expected device before returning
        if check_radix_histogram_on_sequential_device:
            k_in_subset = k_in_subset.to_device(device)
            threshold_count = threshold_count.to_device(device)
            desired_masked_value = desired_masked_value.to_device(device)

        # reinterpret the threshold back to a dtype item
        threshold = dpt.usm_ndarray(
            shape=desired_masked_value.shape, dtype=dtype, buffer=desired_masked_value
        )

        return (
            threshold,
            k_in_subset,
            threshold_count,
        )

    return work_group_size, _get_topk_threshold


def _get_n_bits_per_item(dtype):
    """Returns number of bits in items with given dtype
    e.g, returns:
        - 32 for float32
        - 64 for float64
    """
    return np.dtype(dtype).itemsize * 8


@lru_cache
def _make_create_radix_histogram_kernel(
    n_rows,
    n_cols,
    work_group_size,
    sub_group_size,
    global_mem_cache_size,
    counts_private_copies_max_cache_occupancy,
    dtype,
    device,
):
    histogram_dtype = np.int64

    check_power_of_2(sub_group_size)

    (
        work_group_size,
        n_sub_groups_for_local_histograms,
        n_sub_groups_for_local_histograms_log2,
    ) = _enforce_matmul_like_work_group_geometry(
        work_group_size,
        sub_group_size,
        device,
        required_local_memory_per_item=np.dtype(histogram_dtype).itemsize,
    )

    n_local_histograms = work_group_size // sub_group_size
    work_group_shape = (sub_group_size, 1, n_local_histograms)

    # The size of the radix is chosen such as the size of intermediate objects that
    # build in shared memory amounts to one int64 item per work item.
    radix_size = sub_group_size
    radix_bits = int(math.log2(radix_size))
    local_counts_size = (n_local_histograms, radix_size)

    # Number of iterations when reducing the per-sub group histograms to per-work group
    # histogram in work groups
    n_sum_reduction_steps = int(math.log2(n_local_histograms))

    n_work_groups_per_row = math.ceil(n_cols / work_group_size)

    # Privatization parameters
    # TODO: is privatization really interesting here ?
    n_counts_items = radix_size
    n_counts_bytes = np.dtype(histogram_dtype).itemsize * n_counts_items * n_rows
    n_counts_private_copies = (
        global_mem_cache_size * counts_private_copies_max_cache_occupancy
    ) // n_counts_bytes

    # TODO: `nb_concurrent_sub_groups` is considered equal to
    # `device.max_compute_units`. We're not sure that this is the correct
    # read of the device specs. Confirm or fix once it's made clearer. Suggested reads
    # that highlight complexity of the execution model:
    # - https://github.com/IntelPython/dpctl/issues/1033
    # - https://stackoverflow.com/a/6490897
    n_counts_private_copies = int(
        min(n_work_groups_per_row, n_counts_private_copies, device.max_compute_units)
    )

    # Safety check for edge case where `n_counts_private_copies` equals 0 because
    # `n_counts_bytes` is too large
    n_counts_private_copies = max(n_counts_private_copies, 1)

    # TODO: this privatization parameter could be adjusted to actual `active_n_rows`
    # rather than using `n_rows`, this might improve privatization performance,
    # maybe with some performance hit for kernels that should adapt by variabilizing
    # the `n_rows` arguments rather than declaring it as a compile-time constant (or
    # suffer a much higher compile time for each possible value of `n_rows`).
    # Which is better ?

    lexicographical_mapping = _make_lexicographical_mapping_kernel_func(dtype)

    uint_type = uint_type_mapping[dtype]
    zero_as_uint_dtype = uint_type(0)
    one_as_uint_dtype = uint_type(1)
    two_as_a_long = np.int64(2)
    minus_one_idx = -np.int64(1)

    select_last_radix_bits_mask = (
        one_as_uint_dtype << np.uint32(radix_bits)
    ) - one_as_uint_dtype

    @dpex.kernel
    # fmt: off
    def create_radix_histogram(
        array_in_uint,                # IN READ-ONLY  (n_rows, n_items)
        active_rows_mapping,          # IN            (n_rows,)
        mask_for_desired_value,       # IN            (1,)
        desired_masked_value,         # IN            (n_rows,)
        radix_position,               # IN            (1,)
        privatized_counts             # OUT           (n_counts_private_copies, n_rows, radix_size)  # noqa
    ):
        # fmt: on
        """
        This kernel is the core of the radix top-k algorithm. This top-k implementation
        is well adapted to GPU architectures, but can suffer from bad performance
        depending on the distribution of the input data. It is planned to be
        supplemented with additional optimizations to ensure base performance for all
        input distributions.

        Radix top-k consists in computing the histogram of the number of occurences of
        all possible radixes (i.e subsequence of the sequence of bits) at a given
        radix position for all the items in a subset of `array_in_uint`. This
        histogram gives partial information on the ordering of the items. Computing
        sequentially the histogram for different radixes finally results in converging
        to the top-k greatest items.

        See e.g https://en.wikipedia.org/wiki/Radix_sort for more extensive description
        of radix-based sorting algorithms.

        At the given iteration, the exact subset of items that are considered is framed
        by bitwise conditions depending on `mask_for_desired_value` and
        `desired_mask_value`, that are defined such that only items that have not been
        discarded by the previous iterations match the condition. During the first
        iteration, the condition is true for all items.
        """
        # Row and column indices of the value in `array_in_uint` whose radix will be
        # computed by the current work item
        row_idx = active_rows_mapping[dpex.get_global_id(one_idx)]
        col_idx = dpex.get_global_id(zero_idx) + (
            sub_group_size * dpex.get_global_id(two_idx))

        # Index of the subgroup and position within this sub group. Incidentally, this
        # also indexes the location to which the radix value will be written in the
        # shared memory buffer.
        local_subgroup = dpex.get_local_id(two_idx)
        local_subgroup_work_id = dpex.get_local_id(zero_idx)

        # Like `col_idx`, but where the first value of `array_in_uint` covered by the
        # current work group is indexed with zero.
        local_item_idx = ((local_subgroup * sub_group_size) + local_subgroup_work_id)

        # The first `n_local_histograms` work items are special, they are used to
        # build the histogram of radix counts. The following variable tells wether the
        # current work item is one of those.
        is_histogram_item = local_item_idx < n_local_histograms

        # Initialize the shared memory in the work group
        # NB: for clarity in the code, two variables refer to the same buffer. The
        # buffer will indeed be used twice for different purposes each time.
        radix_values = local_counts = dpex.local.array(
            local_counts_size, dtype=histogram_dtype
        )

        # Initialize private memory
        # NB: private arrays are assumed to be created already initialized with
        # zero values
        private_counts = dpex.private.array(sub_group_size, dtype=histogram_dtype)

        # Compute the radix value of `array_in_uint` at location `(row_idx, col_idx)`,
        # and store it in `radix_values[local_subgroup, local_subgroup_work_id]`. If
        # the value is out of bounds, or if it doesn't match the mask, store `-1`
        # instead.
        compute_radixes(
            row_idx,
            col_idx,
            local_subgroup,
            local_subgroup_work_id,
            radix_position,
            mask_for_desired_value,
            desired_masked_value,
            array_in_uint,
            # OUT
            radix_values
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # The first `n_local_histograms` work items read `sub_group_size`
        # values each and compute the histogram of their occurences in private memory.
        # During this step, all other work items in the work group are idle.
        # NB: this is an order of magnitude faster than cooperatively summing the
        # counts in the histogram using `dpex.atomics.add` (probably because a high
        # occurence of conflicts)
        compute_private_histogram(
            col_idx,
            local_item_idx,
            local_subgroup,
            local_subgroup_work_id,
            is_histogram_item,
            radix_values,
            # OUT
            private_counts
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # The first `n_local_histograms` work items  write their private histogram
        # into the shared memory buffer, effectively sharing it with all other work
        # items. Each work item write to a different row in `local_counts`.
        share_private_histograms(
            local_subgroup,
            local_subgroup_work_id,
            is_histogram_item,
            private_counts,
            # INOUT
            local_counts
        )

        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # This is the merge step, where all shared histograms are summed
        # together into the first buffer local_counts[0], in a bracket manner.
        # NB: apparently cuda have much more powerful intrinsics to perform steps like
        # this, such as ballot voting ? are there `SYCL` or `numba_dpex` roadmaps to
        # enable the same intrinsics ?
        reduction_active_subgroups = n_local_histograms
        for _ in range(n_sum_reduction_steps):
            reduction_active_subgroups = reduction_active_subgroups // two_as_a_long
            partial_local_histograms_reduction(
                local_subgroup,
                local_subgroup_work_id,
                reduction_active_subgroups,
                # OUT
                local_counts
            )
            dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # The current histogram is local to the current work group. Summing right away
        # all histograms to a unique, global histogram in global memory might give poor
        # performance because it would require atomics that would be subject to a lot
        # of conflicts. Likewise, to circumvent this risk partial sums are written to
        # privatized buffers in global memory. The partial buffers will be reduced to a
        # single global histogram in a complementary kernel.
        merge_histogram_in_global_memory(
            row_idx,
            col_idx,
            local_subgroup,
            local_subgroup_work_id,
            local_counts,
            # OUT
            privatized_counts
        )

    # HACK 906: all instructions inbetween barriers must be defined in `dpex.func`
    # device functions.
    # See sklearn_numba_dpex.patches.tests.test_patches.test_need_to_workaround_numba_dpex_906  # noqa

    # HACK 906: start

    @dpex.func
    # fmt: off
    def compute_radixes(
        row_idx,                    # PARAM
        col_idx,                    # PARAM
        local_subgroup,             # PARAM
        local_subgroup_work_id,     # PARAM
        radix_position,             # IN            (1,)
        mask_for_desired_value,     # IN            (1,)
        desired_masked_value,       # IN            (n_rows,)
        array_in_uint,              # IN READ-ONLY  (n_rows, n_cols)
        radix_values,               # OUT           (n_local_histograms, radix_size)
    ):
        # fmt: on
        # If `col_idx` is outside the bounds of the input, ignore this location.
        is_in_bounds = col_idx < n_cols
        if is_in_bounds:
            item = array_in_uint[row_idx, col_idx]

            # Biject the item such as lexicographical order in the target space is
            # equivalent to the natural order in the the source space.
            item_lexicographically_mapped = lexicographical_mapping(item)

            radix_position_ = radix_position[zero_idx]
            mask_for_desired_value_ = mask_for_desired_value[zero_idx]

            desired_masked_value_ = desired_masked_value[row_idx]

        # The item is included to the radix histogram if the sequence of bits at the
        # positions defined by `mask_for_desired_value_` is equal to the sequence of
        # bits in `desired_masked_value_`
        includes_in_histogram = is_in_bounds and (
            (mask_for_desired_value_ == zero_as_uint_dtype)
            or (
                (item_lexicographically_mapped & mask_for_desired_value_)
                == desired_masked_value_
            )
        )

        if includes_in_histogram:
            # Extract the value encoded by the next radix_bits bits starting at
            # position radix_position_ (reading bits from left to right)
            # NB: resulting value is in interval [0, radix_size[
            value = histogram_dtype(
                (item_lexicographically_mapped >> radix_position_)
                & select_last_radix_bits_mask
            )

        else:
            # write `-1` if the index is out of bounds, or if the value doesn't match
            # the mask.
            value = histogram_dtype(minus_one_idx)

        radix_values[local_subgroup, local_subgroup_work_id] = value

    # The `compute_private_histogram` function is written differently depending on how
    # the number of histogram work items compare to the size of the sub groups.

    # First case: work items used for building the histogram span several sub groups
    if n_sub_groups_for_local_histograms_log2 >= 0:
        # NB: because of how parameters have been validated,
        # `n_sub_groups_for_local_histograms` is always divisible by
        # `sub_group_size` here.
        col_idx_increment_per_step = n_sub_groups_for_local_histograms * sub_group_size

        @dpex.func
        # fmt: off
        def compute_private_histogram(
            col_idx,                    # PARAM
            local_item_idx,             # PARAM  (UNUSED)
            local_subgroup,             # PARAM
            local_subgroup_work_id,     # PARAM
            is_histogram_item,          # PARAM
            radix_values,               # IN      (n_local_histograms, sub_group_size)
            private_counts,             # OUT     (sub_group_size,)
        ):
            # fmt: on
            if is_histogram_item:
                current_subgroup = local_subgroup
                current_col_idx = col_idx
                for _ in range(sub_group_size):
                    if current_col_idx < n_cols:
                        radix_value = radix_values[
                            current_subgroup, local_subgroup_work_id
                        ]
                        # `radix_value` can be equal to `-1` which means the value
                        # must be skipped
                        if radix_value >= zero_idx:
                            private_counts[radix_value] += count_one_as_a_long
                    current_subgroup += n_sub_groups_for_local_histograms
                    current_col_idx += col_idx_increment_per_step

    # Second case: histogram items span less than one sub group, and each work item
    # must span several values in each row of `radix_values`
    else:
        # NB: because of how parameters have been validated, `sub_group_size` is
        # always divisible by `n_local_histograms` here.
        n_iter_for_radixes = sub_group_size // n_local_histograms

        @dpex.func
        # fmt: off
        def compute_private_histogram(
            col_idx,                    # PARAM
            local_item_idx,             # PARAM
            local_subgroup,             # PARAM   (UNUSED)
            local_subgroup_work_id,     # PARAM   (UNUSED)
            is_histogram_item,          # PARAM
            radix_values,               # IN      (n_local_histograms, sub_group_size)
            private_counts,             # OUT     (sub_group_size,)
        ):
            # fmt: on
            if is_histogram_item:
                starting_col_idx = col_idx
                for histogram_idx in range(n_local_histograms):
                    current_col_idx = starting_col_idx
                    radix_value_idx = local_item_idx
                    for _ in range(n_iter_for_radixes):
                        if current_col_idx < n_cols:
                            radix_value = radix_values[histogram_idx, radix_value_idx]
                            if radix_value >= zero_idx:
                                private_counts[radix_value] += count_one_as_a_long
                            radix_value_idx += n_local_histograms
                            current_col_idx += n_local_histograms
                    starting_col_idx += sub_group_size

    @dpex.func
    # fmt: off
    def share_private_histograms(
        local_subgroup,             # PARAM
        local_subgroup_work_id,     # PARAM
        is_histogram_item,          # PARAM
        private_counts,             # IN     (sub_group_size,)
        local_counts,               # OUT    (n_local_histograms, radix_size)
    ):
        # fmt: on
        if is_histogram_item:
            col_idx = local_subgroup_work_id
            starting_row_idx = local_subgroup * sub_group_size

            # The following indexing enable nicer memory RW patterns since it ensures
            # that contiguous work items in a sub group access contiguous values.
            for i in range(sub_group_size):
                local_counts[
                    (starting_row_idx + i) % n_local_histograms, col_idx
                ] = private_counts[col_idx]
                col_idx = (col_idx + one_idx) % sub_group_size

    @dpex.func
    # fmt: off
    def partial_local_histograms_reduction(
        local_subgroup,                 # PARAM
        local_subgroup_work_id,         # PARAM
        reduction_active_subgroups,     # PARAM
        local_counts                    # INOUT   (n_local_histograms, sub_group_size)
    ):
        # fmt: on
        if local_subgroup < reduction_active_subgroups:
            local_counts[local_subgroup, local_subgroup_work_id] += local_counts[
                local_subgroup + reduction_active_subgroups, local_subgroup_work_id
            ]

    @dpex.func
    # fmt: off
    def merge_histogram_in_global_memory(
        row_idx,                    # PARAM
        col_idx,                    # PARAM
        local_subgroup,             # PARAM
        local_subgroup_work_id,     # PARAM
        local_counts,               # IN    (n_local_histograms, sub_group_size)
        privatized_counts,          # OUT   (n_counts_private_copies, n_rows, radix_size)  # noqa
    ):
        # fmt: on
        # Each work group is assigned an array of centroids in a round robin manner
        privatization_idx = (col_idx // work_group_size) % n_counts_private_copies

        if local_subgroup == zero_idx:
            dpex.atomic.add(
                privatized_counts,
                (privatization_idx, row_idx, local_subgroup_work_id),
                local_counts[zero_idx, local_subgroup_work_id],
            )

    # HACK 906: end

    # Adjust group size dynamically depending on the number of rows that are active for
    # the ongoing iteration
    def _create_radix_histogram(
        array_in_uint,
        n_active_rows,
        active_rows_mapping,
        mask_for_desired_value,
        desired_masked_value,
        radix_position,
        privatized_counts,
    ):
        global_shape = (
            sub_group_size,
            n_active_rows,
            n_local_histograms * n_work_groups_per_row,
        )
        create_radix_histogram[global_shape, work_group_shape](
            array_in_uint,
            active_rows_mapping,
            mask_for_desired_value,
            desired_masked_value,
            radix_position,
            privatized_counts,
        )

    return (
        radix_size,
        radix_bits,
        n_counts_private_copies,
        _create_radix_histogram,
    )


@lru_cache
def _make_check_radix_histogram_kernel(radix_size, dtype, work_group_size):
    radix_bits = int(math.log2(radix_size))
    lexicographical_unmapping = _make_lexicographical_unmapping_kernel_func(dtype)
    uint_type = uint_type_mapping[dtype]
    zero_as_uint_dtype = uint_type(0)

    @dpex.kernel
    # fmt: off
    def check_radix_histogram(
        counts,                        # IN           (n_rows, radix_size,)
        radix_position,                # IN           (1,)
        n_active_rows,                 # IN           (1,)
        active_rows_mapping,           # IN           (n_rows,)
        k_in_subset,                   # INOUT        (n_rows,)
        desired_masked_value,          # INOUT        (n_rows,)
        threshold_count,               # OUT          (n_rows,)
        new_n_active_rows,             # OUT          (1,)
        new_active_rows_mapping,       # OUT          (n_rows,)
    ):
        # fmt: on
        work_item_idx = dpex.get_global_id(zero_idx)
        if work_item_idx >= n_active_rows[zero_idx]:
            return

        row_idx = active_rows_mapping[work_item_idx]

        k_in_subset_ = k_in_subset[row_idx]
        radix_position_ = radix_position[zero_idx]
        desired_masked_value_ = desired_masked_value[row_idx]

        # Read the histogram starting from the bucket corresponding to the highest
        # value for the current radix position, and in decreasing order.
        current_count_idx = radix_size - count_one_as_a_long
        # NB: `numba_dpex` seem to produce inefficient (branching) code for `break`,
        # use `if/else` instead
        desired_mask_value_search = True
        for _ in range(radix_size):
            if desired_mask_value_search:
                count = counts[row_idx, current_count_idx]
                if count >= k_in_subset_:
                    # The bucket of items matching the value for the current radix
                    # position (equal to `current_count_idx`) contain the k-th highest
                    # value. New mask parameters are chosen such that the next
                    # iteration will scan for the k-th value in this bucket only.
                    desired_masked_value_ = desired_masked_value_ | (
                        uint_type(current_count_idx) << radix_position_
                    )
                    desired_mask_value_search = False

                else:
                    # The k-th greatest value is not in the current bucket of size
                    # `count`. The k-th greatest value is also the (k-count)-th greatest
                    # value among items whose value for the current radix position is
                    # strictly smaller than the value of the current bucket of items
                    # (equal to `current_count_idx`).
                    k_in_subset_ -= count
                    current_count_idx -= count_one_as_a_long

        # The top-k search has converged either if the last radix position that was
        # scanned is 0, or if `k` is 1 and a bucket of size 1 has also been found. In
        # any other case, creation of a new histogram will be computed for items
        # in the bucket with index `current_count_idx` only (and so on).
        terminate_row = (radix_position_ == zero_as_uint_dtype) or (
            (k_in_subset_ == count_one_as_a_long) and (count == count_one_as_a_long)
        )

        k_in_subset[row_idx] = k_in_subset_

        if terminate_row:
            # At this point:
            # - any value in the input data equal to `desired_masked_value_` is the
            # k-th greatest value
            # - the number of values equal to `desired_masked_value_` among the top-k
            # values is exactly `k_in_subset_`.
            # - the number of values equal to `desired_masked_value_` in the data is
            # exactly `count`
            threshold_count[row_idx] = count
            desired_masked_value_ = lexicographical_unmapping(desired_masked_value_)

        else:
            new_active_row_idx = dpex.atomic.add(
                new_n_active_rows, zero_idx, count_one_as_an_int
            )
            new_active_rows_mapping[new_active_row_idx] = row_idx

        desired_masked_value[row_idx] = desired_masked_value_

    @dpex.kernel
    def update_radix_position(radix_position, mask_for_desired_value):
        # The current partial analysis with the current radixes seen was not enough
        # to find the k-th element. Let's inspect the next `radix_bits`.
        radix_position_ = radix_position[zero_idx]
        new_radix_position = radix_position_ - radix_bits
        if new_radix_position < zero_as_uint_dtype:
            new_radix_position = zero_as_uint_dtype
        radix_position[zero_idx] = new_radix_position
        mask_for_desired_value[zero_idx] |= (radix_size - 1) << radix_position_

    # adjust group size dynamically depending on the number of rows that require the
    # next iteration
    def _check_radix_histogram(
        counts,
        active_rows_mapping,
        n_active_rows,
        k_in_subset,
        radix_position,
        desired_masked_value,
        threshold_count,
        new_active_rows_mapping,
        new_n_active_rows,
    ):
        n_active_rows_ = int(n_active_rows[0])
        global_size = math.ceil(n_active_rows_ / work_group_size) * work_group_size

        check_radix_histogram[global_size, work_group_size](
            counts,
            active_rows_mapping,
            n_active_rows,
            k_in_subset,
            radix_position,
            desired_masked_value,
            threshold_count,
            new_active_rows_mapping,
            new_n_active_rows,
        )

    return update_radix_position[1, 1], _check_radix_histogram


@lru_cache
def _make_gather_topk_kernel(
    n_rows,
    n_cols,
    k,
    work_group_size,
):
    """The gather_topk kernel is the last step. By now the k-th greatest values and
    its number of occurences among the top-k values in the search space have been
    identified in reach row. The top-k values that are equal or greater than k,
    including the `n_threshold_occurrences` occurrences equal to the k-th greatest
    value, are written into the result array.
    """
    n_work_groups_per_row = math.ceil(n_cols / work_group_size)
    work_group_shape = (1, work_group_size)
    global_shape = (n_rows, n_work_groups_per_row * work_group_size)

    @dpex.kernel
    # fmt: off
    def gather_topk(
        array_in,                           # IN READONLY    (n_rows, n_cols)
        threshold,                          # IN             (n_rows,)
        n_threshold_occurences_in_topk,     # IN             (n_rows,)
        n_threshold_occurences_in_data,     # IN             (n_rows,)
        result_col_idx,                         # BUFFER         (n_rows,)
        result,                             # OUT            (n_rows, k)
    ):
        # fmt: on
        row_idx = dpex.get_global_id(zero_idx)
        col_idx = dpex.get_global_id(one_idx)

        n_threshold_occurences_in_topk_ = n_threshold_occurences_in_topk[row_idx]

        threshold_ = threshold[row_idx]

        # Different branches depending on the value of `n_threshold_occurences`, since
        # some optimizations are possible depending on the value.
        if n_threshold_occurences_in_data[row_idx] == n_threshold_occurences_in_topk_:
            gather_topk_include_all_threshold_occurences(
                row_idx,
                col_idx,
                threshold_,
                array_in,
                result_col_idx,
                # OUT
                result
            )
        else:
            gather_topk_generic(
                row_idx,
                col_idx,
                threshold_,
                n_threshold_occurences_in_topk_,
                array_in,
                result_col_idx,
                # OUT
                result,
            )

    # Both the number of occurences of the threshold value in the data and the number
    # in the top-k values are known. When those two numbers are equal, the kernel can
    # be written more efficient and much simpler, and the condition is not unusual.
    # Let's write a separate kernel for this special case.
    @dpex.func
    # fmt: off
    def gather_topk_include_all_threshold_occurences(
        row_idx,                           # PARAM
        col_idx,                           # PARAM
        threshold,                         # PARAM
        array_in,                          # IN READ-ONLY (n_rows, n_cols)
        result_col_idx,                    # BUFFER       (n_rows,)
        result,                            # OUT          (n_rows, k)
    ):
        # fmt: on
        if col_idx >= n_cols:
            return

        if result_col_idx[row_idx] >= k:
            return

        item = array_in[row_idx, col_idx]

        if item >= threshold:
            result_col_idx_ = dpex.atomic.add(
                result_col_idx, row_idx, count_one_as_an_int)
            result[row_idx, result_col_idx_] = item

    @dpex.func
    # fmt: off
    def gather_topk_generic(
        row_idx,                       # PARAM
        col_idx,                       # PARAM
        threshold,                     # PARAM
        n_threshold_occurences,        # PARAM
        array_in,                      # IN READ-ONLY (n_rows, n_cols,)
        result_col_idx,                # BUFFER       (n_rows,)
        result,                        # OUT          (n_rows, k)
    ):
        # fmt: on
        if col_idx >= n_cols:
            return

        # The `n_threshold_occurences` first work items write the value of the
        # threshold at the end of the result array.
        if col_idx < n_threshold_occurences:
            result[row_idx, k-col_idx-one_idx] = threshold

        # Then write the remaining `k - n_threshold_occurences` that are strictly
        # greater than the threshold.
        if result_col_idx[row_idx] >= (k - n_threshold_occurences):
            return

        item = array_in[row_idx, col_idx]

        if item <= threshold:
            return

        result_col_idx_ = dpex.atomic.add(result_col_idx, row_idx, count_one_as_an_int)
        result[row_idx, result_col_idx_] = item

    return gather_topk[global_shape, work_group_shape]


@lru_cache
def _make_gather_topk_idx_kernel(
    n_rows,
    n_cols,
    k,
    work_group_size,
):
    """Same than gather_topk kernel but return top-k indices rather than top-k values"""
    n_work_groups_per_row = math.ceil(n_cols / work_group_size)
    work_group_shape = (1, work_group_size)
    global_shape = (n_rows, n_work_groups_per_row * work_group_size)

    @dpex.kernel
    # fmt: off
    def gather_topk_idx(
        array_in,                           # IN READONLY    (n_rows, n_cols)
        threshold,                          # IN             (n_rows,)
        n_threshold_occurences_in_topk,     # IN             (n_rows,)
        n_threshold_occurences_in_data,     # IN             (n_rows,)
        result_col_idx,                     # BUFFER         (n_rows,)
        result,                             # OUT            (n_rows, k)
    ):
        # fmt: on
        row_idx = dpex.get_global_id(zero_idx)
        col_idx = dpex.get_global_id(one_idx)

        n_threshold_occurences_in_topk_ = n_threshold_occurences_in_topk[row_idx]

        if n_threshold_occurences_in_data[row_idx] == n_threshold_occurences_in_topk_:
            gather_topk_idx_include_all_threshold_occurences(
                row_idx,
                col_idx,
                threshold[row_idx],
                array_in,
                result_col_idx,
                # OUT
                result
            )
        else:
            gather_topk_idx_generic(
                row_idx,
                col_idx,
                threshold[row_idx],
                n_threshold_occurences_in_topk,
                array_in,
                result_col_idx,
                # OUT
                result,
            )

    @dpex.func
    # fmt: off
    def gather_topk_idx_include_all_threshold_occurences(
        row_idx,                           # PARAM
        col_idx,                           # PARAM
        threshold,                         # PARAM
        array_in,                          # IN READ-ONLY (n_rows, n_cols)
        result_col_idx,                    # BUFFER       (n_rows,)
        result,                            # OUT          (n_rows, k)
    ):
        # fmt: on
        if col_idx >= n_cols:
            return

        if result_col_idx[row_idx] >= k:
            return

        item = array_in[row_idx, col_idx]

        if item >= threshold:
            result_col_idx_ = dpex.atomic.add(
                result_col_idx, row_idx, count_one_as_an_int)
            result[row_idx, result_col_idx_] = col_idx

    @dpex.func
    # fmt: off
    def gather_topk_idx_generic(
        row_idx,                       # PARAM
        col_idx,                       # PARAM
        threshold,                     # PARAM
        n_threshold_occurences,        # PARAM
        array_in,                      # IN READ-ONLY (n_rows, n_cols)
        result_col_idx,                # BUFFER       (n_rows,)
        result,                        # OUT          (n_rows, k)
    ):
        # fmt: on
        if col_idx >= n_cols:
            return

        if result_col_idx[row_idx] >= k:
            return

        item = array_in[row_idx, col_idx]

        if item < threshold:
            return

        if item > threshold:
            result_col_idx_ = dpex.atomic.add(
                result_col_idx, row_idx, count_one_as_an_int
            )
            result[row_idx, result_col_idx_] = col_idx
            return

        if n_threshold_occurences[row_idx] <= zero_idx:
            return

        remaining_n_threshold_occurences = dpex.atomic.sub(
            n_threshold_occurences, row_idx, count_one_as_an_int)

        if remaining_n_threshold_occurences > zero_idx:
            result_col_idx_ = dpex.atomic.add(
                result_col_idx, row_idx, count_one_as_an_int
            )
            result[row_idx, result_col_idx_] = col_idx

    return gather_topk_idx[global_shape, work_group_shape]
