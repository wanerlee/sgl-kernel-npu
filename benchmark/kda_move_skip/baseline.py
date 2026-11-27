"""Unmodified KDA mover from upstream 458e72caf6cdfb5b9c06711e6f84d134d15e3f97."""

import triton
import triton.language as tl


@triton.jit
def move_cache_dynamic_last_kernel_h_block_kda(
    dst_cache_ptr,
    src_cache_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    dst_layer_stride,
    dst_size_stride,
    dst_h_stride,
    dst_v_stride,
    dst_k_stride,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """KDA-specific mover that respects non-contiguous destination strides.

    On NPU the temporal state (dst) is transposed (-1, -2), so its
    (H, V, K) layout differs from the source. This kernel indexes the
    destination through its real per-element strides and splits dim_v
    into BLOCK_V-sized chunks to keep the on-chip tile within budget.
    """
    valid_id = tl.program_id(0)

    dst_idx_val = tl.load(dst_indices_ptr + valid_id)
    src_idx_val = tl.load(src_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if last_step_val < 0:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_K)

    for l in range(num_layers):
        src_base_addr = (
            src_cache_ptr
            + tl.cast(l, tl.int64) * layer_stride
            + tl.cast(src_idx_val, tl.int64) * size_stride
        )
        dst_base_addr = (
            dst_cache_ptr
            + tl.cast(l, tl.int64) * dst_layer_stride
            + tl.cast(dst_idx_val, tl.int64) * dst_size_stride
        )
        src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride

        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            h_real = h_start + h_offsets
            h_mask = h_real < h_dim
            k_mask = k_offsets < dim_k

            for v_start in range(0, dim_v, BLOCK_V):
                v_offsets = v_start + tl.arange(0, BLOCK_V)
                v_mask = v_offsets < dim_v

                mask = (
                    h_mask[:, None, None]
                    & v_mask[None, :, None]
                    & k_mask[None, None, :]
                )

                # src is contiguous in (H, V, K) -> flat offset.
                src_linear_offset = (
                    h_real[:, None, None] * dim_v * dim_k
                    + v_offsets[None, :, None] * dim_k
                    + k_offsets[None, None, :]
                )
                # dst uses its real per-element strides.
                dst_linear_offset = (
                    h_real[:, None, None] * dst_h_stride
                    + v_offsets[None, :, None] * dst_v_stride
                    + k_offsets[None, None, :] * dst_k_stride
                )

                src_block = tl.load(src_addr + src_linear_offset, mask=mask, other=0)
                tl.store(dst_base_addr + dst_linear_offset, src_block, mask=mask)


def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    """Move intermediate cache to SSM states (KDA-transposed-dst aware).

    Compared with ``move_intermediate_cache``, this variant preserves the
    destination (H, V, K) per-element strides and tiles dim_v in 64-wide
    chunks. Required when the SSM temporal state is not contiguous in the
    (V, K) plane -- e.g. Kimi-K3 on Ascend where the cache is transposed
    (-1, -2).
    """
    L, S, D, H, V, K = intermediate_state_cache.shape

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = (
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
    )
    dst_strides = ssm_states.stride()
    dst_layer_stride, dst_size_stride = int(dst_strides[0]), int(dst_strides[1])
    dst_h_stride, dst_v_stride, dst_k_stride = (
        int(dst_strides[2]),
        int(dst_strides[3]),
        int(dst_strides[4]),
    )
    assert len(dst_indices_tensor) == len(
        last_steps_tensor
    ), "Destination indices lengths must match"
    assert len(src_indices_tensor) == len(
        last_steps_tensor
    ), "Source indices lengths must match"

    if len(dst_indices_tensor) == 0:
        return ssm_states

    grid = (len(dst_indices_tensor),)

    move_cache_dynamic_last_kernel_h_block_kda[grid](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        dst_indices_ptr=dst_indices_tensor,
        src_indices_ptr=src_indices_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        dst_layer_stride=dst_layer_stride,
        dst_size_stride=dst_size_stride,
        dst_h_stride=dst_h_stride,
        dst_v_stride=dst_v_stride,
        dst_k_stride=dst_k_stride,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,
        BLOCK_V=64,
        BLOCK_K=triton.next_power_of_2(K),
    )

    return ssm_states
