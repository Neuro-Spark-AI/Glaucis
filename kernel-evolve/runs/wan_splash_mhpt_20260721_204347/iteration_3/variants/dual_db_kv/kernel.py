# Source: maxdiffusion splash_attention_kernel.flash_attention_kernel_mhpt (production
# ring MHA fwd, hpt=5). Self-contained single-card mirror of that compute
# (== custom_splash_attention._flash_attention_kernel_mhpt): online-softmax flash,
# bf16 QKᵀ + f32 PV, base-2 exp, V1 VPU register tiling. Ring/mask/prefetch plumbing
# stripped (full attention, single card) so it runs in pallas-evolve's evaluate.py.
"""Wan2.2 DiT splash MHA (mhpt) — template for evolution.

The EVOLVE-BLOCK holds the production flash-attention compute kernel + its Pallas
launcher. optimized_compute pre-scales Q by (1/sqrt(d))*log2(e) and casts to bf16
(so base-2 exp reproduces natural softmax with 1/sqrt(d) scaling), runs the kernel,
and transposes the [H, head_dim_v, S] kernel output back to [H, S, D].

Frozen (outside EVOLVE-BLOCK): imports, constants, _make_test_data. The optimizer
may change the kernel body, the launcher, block sizes (bq/bkv/bkv_compute/
bkv_compute_in/heads_per_tile), scratch, and scheduling.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # contract last dim of both (k·qᵀ)
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
LOG2E = math.log2(math.e)


def _make_test_data(H=5, S=8192, D=128):
  """Deterministic f32 q,k,v of shape [H, S, D]. Identical to the reference's copy."""
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(kq, (H, S, D), dtype=jnp.float32) * 0.5
  k = jax.random.normal(kk, (H, S, D), dtype=jnp.float32) * 0.5
  v = jax.random.normal(kv, (H, S, D), dtype=jnp.float32) * 0.5
  return q, k, v


# EVOLVE-BLOCK-START
# Production block config (Wan2.2 i2v 27b flash_block_sizes), hpt=5.
# dual_db_kv: keeps the dual_mxu compute EXACTLY (head-batched 3D QKᵀ / PV dots,
# online softmax, bf16 QKᵀ + f32 PV, base-2 exp). Only the K/V staging changes:
# K and V are handed to the kernel as HBM (ANY) refs and streamed block-by-block
# into a 2-slot VMEM scratch with pltpu.make_async_copy. Block j's compute runs on
# slot (j%2) while block j+1's HBM->VMEM DMA into slot ((j+1)%2) is already in
# flight, so the K/V transfer overlaps the MXU/VPU compute. This is a manual
# double buffer (NOT pltpu.emit_pipeline, which regressed correctness in R2, F005).
BQ = 1024
BKV = 2048
BKV_COMPUTE = 512
BKV_COMPUTE_IN = 256
HEADS_PER_TILE = 5
VMEM_LIMIT_BYTES = 100663296  # 96 MiB
USE_BASE2_EXP = True

# Batched (head-axis) dot dimension numbers. dual_mxu direction: instead of a
# serial per-head `for h in range` of separate 2D dots (which kept only mxu0
# busy, dual_ratio 0.0), fold the head-tile into one 3D dot_general with head as
# a batch dim. The compiler can then spread the batched matmul across mxu0+mxu1.
# QKᵀ: lhs k[hpt,kv,d], rhs q[hpt,q,d]; contract d (dim 2 of both), batch head
# (dim 0). -> [hpt, kv, q], per-head identical to k·qᵀ.
QK_BATCH_DIM_NUMBERS = (((2,), (2,)), ((0,), (0,)))
# PV: lhs v[hpt,kv,d], rhs s[hpt,kv,q]; contract kv (dim 1 of both), batch head
# (dim 0). -> [hpt, d, q], per-head identical to vᵀ·s.
SV_BATCH_DIM_NUMBERS = (((1,), (1,)), ((0,), (0,)))


def _flash_attention_kernel_mhpt(
    q_ref, k_hbm_ref, v_hbm_ref,
    o_ref,
    m_scratch_ref, l_scratch_ref, o_scratch_ref,
    k_vmem_ref, v_vmem_ref, k_sem_ref, v_sem_ref,
    *,
    mask_value, grid_width, bq, bkv, bkv_compute, bkv_compute_in,
    head_dim_v, q_seq_len, kv_seq_len, heads_per_tile, use_base2_exp=True,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {NUM_SUBLANES}")

  h_pid = pl.program_id(0)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  # Online-softmax accumulators live in VMEM scratch and are accumulated across
  # the whole KV axis inside this single kernel invocation (the KV-block loop is
  # now internal to the kernel, so init runs once here, normalize once at the end).
  o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
  m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
  l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def _one_chunk(k_blk_ref, v_blk_ref, slice_k):
    # k_blk_ref / v_blk_ref are the current KV block's VMEM slot (already streamed
    # in by the async copy waited on below), shape [hpt, bkv, d]. slice_k selects a
    # bkv_compute sub-chunk. Compute is byte-for-byte the dual_mxu body: one batched
    # 3D dot_general per QKᵀ / PV with head as the batch dim.
    q_all = q_ref[...]                       # [hpt, bq, d]         (bf16)
    k_chunk = k_blk_ref[:, slice_k, :]       # [hpt, bkv_compute, d]
    v_chunk = v_blk_ref[:, slice_k, :]       # [hpt, bkv_compute, d]

    # Batched QKᵀ across the head-tile -> [hpt, bkv_compute, bq].
    qk = lax.dot_general(k_chunk, q_all, QK_BATCH_DIM_NUMBERS,
                         preferred_element_type=float32)

    m_prev = m_scratch_ref[...]              # [hpt, NUM_SUBLANES, bq]
    l_prev = l_scratch_ref[...]              # [hpt, NUM_SUBLANES, bq]
    o_prev = o_scratch_ref[...]              # [hpt, head_dim_v, bq]

    # --- V1 VPU register tiling (vectorized over the head axis) ---
    step = bkv_compute_in
    for i in range(0, qk.shape[1], step):
      qk_slice = qk[:, i:i + step, :]                       # [hpt, step, bq]
      m_curr = qk_slice.max(axis=1)[:, None, :]             # [hpt, 1, bq]
      m_next = jnp.maximum(m_prev, m_curr)                  # [hpt, NUM_SUBLANES, bq]
      s_curr = exp(qk_slice - m_next[:, 0:1, :])            # [hpt, step, bq]
      l_curr = s_curr.sum(axis=1, keepdims=True)            # [hpt, 1, bq]
      alpha = exp(m_prev - m_next)                          # [hpt, NUM_SUBLANES, bq]
      l_next = l_curr + alpha * l_prev
      # Batched PV across the head-tile -> [hpt, head_dim_v, bq].
      o_curr = lax.dot_general(
          v_chunk[:, i:i + step, :], s_curr.astype(q_ref.dtype),
          SV_BATCH_DIM_NUMBERS, preferred_element_type=float32)
      alpha_o = alpha[:, 0:1, :]                            # [hpt, 1, bq]
      o_prev = alpha_o * o_prev + o_curr
      m_prev, l_prev = m_next, l_next
    # --- end V1 tiling ---

    m_scratch_ref[...] = m_prev
    l_scratch_ref[...] = l_prev
    o_scratch_ref[...] = o_prev

  # Manual K/V double buffer. kv_seq_len is a multiple of bkv (asserted in the
  # launcher) and bkv a multiple of bkv_compute, so every block is full: the loop
  # is python-unrolled over grid_width and every slot / block index is static.
  assert kv_seq_len % bkv == 0
  assert bkv % bkv_compute == 0

  def _kv_copies(j, slot):
    src_k = k_hbm_ref.at[pl.ds(h_pid * heads_per_tile, heads_per_tile),
                         pl.ds(j * bkv, bkv), :]
    src_v = v_hbm_ref.at[pl.ds(h_pid * heads_per_tile, heads_per_tile),
                         pl.ds(j * bkv, bkv), :]
    copy_k = pltpu.make_async_copy(src_k, k_vmem_ref.at[slot], k_sem_ref.at[slot])
    copy_v = pltpu.make_async_copy(src_v, v_vmem_ref.at[slot], v_sem_ref.at[slot])
    return copy_k, copy_v

  # Prologue: kick off block 0's DMA into slot 0.
  ck0, cv0 = _kv_copies(0, 0)
  ck0.start()
  cv0.start()

  for j in range(grid_width):
    cur = j % 2
    # Prefetch block j+1 into the other slot before touching block j, so its DMA
    # overlaps the compute below.
    if j + 1 < grid_width:
      nk, nv = _kv_copies(j + 1, (j + 1) % 2)
      nk.start()
      nv.start()
    # Block j must be resident before we read it.
    ck, cv = _kv_copies(j, cur)
    ck.wait()
    cv.wait()

    k_cur = k_vmem_ref.at[cur]
    v_cur = v_vmem_ref.at[cur]
    for c in range(bkv // bkv_compute):
      _one_chunk(k_cur, v_cur, pl.ds(c * bkv_compute, bkv_compute))

  # Normalize by the softmax denominator and write the output block.
  for h_local in range(heads_per_tile):
    l = l_scratch_ref[h_local]
    l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
    o_ref[h_local] = (o_scratch_ref[h_local] * l_inv).astype(o_ref.dtype)


def _splash_mhpt_forward(q, k, v, *, bq, bkv, bkv_compute, bkv_compute_in,
                         heads_per_tile, use_base2_exp, vmem_limit_bytes):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  kv_seq_len = k.shape[1]
  hpt = heads_per_tile
  assert num_q_heads % hpt == 0
  assert kv_seq_len % bkv == 0  # manual double buffer streams full bkv blocks

  def q_index_map(h, i, *_):
    return (h, i, 0)

  def out_index_map(h, i, *_):
    return (h, 0, i)

  # Q is a normal VMEM BlockSpec (resident, reused across all KV blocks). K and V
  # are handed in as ANY (HBM) refs so the kernel can double-buffer their
  # HBM->VMEM streaming itself via make_async_copy.
  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim_qk), q_index_map),
      pl.BlockSpec(memory_space=pltpu.ANY),
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_shape = jax.ShapeDtypeStruct(
      (num_q_heads, head_dim_v, q_seq_len), q.dtype)
  out_specs = pl.BlockSpec((hpt, head_dim_v, bq), out_index_map)

  scratch_shapes = [
      pltpu.VMEM((hpt, NUM_SUBLANES, bq), jnp.float32),   # m
      pltpu.VMEM((hpt, NUM_SUBLANES, bq), jnp.float32),   # l
      pltpu.VMEM((hpt, head_dim_v, bq), jnp.float32),     # o
      pltpu.VMEM((2, hpt, bkv, head_dim_qk), k.dtype),    # K double buffer
      pltpu.VMEM((2, hpt, bkv, head_dim_v), v.dtype),     # V double buffer
      pltpu.SemaphoreType.DMA((2,)),                      # K DMA semaphores
      pltpu.SemaphoreType.DMA((2,)),                      # V DMA semaphores
  ]

  grid_width = (kv_seq_len + bkv - 1) // bkv
  grid_height = (q_seq_len + bq - 1) // bq
  grid = (num_q_heads // hpt, grid_height)

  out = pl.pallas_call(
      functools.partial(
          _flash_attention_kernel_mhpt,
          mask_value=DEFAULT_MASK_VALUE, grid_width=grid_width, bq=bq, bkv=bkv,
          bkv_compute=bkv_compute, bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v, q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
          heads_per_tile=hpt, use_base2_exp=use_base2_exp),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0, in_specs=in_specs, out_specs=out_specs,
          grid=grid, scratch_shapes=scratch_shapes),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes),
      out_shape=out_shape,
  )(q, k, v)
  return out  # [num_q_heads, head_dim_v, q_seq_len]


def optimized_compute(H=5, S=8192, D=128):
  q, k, v = _make_test_data(H, S, D)
  scale = 1.0 / (D ** 0.5)
  # base-2 exp in the kernel: pre-scale q by scale*log2(e) so exp2 reproduces
  # natural softmax with 1/sqrt(d) scaling.
  q_scaled = (q * (scale * LOG2E)).astype(jnp.bfloat16) if USE_BASE2_EXP else (q * scale).astype(jnp.bfloat16)
  k_bf = k.astype(jnp.bfloat16)
  v_bf = v.astype(jnp.bfloat16)

  out = _splash_mhpt_forward(
      q_scaled, k_bf, v_bf,
      bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, bkv_compute_in=BKV_COMPUTE_IN,
      heads_per_tile=HEADS_PER_TILE, use_base2_exp=USE_BASE2_EXP,
      vmem_limit_bytes=VMEM_LIMIT_BYTES)  # [H, D, S]
  return jnp.swapaxes(out, 1, 2)  # [H, S, D]
# EVOLVE-BLOCK-END
