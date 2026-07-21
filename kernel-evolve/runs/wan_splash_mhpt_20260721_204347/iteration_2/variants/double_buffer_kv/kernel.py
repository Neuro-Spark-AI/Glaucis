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
# Block sizes are frozen at production values; this variant only changes the
# DMA/compute scheduling. The KV-block loop over grid_width is moved inside the
# kernel and driven by pltpu.emit_pipeline, which double-buffers each K/V block's
# HBM->VMEM DMA against the MXU compute of the previous block.
BQ = 1024
BKV = 2048
BKV_COMPUTE = 512
BKV_COMPUTE_IN = 256
HEADS_PER_TILE = 5
VMEM_LIMIT_BYTES = 100663296  # 96 MiB
USE_BASE2_EXP = True


def _flash_attention_kernel_mhpt(
    q_ref, k_hbm_ref, v_hbm_ref,
    o_ref,
    m_scratch_ref, l_scratch_ref, o_scratch_ref,
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

  # Initialize the online-softmax accumulators once per (head-tile, q-block),
  # before streaming the KV blocks. The pipeline below accumulates into these
  # scratch buffers across all KV blocks, then end_normalize() divides out l.
  o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
  m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
  l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  def _one_chunk(k_ref, v_ref, slice_k):
    # k_ref / v_ref are the VMEM blocks for the current KV block, already
    # streamed in by emit_pipeline. slice_k selects a bkv_compute sub-chunk.
    for h_local in range(heads_per_tile):
      m_prev = m_scratch_ref[h_local]
      l_prev = l_scratch_ref[h_local]
      q = q_ref[h_local]
      o_prev = o_scratch_ref[h_local]

      k_chunk = k_ref[h_local, slice_k, :]
      qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
      v_chunk = v_ref[h_local, slice_k, :]

      # --- V1 VPU register tiling ---
      step = bkv_compute_in
      for i in range(0, qk.shape[0], step):
        qk_slice = qk[i:i + step]
        m_curr = qk_slice.max(axis=0)[None, :]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = exp(qk_slice - m_next[0:1])
        l_curr = s_curr.sum(axis=0, keepdims=True)
        alpha = exp(m_prev - m_next)
        l_next = l_curr + alpha * l_prev
        sv_dims = (((0,), (0,)), ((), ()))
        o_curr = lax.dot_general(
            v_chunk[i:i + step], s_curr.astype(q_ref.dtype), sv_dims,
            preferred_element_type=float32)
        alpha_o = alpha[0:1, ...]
        o_prev = alpha_o * o_prev + o_curr
        m_prev, l_prev = m_next, l_next
      # --- end V1 tiling ---

      m_scratch_ref[h_local] = m_prev
      l_scratch_ref[h_local] = l_prev
      o_scratch_ref[h_local] = o_prev

  assert bkv % bkv_compute == 0

  def kv_pipeline_body(j, k_ref, v_ref):
    # One pipelined KV block (size bkv). While the MXU runs the loop below on
    # this block, emit_pipeline is already DMAing block j+1 into the other
    # VMEM buffer, so the K/V HBM->VMEM transfer overlaps compute.
    def compute_body(kv_compute_index, _):
      _one_chunk(k_ref, v_ref, pl.ds(kv_compute_index * bkv_compute, bkv_compute))

    def last_compute_body(kv_compute_index):
      slice_k_len = kv_seq_len % bkv_compute
      _one_chunk(k_ref, v_ref, pl.ds(kv_compute_index * bkv_compute, slice_k_len))

    @pl.when(j != grid_width - 1)
    def body():
      lax.fori_loop(0, (bkv // bkv_compute), compute_body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def last_body():
      if kv_seq_len % bkv == 0:
        lax.fori_loop(0, bkv // bkv_compute, compute_body, None, unroll=True)
      else:
        remain = kv_seq_len % bkv
        iter_num = (remain + bkv_compute - 1) // bkv_compute
        if remain % bkv_compute == 0:
          lax.fori_loop(0, iter_num, compute_body, None, unroll=True)
        else:
          lax.fori_loop(0, iter_num - 1, compute_body, None, unroll=True)
          last_compute_body(iter_num - 1)

  # emit_pipeline streams the KV blocks (the old grid_width dimension) with
  # automatic double-buffering of the K and V VMEM blocks. The index maps pull
  # block j of the current head-tile (h_pid) straight from the HBM refs.
  pltpu.emit_pipeline(
      kv_pipeline_body,
      in_specs=[
          pl.BlockSpec((heads_per_tile, bkv, k_hbm_ref.shape[-1]),
                       lambda j: (h_pid, j, 0)),
          pl.BlockSpec((heads_per_tile, bkv, head_dim_v),
                       lambda j: (h_pid, j, 0)),
      ],
      out_specs=[],
      grid=(grid_width,),
  )(k_hbm_ref, v_hbm_ref)

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

  def q_index_map(h, i, *_):
    return (h, i, 0)

  def out_index_map(h, i, *_):
    return (h, 0, i)

  # Q stays a normal VMEM BlockSpec (reused across all KV blocks). K and V are
  # handed to the kernel in ANY (HBM) memory space so emit_pipeline can manage
  # the double-buffered HBM->VMEM streaming of each KV block itself.
  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim_qk), q_index_map),
      pl.BlockSpec(memory_space=pltpu.ANY),
      pl.BlockSpec(memory_space=pltpu.ANY),
  ]
  out_shape = jax.ShapeDtypeStruct(
      (num_q_heads, head_dim_v, q_seq_len), q.dtype)
  out_specs = pl.BlockSpec((hpt, head_dim_v, bq), out_index_map)

  scratch_shapes = [
      pltpu.VMEM((hpt, NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((hpt, NUM_SUBLANES, bq), jnp.float32),
      pltpu.VMEM((hpt, head_dim_v, bq), jnp.float32),
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
