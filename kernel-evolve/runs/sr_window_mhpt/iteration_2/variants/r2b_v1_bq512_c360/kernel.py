# Source: kernels/wan_splash_mhpt.py (maxdiffusion flash_attention_kernel_mhpt mirror)
# plus the frame-window mask from avatar-turbo-edge-repro pallas_sparse_attention.
"""Wan2.2 SR windowed sparse MHA (mhpt) — template for evolution.

Same online-softmax flash kernel as wan_splash_mhpt (bf16 QKᵀ + f32 PV, base-2 exp,
V1 VPU register tiling), with the SR window mask applied inside the kernel:
q is this rank's cp shard [H, SQ, D]; keys are the cp-gathered sequence [H, SKV, D]
in rank-major layout, so a key's frame is (k_idx % SQ) // LOCAL_FRAME. A query in
frame f attends to frames [f-2, f+3) clipped plus the last frame when f+3 < N_FRAMES.

Baseline block choices are frame-aligned so every kv compute chunk lies inside one
frame of one shard: BKV=LOCAL_FRAME, BKV_COMPUTE divides it, and BQ divides SQ. A
chunk whose frame fails the window test for the whole q block is skipped with
pl.when; chunks that pass apply the exact element mask before the exponent. k/v
blocks are still fetched for skipped chunks — restructuring the grid or index maps
to skip the loads too is prime evolution territory, as are the block sizes and the
mask arithmetic itself.

Frozen (outside EVOLVE-BLOCK): imports, mask geometry constants, _make_test_data.
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

# Mask geometry (problem spec, frozen).
LOCAL_FRAME = 1800
N_FRAMES = 22
WIN_LEFT = 2
WIN_RIGHT = 3
ADD_LAST = True


def _make_test_data(H=5, SQ=39600, SKV=79200, D=128):
  """Deterministic f32 q [H,SQ,D], k/v [H,SKV,D]. Identical in ref and template."""
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(kq, (H, SQ, D), dtype=jnp.float32) * 0.5
  k = jax.random.normal(kk, (H, SKV, D), dtype=jnp.float32) * 0.5
  v = jax.random.normal(kv, (H, SKV, D), dtype=jnp.float32) * 0.5
  return q, k, v


# EVOLVE-BLOCK-START
# Baseline blocks. The kv side is frame-aligned (BKV == LOCAL_FRAME, chunk inside one
# frame of one shard) which only needs a multiple of 8. The q side cannot be: bq is the
# lane dimension of the output block and Mosaic requires a multiple of 128 there, and
# 39600 has no such divisor -- so the last q block runs past the array and relies on
# Pallas partial-block semantics (row-isolated garbage, writes masked).
BQ = 512
BKV = 1800
BKV_COMPUTE = 360
BKV_COMPUTE_IN = 120
HEADS_PER_TILE = 5
VMEM_LIMIT_BYTES = 100663296  # 96 MiB
USE_BASE2_EXP = True


# Grid j enumerates only the frames a q block can see: 6 window slots (frames
# qf0-2 .. qf0+3, clipped) plus one last-frame slot, per shard. 14 kv fetches per
# q block instead of 44. Clipping creates duplicate frames at the edges; a scalar
# predicate skips revisits so each frame contributes exactly once.
SLOTS = 7


def _slot_frame(i, j, bq):
  qf0 = (i * bq) // LOCAL_FRAME
  slot = j % SLOTS
  frame_win = jnp.clip(qf0 - WIN_LEFT + slot, 0, N_FRAMES - 1)
  return jnp.where(slot == SLOTS - 1, N_FRAMES - 1, frame_win)


def _window_sparse_kernel_mhpt(
    q_ref, k_ref, v_ref,
    m_scratch_ref, l_scratch_ref, o_scratch_ref,
    o_ref,
    *,
    mask_value, grid_width, bq, bkv, bkv_compute, bkv_compute_in,
    head_dim_v, q_seq_len, kv_seq_len, heads_per_tile, use_base2_exp=True,
):
  float32 = jnp.float32
  head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
  if rem != 0:
    raise NotImplementedError(f"{head_dim_v=} must be a multiple of {NUM_SUBLANES}")

  _, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  exp = jnp.exp2 if use_base2_exp else jnp.exp

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  # Frame of every q row in this block, and of every key row in one compute chunk.
  # Keys are rank-major: shard offset comes off before the frame is taken.
  q_pos = i * bq + lax.broadcasted_iota(jnp.int32, (1, bq), 1)  # [1, bq]
  q_frame = q_pos // LOCAL_FRAME

  qf0 = (i * bq) // LOCAL_FRAME
  qf1 = (i * bq + bq - 1) // LOCAL_FRAME
  slot = j % SLOTS
  k_frame = _slot_frame(i, j, bq)
  k_frame_prev = _slot_frame(i, j - 1, bq)
  is_window_slot = slot < SLOTS - 1
  # Edge clipping repeats a frame; visit each exactly once. The last-frame slot is
  # a revisit when the window slots already reach frame N_FRAMES-1.
  dup_window = is_window_slot & (slot > 0) & (k_frame == k_frame_prev)
  dup_last = (~is_window_slot) & (qf0 + WIN_RIGHT + 2 >= N_FRAMES)
  visit = ~(dup_window | dup_last)
  in_window = (k_frame >= qf0 - WIN_LEFT) & (k_frame < qf1 + WIN_RIGHT)
  is_last = ADD_LAST & (k_frame == N_FRAMES - 1) & (qf0 + WIN_RIGHT < N_FRAMES)

  def _one_chunk(chunk_idx):
    @pl.when(visit & (in_window | is_last))
    def compute():
      slice_k = pl.ds(chunk_idx * bkv_compute, bkv_compute)
      # Element mask over [bkv_compute, bq]: exact window test per (key, query).
      kf = k_frame  # scalar: whole chunk shares one frame
      allowed = (kf >= q_frame - WIN_LEFT) & (kf < q_frame + WIN_RIGHT)
      if ADD_LAST:
        allowed = allowed | ((kf == N_FRAMES - 1) & (q_frame + WIN_RIGHT < N_FRAMES))
      allowed_b = jnp.broadcast_to(allowed, (bkv_compute, bq))

      for h_local in range(heads_per_tile):
        m_prev = m_scratch_ref[h_local]
        l_prev = l_scratch_ref[h_local]
        q = q_ref[h_local]
        o_prev = o_scratch_ref[h_local]

        k_chunk = k_ref[h_local, slice_k, :]
        qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
        qk = jnp.where(allowed_b, qk, mask_value)
        v_chunk = v_ref[h_local, slice_k, :]

        # --- V1 VPU register tiling ---
        step = bkv_compute_in
        for s0 in range(0, qk.shape[0], step):
          qk_slice = qk[s0:s0 + step]
          m_curr = qk_slice.max(axis=0)[None, :]
          m_next = jnp.maximum(m_prev, m_curr)
          s_curr = exp(qk_slice - m_next[0:1])
          l_curr = s_curr.sum(axis=0, keepdims=True)
          alpha = exp(m_prev - m_next)
          l_next = l_curr + alpha * l_prev
          sv_dims = (((0,), (0,)), ((), ()))
          o_curr = lax.dot_general(
              v_chunk[s0:s0 + step], s_curr.astype(q_ref.dtype), sv_dims,
              preferred_element_type=float32)
          alpha_o = alpha[0:1, ...]
          o_prev = alpha_o * o_prev + o_curr
          m_prev, l_prev = m_next, l_next
        # --- end V1 tiling ---

        m_scratch_ref[h_local] = m_prev
        l_scratch_ref[h_local] = l_prev
        o_scratch_ref[h_local] = o_prev

  assert bkv % bkv_compute == 0
  assert kv_seq_len % bkv == 0, "frame-aligned baseline: BKV divides SKV"
  for idx in range(bkv // bkv_compute):
    _one_chunk(idx)

  @pl.when(j == grid_width - 1)
  def end():
    for h_local in range(heads_per_tile):
      l = l_scratch_ref[h_local]
      l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
      o_ref[h_local] = (o_scratch_ref[h_local] * l_inv).astype(o_ref.dtype)


def _window_sparse_forward(q, k, v, *, bq, bkv, bkv_compute, bkv_compute_in,
                           heads_per_tile, use_base2_exp, vmem_limit_bytes):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  kv_seq_len = k.shape[1]
  hpt = heads_per_tile
  assert num_q_heads % hpt == 0

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def kv_index_map(h, i, j, *_):
    shard = j // SLOTS
    slot = j % SLOTS
    qf0 = (i * bq) // LOCAL_FRAME
    frame_win = jnp.clip(qf0 - WIN_LEFT + slot, 0, N_FRAMES - 1)
    frame = jnp.where(slot == SLOTS - 1, N_FRAMES - 1, frame_win)
    return (h, shard * N_FRAMES + frame, 0)

  def out_index_map(h, i, j, *_):
    return (h, 0, i)

  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_qk), kv_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_v), kv_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, q_seq_len), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, head_dim_v, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, head_dim_v, bq), out_index_map),
  ]
  grid_width = (kv_seq_len // q_seq_len) * SLOTS  # shards x slots
  grid_height = (q_seq_len + bq - 1) // bq
  grid = (num_q_heads // hpt, grid_height, grid_width)

  all_out = pl.pallas_call(
      functools.partial(
          _window_sparse_kernel_mhpt,
          mask_value=DEFAULT_MASK_VALUE, grid_width=grid_width, bq=bq, bkv=bkv,
          bkv_compute=bkv_compute, bkv_compute_in=bkv_compute_in,
          head_dim_v=head_dim_v, q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
          heads_per_tile=hpt, use_base2_exp=use_base2_exp),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0, in_specs=in_specs, out_specs=out_specs, grid=grid),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes),
      out_shape=out_shapes,
  )(q, k, v)
  return all_out[-1]  # [num_q_heads, head_dim_v, q_seq_len]


def optimized_compute(H=5, SQ=39600, SKV=79200, D=128):
  q, k, v = _make_test_data(H, SQ, SKV, D)
  scale = 1.0 / (D ** 0.5)
  # base-2 exp in the kernel: pre-scale q by scale*log2(e) so exp2 reproduces
  # natural softmax with 1/sqrt(d) scaling. The additive mask value is large enough
  # that the base change does not matter for masked entries.
  q_scaled = (q * (scale * LOG2E)).astype(jnp.bfloat16) if USE_BASE2_EXP else (q * scale).astype(jnp.bfloat16)
  k_bf = k.astype(jnp.bfloat16)
  v_bf = v.astype(jnp.bfloat16)

  out = _window_sparse_forward(
      q_scaled, k_bf, v_bf,
      bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, bkv_compute_in=BKV_COMPUTE_IN,
      heads_per_tile=HEADS_PER_TILE, use_base2_exp=USE_BASE2_EXP,
      vmem_limit_bytes=VMEM_LIMIT_BYTES)  # [H, D, SQ]
  return jnp.swapaxes(out, 1, 2)  # [H, SQ, D]
# EVOLVE-BLOCK-END
