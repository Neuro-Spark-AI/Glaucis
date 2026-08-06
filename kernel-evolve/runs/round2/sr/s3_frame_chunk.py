# Source: sr_window_mhpt_champion (r3b_full_partial) plus in-kernel rope rotation.
"""Wan2.2 SR windowed sparse MHA with fused rope — template for evolution.

q and k arrive unrotated; four extra inputs carry cos/sin duplicated over interleaved
pairs ([S, D], table[:, 2i] == table[:, 2i+1]), fetched with the same index maps as q
and k so a key block and its rotation multipliers always travel together. The q tables
come pre-multiplied by scale*log2(e), so rotation and softmax scaling fold into one
pass; rotation is 7 full-width VPU ops (two lane rolls, a parity select, three
multiply-adds) with no interleaved-pair reshape anywhere.

The q block is rotated once per grid row (at j == 0) into a bf16 scratch; key chunks
are rotated per visit -- a frame is re-rotated by every q block that can see it, which
is pure VPU work overlapped against the MXU dots.

Below the fused-rope docstring, everything else is r3b_full_partial unchanged:

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

# The production patch grid at tp=8/cp=2 and the rope frequency ladder (problem spec,
# frozen; kept identical to sr_window_mhpt_rope_ref).
GRID_F, GRID_H, GRID_W = 22, 45, 80
SHARDS = 2
THETA = 10000.0


def _tables(D):
  import numpy as np
  c = D // 2
  inv = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
  ct = c - 2 * (c // 3)
  inv_t, inv_h, inv_w = inv[:ct], inv[ct : ct + c // 3], inv[ct + c // 3 :]
  w_local = GRID_W // SHARDS

  def rank_angles(rank):
    f = np.arange(GRID_F)[:, None, None, None]
    y = np.arange(GRID_H)[None, :, None, None]
    x = np.arange(w_local * rank, w_local * (rank + 1))[None, None, :, None]
    ang = np.concatenate(
        [
            np.broadcast_to(f * inv_t, (GRID_F, GRID_H, w_local, len(inv_t))),
            np.broadcast_to(y * inv_h, (GRID_F, GRID_H, w_local, len(inv_h))),
            np.broadcast_to(x * inv_w, (GRID_F, GRID_H, w_local, len(inv_w))),
        ],
        axis=-1,
    )
    return ang.reshape(GRID_F * GRID_H * w_local, c)

  q_ang = rank_angles(0)
  kv_ang = np.concatenate([rank_angles(r) for r in range(SHARDS)], axis=0)

  def dup(a):
    return np.repeat(a, 2, axis=-1).astype(np.float32)

  return (
      jnp.asarray(dup(np.cos(q_ang))),
      jnp.asarray(dup(np.sin(q_ang))),
      jnp.asarray(dup(np.cos(kv_ang))),
      jnp.asarray(dup(np.sin(kv_ang))),
  )


def _make_test_data(H=5, SQ=39600, SKV=79200, D=128):
  """Unrotated f32 q/k/v plus rotation tables. Identical in ref and template."""
  assert SQ == GRID_F * GRID_H * GRID_W // SHARDS and SKV == SQ * SHARDS
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(kq, (H, SQ, D), dtype=jnp.float32) * 0.5
  k = jax.random.normal(kk, (H, SKV, D), dtype=jnp.float32) * 0.5
  v = jax.random.normal(kv, (H, SKV, D), dtype=jnp.float32) * 0.5
  return (q, k, v, *_tables(D))


# EVOLVE-BLOCK-START
# Baseline blocks. The kv side is frame-aligned (BKV == LOCAL_FRAME, chunk inside one
# frame of one shard) which only needs a multiple of 8. The q side cannot be: bq is the
# lane dimension of the output block and Mosaic requires a multiple of 128 there, and
# 39600 has no such divisor -- so the last q block runs past the array and relies on
# Pallas partial-block semantics (row-isolated garbage, writes masked).
BQ = 1024
BKV = 1800
BKV_COMPUTE = 1800
BKV_COMPUTE_IN = 360
HEADS_PER_TILE = 5
VMEM_LIMIT_BYTES = 100663296  # 96 MiB
USE_BASE2_EXP = True


# Grid j enumerates only the frames a q block can see: 6 window slots (frames
# qf0-2 .. qf0+3, clipped) plus one last-frame slot, per shard. 14 kv fetches per
# q block instead of 44. Clipping creates duplicate frames at the edges; a scalar
# predicate skips revisits so each frame contributes exactly once.
# Window slots cover the union of windows over the q rows of one block,
# [qf0-WIN_LEFT, qf1+WIN_RIGHT); with bq <= LOCAL_FRAME a block straddles at most one
# frame boundary, so qf1 <= qf0+1 and the union spans WIN_LEFT+WIN_RIGHT+1 frames.
WINDOW_SLOTS = WIN_LEFT + WIN_RIGHT + 1
SLOTS = WINDOW_SLOTS + (1 if ADD_LAST else 0)


def _slot_frame(i, j, bq, xp=jnp):
  qf0 = (i * bq) // LOCAL_FRAME
  slot = j % SLOTS
  frame_win = xp.clip(qf0 - WIN_LEFT + slot, 0, N_FRAMES - 1)
  return xp.where(slot >= WINDOW_SLOTS, N_FRAMES - 1, frame_win)


def _slot_state(i, j, bq, xp=jnp):
  """Slot geometry for grid step (i, j): frame, whether to run it, and how to mask it.

  Scalar arithmetic only, and the array namespace is a parameter, so the host-side
  coverage check can walk the same predicate with numpy that the kernel traces with jnp.
  """
  qf0 = (i * bq) // LOCAL_FRAME
  qf1 = (i * bq + bq - 1) // LOCAL_FRAME
  slot = j % SLOTS
  k_frame = _slot_frame(i, j, bq, xp)
  is_window_slot = slot < WINDOW_SLOTS
  # Edge clipping repeats a frame; visit each exactly once. The last-frame slot is a
  # revisit only when the window slots themselves already reach the last frame, i.e.
  # when the highest window frame qf0 - WIN_LEFT + WINDOW_SLOTS - 1 gets there. Testing
  # one frame earlier drops the reference frame for the two q blocks whose window stops
  # exactly one short of it, which random-data correctness checks cannot see: 1800 of
  # 21600 keys go missing and the softmax average moves by ~1e-3.
  dup_window = is_window_slot & (slot > 0) & (k_frame == _slot_frame(i, j - 1, bq, xp))
  dup_last = (~is_window_slot) & (qf0 - WIN_LEFT + WINDOW_SLOTS - 1 >= N_FRAMES - 1)
  visit = ~(dup_window | dup_last)
  in_window = (k_frame >= qf0 - WIN_LEFT) & (k_frame < qf1 + WIN_RIGHT)
  is_last = ADD_LAST & (k_frame == N_FRAMES - 1) & (qf0 + WIN_RIGHT < N_FRAMES)
  # A block is fully allowed when every q row in [qf0, qf1] passes the window test for
  # this frame, or when it is the last frame: a row either has it in its window
  # (qf >= L-WIN_RIGHT+1) or takes it as the reference frame (qf <= L-WIN_RIGHT), so the
  # two cases meet with no gap. Fully allowed blocks skip the element mask entirely.
  full_window = (qf1 <= k_frame + WIN_LEFT) & (qf0 >= k_frame - WIN_RIGHT + 1)
  full = full_window | (ADD_LAST & (k_frame == N_FRAMES - 1))
  return k_frame, visit & (in_window | is_last), full


def _window_sparse_kernel_mhpt(
    q_ref, k_ref, v_ref, qcos_ref, qsin_ref, kcos_ref, ksin_ref,
    m_scratch_ref, l_scratch_ref, o_scratch_ref, qrot_scratch_ref,
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

  def rotate(x, cos2, sin2):
    # Interleaved-pair rotation without touching the pair layout: for pair (x0, x1),
    # the partner vector is (-x1, x0), built from two lane rolls and a parity select.
    xf = x.astype(jnp.float32)
    even = lax.broadcasted_iota(jnp.int32, xf.shape, xf.ndim - 1) % 2 == 0
    swap = jnp.where(even, -jnp.roll(xf, -1, axis=-1), jnp.roll(xf, 1, axis=-1))
    return xf * cos2 + swap * sin2

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    # Rotate this grid row's q block once; the q tables carry the softmax scale.
    for h_local in range(heads_per_tile):
      qrot_scratch_ref[h_local] = rotate(
          q_ref[h_local], qcos_ref[...], qsin_ref[...]).astype(q_ref.dtype)

  # Frame of every q row in this block, and of every key row in one compute chunk.
  # Keys are rank-major: shard offset comes off before the frame is taken.
  q_pos = i * bq + lax.broadcasted_iota(jnp.int32, (1, bq), 1)  # [1, bq]
  q_frame = q_pos // LOCAL_FRAME

  # Predicates and the element mask are block-level: with BKV == LOCAL_FRAME every chunk
  # of a block shares one frame, so they are computed once per grid step instead of once
  # per chunk -- the scalar unit was saturated at 97.7%.
  k_frame, run, full = _slot_state(i, j, bq)

  def _block(masked):
    kf = k_frame
    if masked:
      allowed = (kf >= q_frame - WIN_LEFT) & (kf < q_frame + WIN_RIGHT)
      if ADD_LAST:
        allowed = allowed | ((kf == N_FRAMES - 1) & (q_frame + WIN_RIGHT < N_FRAMES))
      allowed_b = jnp.broadcast_to(allowed, (bkv_compute, bq))
    for chunk_idx in range(bkv // bkv_compute):
      slice_k = pl.ds(chunk_idx * bkv_compute, bkv_compute)

      for h_local in range(heads_per_tile):
        m_prev = m_scratch_ref[h_local]
        l_prev = l_scratch_ref[h_local]
        q = qrot_scratch_ref[h_local]
        o_prev = o_scratch_ref[h_local]

        k_chunk = rotate(
            k_ref[h_local, slice_k, :], kcos_ref[slice_k, :], ksin_ref[slice_k, :]
        ).astype(k_ref.dtype)
        qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)
        if masked:
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

  @pl.when(run & full)
  def compute_full():
    _block(masked=False)

  @pl.when(run & (~full))
  def compute_partial():
    _block(masked=True)

  assert bkv % bkv_compute == 0
  assert kv_seq_len % bkv == 0, "frame-aligned baseline: BKV divides SKV"

  @pl.when(j == grid_width - 1)
  def end():
    for h_local in range(heads_per_tile):
      l = l_scratch_ref[h_local]
      l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
      o_ref[h_local] = (o_scratch_ref[h_local] * l_inv).astype(o_ref.dtype)


def _window_sparse_forward(q, k, v, qcos2, qsin2, kcos2, ksin2, *,
                           bq, bkv, bkv_compute, bkv_compute_in,
                           heads_per_tile, use_base2_exp, vmem_limit_bytes):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  kv_seq_len = k.shape[1]
  hpt = heads_per_tile
  assert num_q_heads % hpt == 0

  def q_index_map(h, i, j, *_):
    return (h, i, 0)

  def kv_index_map(h, i, j, *_):
    # bkv == LOCAL_FRAME, so a kv block index is (shard, frame) in frame units.
    return (h, (j // SLOTS) * N_FRAMES + _slot_frame(i, j, bq), 0)

  def out_index_map(h, i, j, *_):
    return (h, 0, i)

  # The tables are 2D; their index maps drop the head coordinate but keep the exact
  # block arithmetic of their tensor, so multipliers and data always co-travel.
  def qtab_index_map(h, i, j, *_):
    return (i, 0)

  def ktab_index_map(h, i, j, *_):
    return ((j // SLOTS) * N_FRAMES + _slot_frame(i, j, bq, jnp), 0)

  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim_qk), q_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_qk), kv_index_map),
      pl.BlockSpec((hpt, bkv, head_dim_v), kv_index_map),
      pl.BlockSpec((bq, head_dim_qk), qtab_index_map),
      pl.BlockSpec((bq, head_dim_qk), qtab_index_map),
      pl.BlockSpec((bkv, head_dim_qk), ktab_index_map),
      pl.BlockSpec((bkv, head_dim_qk), ktab_index_map),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, head_dim_v, bq), jnp.float32),
      jax.ShapeDtypeStruct((hpt, bq, head_dim_qk), q.dtype),
      jax.ShapeDtypeStruct((num_q_heads, head_dim_v, q_seq_len), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, NUM_SUBLANES, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, head_dim_v, bq), lambda *_: (0, 0, 0)),
      pl.BlockSpec((hpt, bq, head_dim_qk), lambda *_: (0, 0, 0)),
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
  )(q, k, v, qcos2, qsin2, kcos2, ksin2)
  return all_out[-1]  # [num_q_heads, head_dim_v, q_seq_len]


def make_inputs(H=5, SQ=39600, SKV=79200, D=128):
  """Kernel-ready operands: bf16 unrotated q/k/v plus the tables, the q tables carrying
  scale*log2(e) so exp2 in the kernel reproduces the natural scaled softmax. In the
  pipeline the casts and the one-time table build live outside the custom call."""
  q, k, v, qcos2, qsin2, kcos2, ksin2 = _make_test_data(H, SQ, SKV, D)
  scale = 1.0 / (D ** 0.5)
  s = scale * LOG2E if USE_BASE2_EXP else scale
  return (
      q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
      qcos2 * s, qsin2 * s, kcos2, ksin2,
  )


def timed_compute(q, k, v, qcos2, qsin2, kcos2, ksin2):
  out = _window_sparse_forward(
      q, k, v, qcos2, qsin2, kcos2, ksin2,
      bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, bkv_compute_in=BKV_COMPUTE_IN,
      heads_per_tile=HEADS_PER_TILE, use_base2_exp=USE_BASE2_EXP,
      vmem_limit_bytes=VMEM_LIMIT_BYTES)  # [H, D, SQ]
  return jnp.swapaxes(out, 1, 2)  # [H, SQ, D]


def optimized_compute(H=5, SQ=39600, SKV=79200, D=128):
  return timed_compute(*make_inputs(H, SQ, SKV, D))
# EVOLVE-BLOCK-END
