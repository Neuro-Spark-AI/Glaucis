# Source: avatar-turbo-edge-repro mhpt_sparse_attention._dense_kernel (fused rope), as
# deployed in the pipeline at [5, 10120 x 20240, 128]: 1.645 ms standalone against the
# library flash kernel's 3.50.
"""a2v dense MHA (mhpt, hpt=5) with fused rope — template for evolution.

One grid step handles all 5 local heads, so k/v stream from HBM once per query block.
q/k arrive unrotated bf16; four f32 tables carry cos/sin duplicated over interleaved
pairs, the q tables pre-scaled by softmax scale x log2(e). Rotation is two lane rolls,
a parity select and three multiply-adds; the q block rotates once per grid row into a
bf16 scratch, key chunks rotate per visit.

Frozen (outside EVOLVE-BLOCK): imports, grid geometry, _make_test_data, tables.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
PV_DIM_NUMBERS = (((0,), (0,)), ((), ()))
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
LOG2E = math.log2(math.e)

GRID_F, GRID_H, GRID_W = 22, 23, 40
SHARDS = 2
THETA = 10000.0


def _tables(D):
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
    ).reshape(GRID_F * GRID_H * w_local, c)
    return ang

  qa = rank_angles(0)
  ka = np.concatenate([rank_angles(r) for r in range(SHARDS)], axis=0)

  def dup(a):
    return jnp.asarray(np.repeat(a, 2, axis=-1).astype(np.float32))

  return dup(np.cos(qa)), dup(np.sin(qa)), dup(np.cos(ka)), dup(np.sin(ka))


def _make_test_data(H=5, SQ=10120, SKV=20240, D=128):
  """Unrotated f32 q/k/v plus rotation tables. Identical in ref and template."""
  assert SQ == GRID_F * GRID_H * GRID_W // SHARDS and SKV == SQ * SHARDS
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(kq, (H, SQ, D), dtype=jnp.float32) * 0.5
  k = jax.random.normal(kk, (H, SKV, D), dtype=jnp.float32) * 0.5
  v = jax.random.normal(kv, (H, SKV, D), dtype=jnp.float32) * 0.5
  return (q, k, v, *_tables(D))


# EVOLVE-BLOCK-START
# Swept at these shapes: bq 1024 and bkv 2024 win, the key tile is flat within 0.16 ms
# over {920, 1840, 2024, 4048}; the in-kernel rotation costs 0.12 ms.
BQ = 1024
BKV = 4048
BKV_COMPUTE = 2024
INNER = 184
HEADS_PER_TILE = 5
VMEM_LIMIT_BYTES = 100663296


def _dense_kernel_body(
    q_ref, k_ref, v_ref, qcos_ref, qsin_ref, kcos_ref, ksin_ref,
    m_ref, l_ref, o_ref, qrot_ref, out_ref,
    *, grid_w, bq, bkv, bkv_compute, inner, head_dim, heads_per_tile,
):
  j = pl.program_id(2)

  def rotate(x, cos2, sin2):
    xf = x.astype(jnp.float32)
    even = lax.broadcasted_iota(jnp.int32, xf.shape, xf.ndim - 1) % 2 == 0
    swap = jnp.where(even, -jnp.roll(xf, -1, axis=-1), jnp.roll(xf, 1, axis=-1))
    return xf * cos2 + swap * sin2

  @pl.when(j == 0)
  def init():
    o_ref[...] = jnp.zeros_like(o_ref)
    m_ref[...] = jnp.full_like(m_ref, DEFAULT_MASK_VALUE)
    l_ref[...] = jnp.zeros_like(l_ref)
    for h in range(heads_per_tile):
      qrot_ref[h] = rotate(q_ref[h], qcos_ref[...], qsin_ref[...]).astype(q_ref.dtype)

  for chunk in range(bkv // bkv_compute):
    keys = pl.ds(chunk * bkv_compute, bkv_compute)
    for h in range(heads_per_tile):
      m_prev, l_prev, o_prev = m_ref[h], l_ref[h], o_ref[h]
      k_chunk = rotate(k_ref[h, keys, :], kcos_ref[keys, :], ksin_ref[keys, :]).astype(k_ref.dtype)
      scores = lax.dot_general(k_chunk, qrot_ref[h], NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
      values = v_ref[h, keys, :]
      for s0 in range(0, bkv_compute, inner):
        tile = scores[s0 : s0 + inner]
        m_next = jnp.maximum(m_prev, tile.max(axis=0)[None, :])
        w = jnp.exp2(tile - m_next[0:1])
        alpha = jnp.exp2(m_prev - m_next)
        l_prev = w.sum(axis=0, keepdims=True) + alpha * l_prev
        o_prev = alpha[0:1, ...] * o_prev + lax.dot_general(
            values[s0 : s0 + inner], w.astype(q_ref.dtype), PV_DIM_NUMBERS,
            preferred_element_type=jnp.float32)
        m_prev = m_next
      m_ref[h], l_ref[h], o_ref[h] = m_prev, l_prev, o_prev

  @pl.when(j == grid_w - 1)
  def finish():
    for h in range(heads_per_tile):
      s = jnp.tile(1.0 / l_ref[h], (head_dim // NUM_SUBLANES, 1))
      out_ref[h] = (o_ref[h] * s).astype(out_ref.dtype)


def _dense_forward(q, k, v, qcos2, qsin2, kcos2, ksin2, *,
                   bq, bkv, bkv_compute, inner, heads_per_tile, vmem_limit_bytes):
  heads, q_len, head_dim = q.shape
  kv_len = k.shape[1]
  hpt = heads_per_tile
  assert kv_len % bkv == 0 and bkv % bkv_compute == 0 and bkv_compute % inner == 0
  grid_w = kv_len // bkv
  grid = (heads // hpt, -(-q_len // bq), grid_w)

  in_specs = [
      pl.BlockSpec((hpt, bq, head_dim), lambda h, i, j: (h, i, 0)),
      pl.BlockSpec((hpt, bkv, head_dim), lambda h, i, j: (h, j, 0)),
      pl.BlockSpec((hpt, bkv, head_dim), lambda h, i, j: (h, j, 0)),
      pl.BlockSpec((bq, head_dim), lambda h, i, j: (i, 0)),
      pl.BlockSpec((bq, head_dim), lambda h, i, j: (i, 0)),
      pl.BlockSpec((bkv, head_dim), lambda h, i, j: (j, 0)),
      pl.BlockSpec((bkv, head_dim), lambda h, i, j: (j, 0)),
  ]
  accs = [
      (jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32), (hpt, NUM_SUBLANES, bq)),
      (jax.ShapeDtypeStruct((hpt, NUM_SUBLANES, bq), jnp.float32), (hpt, NUM_SUBLANES, bq)),
      (jax.ShapeDtypeStruct((hpt, head_dim, bq), jnp.float32), (hpt, head_dim, bq)),
      (jax.ShapeDtypeStruct((hpt, bq, head_dim), jnp.bfloat16), (hpt, bq, head_dim)),
  ]

  out = pl.pallas_call(
      functools.partial(
          _dense_kernel_body, grid_w=grid_w, bq=bq, bkv=bkv, bkv_compute=bkv_compute,
          inner=inner, head_dim=head_dim, heads_per_tile=hpt),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=[
              *[pl.BlockSpec(shape, lambda *_: (0, 0, 0)) for _, shape in accs],
              pl.BlockSpec((hpt, head_dim, bq), lambda h, i, j: (h, 0, i)),
          ],
          grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          disable_bounds_checks=True,
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      out_shape=[*[sd for sd, _ in accs],
                 jax.ShapeDtypeStruct((heads, head_dim, q_len), jnp.bfloat16)],
  )(q, k, v, qcos2, qsin2, kcos2, ksin2)
  return jnp.swapaxes(out[-1], 1, 2)


def make_inputs(H=5, SQ=10120, SKV=20240, D=128):
  """bf16 unrotated q/k/v plus the tables, q tables carrying scale*log2(e)."""
  q, k, v, qcos2, qsin2, kcos2, ksin2 = _make_test_data(H, SQ, SKV, D)
  s = (1.0 / (D ** 0.5)) * LOG2E
  return (q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
          qcos2 * s, qsin2 * s, kcos2, ksin2)


def timed_compute(q, k, v, qcos2, qsin2, kcos2, ksin2):
  return _dense_forward(
      q, k, v, qcos2, qsin2, kcos2, ksin2,
      bq=BQ, bkv=BKV, bkv_compute=BKV_COMPUTE, inner=INNER,
      heads_per_tile=HEADS_PER_TILE, vmem_limit_bytes=VMEM_LIMIT_BYTES)


def optimized_compute(H=5, SQ=10120, SKV=20240, D=128):
  return timed_compute(*make_inputs(H, SQ, SKV, D))
# EVOLVE-BLOCK-END
