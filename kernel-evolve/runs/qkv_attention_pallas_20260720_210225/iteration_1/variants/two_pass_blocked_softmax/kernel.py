# Source: HW_Optimization probe — tpu/wan22/test/qkv_attention_pallas.py
"""qkv self-attention with a Pallas attention core — template for evolution."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _make_test_data(B=1, S=512, H=8, D=128):
  d_model = H * D
  kx, kq, ko = jax.random.split(jax.random.PRNGKey(0), 3)
  x = jax.random.normal(kx, (B, S, d_model), dtype=jnp.float32) * 0.1
  w_qkv = jax.random.normal(kq, (d_model, 3 * d_model), dtype=jnp.float32) * 0.02
  w_o = jax.random.normal(ko, (d_model, d_model), dtype=jnp.float32) * 0.02
  return x, w_qkv, w_o


# EVOLVE-BLOCK-START
def _attn_kernel(q_ref, k_ref, v_ref, o_ref, *, scale):
  q = q_ref[0]
  s = q.shape[0]
  d = q.shape[1]

  block_kv = 256
  if s % block_kv != 0:
    block_kv = s
  n_blocks = s // block_kv

  neg_inf = jnp.float32(-1e30)

  # Pass 1: global per-row max over all KV blocks. Only one (s, block_kv)
  # score tile is live at a time, never the full (s, s) matrix.
  row_max = jnp.full((s,), neg_inf, dtype=jnp.float32)
  for j in range(n_blocks):
    k_blk = k_ref[0][pl.ds(j * block_kv, block_kv)]
    scores = jnp.dot(q, k_blk.T, preferred_element_type=jnp.float32) * scale
    row_max = jnp.maximum(row_max, jnp.max(scores, axis=-1))

  # Pass 2: re-read K, recompute scores, accumulate denom and unnormalized
  # output using the fixed row_max from pass 1.
  denom = jnp.zeros((s,), dtype=jnp.float32)
  acc = jnp.zeros((s, d), dtype=jnp.float32)
  for j in range(n_blocks):
    k_blk = k_ref[0][pl.ds(j * block_kv, block_kv)]
    v_blk = v_ref[0][pl.ds(j * block_kv, block_kv)]
    scores = jnp.dot(q, k_blk.T, preferred_element_type=jnp.float32) * scale
    p = jnp.exp(scores - row_max[:, None])
    denom = denom + jnp.sum(p, axis=-1)
    acc = acc + jnp.dot(p, v_blk, preferred_element_type=jnp.float32)

  o_ref[0] = acc / denom[:, None]


def optimized_compute(B=1, S=512, H=8, D=128):
  x, w_qkv, w_o = _make_test_data(B, S, H, D)
  b, s, d = x.shape
  scale = 1.0 / (D ** 0.5)

  qkv = jnp.einsum("bsd,de->bse", x, w_qkv)
  q, k, v = jnp.split(qkv, 3, axis=-1)
  to_heads = lambda t: t.reshape(b, s, H, D).transpose(0, 2, 1, 3)
  q, k, v = to_heads(q), to_heads(k), to_heads(v)

  bh = b * H
  qf, kf, vf = (t.reshape(bh, s, D) for t in (q, k, v))
  block = pl.BlockSpec((1, s, D), lambda i: (i, 0, 0))
  out = pl.pallas_call(
    functools.partial(_attn_kernel, scale=scale),
    grid=(bh,),
    in_specs=[block, block, block],
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((bh, s, D), jnp.float32),
  )(qf, kf, vf)

  ctx = out.reshape(b, H, s, D).transpose(0, 2, 1, 3).reshape(b, s, d)
  return jnp.einsum("bsd,de->bse", ctx, w_o)
# EVOLVE-BLOCK-END
