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
# Process HEADS_PER_TILE heads per Pallas program so the two matmuls
# (scores = Q@K.T and out = attn@V) have several independent per-head
# instances available at once. Baseline ran one head per program
# (grid=(8,)) which left MXU1 idle (dual_ratio 0.0) and packed few ops
# per VLIW bundle (avg 1.21). Emitting HEADS_PER_TILE independent dots
# back-to-back gives the scheduler more to co-issue across both MXUs and
# to pack into bundles. Must divide bh = B*H = 8.
HEADS_PER_TILE = 2


def _attn_kernel(q_ref, k_ref, v_ref, o_ref, *, scale, heads_per_tile):
  for h in range(heads_per_tile):
    q = q_ref[h]
    k = k_ref[h]
    v = v_ref[h]
    scores = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    o_ref[h] = jnp.dot(attn, v, preferred_element_type=jnp.float32)


def optimized_compute(B=1, S=512, H=8, D=128):
  x, w_qkv, w_o = _make_test_data(B, S, H, D)
  b, s, d = x.shape
  scale = 1.0 / (D ** 0.5)

  qkv = jnp.einsum("bsd,de->bse", x, w_qkv)
  q, k, v = jnp.split(qkv, 3, axis=-1)
  to_heads = lambda t: t.reshape(b, s, H, D).transpose(0, 2, 1, 3)
  q, k, v = to_heads(q), to_heads(k), to_heads(v)

  bh = b * H
  heads_per_tile = HEADS_PER_TILE
  assert bh % heads_per_tile == 0
  qf, kf, vf = (t.reshape(bh, s, D) for t in (q, k, v))
  block = pl.BlockSpec((heads_per_tile, s, D), lambda i: (i, 0, 0))
  out = pl.pallas_call(
    functools.partial(_attn_kernel, scale=scale, heads_per_tile=heads_per_tile),
    grid=(bh // heads_per_tile,),
    in_specs=[block, block, block],
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((bh, s, D), jnp.float32),
  )(qf, kf, vf)

  ctx = out.reshape(b, H, s, D).transpose(0, 2, 1, 3).reshape(b, s, d)
  return jnp.einsum("bsd,de->bse", ctx, w_o)
# EVOLVE-BLOCK-END
