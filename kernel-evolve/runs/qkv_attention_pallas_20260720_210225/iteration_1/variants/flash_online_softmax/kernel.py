# Source: HW_Optimization probe — tpu/wan22/test/qkv_attention_pallas.py
"""qkv self-attention with a Pallas attention core — template for evolution."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _make_test_data(B=1, S=512, H=8, D=128):
  """Deterministic f32 inputs. MUST be identical to the reference's copy."""
  d_model = H * D
  kx, kq, ko = jax.random.split(jax.random.PRNGKey(0), 3)
  x = jax.random.normal(kx, (B, S, d_model), dtype=jnp.float32) * 0.1
  w_qkv = jax.random.normal(kq, (d_model, 3 * d_model), dtype=jnp.float32) * 0.02
  w_o = jax.random.normal(ko, (d_model, d_model), dtype=jnp.float32) * 0.02
  return x, w_qkv, w_o


# EVOLVE-BLOCK-START
def _attn_kernel(q_ref, k_ref, v_ref, o_ref, *, scale, block_kv):
  """One (batch*head) slice. Each ref is (1, S, D); index [0] -> 2D.

  Flash online-softmax: tile the KV sequence into blocks of block_kv and keep a
  running max m, running denominator l, and running weighted output acc. This
  never materializes the full (S, S) score matrix; only (S, block_kv) blocks.
  """
  q = q_ref[0]  # (S, D)
  s = q.shape[0]
  d = q.shape[1]
  n_blocks = s // block_kv

  neg_inf = jnp.float32(-1e30)
  m = jnp.full((s, 1), neg_inf, dtype=jnp.float32)   # running row max
  l = jnp.zeros((s, 1), dtype=jnp.float32)           # running denominator
  acc = jnp.zeros((s, d), dtype=jnp.float32)         # running weighted output

  for j in range(n_blocks):
    k_blk = k_ref[0, pl.ds(j * block_kv, block_kv), :]  # (block_kv, D)
    v_blk = v_ref[0, pl.ds(j * block_kv, block_kv), :]  # (block_kv, D)

    # scores for this KV block: (S, block_kv)
    scores = jnp.dot(q, k_blk.T, preferred_element_type=jnp.float32) * scale

    blk_max = jnp.max(scores, axis=-1, keepdims=True)     # (S, 1)
    m_new = jnp.maximum(m, blk_max)                       # (S, 1)
    p = jnp.exp(scores - m_new)                           # (S, block_kv)
    alpha = jnp.exp(m - m_new)                            # (S, 1) rescale factor

    l = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
    acc = acc * alpha + jnp.dot(p, v_blk, preferred_element_type=jnp.float32)
    m = m_new

  o_ref[0] = acc / l  # (S, D)


def optimized_compute(B=1, S=512, H=8, D=128):
  x, w_qkv, w_o = _make_test_data(B, S, H, D)
  b, s, d = x.shape
  scale = 1.0 / (D ** 0.5)

  qkv = jnp.einsum("bsd,de->bse", x, w_qkv)
  q, k, v = jnp.split(qkv, 3, axis=-1)
  to_heads = lambda t: t.reshape(b, s, H, D).transpose(0, 2, 1, 3)
  q, k, v = to_heads(q), to_heads(k), to_heads(v)  # (b, H, s, D)

  bh = b * H
  qf, kf, vf = (t.reshape(bh, s, D) for t in (q, k, v))

  # Pick a KV block size that divides the sequence length.
  block_kv = 256
  if s % block_kv != 0:
    block_kv = s

  block = pl.BlockSpec((1, s, D), lambda i: (i, 0, 0))
  out = pl.pallas_call(
    functools.partial(_attn_kernel, scale=scale, block_kv=block_kv),
    grid=(bh,),
    in_specs=[block, block, block],
    out_specs=block,
    out_shape=jax.ShapeDtypeStruct((bh, s, D), jnp.float32),
  )(qf, kf, vf)

  ctx = out.reshape(b, H, s, D).transpose(0, 2, 1, 3).reshape(b, s, d)  # merge heads
  return jnp.einsum("bsd,de->bse", ctx, w_o)
# EVOLVE-BLOCK-END
