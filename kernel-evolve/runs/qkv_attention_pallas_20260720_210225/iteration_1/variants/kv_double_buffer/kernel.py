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
from jax.experimental.pallas import tpu as pltpu


def _attn_kernel(q_ref, k_ref, v_ref, o_ref, acc_ref, m_ref, l_ref,
                 *, scale, num_kv):
  # Grid is (batch*head, kv_block). Q is resident (same block every kv step,
  # so Pallas does not re-DMA it); K/V blocks advance each step and are
  # double-buffered — the next block's HBM->VMEM DMA overlaps this block's MXU
  # matmuls. Softmax is carried online in VMEM scratch (running max/sum/acc).
  j = pl.program_id(1)

  @pl.when(j == 0)
  def _init():
    m_ref[...] = jnp.full_like(m_ref, -1e30)   # running row max, (s, 1)
    l_ref[...] = jnp.zeros_like(l_ref)         # running denominator, (s, 1)
    acc_ref[...] = jnp.zeros_like(acc_ref)     # running output, (s, D)

  q = q_ref[0]                          # (s, D)
  k = k_ref[0]                          # (block_kv, D)
  v = v_ref[0]                          # (block_kv, D)

  scores = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * scale  # (s,bkv)

  m_prev = m_ref[...]                              # (s, 1)
  blk_max = jnp.max(scores, axis=-1, keepdims=True)  # (s, 1)
  m_new = jnp.maximum(m_prev, blk_max)            # (s, 1)

  p = jnp.exp(scores - m_new)                     # (s, block_kv)
  alpha = jnp.exp(m_prev - m_new)                 # (s, 1) rescale of prior state

  l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)
  acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
      p, v, preferred_element_type=jnp.float32)
  m_ref[...] = m_new

  @pl.when(j == num_kv - 1)
  def _finalize():
    o_ref[0] = acc_ref[...] / l_ref[...]


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

  block_kv = 256 if s % 256 == 0 else s
  num_kv = s // block_kv

  q_spec = pl.BlockSpec((1, s, D), lambda i, j: (i, 0, 0))
  kv_spec = pl.BlockSpec((1, block_kv, D), lambda i, j: (i, j, 0))
  o_spec = pl.BlockSpec((1, s, D), lambda i, j: (i, 0, 0))

  out = pl.pallas_call(
    functools.partial(_attn_kernel, scale=scale, num_kv=num_kv),
    grid=(bh, num_kv),
    in_specs=[q_spec, kv_spec, kv_spec],
    out_specs=o_spec,
    out_shape=jax.ShapeDtypeStruct((bh, s, D), jnp.float32),
    scratch_shapes=[
      pltpu.VMEM((s, D), jnp.float32),   # acc (running weighted output)
      pltpu.VMEM((s, 1), jnp.float32),   # running max
      pltpu.VMEM((s, 1), jnp.float32),   # running sum
    ],
  )(qf, kf, vf)

  ctx = out.reshape(b, H, s, D).transpose(0, 2, 1, 3).reshape(b, s, d)
  return jnp.einsum("bsd,de->bse", ctx, w_o)
# EVOLVE-BLOCK-END
