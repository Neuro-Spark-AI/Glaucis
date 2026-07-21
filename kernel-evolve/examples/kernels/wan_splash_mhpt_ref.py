# Source: maxdiffusion splash_attention_kernel.flash_attention_kernel_mhpt (production
# ring MHA fwd, hpt=5) — self-contained mirror = custom_splash_attention._flash_attention_kernel_mhpt.
"""Reference full multi-head attention for the Wan2.2 splash MHA kernel.

Per-head softmax(Q·Kᵀ / sqrt(d)) · V, in f32. Query-blocked so it never
materializes the full (S x S) score matrix (S is large: 8192 .. 86016), keeping
memory at O(S) per block while staying numerically exact (full softmax per query
row over all keys). Ground truth for the flash kernel in wan_splash_mhpt.py.
Inputs come from the SAME _make_test_data as the template.
"""

import jax
import jax.numpy as jnp


def _make_test_data(H=5, S=8192, D=128):
  """Deterministic f32 q,k,v of shape [H, S, D]. Identical to the template's copy."""
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
  q = jax.random.normal(kq, (H, S, D), dtype=jnp.float32) * 0.5
  k = jax.random.normal(kk, (H, S, D), dtype=jnp.float32) * 0.5
  v = jax.random.normal(kv, (H, S, D), dtype=jnp.float32) * 0.5
  return q, k, v


def simple_compute(H=5, S=8192, D=128, q_block=1024):
  q, k, v = _make_test_data(H, S, D)
  scale = 1.0 / (D ** 0.5)

  def per_head(qh, kh, vh):  # qh,kh,vh: [S, D]
    def qblock(carry, q_blk):  # q_blk: [q_block, D]
      scores = jnp.einsum("qd,kd->qk", q_blk, kh) * scale  # [q_block, S]
      attn = jax.nn.softmax(scores, axis=-1)
      return carry, jnp.einsum("qk,kd->qd", attn, vh)  # [q_block, D]

    nb = S // q_block
    rem = S - nb * q_block
    q_main = qh[: nb * q_block].reshape(nb, q_block, D)
    _, out_main = jax.lax.scan(qblock, None, q_main)
    out = out_main.reshape(nb * q_block, D)
    if rem:
      scores = jnp.einsum("qd,kd->qk", qh[nb * q_block:], kh) * scale
      out_rem = jnp.einsum("qk,kd->qd", jax.nn.softmax(scores, axis=-1), vh)
      out = jnp.concatenate([out, out_rem], axis=0)
    return out

  return jax.vmap(per_head)(q, k, v)  # [H, S, D]


def reference_fn(**kwargs):
  return simple_compute(**kwargs)
