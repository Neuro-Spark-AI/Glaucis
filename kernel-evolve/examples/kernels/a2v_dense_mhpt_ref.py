# Reference for the a2v dense attention with fused rope, single card, f32.
# q [H, SQ, D] is one rank's cp shard, k/v [H, SKV, D] the rank-major gather; all arrive
# unrotated with cos/sin tables duplicated over interleaved pairs. Exact f32 rotation
# then dense softmax attention, chunked over q to bound the score matrix.
"""Reference a2v_dense_mhpt kernel: f32 rope rotation + exact dense attention."""

import jax
import jax.numpy as jnp
import numpy as np

# The production patch grid at tp=8/cp=2: (F, H, W_global) = (22, 23, 40), width split
# over 2 ranks; latent tokens 22*23*20 = 10120 per rank.
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
        )
        return ang.reshape(GRID_F * GRID_H * w_local, c)

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


def _rotate(x, cos2, sin2):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos2[None, :, 0::2], sin2[None, :, 0::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return jnp.stack((r0, r1), axis=-1).reshape(x.shape)


def make_inputs(H=5, SQ=10120, SKV=20240, D=128):
    return _make_test_data(H, SQ, SKV, D)


def timed_compute(q, k, v, qcos2, qsin2, kcos2, ksin2):
    H, SQ, D = q.shape
    scale = 1.0 / (D**0.5)
    q = _rotate(q, qcos2, qsin2)
    k = _rotate(k, kcos2, ksin2)
    outs = []
    for s0 in range(0, SQ, 1012):
        sc = jnp.einsum("hqd,hkd->hqk", q[:, s0 : s0 + 1012], k) * scale
        p = jax.nn.softmax(sc, axis=-1)
        outs.append(jnp.einsum("hqk,hkd->hqd", p, v))
    return jnp.concatenate(outs, axis=1)


def simple_compute(H=5, SQ=10120, SKV=20240, D=128):
    return timed_compute(*make_inputs(H, SQ, SKV, D))


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
