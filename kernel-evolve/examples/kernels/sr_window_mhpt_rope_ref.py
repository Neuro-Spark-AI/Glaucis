# Reference for the rope-fused SR windowed sparse attention, single card, f32.
# Same interval-attention semantics as sr_window_mhpt_ref, but q and k arrive
# unrotated together with per-position rotation tables, and the rotation is applied
# in f32 before the attention -- which is exactly what the fused kernel must
# reproduce. The tables are cos/sin duplicated over interleaved pairs ([S, D] with
# table[:, 2i] == table[:, 2i+1]), the layout the kernel's lane-roll rotation wants.
"""Reference sr_window_mhpt_rope kernel: f32 rope rotation + exact interval attention."""

import jax
import jax.numpy as jnp
import numpy as np

LOCAL_FRAME = 1800
N_FRAMES = 22
WIN_LEFT = 2
WIN_RIGHT = 3
ADD_LAST = True

# The production patch grid at tp=8/cp=2: (F, H, W_global) = (22, 45, 80), width split
# over 2 ranks. This card models rank 0; keys are the rank-major gather over both.
GRID_F, GRID_H, GRID_W = 22, 45, 80
SHARDS = 2
THETA = 10000.0


def _window_frames(f):
    frames = list(range(max(0, f - WIN_LEFT), min(N_FRAMES, f + WIN_RIGHT)))
    if ADD_LAST and f + WIN_RIGHT < N_FRAMES:
        frames.append(N_FRAMES - 1)
    return sorted(set(frames))


def _tables(D):
    """cos2/sin2 [S, D] for this rank's q shard and for the rank-major gathered keys.

    Mirrors the model's rope: one theta^(-2c/D) frequency ladder over D/2 channels,
    split [t, h, w] = [22, 21, 21], the angle of a token at (f, y, x) being the
    concatenation of f, y and the global x against their channel groups. Each rank's
    x runs over its own width slice, so the kv table is the rank-major concatenation.
    """
    c = D // 2
    inv = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)  # [c]
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
        return np.repeat(a, 2, axis=-1).astype(np.float32)  # [S, D], pairs duplicated

    return (
        jnp.asarray(dup(np.cos(q_ang))),
        jnp.asarray(dup(np.sin(q_ang))),
        jnp.asarray(dup(np.cos(kv_ang))),
        jnp.asarray(dup(np.sin(kv_ang))),
    )


def _make_test_data(H=5, SQ=39600, SKV=79200, D=128):
    """Unrotated f32 q/k/v plus the four rotation tables. Identical in ref and template."""
    assert SQ == GRID_F * GRID_H * GRID_W // SHARDS and SKV == SQ * SHARDS
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(kq, (H, SQ, D), dtype=jnp.float32) * 0.5
    k = jax.random.normal(kk, (H, SKV, D), dtype=jnp.float32) * 0.5
    v = jax.random.normal(kv, (H, SKV, D), dtype=jnp.float32) * 0.5
    return (q, k, v, *_tables(D))


def _rotate(x, cos2, sin2):
    """Interleaved-pair rotation in f32: (x0+ix1)(cos+isin), pairs on the last axis."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos2[None, :, 0::2], sin2[None, :, 0::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return jnp.stack((r0, r1), axis=-1).reshape(x.shape)


def make_inputs(H=5, SQ=39600, SKV=79200, D=128):
    return _make_test_data(H, SQ, SKV, D)


def timed_compute(q, k, v, qcos2, qsin2, kcos2, ksin2):
    H, SQ, D = q.shape
    SKV = k.shape[1]
    shards = SKV // SQ
    scale = 1.0 / (D**0.5)
    q = _rotate(q, qcos2, qsin2)
    k = _rotate(k, kcos2, ksin2)
    outs = []
    for f in range(N_FRAMES):
        qf = q[:, f * LOCAL_FRAME : (f + 1) * LOCAL_FRAME]
        sel = [
            slice(s * SQ + kf * LOCAL_FRAME, s * SQ + (kf + 1) * LOCAL_FRAME)
            for s in range(shards)
            for kf in _window_frames(f)
        ]
        k_sel = jnp.concatenate([k[:, sl] for sl in sel], axis=1)
        v_sel = jnp.concatenate([v[:, sl] for sl in sel], axis=1)
        scores = jnp.einsum("hqd,hkd->hqk", qf, k_sel) * scale
        p = jax.nn.softmax(scores, axis=-1)
        outs.append(jnp.einsum("hqk,hkd->hqd", p, v_sel))
    return jnp.concatenate(outs, axis=1)


def simple_compute(H=5, SQ=39600, SKV=79200, D=128):
    return timed_compute(*make_inputs(H, SQ, SKV, D))


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
