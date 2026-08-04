# Reference for the Wan2.2 SR windowed sparse self-attention, single card, f32.
# Geometry from avatar-turbo-edge-repro at tp=8/cp=2: q is this rank's shard
# [H=5, SQ=39600, D=128], k/v are the cp-gathered sequence [H, SKV=79200, D] laid out
# rank-major (SHARDS blocks of SQ, each holding all N_FRAMES frames of LOCAL_FRAME
# tokens). A query in frame f attends to frames [f-2, f+3) clipped, plus the last
# frame when f+3 < N_FRAMES; the shard a key came from does not enter the window test.
"""Reference sr_window_mhpt kernel: exact f32 attention over the mask's intervals.

Mirrors heygenai's pure-PyTorch sparse_attn: for each q frame, concatenate its
allowed key intervals (every window frame appears once per shard) and run one exact
softmax attention. Used as the correctness baseline for evolutionary optimization.
"""

import jax
import jax.numpy as jnp

LOCAL_FRAME = 1800
N_FRAMES = 22
WIN_LEFT = 2
WIN_RIGHT = 3
ADD_LAST = True


def _window_frames(f):
    frames = list(range(max(0, f - WIN_LEFT), min(N_FRAMES, f + WIN_RIGHT)))
    if ADD_LAST and f + WIN_RIGHT < N_FRAMES:
        frames.append(N_FRAMES - 1)
    return sorted(set(frames))


def _make_test_data(H=5, SQ=39600, SKV=79200, D=128):
    """Deterministic f32 q [H,SQ,D], k/v [H,SKV,D]. Identical in ref and template."""
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(kq, (H, SQ, D), dtype=jnp.float32) * 0.5
    k = jax.random.normal(kk, (H, SKV, D), dtype=jnp.float32) * 0.5
    v = jax.random.normal(kv, (H, SKV, D), dtype=jnp.float32) * 0.5
    return q, k, v


def simple_compute(H=5, SQ=39600, SKV=79200, D=128):
    q, k, v = _make_test_data(H, SQ, SKV, D)
    shards = SKV // SQ
    scale = 1.0 / (D**0.5)
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


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
