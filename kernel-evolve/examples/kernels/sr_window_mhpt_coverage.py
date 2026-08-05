"""Host-side grid coverage check for the sr_window_mhpt slot grid.

The correctness stage compares against an f32 reference on random normal data, which
cannot see a dropped key frame: 1800 missing keys out of 21600 move the softmax average
by ~1e-3, well under atol. This walks the variant's own `_slot_state` predicate with
numpy instead of jnp and asserts that every q block runs exactly the (shard, frame) pairs
its rows are allowed to attend to. Run it on any variant that touches the slot
arithmetic; needs jax importable but no TPU.
"""

import argparse
import importlib.util
import sys

import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location("variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(m, sq=39600, skv=79200):
    frame, nf = m.LOCAL_FRAME, m.N_FRAMES
    left, right, last = m.WIN_LEFT, m.WIN_RIGHT, m.ADD_LAST
    bq, slots = m.BQ, m.SLOTS
    shards = skv // sq

    def allowed(qf):
        fs = set(range(max(0, qf - left), min(nf, qf + right)))
        if last and qf + right < nf:
            fs.add(nf - 1)
        return fs

    gh, gw = (sq + bq - 1) // bq, shards * slots
    bad = []
    for i in range(gh):
        lo, hi = i * bq, min(i * bq + bq, sq)
        if lo >= sq:
            continue
        need_f = set()
        for qf in range(lo // frame, (hi - 1) // frame + 1):
            need_f |= allowed(qf)
        need = {(s, f) for s in range(shards) for f in need_f}
        got, dups = set(), []
        for j in range(gw):
            shard = j // slots
            # The variant's own predicate, walked with numpy in place of jnp.
            kf, run, full = m._slot_state(i, j, bq, np)  # noqa: SLF001
            if not bool(run):
                continue
            if (shard, int(kf)) in got:
                dups.append((j, shard, int(kf)))
            got.add((shard, int(kf)))
            # A block declared fully allowed must really be allowed for every q row.
            if bool(full):
                rows = range(lo // frame, (hi - 1) // frame + 1)
                if any(int(kf) not in allowed(qf) for qf in rows):
                    dups.append((j, shard, f"full-but-masked f{int(kf)}"))
        if need - got or got - need or dups:
            bad.append((i, lo // frame, (hi - 1) // frame, sorted(need - got), sorted(got - need), dups))
    return gh, gw, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", nargs="?", default="kernel-evolve/examples/kernels/sr_window_mhpt_champion.py")
    a = ap.parse_args()
    gh, gw, bad = check(load(a.kernel))
    print(f"{a.kernel}: {gh} q blocks x grid width {gw}")
    if not bad:
        print("COVERAGE OK")
        sys.exit(0)
    print(f"COVERAGE BROKEN in {len(bad)}/{gh} blocks:")
    for i, f0, f1, missing, extra, dups in bad:
        print(f"  block {i} qframes {f0}-{f1}: missing={missing} extra={extra} dup={dups}")
    sys.exit(1)
