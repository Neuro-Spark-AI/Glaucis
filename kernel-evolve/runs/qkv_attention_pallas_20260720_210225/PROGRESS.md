# Progress: qkv_attention_pallas evolution (paused 2026-07-21)

Repo now **private + self-contained** (`edgexyz/Glaucis` made PRIVATE, issues
enabled). Tracking Issue #1 exists. No separate pallas-kernels repo referenced
anymore (that repo still exists at edgexyz/pallas-kernels but unused).

## Run state
`kernel-evolve/runs/qkv_attention_pallas_20260720_210225/`
- `baseline/` + `iteration_1/` committed (commit f58ff07).
- Round 1 THINK done: 4 variants in `iteration_1/variants/*/kernel.py`
  (flash_online_softmax, kv_double_buffer, two_pass_blocked_softmax,
  head_tile_dual_mxu). All syntax-OK, frozen regions intact (only comments/
  docstrings trimmed by sub-agents — harmless, verified via AST on the
  functional parts).

## Phase 2 SUBMIT — NOT complete. Two infra bugs found (both diagnosed):

1. **Disk fill.** evaluate.py writes IR dump to `/tmp/ir_dumps/{variant}` and
   NEVER cleans up. ~15 GB/variant → a 4-variant round hits ~60 GB and the
   small VM root disk (97 GB, 3 hosts at 100%) fills. Symptom: late variants
   abort `RESOURCE_EXHAUSTED: No space left`; primary can hang to its timeout.
   **Fixed** in evaluate.py: added `EVAL_DISABLE_DUMPS=1` (skip dump flags entirely)
   and `EVAL_DUMP_DIR=<path>` (relocate dumps). Patched file synced to all 4 VMs
   (commit pending on Glaucis main — NOT yet pushed).

2. **Dump-to-/dev/shm HANGS.** VM has RAM-backed tmpfs: `/dev/shm` 355 G free,
   `/mnt/wram` 113 G free (both writable). Routing the IR dump to
   `/dev/shm/ir_dumps` makes all 4 hosts **silently hang** at libtpu init
   (0 bytes of output, timeout) — XLA's dump subsystem deadlocks on tmpfs.
   `/mnt/wram` untested but likely same. **Workaround that WORKS: dumps OFF
   (`EVAL_DISABLE_DUMPS=1`).** With dumps off, a 1-variant eval on all 4 hosts
   completes in ~36 s. Deep profile then returns ok:false (no files), but
   correctness + benchmark are intact.

## Benchmark trustworthiness (IMPORTANT for analysis)
- Multi-host co-launch: only **jax process 0 = IP 34.162.213.160** (script's
  `IPS[0]`, log `r1*_w0.log`) has a clean benchmark. The other 3 hosts are
  **noisy** — on the same kernel their `evaluation_times_ms` swings wildly
  (0.0005–0.87 ms, CV > 50%) because the single-device kernel only pins one of
  their 4 chips and the rest is free for interference. ALWAYS read results from
  the primary log, never aggregate across hosts.
- Reference latency is stable (~0.188 ms) across hosts when the primary is clean
  (earlier 0.062-vs-0.189 drift was from the disk-full hang corrupting timing).

## What a clean run looked like (1-variant, dumps off, mini test)
Primary (proc 0): baseline kernel = **0.964x** vs jnp (kern 0.195 ms, ref 0.188 ms,
CV 0.1%). So at S=1024 the non-flash Pallas baseline ≈ jnp (the 0.985x from the
committed baseline eval is consistent within noise).

## Next step (resume here)
Run the **4-variant** eval with `EVAL_DISABLE_DUMPS=1`, primary-trusted. The
attempt was launched but the harness tool call errored mid-flight; remote procs
were killed and cleaned. Re-launch the same 4-variant payload (already staged as
`/tmp/.../scratchpad/r1.payload`) on all 4 hosts with dumps off, parse only
`r1*_w0.log`, then proceed to Phase 3 ANALYZE / Phase 4 REFLECT.

Suggested launcher: `/tmp/.../scratchpad/r1_final.sh` but change
`EVAL_DUMP_DIR=/dev/shm/ir_dumps` → `EVAL_DISABLE_DUMPS=1` (remove the shm env).
A clean script is straightforward from the mini-test pattern.

## Uncommitted code changes (Glaucis working tree)
- `docker/evaluate.py`: + `EVAL_DISABLE_DUMPS` guard in `_setup_dump_env`,
  + `EVAL_DUMP_DIR` override in `_get_dump_dir`. NOT yet committed/pushed.
- VMs have this patched evaluate.py (manually scp'd); Glaucis main does not.
