# Running pallas-evolve on a TPU-VM (v6e) over SSH

This fork adds an SSH evaluation path so the plugin runs on a Cloud TPU-VM
(e.g. a v6e-16 slice) without any GKE cluster, and defaults the roofline metrics
to TPU v6e (Trillium).

## What changed vs upstream

- `src/kernel_evolve/ssh_evaluator.py` — `SSHEvaluator` (parallel to `KubeEvaluator`).
  Stages the base64 payload on the VM, runs `docker/evaluate.py` there, parses
  `EVAL_RESULT:` lines, and scp's artifacts back. No kubectl / ConfigMap / GCS.
- `docker/evaluate.py` — roofline peaks default to v6e and are env-overridable:
  `PEAK_FLOPS` (918e12), `PEAK_HBM_BW` (1759e9), `HBM_CAPACITY_GB` (32),
  `VMEM_CAPACITY_MIB` (128). Also honors `ARTIFACTS_LOCAL_DIR` to copy
  LLO/HLO/trace to a local dir per variant (so SSH can scp them back).
- `src/kernel_evolve/profiler.py` — `compute_derived_metrics` defaults to v6e peaks.
- `src/kernel_evolve/config.py` — `evaluator.type` (`kube` | `ssh`) + `ssh_*`
  fields; a `roofline` block; `tpu` is now optional.
- Skills (`submit`, `start`, `profile-brief`, `analyze`, `init-kernel`) retargeted
  from GKE/v7x to TPU-VM/SSH/v6e.

## VM prerequisites (once)

On the TPU-VM (as `cloud-user`):

```bash
git clone https://github.com/edgexyz/Glaucis.git /home/cloud-user/Glaucis
# install kernel-evolve into the tpu-venv that has jax/libtpu:
/home/cloud-user/tpu-venv/bin/pip install -e /home/cloud-user/Glaucis/kernel-evolve
# sanity: the VM must see the TPU
/home/cloud-user/tpu-venv/bin/python -c "import jax; print(jax.devices())"
```

## Config

Use `evaluator.type: ssh` + a `roofline` block. See
`examples/matmul_v6e_ssh.yaml`. Fill in the current worker IP (they go stale —
re-discover with gcloud) and your SSH key path.

## Run

- `/pallas-evolve:init-kernel <kernel>` — generates a v6e SSH config + baseline.
- `/pallas-evolve:start <config.yaml>` — the optimization loop (SSH connectivity
  check → think → submit over SSH → analyze → reflect).
- Standalone: `/pallas-evolve:submit` (one round) then `/pallas-evolve:profile-brief <dir>`.

## Roofline peaks (v6e defaults, override if needed)

| Env / config | v6e default | Meaning |
|---|---|---|
| `PEAK_FLOPS` / `roofline.peak_flops` | 918e12 | bf16 TFLOP/s per chip |
| `PEAK_HBM_BW` / `roofline.peak_hbm_bw` | 1759e9 | HBM GB/s per chip |
| `HBM_CAPACITY_GB` / `roofline.hbm_capacity_gb` | 32 | HBM per chip |
| `VMEM_CAPACITY_MIB` / `roofline.vmem_capacity_mib` | 128 | physical VMEM per chip |

## Known caveats

- **VMEM capacity (128 MiB) is a default** — confirm the exact v6e physical VMEM
  and override `vmem_capacity_mib` if it differs. Only the `vmem_utilization_pct`
  figure depends on it; raw counts are unaffected.
- **LLO text-format parsing** in `docker/evaluate.py` (`stage_profile_deep`) was
  written against v7x MLIR LLO. On v6e the LLO dump format may differ; if the
  deep-profile metrics come back empty on the first run, the parser needs a v6e
  format branch. Raw benchmark latency/speedup and correctness are unaffected.
- The GCS upload path is retained for the kube transport but is a no-op on the
  SSH path (artifacts travel by scp).
