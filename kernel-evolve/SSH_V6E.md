# Running pallas-evolve on a TPU-VM (v6e) over SSH

This fork adds an SSH evaluation path so the plugin runs on a Cloud TPU-VM
(e.g. a v6e-16 slice) without any GKE cluster, and defaults the roofline metrics
to TPU v6e (Trillium).

## Multi-host slices (v6e-16 etc.)

A v6e-16 is **4 hosts** (4 chips each) sharing one ICI domain. libtpu does a
slice-wide init barrier, so `import jax; jax.devices()` on a single host **hangs
forever** waiting for the other three. Every TPU program — including
`evaluate.py` and any connectivity check — must be **co-launched on all hosts at
once**. This was validated end-to-end on a live `v6e-16` (device kind
`TPU v6 lite`, 16 global / 4 local devices per host).

The evaluator handles this by co-launching `evaluate.py` on all hosts and
reading results + artifacts from the **first** host (the primary). The kernel
template is single-device, so each host independently runs the same kernel on
its local chip and prints an identical `EVAL_RESULT:`; the other hosts run only
to satisfy the barrier. List all worker IPs under `evaluator.ssh_hosts`
(the first is the primary). Single-host slices (v6e-1/-4/-8) use `ssh_host`.

## What changed vs upstream

- `src/kernel_evolve/ssh_evaluator.py` — `SSHEvaluator` (parallel to `KubeEvaluator`).
  Stages the base64 payload on every host, **co-launches** `docker/evaluate.py`
  on all hosts of the slice (barrier-safe), parses `EVAL_RESULT:` from the
  primary, and scp's artifacts back from the primary. No kubectl / ConfigMap / GCS.
- `docker/evaluate.py` — roofline peaks default to v6e and are env-overridable:
  `PEAK_FLOPS` (918e12), `PEAK_HBM_BW` (1759e9), `HBM_CAPACITY_GB` (32),
  `VMEM_CAPACITY_MIB` (128). Also honors `ARTIFACTS_LOCAL_DIR` to copy
  LLO/HLO/trace to a local dir per variant (so SSH can scp them back).
- `src/kernel_evolve/profiler.py` — `compute_derived_metrics` defaults to v6e peaks.
- `src/kernel_evolve/config.py` — `evaluator.type` (`kube` | `ssh`) + `ssh_*`
  fields including `ssh_hosts` (list, for multi-host slices) with a
  `resolved_ssh_hosts()` helper; a `roofline` block; `tpu` is now optional.
- Skills (`submit`, `start`, `profile-brief`, `analyze`, `init-kernel`) retargeted
  from GKE/v7x to TPU-VM/SSH/v6e.

## VM prerequisites (once, on EVERY host)

On a multi-host slice, run this on **all** workers (the checkout + install must
exist on each host, since each runs `evaluate.py` locally). Discover the worker
IPs with:

```bash
gcloud compute tpus tpu-vm describe <name> --zone <zone> \
  --format="value(networkEndpoints[].accessConfig.externalIp)"
```

Then, on each worker (as `cloud-user`):

```bash
git clone https://github.com/edgexyz/Glaucis.git /home/cloud-user/Glaucis
# install kernel-evolve into the tpu-venv that has jax/libtpu (deps: pyyaml,
# pydantic — already present in the tpu-venv; jax/libtpu are NOT touched):
/home/cloud-user/tpu-venv/bin/pip install -e /home/cloud-user/Glaucis/kernel-evolve
```

Sanity check the TPU — **co-launch on all hosts at once** (a single host hangs):

```bash
for H in <ip0> <ip1> <ip2> <ip3>; do
  ssh -i <key> -o BatchMode=yes cloud-user@"$H" \
    "/home/cloud-user/tpu-venv/bin/python -c 'import jax; print(jax.process_index(), jax.devices()[0].device_kind, len(jax.devices()))'" &
done; wait
```

## Config

Use `evaluator.type: ssh` + a `roofline` block. See
`examples/matmul_v6e_ssh.yaml`. Fill in `evaluator.ssh_hosts` with all worker
IPs (they go stale on stop/start — re-discover with gcloud) and your SSH key
path. For a single-host slice use `evaluator.ssh_host` instead.

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

## Confirmed on v6e (measured, 2026-07-20)

- **Physical VMEM = 128 MiB** per chip. Measured by driving a Pallas matmul to
  overflow: XLA reports "Used 131.86M of 127.94M vmem". So `vmem_capacity_mib:
  128` is correct — no override needed.
- **Default scoped VMEM budget is only 32 MiB**, not the full 128. A kernel that
  legitimately needs >32 MiB VMEM OOMs at compile
  (`CompileTimeScopedVmemOom: limit 32.00M exceeded`) even though the chip has
  128. Raise it per-compile with
  `compiler_options={"xla_tpu_scoped_vmem_limit_kib": "131072"}`, or globally via
  `XLA_FLAGS=--xla_tpu_scoped_vmem_limit_kib=131072`. Relevant if variants tile
  large blocks into VMEM.
- **LLO parser works on v6e.** The v6e LLO dump is standard Mosaic MLIR
  (`stable_mosaic.version = 15`), the same shape the `stage_profile_deep` parser
  expects. Deep-profile metrics come back populated (vliw bundle count, MXU
  utilization, VMEM allocation, DMA analysis) — no v6e-specific branch needed.

## Known caveats

- **`trace_events.json` is large** (~80 MB for a single matmul). It is scp'd per
  variant. If bandwidth/disk matters, skip it — `analyze`/`profile-brief` work
  from `llo_final.txt` + `eval_result.json` alone.
- The GCS upload path is retained for the kube transport but is a no-op on the
  SSH path (artifacts travel by scp).
