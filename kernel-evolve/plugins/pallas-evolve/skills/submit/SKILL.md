---
name: submit
description: Use when submitting a batch of Pallas kernel variants for TPU evaluation over SSH — runs evaluate.py on a TPU-VM (e.g. v6e-16), collects all EVAL_RESULTs, scp's artifacts back
---

# Submit Batch of Kernel Variants for TPU Evaluation (SSH)

Submit a batch of kernel variants for evaluation on a TPU-VM over SSH. Builds a
combined payload with all variants, stages it on the VM, runs
`kernel-evolve/docker/evaluate.py` there (which evaluates variants serially in
subprocess isolation), collects per-variant results, and scp's IR/trace
artifacts back. No GKE / kubectl / GCS.

## Context

Invoked by `pallas-evolve:start` during the optimization loop, or standalone for
debugging. It expects:
- A run directory with the current iteration's `variants/` subdirectory containing one or more variant kernels
- A config YAML already loaded in context with an `evaluator.type: ssh` block and a `roofline` block

Required config fields (under `evaluator`): `ssh_host`, `ssh_user`, `ssh_key`,
`ssh_remote_repo`, `ssh_python`, `ssh_remote_tmp`, `ssh_artifacts_dir`.
The VM must have the repo checked out at `ssh_remote_repo` with kernel-evolve
installed into `ssh_python`'s env (`uv pip install -e kernel-evolve/`), and a
working TPU runtime.

Define these shell vars from the config before starting:

```bash
SSH_HOST="{evaluator.ssh_host}"
SSH_USER="{evaluator.ssh_user}"
SSH_KEY="{evaluator.ssh_key}"
REMOTE_REPO="{evaluator.ssh_remote_repo}"
REMOTE_PY="{evaluator.ssh_python}"
REMOTE_TMP="{evaluator.ssh_remote_tmp}"
REMOTE_ART="{evaluator.ssh_artifacts_dir}"
PEAK_FLOPS="{roofline.peak_flops}"               # v6e default 918e12
PEAK_HBM_BW="{roofline.peak_hbm_bw}"             # v6e default 1759e9
HBM_CAPACITY_GB="{roofline.hbm_capacity_gb}"     # v6e default 32
VMEM_CAPACITY_MIB="{roofline.vmem_capacity_mib}" # v6e default 128
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ServerAliveInterval=30"
SSH="ssh -i ${SSH_KEY} ${SSH_OPTS} ${SSH_USER}@${SSH_HOST}"
SCP="scp -r -i ${SSH_KEY} ${SSH_OPTS}"
```

## Procedure

### Step 1: Locate files

Find the current iteration directory (the latest `iteration_{N}/` in the run dir). Identify:
- ALL variant kernel files: `iteration_{N}/variants/*/kernel.py`
- The reference kernel from the config's `kernel.reference` path (relative to config dir)
- The config's `shapes`, `correctness.rtol`, `correctness.atol`

Count the number of variants (`N_VARIANTS`). If zero, stop with an error — nothing to submit.

### Step 2: Construct batch payload

Run this Python script via Bash to build the base64-encoded batch payload:

```bash
python3 -c "
import json, base64, sys, glob, os

reference_code = open(sys.argv[1]).read()
shapes = json.loads(sys.argv[2])
rtol = float(sys.argv[3])
atol = float(sys.argv[4])
variant_dir = sys.argv[5]

variants = []
for vdir in sorted(glob.glob(os.path.join(variant_dir, '*/kernel.py'))):
    variant_id = os.path.basename(os.path.dirname(vdir))
    kernel_code = open(vdir).read()
    variants.append({'variant_id': variant_id, 'kernel_code': kernel_code})

payload = json.dumps({
    'batch': True,
    'reference_code': reference_code,
    'shapes': shapes,
    'rtol': rtol,
    'atol': atol,
    'variants': variants
})

print(base64.b64encode(payload.encode()).decode())
" \
  "<path/to/reference_kernel.py>" \
  '<shapes_json_array>' \
  "{rtol}" \
  "{atol}" \
  "<path/to/iteration_N/variants>" > /tmp/batch_payload.b64
```

The base64 payload is now in `/tmp/batch_payload.b64`. (Writing to a file avoids
huge argv/env strings; there is no ConfigMap 900KB limit on the SSH path.)

### Step 3: Generate job name

```bash
KERNEL_NAME_SLUG=$(echo "{kernel_name}" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-' | sed 's/--*/-/g; s/^-//; s/-$//')
TIMESTAMP=$(date +%m%d-%H%M%S)
JOB_NAME=$(echo "${KERNEL_NAME_SLUG}-iter${N}-${TIMESTAMP}" | cut -c1-63 | sed 's/-$//')
REMOTE_PAYLOAD="${REMOTE_TMP}/${JOB_NAME}.payload"
REMOTE_ART_RUN="${REMOTE_ART}/${JOB_NAME}"
```

### Step 4: Stage payload on the VM

```bash
$SSH "mkdir -p ${REMOTE_TMP} ${REMOTE_ART_RUN}" && \
cat /tmp/batch_payload.b64 | $SSH "cat > ${REMOTE_PAYLOAD}" && \
echo "staged payload -> ${SSH_USER}@${SSH_HOST}:${REMOTE_PAYLOAD}"
```

If staging fails, save synthetic error results for ALL variants and stop.

### Step 5: Run evaluate.py on the VM

`evaluate.py` sets its own IR-dump flags (region trace + LLO debug info) and
preserves existing env. Pass the v6e roofline peaks and `ARTIFACTS_LOCAL_DIR` so
each variant writes LLO/HLO/trace under `${REMOTE_ART_RUN}/<variant_id>/`.

```bash
TIMEOUT=$((300 * N_VARIANTS + 300))
$SSH "cd ${REMOTE_REPO}/kernel-evolve && \
  JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true \
  PEAK_FLOPS=${PEAK_FLOPS} PEAK_HBM_BW=${PEAK_HBM_BW} \
  HBM_CAPACITY_GB=${HBM_CAPACITY_GB} VMEM_CAPACITY_MIB=${VMEM_CAPACITY_MIB} \
  ARTIFACTS_LOCAL_DIR=${REMOTE_ART_RUN} \
  ${REMOTE_PY} docker/evaluate.py --eval-payload-file ${REMOTE_PAYLOAD}" \
  > /tmp/${JOB_NAME}.log 2>&1
echo "exit=$?"
```

Use a Bash tool call with `timeout: {TIMEOUT * 1000 + 30000}` (ms, plus buffer).
Even on non-zero exit, proceed to Step 6 — some variants may have printed
`EVAL_RESULT:` before a later one failed.

### Step 6: Collect and parse results

Scan `/tmp/${JOB_NAME}.log` for ALL lines containing `EVAL_RESULT:`. Each line is
one variant's result JSON. Distribute to per-variant `eval_result.json`:

```bash
python3 -c "
import json, sys, os

logs = open(sys.argv[1]).read()
variants_dir = sys.argv[2]

results = {}
for line in logs.splitlines():
    if 'EVAL_RESULT:' in line:
        try:
            r = json.loads(line.split('EVAL_RESULT:', 1)[1].strip())
            results[r.get('variant_id', '')] = r
        except json.JSONDecodeError:
            pass

for vdir in [d for d in os.listdir(variants_dir) if os.path.isdir(os.path.join(variants_dir, d))]:
    out = os.path.join(variants_dir, vdir, 'eval_result.json')
    if vdir in results:
        json.dump(results[vdir], open(out, 'w'), indent=2)
        print(f'Saved {vdir}: status={results[vdir].get(\"status\")}')
    else:
        json.dump({'variant_id': vdir, 'status': 'COMPILE_ERROR', 'fitness': 0.0,
                   'error': 'No EVAL_RESULT in remote logs', 'latency_ms': 0.0, 'speedup': 0.0},
                  open(out, 'w'), indent=2)
        print(f'Saved synthetic error for {vdir}')
" "/tmp/${JOB_NAME}.log" "<path/to/iteration_N/variants>"
```

### Step 7: Pull artifacts back (scp)

For each variant whose `eval_result.json` has `metadata.artifacts_local_dir`,
scp the contents of that remote dir into the local variant dir:

```bash
for VARIANT_DIR in iteration_N/variants/*/; do
  VARIANT_NAME=$(basename "$VARIANT_DIR")
  RESULT_FILE="${VARIANT_DIR}eval_result.json"
  [ -f "$RESULT_FILE" ] || continue
  REMOTE_ART_DIR=$(python3 -c "import json,sys; print((json.load(open(sys.argv[1])).get('metadata') or {}).get('artifacts_local_dir',''))" "$RESULT_FILE")
  if [ -n "$REMOTE_ART_DIR" ]; then
    $SCP "${SSH_USER}@${SSH_HOST}:${REMOTE_ART_DIR}/." "${VARIANT_DIR}" 2>/dev/null && \
      echo "pulled artifacts for ${VARIANT_NAME}" || \
      echo "artifact pull skipped for ${VARIANT_NAME}"
  fi
done
```

This downloads up to three files per variant:
- `hlo_post_opt.txt` — post-optimization HLO IR text
- `llo_final.txt` — final-pass LLO IR text with VLIW bundles
- `trace_events.json` — expanded XPlane trace events (Chrome trace format)

These files are optional. If pull fails, the analyze skill falls back to metrics-only analysis.

### Step 8: Cleanup

Always clean up remote temp files and the local log:

```bash
$SSH "rm -rf ${REMOTE_PAYLOAD} ${REMOTE_ART_RUN}"
rm -f /tmp/${JOB_NAME}.log /tmp/batch_payload.b64
```

## Error Handling

- If payload staging (Step 4) fails: save synthetic error results for ALL variants, skip run, clean up.
- If the SSH run (Step 5) times out or errors: still parse partial results from the log (some variants may have succeeded), then clean up.
- If a variant has no matching `EVAL_RESULT:` in the log: save a synthetic COMPILE_ERROR for that variant.
- Partial results are acceptable — some variants may succeed while others fail.
- Always run the cleanup step regardless of outcome.
- SSH host key / auth failures: verify `ssh_key` path and that the current `ssh_host` is the live worker IP (IPs go stale; re-discover with gcloud if needed).

### Step 9: Context note

All evaluation results have been distributed to per-variant `eval_result.json`
files, artifacts pulled, and remote temp files cleaned up.

**Do NOT compact here** — this skill is typically invoked within the
`pallas-evolve:start` loop (Phase 2), and the subsequent Phase 3 (ANALYZE) needs
orchestration context (iteration number, run directory) to locate results. The
start loop manages compaction at Phase 5.

If invoked **standalone** (outside the start loop), invoke `/compact` after this
skill completes — the SSH logs, base64 payload, and run output are no longer
needed in context.
