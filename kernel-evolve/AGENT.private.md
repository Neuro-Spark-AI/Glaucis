# Pallas Kernel Optimization Agent Knowledge (private)

## Failure Patterns

## Successful Optimizations

### [F001] Two-pass blocked softmax numerical/indexing bug (qkv_attention_pallas R1)
- Symptom: variant two_pass_blocked_softmax returns INCORRECT (correctness fail vs jnp).
- Root cause: the explicit two-pass (global row_max then re-accumulate) as written diverges from reference beyond rtol=1e-2; likely a slicing/max-init or renorm error in the second pass.
- Fix: prefer single-pass online softmax (algebraically exact); if two-pass, verify against jnp at small S first.

### [F002] Head-tiling does NOT engage the 2nd MXU (qkv_attention_pallas R1)
- Symptom: head_tile_dual_mxu ran HEADS_PER_TILE heads via a Python for-loop per program, dual_ratio stayed 0.0.
- Root cause: a serial for-loop over heads emits sequential matmuls; Mosaic did not co-issue them on mxu0+mxu1. Looping ≠ dual-MXU.
- Fix: to use both MXUs, need batched/stacked matmuls (e.g. a single (H_tile*S, D) dot) or explicit scheduling, not a serial loop.

### [S-none-R1] No win at S=1024
- flash_online_softmax 0.977x, kv_double_buffer 0.776x, head_tile 0.996x — none beat baseline 0.985x.
- Insight: at S=1024,H=8,D=128 the full (S×S) score matrix already fits VMEM, so flash/double-buffer add loop overhead without a memory win; the pallas attention core is near parity with jnp. Untried high-leverage lever: bf16 MXU operands (f32 inputs cast to bf16 for the two dots) — MXU throughput lever.

### [F003] bf16 MXU operands give no win when not compute-bound (qkv_attention_pallas R2)
- Symptom: casting the attention dots (and projections) to bf16 did not beat f32 (best 0.987x vs baseline 0.985x, vs R1 0.996x).
- Root cause: at B=1,S=1024,H=8,D=128 the op is not MXU-compute-bound — kernel-launch overhead + the two jnp projections (identical on ref and variant) dominate, diluting any matmul-throughput gain. bf16 on the projections added cast overhead for no benefit.
- Fix: apply bf16 only where matmul FLOPs actually dominate (large S / large D / compute-bound shapes). For small-headroom shapes, don't expect bf16 to help; measure the projection vs attention time split first.

### [S001 / F004] wan_splash_mhpt block-size sweep — block sizes are e2e-optimal, not a lever
- R1 @ [5,8192,128]: baseline (bq1024/bkv2048/bkv_compute512) 2.13x. bkv_compute=1024 → 2.27x (+6% in isolation); bkv=4096 → 1.61x, bq=2048 → 1.58x, bkv_compute_in=512 → 2.01x (all worse).
- User confirms: block sizes were already tuned; the production bq=1024/bkv=2048 are e2e-optimal (full DiT pipeline / large S / ring). Isolated-S8192 gains from bigger blocks don't hold e2e.
- Fix: treat block sizes as FIXED at production values. Optimize kernel STRUCTURE instead.
- Real levers from the baseline profile: dual_ratio=0.0 (2nd MXU idle), double_buffering=false (23480 DMAs not overlapped), low ILP 1.31. These are structural, not block-size.

### [S002] Batched 3D dot engages the 2nd MXU on large attention (wan_splash_mhpt R2)
- dual_mxu variant: replaced the per-head `for h_local in range(5)` loop of separate 2D dots with two single 3D `lax.dot_general`s batched over the head axis (QKᵀ and PV). 2.13x → 2.327x (+9%), correctness clean.
- Why it worked here but not in qkv_attention_pallas R2 (where batched dot gave no win): here S=8192, hpt=5, head_dim=128 — the batched matmul is large enough for the compiler to spread per-head slices across mxu0+mxu1; the qkv case (S=1024, tiny) was overhead-bound, not MXU-bound.
- Takeaway: batch independent matmuls over a real batch axis (not a serial loop) to use both MXUs, and only expect a win when the op is genuinely MXU-bound.

### [F005] emit_pipeline KV double-buffer rewrite is correctness-fragile (wan_splash_mhpt R2)
- double_buffer_kv (moved KV loop into pltpu.emit_pipeline, K/V as ANY/HBM refs, m/l/o to scratch) threw at runtime (INCORRECT). The emit_pipeline restructure + accumulator-scratch closure is error-prone; needs careful shape/index-map work and CPU-shape validation before TPU.
