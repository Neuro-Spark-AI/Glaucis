## Profile Brief for Round 0 (wan_splash_mhpt @ [5,8192,128])

### Source
- Kernel: production splash MHA mhpt (hpt=5), flash online-softmax, bf16 QKᵀ + f32 PV, base-2 exp.
- Speedup 2.13x vs query-blocked jnp reference | kern 1.10 ms / ref 2.35 ms.
- Block config: bq=1024, bkv=2048, bkv_compute=512, bkv_compute_in=256, vmem_limit=96MiB.

### Deep profile
| Metric | Value | Assessment |
|---|---|---|
| VLIW bundles | 454888 | very high (unrolled KV×head loops) |
| MXU dual_ratio | 0.0 (mxu0=7680, mxu1=0) | 2nd MXU idle |
| Avg ops/bundle | 1.31 | low ILP |
| VMEM utilization | 8.06% (10.8 MB / 128 MB) | large headroom — blocks can grow |
| DMA count | 23480 | double_buffering: FALSE — K/V not overlapped with compute |
| EUP (exp) ops | 21120 | exp2-heavy softmax |
| HBM | negligible | not HBM-bound |

### Bottleneck
Compute/latency-bound on a single MXU with heavy DMA count and no double-buffering; VMEM is
88% idle. The kernel does many small chunks (bkv_compute=512) → many DMAs + bundles + low ILP.

### Optimization priorities
1. **Grow bkv_compute (512→1024)**: VMEM has 92% headroom; larger MXU chunks = more compute per
   DMA, fewer chunk iterations/bundles. Source's own default hint: "1024 for massive MXU throughput".
2. **Larger KV/Q blocks (bkv 2048→4096, bq 1024→2048)**: fewer grid steps → fewer DMAs (23480 now).
3. **Double-buffer K/V**: overlap the next block's HBM→VMEM DMA with current MXU compute.
4. **VPU inner tile (bkv_compute_in 256→512)**: fewer inner iterations → fewer bundles, higher ILP.

### What NOT to try
- Shrinking blocks (VMEM 8%, no spills — nothing to relieve).
- HBM-bandwidth tricks (HBM negligible).
