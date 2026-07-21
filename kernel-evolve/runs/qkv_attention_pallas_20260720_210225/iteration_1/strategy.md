## Round 1 Strategy

Generating 4 variants from baseline, each a different technical direction derived
from the baseline profile (memory-bound compute_ratio 0.49, dual_ratio 0.0, VMEM
3.9%, full S×S score materialization, double_buffering false). Variants generated in
parallel via sub-agents.

### Variant: flash_online_softmax
**Technical direction**: flash online-softmax (single-pass)
**Profile motivation**: memory-bound + materializes full 1024×1024 score matrix (26095 bundles)
**Approach**: keep Q whole, loop KV in block_kv=256 tiles, carry running (max, denom, acc); never form full score matrix
**Target metric**: lower VLIW bundle count + memory traffic; push past 0.985x

### Variant: kv_double_buffer
**Technical direction**: pipeline K/V movement (double buffering)
**Profile motivation**: double_buffering=false, 1792 DMAs not overlapped with compute
**Approach**: grid (bh, num_kv) with per-block K/V BlockSpec advancing along the pipelined dim; online-softmax carry in persistent VMEM scratch; Pallas built-in prefetch hides DMA
**Target metric**: double_buffering=true; DMA hidden behind MXU compute

### Variant: two_pass_blocked_softmax
**Technical direction**: classic two-pass blocked softmax
**Profile motivation**: same as flash (full matrix); distinct approach for diversity
**Approach**: pass 1 = global row_max over KV blocks; pass 2 = accumulate denom + acc with exp(scores-row_max); only one S×block_kv tile live
**Target metric**: lower peak score-matrix memory (vs full S×S); trades an extra K re-read

### Variant: head_tile_dual_mxu
**Technical direction**: head-tile to fill both MXUs / raise ILP
**Profile motivation**: dual_ratio 0.0 (only mxu0), avg ops/bundle 1.21
**Approach**: grid (bh//HEADS_PER_TILE,) with HEADS_PER_TILE=2 heads per program; loop over heads so independent per-head matmuls co-issue
**Target metric**: dual_ratio > 0, ops/bundle up
