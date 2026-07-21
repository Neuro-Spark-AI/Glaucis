## Round 1 Strategy (wan_splash_mhpt, block-size sweep)

Profile: single-MXU (dual 0), no double-buffer (23480 DMAs), VMEM 8% (headroom), low ILP 1.31.
4 variants tune block sizes to use the idle VMEM for more MXU work per DMA:

- bkv_compute_1024: BKV_COMPUTE 512→1024 (source hint "1024 for massive MXU throughput").
- bkv4096_compute1024: BKV 2048→4096 + BKV_COMPUTE 1024 (fewer grid steps/DMAs).
- bq_2048: BQ 1024→2048 (larger query block, more K/V reuse per program).
- bkv_compute_in_512: BKV_COMPUTE_IN 256→512 (fewer VPU inner iterations, higher ILP).
