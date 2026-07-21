# wan_splash_mhpt 优化结果:head-batched 3D dot(dual_mxu)

pallas-evolve 在 Wan2.2 DiT splash MHA(mhpt, hpt=5)单卡提取版上找到的加速。赢家 kernel 见同目录 `wan_splash_mhpt.py`。

## 数字

| shape | baseline kern | winner kern | winner / baseline |
|---|---|---|---|
| [5, 8192, 128] | 1.10 ms (2.13x vs jnp) | 1.013 ms (2.327x) | +9% |
| [5, 86016, 128](1080p 每卡) | 58.48 ms (4.76x vs jnp) | 51.84 ms (5.37x) | +12.8% |

correctness 在 rtol/atol 2e-2 内通过(bf16 QKᵀ + f32 PV vs query-blocked f32 参考)。

## 改了什么

只动 `_flash_attention_kernel_mhpt` 内核体。baseline 用 `for h_local in range(heads_per_tile)` 串行发每个头的两个 2D `lax.dot_general`(QKᵀ、PV)。赢家把 head-tile 折成一个带 head batch 维的 3D `dot_general`,两个矩阵乘各一次:

- QKᵀ:`dot_general(k[hpt,kv,d], q[hpt,bq,d], (((2,),(2,)),((0,),(0,))))` → `[hpt, kv, bq]`,contract head_dim、batch head。
- PV:`dot_general(v[hpt,kv,d], s[hpt,kv,bq], (((1,),(1,)),((0,),(0,))))` → `[hpt, head_dim, bq]`,contract kv、batch head。

online-softmax 的 V1 VPU register tiling 内层循环保留,只是所有 reduce/broadcast 多带一个 head 前导轴(`axis=0→1`、`[None,:]→[:,None,:]` 等)。block 尺寸、bf16/f32 精度、base-2 exp 都不变。

## 机制(诚实版)

不是"激活了闲置的第二个 MXU"——profiler 修好后确认 baseline 两个 MXU 就已均衡(dual_ratio ~1.0)。加速来自:一个 batched 大矩阵乘比 5 个串行小矩阵乘把 MXU 喂得更连续、per-head 调度开销更少(occupancy/流水)。大 S 更 MXU-bound,所以 +9%(8192)在生产尺寸涨到 +12.8%(86016)。

## 怎么搬回生产 kernel

生产 kernel `flash_attention_kernel_mhpt`(maxdiffusion `kernels/splash_attention/splash_attention_kernel.py`,以及镜像 `custom_splash_attention.py`)的内核体同样是 `for h_local in range(heads_per_tile)` 串行两个 2D dot。把那段换成上面两个 head-batched 3D `dot_general`、并让 m/l/o 的 online-softmax bookkeeping 多带 head 前导轴,即可移植这个 +12.8%。生产版还有 mask/segment_ids/ring 残差,这些不受影响(逐头数学不变,只是 batch 起来)。

## 探过但没用的(避免重复)

- block 尺寸(bq/bkv/bkv_compute):已 e2e 最优,加大反而慢。
- micro-opt(减 cast、合并 exp/rescale、内层 tile 变化):±1% 噪声,无净收益。
- K/V 双缓冲(emit_pipeline、make_async_copy 两种):在 jax 0.11 这个 mhpt kernel 上都编译/正确性失败。
- 唯一未开采的硬杠杆:head_dim=128 < 256,QK contraction 只填 MXU 阵列一半;要填满得跨头拼 head_dim,会改变数学,未做。
