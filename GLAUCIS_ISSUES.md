# Glaucis / kernel-evolve 问题记录

2026-07-20-21 在这个 fork 上跑 pallas-evolve 的发现。合作方交付的是 GKE+GCS 评测路径;单卡 SSH/v6e 路径是本 fork 自加的新增能力,不算上游问题。第一、二节是 LLO/profile 分析与优化流程层面的问题,第三节是自加路径的设计要点,第四、五节是本次成果与使用意见。

## 一、LLO / profile 分析

### 1.1 profiler 的 MXU per-unit 计数错误,dual_ratio 恒为 0

现象:`eval_result.json` 的 `mxu_utilization.dual_ratio` 一直是 0.0、mxu1=0,看起来第二个 MXU 空闲。据此得出"第二个 MXU 空闲、可作优化项"的判断是错的;实测两个 MXU 都在用、且分配均衡。

位置:`docker/evaluate.py` `stage_profile_deep`,MXU 计数选单个 `best_file` 后 findall `.mxu0`/`.mxu1`。

根因:per-unit 的 `.mxu0`/`.mxu1` 标注只出现在 `auto-mxu-assigner` 之后的 late-finalization LLO pass;parser 选的是 mosaic / 分配前的 pass(没有这些标注),于是回退到数 `vmatmul` 并硬编码 mxu1=0。grep 最终 pass 的 ground truth:单卡 wan kernel 与生产 ring kernel 两个 MXU 都在用(dual_ratio 实测约 0.6-1.0,取决于取哪个 pass)。这个假 0 一度把优化方向带偏(以为要去"激活"第二个 MXU)。

状态:已修。改为在 finalization pass(`.mxuN` 标注最多的那个)上数 MXU/DMA/double-buffering,VMEM 与 bundle density 仍从 mosaic pass 取。修完 parser 报 dual_ratio 非 0。教训:信 post-assigner 的最终 LLO,别信单个分配前 pass。

### 1.2 double_buffering 标记未经核实

现象:`dma_analysis.double_buffering` 长期 false;和 dual_ratio 同类,可能也是选错 pass 的假象。

位置:`docker/evaluate.py` `stage_profile_deep`,`re.search(r"\bsand\.u32\s+1\b", ...)`。

根因:`sand.u32 1` 这个双缓冲判定标记对 v6e LLO 是否成立、是否只在 finalization pass 出现,都没确认。

状态:未定。1.1 的修法已让它改读 finalization pass,但标记本身仍是启发式,需单独核对一个已知双缓冲的 kernel 才能确认。

### 1.3 benchmark 与 deep profile 只用 shapes[0]

现象:config 给多个 shape 时,correctness 遍历所有 shape,但 speedup/latency 与 deep profile 只来自第一个 shape;不知道这点时,调换 shape 顺序会得到看似矛盾的 speedup。

位置:`docker/evaluate.py` `stage_benchmark`(`shape = shapes[0]`)、`stage_profile_deep`(`s = shapes[0]`)。

根因:设计如此,但 config/skill 未说明。

状态:未改(行为合理)。用单 shape,或明确 shapes[0] 是被测项。建议在文档标注。

### 1.4 大 seq 缺内存安全的参考实现

现象:注意力这类 O(S²) kernel,seq 大(37888/86016)时朴素 einsum 参考会 materialize `[H,S,S]` 分数矩阵(TB 级)OOM;框架没有提供内存安全的参考模板。

位置:kernel-evolve 的 reference 约定(`simple_compute`)未涉及分块。

状态:本次手写 query-blocked 精确注意力参考(`wan_splash_mhpt_ref.py`)绕过。可作为一个可复用参考模式沉淀。

## 二、优化流程

### 2.1 init-kernel 只认 primatrix/pallas-kernel 布局

现象:`init-kernel` 只能从 `primatrix/pallas-kernel` 的 `tops/ops/<kernel>/` 多文件布局导入;对已经是单文件、或来自别的 repo(maxdiffusion)的 kernel 不适用,得手搓 ref/template/yaml。

位置:skill `init-kernel` Step 2/2.5/3(依赖 `tops/ops/$KERNEL_NAME/` 与逐文件 tracing)。

状态:未改。本次两个 kernel 都是手搓三件套。可加一个 `--from-file` 轻量 ingest 路径处理单文件 kernel。

### 2.2 IR dump 无清理、无上限

现象:一轮多个变体后,dump 累积占满磁盘,靠后的变体因满盘报 COMPILE_ERROR。

位置:`docker/evaluate.py` `_setup_dump_env`/`_get_dump_dir`,dump 目录 `/tmp/ir_dumps/{variant}`。

根因:每变体 dump HLO+Mosaic+LLO 约 15 GB,评完从不删除。GKE 每变体一个 pod、盘独立,这个问题被掩盖;单机连续跑才暴露。

状态:部分修(自加 SSH 路径侧)。加了 `EVAL_DISABLE_DUMPS=1` 与 `EVAL_DUMP_DIR=<path>` 两个开关。根本修法(每变体评完即删、或限制文件数)未做。

### 2.3 跟踪(Issue/PR/AGENT.md)默认落 git origin

现象:`start`/`reflect` 的 `gh` 调用不带 `--repo`,默认作用在当前 checkout 的 origin;引擎在公开 fork、kernel 私有时,变体代码、结果、优化后 kernel、AGENT.md 会被写进公开 fork。

位置:skills `start`(建 Issue、结束建 PR)、`reflect`(评论、提交 AGENT.md)。

状态:已修。config 加 `session.tracking_repo` 与 `session.agent_md_path`;skill 相应加 `--repo`,并把提交 AGENT.md 限定在拥有该文件的 repo,结束时禁止把优化 kernel push 到公开 fork。

### 2.4 GCS 桶名硬编码为上游的 glaucis-profiles

现象:GKE 路径上传 profile artifacts 用固定桶 `glaucis-profiles`,本账号无权限、桶不属于本项目。

位置:`docker/evaluate.py` `upload_to_gcs(bucket_name="glaucis-profiles")`。

状态:已修。加 `GCS_BUCKET` 环境覆盖。

### 2.5 kube job 模板停留在 v7x,且运行时公开 clone

现象:`.github/ci/kernel-eval-job.yaml` nodeSelector 是 `tpu7x`/topology `2x2x1`(v7x),不是 v6e;pod 启动时 `pip install jax` + `git clone https://github.com/$REPO`(仓库私有会失败)。

位置:`.github/ci/kernel-eval-job.yaml`。

状态:未改。本次走 GKE 时改用预制 kernel_evolve 的镜像 + v6e node label 绕过运行时 clone;模板本身待更新。

### 2.6 config 只暴露物理 VMEM,不暴露 scoped VMEM 上限

现象:变体用到超过 32 MiB VMEM 时编译期报 `CompileTimeScopedVmemOom: limit 32.00M`,即便物理 VMEM 有 128 MiB;config 没有调这个上限的入口。

位置:`config.py` `roofline.vmem_capacity_mib`(=物理 128)与 evaluate.py 的编译参数。

根因:默认 scoped vmem 预算 32 MiB;抬高需要 `xla_tpu_scoped_vmem_limit_kib`(compiler_options 或 XLA_FLAGS),但没接进 config/evaluate.py。

状态:未改。需要时加一个 config 字段透传该 XLA flag。

## 三、SSH/v6e 单卡路径(本 fork 自加的新增能力,非上游问题)

合作方交付只有 GKE+GCS 路径。本 fork 加了单卡/多卡 SSH 评测路径以在 TPU-VM 上直接跑,不依赖 GKE。相关改动是新增 feature,遇到的"问题"是这条新路径自身的设计要点:

- 多-host slice 的 co-launch:v6e-16 是 4 host 共享一个 ICI domain,单条 ssh 起不来 libtpu(barrier)。加了 `ssh_hosts` 列表,在所有 host co-launch、只从 primary 收结果。单-host slice 用 `ssh_host`。
- dump 落盘控制:加 `EVAL_DISABLE_DUMPS` / `EVAL_DUMP_DIR`(对应 2.2)。
- 多-host benchmark 噪声:co-launch 下只有 jax process 0 那台计时干净,其余是陪跑,须只读 primary。这也是后来改用单卡 v6e-4 评测的原因(单进程、无 barrier、计时干净)。

这些是 `ssh_evaluator.py` + 相关 skill 的自加内容,`SSH_V6E.md` 有说明。

## 四、本次实际成果(splash attention 的改动与提升)

对象:生产 Wan2.2 DiT splash MHA(mhpt, hpt=5),单卡提取版。pallas-evolve 找到的改动只在内核体 `_flash_attention_kernel_mhpt`:把 `for h_local in range(heads_per_tile)` 串行发 5 个头的两个 2D `lax.dot_general`(QKᵀ、PV)换成两个带 head batch 维的 3D `dot_general`,online-softmax 的 m/l/o bookkeeping 各加一个 head 前导轴。block 尺寸、bf16 QKᵀ + f32 PV、base-2 exp 都不变。

实测(v6e 单卡,vs query-blocked f32 参考,correctness 2e-2 内):[5,8192,128] 1.10→1.01 ms(2.13x→2.327x,+9%);生产 1080p 每卡 [5,86016,128] 58.48→51.84 ms(4.76x→5.37x,+12.8%)。大 S 更 MXU-bound,收益更大。

机制:加速来自单个 MXU 的 occupancy 提升,即 batched 大矩阵乘比 5 个串行小矩阵乘把 MXU 喂得更连续(流水更满)。两个 MXU 本就均衡(profiler 修好后 dual_ratio ~1.0),与第二个 MXU 的调度无关。可直接搬回生产 `flash_attention_kernel_mhpt`(逐头数学不变,mask/segment_ids/ring 残差不受影响)。细节与移植说明见 `kernel-evolve/runs/wan_splash_mhpt_20260721_204347/best/WINNER.md`。

## 五、使用体验 / 流程意见(给工具作者/协作方)

1. 每轮优化方向的决策频率过高。一轮是 PROFILE→THINK→SUBMIT→ANALYZE→REFLECT,现在每轮结束都要人确认"下一轮攻哪个方向、是否继续",而方向本可以直接从 profile brief 派生。这次三轮里每轮开头都停下来对齐方向,打断节奏、拉长总时长。建议 loop 默认自主:从 profile brief 派生 N 个方向、跑完直接进下一轮、用停滞规则(连续若干轮无改进)自然收敛,只在真正的岔路(已收敛、要花钱/开云资源、换 kernel 或 shape)才找人。start skill 已有 step-by-step / autonomous 两档,autonomous 档应真正做到"方向自派 + 连续跑",不再逐轮请示。

2. 先信 profile 再动手。dual_ratio 假 0 的 bug(见 1.1)一度把一整轮方向带偏,去"激活"其实没闲的第二个 MXU。loop 在用 profile 指标派方向前应做一次 sanity 校验(例如 dual_ratio 对着 finalization pass 交叉核对),否则会朝幻影优化。

3. 反馈延迟。一轮的墙钟主要花在评测,而评测里 dumps-on 的 deep profile 解析(几万个 LLO 小文件、每变体几分钟)是大头;dumps-off 每变体秒级(约 36s 含编译)。这次多次因为 dumps-on 把 900s 的 ssh 超时撑爆、要拉到 2400s。deep profile 是给"下一轮派方向"用的,一轮取一次(baseline)就够,不必每个变体都付。建议 loop 内层迭代默认 dumps-off(只要 speedup + correctness 排名),只在 round-0 baseline 和最终赢家上开一次 dumps-on 取 deep profile。这样每轮从十几分钟降到几分钟。

4. 区分噪声与改进。R3 几个变体落在 ±1% 内,是噪声不是名次差。建议 benchmark 报 CV 或多次取样,把落在噪声内的 delta 标出来,人和 loop 都别追噪声。

5. speedup 口径稀释。fitness 比的是整个 `optimized_compute` vs 整个 reference;当被优化的 kernel 只占其中一小块(如 qkv 里 QKV/输出投影占大头),再好的核心改动也被稀释、看起来没进展。建议提供"只测 kernel 核心"的选项或单独报核心耗时,让 loop 优化到点子上。

6. 正确性容差。bf16 kernel 对 f32 参考天然有偏差,默认 allclose(rtol/atol 1e-2)偏紧,会把数值正确的变体误判 INCORRECT。建议按 kernel dtype 给 bf16-aware 的默认容差或提示。

7. 建项成本集中在手搓三件套。把一个 kernel 接进 pallas-evolve 的主要人力,在手搓自包含 template(内核体放进 EVOLVE-BLOCK、签名不动)+ 参考实现 + config。init-kernel 只认 primatrix 的 tops/ops 布局(见 2.1),对来自 maxdiffusion、已是单文件的 kernel 用不上;大 seq 还得手写内存安全参考(见 1.4)。这次两个 kernel 都是纯手搓。建议:(a) 一个 `--from-file` 轻量 ingest,吃已自包含的单文件 kernel,自动切 ref/template 并插 EVOLVE-BLOCK;(b) 提供内存安全参考模板(query-blocked / flash-style),避免大 seq 的 einsum 参考 OOM。这两个能把"接一个新 kernel"从半天降到分钟级。
