# Glaucis / kernel-evolve 仓库问题记录

记录 2026-07-20-21 在这个 fork 上跑 pallas-evolve 时发现的仓库层面问题(代码、skill、config、
profiler),不含 TPU 节点侧的运维/容量问题。每条给出现象、位置、根因、状态。

## 1. 多-host slice 下单-host 假设导致挂起

现象:在 v6e-16(4 host 共享一个 ICI domain)上,单条 ssh 跑 `evaluate.py` 或 `jax.devices()`
永久卡住,无输出直到超时。

位置:`src/kernel_evolve/ssh_evaluator.py`、skills `submit`/`start`/`init-kernel`(原来只认单个
`ssh_host`)。

根因:libtpu 对整个 slice 做一次 init barrier,一台 host 起不来,要 4 台同时 co-launch。

状态:已修。config 加 `ssh_hosts` 列表 + `resolved_ssh_hosts()`;`SSHConfig.hosts`/`all_hosts()`/
`primary()` + `_ssh_all` 在所有 host co-launch,只从第一台(primary)收结果与 artifacts;skill 的
stage/eval/cleanup 步骤改为按 host 列表 fan-out。单-host slice(v6e-1/-4)仍用 `ssh_host` 兼容。

## 2. IR dump 无清理、无上限,写爆磁盘

现象:一轮 4 个变体后,靠后的变体报 `RESOURCE_EXHAUSTED: No space left`,primary 可能卡到超时。

位置:`docker/evaluate.py` `_setup_dump_env` / `_get_dump_dir`,dump 目录 `/tmp/ir_dumps/{variant}`。

根因:每个变体 dump HLO+Mosaic+LLO 约 15 GB,评完从不删除;一轮累计约 60 GB,写满 97 GB 根盘。
同一目录内 LLO 是几万个小文件,dump 到 tmpfs 时目录操作退化、变慢(所以曾误判为"卡死")。

状态:部分修。加了 `EVAL_DISABLE_DUMPS=1`(完全跳过 dump)和 `EVAL_DUMP_DIR=<path>`(改 dump 目录)
两个环境开关。根本修法(每变体评完即删 dump、或限制文件数)仍未做。

## 3. profiler 的 MXU per-unit 计数错误,dual_ratio 恒为 0

现象:`eval_result.json` 里 `mxu_utilization.dual_ratio` 一直是 0.0、mxu1=0,像是第二个 MXU 全空。
据此判断"第二个 MXU 空闲、是可优化项"是错的——实测两个 MXU 都在约 61% 使用。

位置:`docker/evaluate.py` `stage_profile_deep` 的 MXU 计数(选单个 `best_file` 后 findall
`.mxu0`/`.mxu1`)。

根因:per-unit 的 `.mxu0`/`.mxu1` 标注只出现在 `auto-mxu-assigner` 之后的 late-finalization LLO
pass;parser 挑的是 mosaic / 分配前的 pass(没有这些标注),于是回退到数 `vmatmul` 并硬编码 mxu1=0。
grep 最终 pass 的 ground truth:单卡 kernel `.mxu0` 5.22M / `.mxu1` 3.20M(~0.61),生产 ring kernel
`.mxu0` 1.45M / `.mxu1` 0.91M(~0.62),两者都用满两个 MXU。

状态:已修(待提交)。改为在 finalization pass(`.mxuN` 标注最多的那个)上数 MXU/DMA/double-buffering,
VMEM 和 bundle density 仍从 mosaic pass 取。教训:信 post-assigner 的最终 LLO,别信单个分配前 pass。

## 4. double_buffering 检测标记未经验证

现象:`dma_analysis.double_buffering` 长期为 false(23480 次 DMA),但这和 dual_ratio 一样可能是选错
pass 的假象,尚未在正确 pass 上核实。

位置:`docker/evaluate.py` `stage_profile_deep`,`re.search(r"\bsand\.u32\s+1\b", ...)`。

根因:`sand.u32 1` 这个双缓冲判定标记对 v6e LLO 是否正确、以及是否只在 finalization pass 出现,都没确认。

状态:未定。第 3 条的修法已让它改读 finalization pass,但标记本身仍是启发式,需要单独核对。

## 5. benchmark 只测 shapes[0],多 shape 下 speedup 口径含糊

现象:config 给多个 shape 时,correctness 会遍历所有 shape,但 speedup/latency 只来自第一个 shape;
不清楚这点时,换 shape 顺序会得到看似矛盾的 speedup。

位置:`docker/evaluate.py` `stage_benchmark`(`shape = shapes[0]`)、`stage_profile_deep`
(`s = shapes[0]`)。

根因:benchmark 与 deep profile 只跑第一个 shape,设计如此但未在 config/skill 里说明。

状态:未改(行为本身合理)。用单 shape 或明确知道 shapes[0] 是被测项即可,建议在文档标注。

## 6. GCS 桶名硬编码为上游的 glaucis-profiles

现象:kube 路径上传 profile artifacts 时用固定桶 `glaucis-profiles`,本账号无权限、桶也不属于本项目。

位置:`docker/evaluate.py` `upload_to_gcs(bucket_name="glaucis-profiles")` 与其调用处。

根因:桶名写死为上游 `sii-xinglong` 项目的桶。

状态:已修。加 `GCS_BUCKET` 环境覆盖,调用处改为读它。

## 7. 跟踪(Issue/PR/AGENT.md)默认落 git origin,私有 kernel 会泄漏到公开 fork

现象:`start`/`reflect` 的 `gh issue create`/`gh pr create`/`gh issue comment` 不带 `--repo`,默认
作用在当前 checkout 的 origin;当引擎在公开 fork、kernel 私有时,变体代码、结果、优化后的 kernel、
AGENT.md 会被写进公开 fork。

位置:skills `start`(建 Issue、结束建 PR)、`reflect`(评论 Issue、写并提交 AGENT.md)。

根因:单仓库假设,跟踪目标 = git origin。

状态:已修。config 加 `session.tracking_repo`(Issue/PR/评论指定 repo)和 `session.agent_md_path`
(AGENT.md 落私有 repo checkout);skill 相应加 `--repo` 并把提交 AGENT.md 的操作限定在拥有该文件的
repo。结束时若设了 tracking_repo,禁止把优化 kernel commit/push 到公开 fork。

## 8. init-kernel 写死 primatrix/pallas-kernel 布局

现象:`init-kernel` 只能从 `primatrix/pallas-kernel` 的 `tops/ops/<kernel>/` 多文件布局导入;对已经是
单文件、或来自别的 repo(如 maxdiffusion)的 kernel 不适用,只能手搓 ref/template/yaml。

位置:skill `init-kernel` Step 2/2.5/3(依赖 `tops/ops/$KERNEL_NAME/` 目录与逐文件 tracing)。

根因:导入流程为上游那一种目录结构设计。

状态:未改。本次两个 kernel(qkv_attention_pallas、wan_splash_mhpt)都是手搓三件套。可加一个
`--from-file` 轻量 ingest 路径处理单文件 kernel。

## 9. kube job 模板仍是 v7x,且运行时公开 clone

现象:`.github/ci/kernel-eval-job.yaml` 的 nodeSelector 是 `tpu7x` / topology `2x2x1`(v7x),不是
v6e;且 pod 在启动时 `pip install jax` + `git clone https://github.com/$REPO`(仓库私有时会失败)。

位置:`.github/ci/kernel-eval-job.yaml`。

根因:模板停留在上游 v7x + 公开仓库假设。

状态:未改(SSH 路径下用不到)。走 GKE 时改用了烤好 kernel_evolve 的镜像 + v6e node label,绕过运行时
clone;模板本身仍待更新。

## 10. 大 seq 下缺内存安全的参考实现

现象:注意力这类 O(S²) kernel,seq 大(37888/86016)时朴素 einsum 参考会 materialize `[H,S,S]` 分数矩阵
(TB 级)直接 OOM;kernel-evolve 没有提供内存安全的参考模板。

位置:kernel-evolve 的 reference 约定(`simple_compute`)未涉及分块。

根因:example 里的参考都是小 shape 的朴素实现。

状态:本次手写了 query-blocked 精确注意力参考(`wan_splash_mhpt_ref.py`)绕过。可作为一个可复用的参考模式沉淀。

## 11. 只暴露物理 VMEM,不暴露 scoped VMEM 上限

现象:变体用到超过 32 MiB VMEM 时编译期报 `CompileTimeScopedVmemOom: limit 32.00M`,即便物理 VMEM 有
128 MiB;config 没有调这个上限的入口。

位置:`config.py` `roofline.vmem_capacity_mib`(=物理 128)与 evaluate.py 的编译参数。

根因:默认 scoped vmem 预算是 32 MiB,与物理容量是两回事;需要 `xla_tpu_scoped_vmem_limit_kib`
(compiler_options 或 XLA_FLAGS)才能抬高,但没接进 config/evaluate.py。

状态:未改。需要时可加一个 config 字段透传该 XLA flag。
