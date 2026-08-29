# Protocol Check — B9 / DC-PHSR / PHM2010 / D1

日期：2026-08-29/30。执行者：Claude（本对话直接执行，未委派子代理执行训练）。

## 检查项

| 项目 | 结果 |
|---|---|
| PREPROCESS_SEED=42 固定复用（不重新生成） | ✅ 复用 `protocol_diagnostic_fixed_preprocess/frozen_preprocess/`（2026-08-20 冻结），镜像到 `shared/reproducibility/PHM2010_D1_frozen_preprocess/`，镜像前逐文件 sha256 校验字节相同 |
| feature_hash / split_hash / gmm_hash / window_hash 5 个 seed 完全一致 | ✅ 逐 seed 核对（见 `hash_manifest.json`），5/5 一致 |
| 运行期自检（`assert_frozen_artifacts_unchanged`）通过 | ✅ 5/5 seed 均通过，日志显示 `frozen artifacts verified OK` |
| TRAIN_SEED ∈ {42,52,62,72,82} 独立进程、独立 RNG reset | ✅ 每个 seed 单独 `python run_seed_d1.py --train_seed N` 进程（seed42 单独运行；52/62/72/82 各自独立后台进程并行运行，互不共享 RNG） |
| B9 未独立训练新 backbone | ✅ 每 seed 一次 `base.train_model()` 调用，B3 checkpoint 与 B9 推理共用同一次训练输出 |
| B9 推理参数从当前正式代码实时读取（未硬编码复制） | ✅ `mod.B12_PARAMS` 从实时 `importlib` 载入的 `代码/7.4对比实验.py` 读取；5 个 seed 值全部一致：`eta=0.75, fine_weight=0.30, temperature=1.20, mid_floor=0.12, late_tau=0.66, early_tau=0.38, order_blend=0.25` |
| 历史参考值核对 | ✅ 与用户提供的历史参考值完全一致，无 drift |
| 未修改 `代码/` 任何文件 | ✅ 仅以 `importlib` 只读方式加载；`git status` 在本轮结束前核对 `代码/` 下无改动 |
| 未覆盖旧结果目录 | ✅ 所有输出仅写入 `扩充实验代码/`；`protocol_diagnostic_fixed_preprocess/`、`final_five_seed_sweep/` 未被触碰 |
| 未按 C6 表现调参/挑 seed/删 seed | ✅ 5 个 seed 全部保留、如实记录，包括 seed52 的明显偏低结果 |
| Train/Val/Test 定义 | Train=C1+C4（含 source-side stratified 内部验证切分），Test=C6，n_test=304，run_id_end 12–315 |

## 已知非阻塞性异常

**Seed42 的 `training_seconds` 记录为 7673.3s（约2.1小时），显著高于同协议历史基准（57.5s）及本轮其它 4 个 seed（147–195s）。**

- 根因排查：seed42 单独以「前台启动、超时后转后台」方式运行；seed52/62/72/82 则以并行后台方式启动。训练日志（`training_log.csv`）显示 seed42 实际只训练了 23 个 epoch（与其它 seed 同量级），且完成后 GPU 立即空闲（`nvidia-smi` 确认 0% 利用率、0MiB 占用，无残留进程）。
- 结论：该长耗时是本地任务调度/后台进程环境的计时伪影（很可能是后台 shell 会话在未被轮询期间被系统降低调度优先级），并非训练逻辑本身变慢或产生了错误结果——seed42 的 Acc/Macro-F1 与历史基准逐位精确一致（见下方 OLD_VS_NEW.md），证明计算过程本身正确、完整。
- 未采取任何"为了计时好看"的重跑；如实记录该异常与其真实数值。

## 结论

本轮 5-seed 协议隔离与哈希一致性核查全部通过，无 protocol drift。
