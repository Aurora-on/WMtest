# 运行与训练

以下命令从 `experiments/damped_pendulum/` 执行，依赖见该目录 `requirements.txt`。代码以 Python 3.12／PyTorch 2.8.0 验证；GPU 训练另需可用 CUDA 环境。

## 数据与时间协议

- `dt=0.1s`，21帧历史跨度2.0秒，训练仅使用前81帧。
- 单阻尼 `b=0.2`；多阻尼 `b=[0,0.05,0.1,0.2,0.5,1]`。
- 世界模型读取图像、动作与时间间隔；物理状态只用于独立读出拟合和评测。
- 多阻尼各工况共享初始条件组，不能当成独立初态。

查看生成计划：

```bash
python scripts/prepare_comparison_data.py --output /path/to/new_data
```

实际生成时添加 `--execute`，使用新的输出目录。生成后有 `a/undamped.h5`、`single/damping.h5`、`multi/seen.h5` 及数据清单。精确恢复历史检查点要求原数据字节哈希一致，重新生成的数据需先验证。

## 最新 H-JEPA 配方

配置：`configs/hjepa_staged_retrain_20261109/{a,single,multi}.json`。

三个阶段均为 batch128、固定训练预测5步、AdamW lr=5e-5、weight_decay=1e-3。保留现有 CNN＋GRU 前端、官方核心和损失；**耗散网络一直可学习**。A 使用无阻尼数据；B 从合格 A20000 复制全部权重，包括耗散网络，各自重置优化器并联合微调。

以下使用绝对路径：

```bash
DP_DATA=/path/to/data
DP_RUN=/path/to/new_runs

# 默认只读预检，不构造优化器，不创建训练结果。
python scripts/run_hjepa_retrain.py \
  --config configs/hjepa_staged_retrain_20261109/a.json \
  --data "$DP_DATA/a/undamped.h5" --data-a "$DP_DATA/a/undamped.h5" \
  --output "$DP_RUN/a" --device cuda

# 添加 --execute：A4000验收通过才继续A20000。
python scripts/run_hjepa_retrain.py \
  --config configs/hjepa_staged_retrain_20261109/a.json \
  --data "$DP_DATA/a/undamped.h5" --data-a "$DP_DATA/a/undamped.h5" \
  --output "$DP_RUN/a" --device cuda --execute

# A20000通过后运行B；multi对应替换配置、数据与输出目录。
python scripts/run_hjepa_retrain.py \
  --config configs/hjepa_staged_retrain_20261109/single.json \
  --data "$DP_DATA/single/damping.h5" --data-a "$DP_DATA/a/undamped.h5" \
  --parent-output "$DP_RUN/a" --output "$DP_RUN/single" --device cuda --execute
```

在独立 GPU 分别启动 single 和 multi 可以并行。每阶段 `--until-step 4000` 可在首轮验收后停止，默认目标20000。续跑同一命令会验证并恢复状态，不应修改总步数来模拟续训。

每个检查点独立用训练集288组拟合读出、96组选k，在64组验证初态上评测未来8/16/32步。多阻尼分别验收每个b；门槛是工程筛查，不是论文标准或最终物理有效性的证明。

## Slurm 和历史记录

`studies/hjepa_staged_retrain_20261109T025341Z/` 保存实际上传包的配置、冻结源码、SHA、提交脚本与记录。这些脚本含原服务器绝对路径，只供原环境使用或迁移时参考。**不要在新机器直接执行历史提交入口**；使用上面的可配置 CLI，并为自己的 Slurm 环境设置路径和资源。

原提交入口使用 `afterok:A_JOB_ID` 和父检查点重新验收控制两个B。每项作业先用独立模型做 GPU 预检查，之后重新初始化正式模型或迁移父权重。

阶段数和更新数与此前一致，但 batch128 是旧 batch32 的4倍样本预算。本配方用于训练诊断，不应直接声称等预算优势。

原实验目录 README 是历史实验日志，包含已取代计划。官方 LeHami 复现的原机器脚本位于另一实验目录；固定来源和环境清单保留，完整旧环境、数据和 checkpoint 未纳入 Git。
