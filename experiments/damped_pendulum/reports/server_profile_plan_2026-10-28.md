# 服务器短测：与本地一致的最小计时方案

日期：2026-10-28。此文件是只读代码审查后的执行建议；本文件作者未连接服务器、未投任务、未修改训练代码。连接和实测由主任务执行。资源申请上限不代表已经使用的 GPU 时间。

## 应采用的短测

用一个 Slurm 作业申请 **1 GPU、4 CPU、32GB 主机内存、30 分钟上限**，两模型顺序运行。保持上传的 HDF5 与本地 SHA256 相同；优先同一 PyTorch 2.8.0、同一份上游源码和适配层。

只测固定阻尼 `b=0.2` 已足以得到首轮速度估计：两种阻尼的数据形状和模型计算路径相同，阻尼不直接作为模型输入。但将这个速度外推到两种阻尼要标作外推，不能声称已经实测全部12组。

建议两模型各 **500 个完整训练步**，batch32、seed0、FP32、unroll8、train_frames81、threads4、workers0、val_every250、val_windows256；之后执行**完整评测**，train/val校准标签4096/1024、MLP500步、全部64条测试轨迹、4/8/16/32秒。该设置可以直接对应本地已完成的两组阻尼试跑。

```bash
# 在项目根目录、已分配的单GPU作业内运行；PENDULUM_PYTHON 指向独立环境。
PYTHON_BIN="${PENDULUM_PYTHON:-.venv/bin/python}"
for model in lehami lewm; do
  RUN_DIR="runs/server_matched_${SLURM_JOB_ID}_${model}"
  mkdir -p "${RUN_DIR}_timing"
  /usr/bin/time -p -o "${RUN_DIR}_timing/train_process_time.txt" \
    "$PYTHON_BIN" -m dpbench.train \
    --data data/basic/pendulum_b0p2.h5 --model "$model" --output "$RUN_DIR" \
    --device cuda --steps 500 --warmup 20 --batch-size 32 --seed 0 \
    --threads 4 --workers 0 --train-frames 81 --unroll 8 \
    --val-every 250 --val-windows 256
  /usr/bin/time -p -o "${RUN_DIR}_timing/evaluate_process_time.txt" \
    "$PYTHON_BIN" -m dpbench.evaluate \
    --data data/basic/pendulum_b0p2.h5 --checkpoint "$RUN_DIR/best.pt" \
    --output "$RUN_DIR/evaluation" --device cuda --seed 0 --threads 4 \
    --labels 4096 --val-labels 1024 --probe-steps 500 --horizons 4 8 16 32
done
```

`--warmup 20` 只改变计时统计，不额外增加训练步。为稳态速度比较，可从本地已有 `history.jsonl` 同样筛选 `step>20`，不需要重跑本地训练。保留全部500步的wall时间用于端到端训练成本，不能把预热成本从实际资源消耗中删掉。若作业队列等待很久，单独报告排队时长；它不属于GPU计算时长。

## 原 profile 脚本的适用边界

现有 `scripts/slurm_profile.sh` 先跑103步benchmark，再用10步、batch8做检查。这能证实CUDA前后向、保存/加载及长时预测可运行，但不支持直接用其总时间比较本地500步实验：

- benchmark分支没有验证和checkpoint写入。
- 检查分支的batch8不同于本地batch32，不能按每步秒数直接比较。
- 检查评测只用128/64校准标签、50步probe和4条测试轨迹；本地为4096/1024、500步probe和64条测试轨迹。
- warmup3仅适合粗略检查，首次CUDA/库初始化可能仍影响统计。

因此，若完整匹配的500步短测已经执行，原来的103+10步两阶段不必重复；失败时保留日志并修正实际故障，而不是把失败配置的速度用于预算。

## 必须区分的计时范围

1. `history.jsonl:wall_step_seconds`：数据读取、搬到设备、前向、反向、裁剪、参数更新和GPU同步；**不包括**该步后面的验证、checkpoint和日志写入。
2. `compute_step_seconds`：去掉该步取数和搬运，仍包含模型执行的CPU诊断。它不是纯GPU kernel时间。
3. `summary.json:wall_seconds`：实际训练循环总时间，含验证、checkpoint和训练日志；**不含**Python启动/导入、数据文件hash、模型/优化器构建等前期准备。
4. `evaluation/metrics.json:total_seconds`：从加载checkpoint前开始，含校准与预测；在写最终metrics和压缩轨迹之前取值，故**不含最终全部输出写入**。
5. 外层 `/usr/bin/time`：涵盖整个训练或评测进程，适合服务器准备好的环境下的实际交付成本；另记录上传、安装和排队。现有本地报告没有相同外层计时，因此不能把服务器外层耗时与本地内部计时叫作完全相同口径。

GPU内存使用 `cuda_peak_allocated_bytes` 和 `cuda_peak_reserved_bytes`；二者含义不同，都不能与本地MPS的末次driver分配量称作同口径“峰值显存”。记录GPU型号、驱动、可见GPU数量、PyTorch/CUDA版本、CPU配额与线程数。还应记录 TF32 开关、float32 matmul precision 和 cuDNN 设置；主比较不临时打开混合精度或torch.compile。当前CPU监控SVD在两个后端都执行，应保留。

## 预算外推与报告方法

设两模型稳态每步分别为 `t_H`、`t_W`，每模型每阻尼每种子训练 `N` 步：

`12组纯训练小时 = 2种阻尼 × 3种子 × N × (t_H+t_W) / 3600`。

用本地已有 `reports/costs.json` 的数值计算，N=5000时为3.47小时，N=28140（本数据35epoch）时为19.54小时；这是此前本地外推，不是服务器结果。

推荐最终同时给出：

- 单模型：匹配500步的训练循环wall、稳态均值/中位/p90、完整评测耗时、CUDA峰值。
- 12组：5000步与28140步两个预算，先给纯训练点估计，再给包含验证/保存/评测的计划范围。
- 用 `train_wall - sum(all history wall_step_seconds)` 审计未进入逐步计时的验证/保存/日志开销。它不能单独当作纯验证耗时。按相同验证频率把500步训练总wall近似放大10倍，可得到5000步的简易全循环预算，但预热被重复放大、长时热/共享负载尚未覆盖，应明确是粗估。
- 完整评测按每次模型训练后一次计算：两模型各6次。不要按训练步数放大评测耗时。
- 排队、首次环境准备、上传时间、未来超参搜索次数单列。若没有实际收费单价或配额，不输出货币报价。

首测若后半段单步明显变慢、共享GPU利用率异常或显存不足，再延长/重复受影响部分；如果稳定且所有完整阶段通过，无需例行扩大短测。
