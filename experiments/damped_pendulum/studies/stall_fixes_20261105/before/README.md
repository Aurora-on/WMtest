# LeHami / LeWM 阻尼摆基础实验

本项目比较两个既有模型在**同一视觉阻尼摆数据**上的自主预测。平台为 Python + PyTorch，仿真为显式单摆 ODE 和固定相机 RGB 渲染，不依赖 MuJoCo。下文原有实验结果均属于 LeHami／LeWM；新增耗散 JEPA 分支的首次训练于第2795步隐式求解失败；稳健求解器的服务器验收已于2026-11-04按用户要求暂停，尚无物理评测结论。

**2026-11-03 新耗散分支首轮协议：无外力、单模型、多阻尼。** 使用 21 帧历史推断固定环境上下文，阻尼为 D(q,v,c)，初速度仍只由末尾 5 帧拟合。正式数据包括六个训练阻尼值，以及独立初态上的已见阻尼、未见阻尼插值和外推测试；配置和生成方式见 [新数据与测试方案](reports/passive_multi_damping_protocol_2026-11-03.md)。服务器已生成完整数据；作业47541实际完成2794/20000步后因Newton残差超限终止，耗时约31分钟，条件续训47542未触发。旧时间估计不是完成记录；历史设置见 [启动记录](reports/passive_training_launch_2026-11-03.md)。原有基础数据及以下旧基线实验记录保持原协议。

## 稳健求解器与训练准备（2026-11-04）

新配置采用逐样本Armijo回溯Newton（上限12次、收敛早停），失败时最多二分4层；保留原残差和链式缺陷容差。仅重放接受的求解路径构建反向图，失败候选不参与参数梯度或能量账。日志记录回溯、细分、接受子步数及残差/容差比。有限细分仍可能耗尽，数值检查不等于真实物理预测准确。

本地108项检查、服务器17项CUDA数值/梯度检查已通过。最终快照的作业47696在2026-11-04 16:42（北京时间）按用户要求中断，正常日程完成2751→3877共1127次更新；可恢复的`last.pt`保存在3750步。服务器记录的`KeyboardInterrupt`来自此次主动停止，不是新的求解失败。依赖作业47701已取消。K16正常日程、500次K32压力更新和384条验证初态的320步推演尚未执行，不能宣告完整验收通过；早先24次初始化短测覆盖过K4/8/16/32，与训练后权重的长阶段验收不同。运行索引见 `studies/robust_solver_20261104/active_validation_state.json`，暂停和计算量证据见同目录`pause_snapshot_20261104.json`，冻结代码与配置哈希见`code_hashes.json`。用户恢复前不自动启动后续作业。

旧检查点须通过 `scripts/migrate_solver_checkpoint.py` 显式迁移：只允许求解器控制项变更，验证原源码/数据/环境，保留模型、Adam、RNG与采样位置，清空旧best并重新验证。不能绕过训练器的严格恢复检查。`--stop-after-step` 可在验证检查点结束有界测试，不改变20,000步学习率或课程安排。

验收后可通过 `scripts/slurm_passive_robust_train.sh` 开展下一轮；需设置`PENDULUM_CODE_DIR`为已验证冻结快照、`PENDULUM_PYTHON`、`PENDULUM_DATA`（seen.h5）、`PENDULUM_RUN_ROOT`。不设置`PENDULUM_RESUME`则从seed0初始化；设置时仅使用同版本正常训练的`last.pt`，并保留其同目录`best.pt`。该脚本尚未提交新一轮20,000步训练；24小时是Slurm上限，不是预计训练时间。

## 新增耗散 JEPA 分支

### 2026-11-05 高效求解分支（3子步训练完成，物理测试已启动）

作业47727已从0完成20000次更新与80次验证，按最终状态文件时间于2026-11-05
北京时间10:15结束，训练耗时约16小时4分钟。最终与最佳检查点、连续日志、Adam步数、
源码哈希及有限数值均已核验；最终/最佳潜空间验证分数为0.615104/0.613587，最佳步数19250。
完成证据见 `studies/efficient3_fresh_20261104T101014Z/completion_audit_20261105.json`。

用户随后授权检查和测试。物理评测作业 **47767** 使用预先选定的最佳检查点19250，
冻结模型，在seen训练标签20000个、验证标签4000个上校准并选择ridge/MLP读出，
再测试全部1408条轨迹（已见768、插值512、外推128），报告1/2/4/8/16/32秒自主预测。
评测保留求解失败的有效前缀与覆盖率，记录完整数值能量账，并比较状态保持、固定末帧、
静止平衡点及使用真实初态的保持基线。仅诊断读出被拟合，世界模型不更新。
评测接口7项检查通过，核心与训练源码未改变。启动记录位于
`studies/efficient3_evaluation_20261105T023318Z/`；服务器目录为
`/home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z`。
该作业启动不代表测试已完成或物理预测已准确；应以最终评测清单、覆盖率和指标为准。

用户在完成短测后授权3子步正式训练，并明确要求**从0开始**。训练作业 **47727**
使用单张L40、batch32、seed0，重新初始化全部模型参数和Adam，从第1次更新开始执行
完整20000步课程。不传入检查点，不执行迁移。先前从3750步续训的作业47726已按要求取消，
其输出仅保留历史记录，不用于本次训练。
当前冻结目录为 `/home/weizhixin/ssd/efficient3_fresh_20261104T101014Z`；
本地启动记录位于 `studies/efficient3_fresh_20261104T101014Z/`。
求解核心与3子步GPU短测版本相同。每250步验证并保存，24小时为调度上限，
并非完成时间承诺；训练启动时未执行物理测试。训练启动命令见
`scripts/slurm_passive_efficient_fresh_train.sh`，实际健康核查以启动目录记录为准。

新增 `dpbench/efficient_dissipative_core.py` 和
`dpbench/automatic_discrete_gradient.py`，采用预测中点系数、预条件Picard和隐式反传。
保持原来学习的连续模型、网络参数和位置/速度/环境相关阻尼；离散映射有意改变，
因此不能把旧检查点直接视为同一实验的无变化续训。

每个子步先计算

\[
\widehat q=q+\tfrac h2p,\quad
\widehat p=(I+\tfrac h2D(q,p,c))^{-1}
 [p-\tfrac h2\nabla V(q)+\tfrac h2B(q)u],\quad
D_*=D(\widehat q,\widehat p,c),\quad b_*=B(\widehat q)u.
\]

固定系数只限本次求根，下一子步重新计算；其对初态、环境与参数的训练梯度全部保留。
令 `A=I+h D*/2`，只进行一次Cholesky分解，主迭代为

\[
v^{k+1}=A^{-1}[p-\tfrac h2\bar\nabla V(q,q+hv^k)+\tfrac h2b_*],\qquad
q^+=q+hv,\quad p^+=2v-p.
\]

离散梯度通过MLP逐层割线链式计算，不再嵌套调用势能的自动求导。Picard不满足原残差/
链式缺陷容差时，继续用**同一个新方程**进行FD-Newton/Armijo或有界二分，不能改回旧方程。
非凸势能下不保证全局收敛。预测器本身非有限时直接拒绝，并不进入细分恢复。

反向传播仅在接受的根处构造一次Jacobian，求解转置线性系统并计算VJP，取消Newton迭代
重放；检查Jacobian条件数、伴随残差及 `||J^-1 F||`。这是收敛根的梯度近似，不是有限次
迭代程序的精确导数，仅支持一阶外层训练梯度。实际能量账继续检查
`DeltaH = work - dissipation + residual_work + chain_defect`。
光滑分支、准确求根和中点预测误差O(h²)条件下为二阶；Softplus阈值和小割线近似仍需实际验收。

2子步候选为 `configs/dissipative_passive_efficient.json`；
3子步精度对照为 `configs/dissipative_passive_efficient_3substeps.json`。
同一个3750步检查点、L40、batch32，原稳健方案与2子步新方案的实际短测结果：

| 预测步数 | 原方案秒/更新 | 新方案秒/更新 | 提速 |
|---|---:|---:|---:|
| 8 | 1.549 | 0.716 | 2.16× |
| 16 | 2.762 | 1.360 | 2.03× |
| 32 | 4.551 | 2.623 | 1.73× |

每种设置只计时3次更新，不能据此证明长训练稳定或物理精度相同。
另一个2795故障样本的冻结ODE精度对照显示：同为2子步，新方案轨迹RMS误差比原方案大72.7%；
增加到3子步后该样本RMS为0.04736，原方案2子步为0.08288。
这是单一困难样本、固定学习到的ODE，不能替代真实物理测试集。
3子步的独立K32 GPU短测中，旧法2子步为4.829秒/更新，新法3子步为4.040秒/更新，
提速1.20倍、节省16.3%时间。两个作业的基线有波动，因此仅报告各自作业内的配对比较。
3子步在这个困难样本上兼顾精度与速度，但还不能推断全测试集等精度优势。

本地70项检查通过，CUDA21项检查通过；第一轮短测共24次更新，作业47712耗时82秒；
3子步对照另做8次更新，作业47717耗时52秒。共32次有界更新，两个作业均已完成且未自动续训。
短测权重不保存为正式续训检查点。证据、完整推导约定、源哈希、误差细化和独立审计均位于
`studies/efficient_solver_20261104/`。自动离散微分参考
[DGNet (NeurIPS2020)](https://arxiv.org/abs/1905.08604)；该论文的相邻已观测状态训练方式
未替代本项目的自主视频rollout训练。

新增入口 `dpbench/train_dissipative.py`，默认只做只读数据与配置预检，只有显式 `--execute` 才允许优化器更新。3子步新方案正式训练已完成，物理读出与测试现已按用户授权启动。下述单阻尼示例与早期报告保留历史配置；旧首轮使用 `configs/dissipative_passive_family.json`；稳健基线配置为 `configs/dissipative_passive_robust.json`，高效方案见上文。

```bash
.venv/bin/python -m dpbench.train_dissipative \
  --data data/basic/pendulum_b0p2.h5 --config configs/dissipative_jepa.json
```

模型使用单帧 GroupNorm 编码、五帧因果速度初始化、位置与速度相关的 PSD 耗散力以及自主多步 JEPA。默认离散梯度积分器可切换为共享同一 H/D/B 的 RK4 对照；严格保留 train/val 前 81 帧和 RGB/action/dt 输入边界。训练方案、公式、配置和适用范围见 [训练方案](reports/dissipative_training_plan_2026-10-29.md)，实际检查记录见 `reports/dissipative_code_checks_2026-10-29.json`。现有常阻尼数据只能验证新模型的常阻尼特例，不能证明真实的速度相关阻尼学习。

以下检查包括小规模CPU优化器更新与恢复测试：

```bash
.venv/bin/python -m pytest tests/test_dissipative_core.py \
  tests/test_dissipative_data.py tests/test_dissipative_model.py \
  tests/test_dissipative_training.py -q
```

只允许非训练检查时，还需从 `test_dissipative_training.py` 排除名称含 `real_adam` 的测试；不要把整套测试描述成不更新参数。旧测试目录另含基线优化器更新和物理读出拟合。

## 实验口径

- LeHami：调用作者固定版本的卷积编码器、Hamiltonian 预测器、SG 动量初始化及多步损失；**不向模型添加阻尼项**。
- LeWM：调用 LeHami 作者提供的 LeWM 卷积、多步预测对照，时间编码使用方向敏感的 free W3、lag0。它是 **LeWM 的摆任务适配版**，不是原论文 224×224 ViT 机器人配方，也不是本文自行重写的近似模型。
- 上游固定为 LeHami `f76f65f8b99afb6b7e3063f573645d3e2f290ede`、LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。`third_party/sources.json` 记录最小源码快照、来源和 Git blob SHA；原始文件未修改，保留 LICENSE/NOTICE。
- 本地适配层仅跳过未使用的 ViT 依赖，将监控用 SVD 显式放在 CPU；训练/梯度仍在所选设备。CPU–MPS 同权重数值核验见 `reports/backend_parity.json`。未启用全局 MPS fallback。
- 当前统一 FP32、AdamW、固定学习率 5e-5、weight decay 1e-3、梯度裁剪 1；与作者 Lightning/bf16 运行配方不同。SG、预测器和损失结构来自作者代码。

## 数据和公平性

基础配置 `configs/basic.json`：每个阻尼设置 **512 条×325 帧，64×64 RGB，dt=0.1s**；质点摆 `m=L=1, g=9.81`。连续方程为

`m L² θ̈ = −m g L sinθ − b θ̇ + u`。

无阻尼 `b=0` 与固定阻尼 `b=0.2 N m s/rad` **分别训练**。每组一半自由运动，一半使用分段常值力矩（范围±2 N m），包含施力、撤去、再次施力。相同初态/动作种子在两个阻尼组使用相同 split；384/64/64 条 train/val/test，先按整条轨迹划分，再切窗口。

模型训练只接收 RGB、动作与固定帧间隔。真实状态、能量、质量、阻尼不进入世界模型损失。训练、验证 checkpoint 选择和读出校准均限制在每条轨迹前 **81 帧，即0–8秒**；测试从第5帧（t=0.4s）初始化，自主预测4/8/16/32秒，不重编码未来真实图像。

共同初始可用历史为5帧。LeHami 实际使用最后4帧构造 `(q,p)`；LeWM 使用5帧形成3个 W3 latent。训练共同窗口15帧，两者均从第5帧开始预测8步；LeHami 额外尾部图像只用于作者的中心差分训练目标。推演中没有这些未来图像。LeWM 收到必要的过去动作和同一未来动作序列。

数据是本项目新建的受控变体，使用 LeHami 风格的简单渲染；不能写成“原 LeHami benchmark”或“原 CpAE 数据”。公开外部验证候选见文献报告，当前不混入首轮主实验。

仿真使用 float64 RK4、每观测帧20子步，独立积分输入功 `W=∫uω dt` 和耗散 `D=∫bω² dt`。生成时与 DOP853 高精度解核对，并审计全部参考轨迹的 `ΔE−W+D`。HDF5 每条轨迹最后一行 action/W/D 为padding，`valid_transition=false`；不跨轨迹计算转移。

## 直接运行

当前机器已准备好项目 `.venv`。它只读复用旧环境的部分包，新增包安装在本目录；精确版本见 `reports/local_environment.json`。迁移新机器请用独立 Python 3.12 环境安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/bootstrap_upstream.py --verify
.venv/bin/python -m pytest tests -q
```

在项目根目录运行。源快照已随项目保留；若有源文件缺失，去掉 `--verify` 可按清单重新下载。CUDA 主机应按其驱动使用对应的 PyTorch 2.8.0 wheel；2026-10-28 已在服务器 L40、PyTorch 2.8.0+cu126 上完成两模型训练和完整评测。

生成数据（**不覆盖已有数据**）：

```bash
.venv/bin/python -m dpbench.generate --config configs/basic.json --output data/basic
```

当前 `data/basic` 已生成并校验，可直接使用。快速检查可选择 `configs/smoke.json`，但其轨迹较短，评测应显式指定 `--horizons 4 8`。

本地四组短实验（一个种子，各500步，主要检查流程）：

```bash
.venv/bin/python scripts/run_matrix.py --output runs/new_pilot500 \
  --device mps --steps 500 --seeds 0
```

扩大到三种子，每组5000步的**第一轮训练预算**：

```bash
.venv/bin/python scripts/run_matrix.py --output runs/basic5000 \
  --device mps --steps 5000 --seeds 0 1 2
```

这是预算起点，不是保证收敛的轮数。每模型每阻尼共有25,728个训练窗口，batch32下每epoch804步；500步不足一个epoch，5000步约6.2epoch。应按验证曲线增加预算，不能用未收敛结果下最终优劣结论。两模型的训练目标不同，训练loss不能直接横比。

单独计时或训练：

```bash
.venv/bin/python -m dpbench.train --data data/basic/pendulum_b0p2.h5 \
  --model lehami --output runs/new_profile --device mps \
  --steps 103 --warmup 3 --batch-size 32 --benchmark

.venv/bin/python -m dpbench.train --data data/basic/pendulum_b0p2.h5 \
  --model lewm --output runs/new_lewm --device mps \
  --steps 5000 --batch-size 32 --seed 0 --val-every 250

.venv/bin/python -m dpbench.evaluate --data data/basic/pendulum_b0p2.h5 \
  --checkpoint runs/new_lewm/best.pt --output runs/new_lewm/evaluation \
  --device mps --horizons 4 8 16 32
```

已有非空输出目录会报错，防止混合实验。训练保存 `best.pt/last.pt`、优化器状态和数据hash；当前 CLI **没有精确断点续训入口**，重新执行应使用新目录。评测按数据SHA256匹配，允许路径迁移。

## 评测读法

冻结世界模型后，另用固定标签预算拟合 ridge 和两层小 MLP 读出 `(sinθ,cosθ,ω)`；仅train拟合，val选正则或checkpoint，test冻结。默认train4096、val1024个标签。LeHami读出使用完整64维 `(q,p)`，LeWM使用全部3帧latent历史（192维），不能只读其瞬时latent而丢掉速度信息。

报告自由/受驱两类的角度RMSE、角速度RMSE、机械能RMSE、自由衰减的能量上升比例，并单列anchor读出误差。`reference_budget_rmse_J` 是预测端点能量与**真实轨迹W/D**的偏差；它不是模型自身离散动力学的精确能量保证。能量单调性不能单独评分：错误地一直预测静止也可能单调，必须同时看轨迹误差。

`evaluation/metrics.json` 给出每轨迹和聚合结果；`trajectories.npz` 保存完整时间曲线；`readouts.pt/calibration.json` 保存读出与实际校准样本。episode标准差不是多次训练的误差条；正式结果需至少三个独立训练种子。

完成后可用 `scripts/summarize_matrix.py --runs-root <12组结果目录> --output <汇总目录> --seeds 0 1 2` 汇总35轮矩阵（默认要求每组28,140步）。它读取已保存结果，不查询服务器；逐种子先对轨迹取均值，再计算跨种子均值和样本标准差。单种子不报告标准差，默认拒绝缺组、未完成或记录不匹配的结果；只有显式 `--allow-partial` 才生成标为PARTIAL的进度汇总。原 `scripts/summarize.py` 仅用于500步pilot。

## 运行位置

本地M5/16GB和服务器单张L40均已通过真实训练、保存加载与评测。2026-10-28 的服务器作业47106已成功完成；同一b=0.2数据、batch32、500步下，服务器训练步比本地快约2.1倍，但完整评测较慢。建议本地做仿真和调试，正式多种子训练用服务器单卡；具体范围和预算见 `reports/server_runtime_comparison.md`。

`scripts/slurm_profile.sh` 保留最初的轻量检查；匹配计时实际执行的是 `scripts/slurm_matched_profile.sh`，它跑500步并完成与本地同预算的全部评测。2026-10-28 已按用户确认提交12组35轮正式矩阵，作业号 **47111**（同一时刻只占1张L40，种子0/1/2）。每组28,140步，执行冻结快照中的 `scripts/slurm_matrix.sh`；2小时是每组申请上限，不是预测时长。所有服务器计算都经Slurm，不在登录节点训练。

正式实验快照：`/home/weizhixin/ssd/damped_pendulum_profile_20261028/studies/epoch35_20261028T133556Z`；结果位于该目录的 `runs/server_47111/`。本地 `reports/epoch35_20261028T133556Z/` 保存配置、39个源文件hash、数据预检、提交回执和初始启动核查。启动后按用户要求未持续监控。

**2026-10-29复查：12/12组训练与4/8/16/32秒评测均完整完成。** 每组28,140步、113次验证，全部损失/梯度有限，数据与协议核对通过，模型及预测轨迹文件已保存。按提交回执至最后输出文件时间计，北京时间10月28日21:36至29日07:25，约9小时49分钟；Slurm历史记录已清除，不把文件核查写成调度器退出码证明。结果和逐种子统计在 `reports/matrix_47111_20261029/`；下载证据在 `received/matrix_47111_review_20261029/`。本配置下LeWM预测误差低于LeHami，但LeHami在无阻尼对照也落后，不能把差距单归因为耗散；完成35轮不等于已充分收敛。

后续原文核查见该报告目录的 `lehami_paper_audit.json`：矩阵内部协议一致，并不等于复现原论文。当前W3-free/192维历史读出比论文主要的W1/单latent比较更强，训练schedule、图像归一化、checkpoint选择和probe配方也有改动。尤其当前probe在因果SG动量上校准，而原作者在中心SG动量上校准。`scripts/diagnose_probe_convention.py` 对两个阻尼的seed0做了同权重、同rollout、配对样本诊断：改变校准约定可降低部分速度误差，但未解释主要差距或逆转排序；原三种子结果未覆盖修改。`scripts/analyze_normalized_metrics.py` 新增事后幅度归一化诊断，保存为 `normalized_metrics.json`，同样保留原绝对误差。

本次服务器目录为 `/home/weizhixin/ssd/damped_pendulum_profile_20261028`。短测结果在 `runs/server_matched_47106`；服务器已重新生成两组完整数据到 `data/server_generated_47106`，均通过数值审计。用于匹配计时的原始b=0.2数据仍在 `data/basic`，两者分别保留hash，不混用checkpoint与数据。正式矩阵可通过 `PENDULUM_DATA_DIR` 指定数据目录。

## 结果与文献

- `reports/experiment_report_2026-10-28.md`：实际完成的训练、评测、成本和结论边界。
- `reports/related_work_2026-10-28.md`：H-JEPA、CpAE、DFLNN、LaWM、ALR、SlotPi、HaM-World的原文与代码核查。
- `reports/server_assessment_2026-10-28.md`：服务器实时连通性、历史库存和单GPU预算流程。
- `reports/server_runtime_comparison.md`：作业47106的真实L40计时、显存和完整训练预算；`received/server_profile_47106`保存已核验的原始结果。
- `reports/backend_parity.json`：CPU/MPS数值核验；`tests/`：独立积分器、物理收支、episode划分、padding和模型适配检查。
