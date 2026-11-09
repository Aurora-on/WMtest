# 阻尼摆实验的相关工作与后续基线

核查日期：2026-10-28。本文服务于当前 LeWM / LeHami 的 RGB 阻尼摆实验，按原论文、作者仓库文件和 GitHub 元数据筛选；未训练或复现下列方法，公开代码不等于已经跑通。这里的“自主 rollout”指初始化后递推预测状态；受控模型可以继续接收给定动作，但不能接收未来真实图像或状态。

**建议先完成 LeWM / LeHami 的共享数据实验，再添加 CpAE 和 H-JEPA。** CpAE 已有被动视觉阻尼摆生成器和方法实现，是当前任务最便于落地的外部参考；H-JEPA 是更直接的视觉耗散 port-Hamiltonian JEPA 比较对象。DFLNN 是必须讨论的离散耗散先行工作，但当前未确认作者公开的完整实现，暂不把重新实现它设为首轮实验的前置条件。

## 七项直接相关工作

“适配成本”是本次针对 RGB 单摆离线预测的判断，不是论文结论；“结构”也不等于真实机械能的无条件保证。

| 工作与最新核查版本 | 输入、自主预测与实际实验 | 约束对象及耗散边界 | 公开实现与当前适配判断 |
|---|---|---|---|
| **H-JEPA** — *Hamiltonian JEPA: Action-Conditioned World Models with an Inherited Control State*；2026-09-27 v1。[论文](https://arxiv.org/abs/2609.33497) | RGB + actions；从视觉表示取固定正交切片作控制状态，使用状态差分上下文，自主预测；原实验含 TwoRoom、DMCReacher、PushT、OGB-Cube，直接比较 LeWM 等 JEPA。 | 预测向量场有反对称项、PSD 阻尼及动作端口，另有 PIC 和表示正则。它明确支持耗散，但公开 `forward` 是显式 Euler；上下文随递推重算，不能从冻结上下文的连续时间能量式推出完整离散 rollout 的能量收支。 | [作者代码](https://github.com/tmz-lab/hjepa)含训练、评估、损失与预测器，MIT。**中等成本、优先加入**：适配同一 RGB / action 数据与观察历史；被动零动作设置可比较预测，但不能声称已覆盖原论文的控制优势。 |
| **CpAE** — *Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images*；ICLR 2025，arXiv 2025-02-02 v1。[会议论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/3cb18c3c31ff1b77b97f4512abd5e4a6-Paper-Conference.pdf)、[版本记录](https://arxiv.org/abs/2502.00754) | 两帧图像输入，连续性正则化自编码器，然后学习 latent Neural ODE；有被动阻尼单摆、RGB 128×128、自主预测。 | 约束编码映射的连续性，随后学习潜动力学；不是 Hamiltonian / PSD 耗散保证。它检验“更合适的表示 + 通用动力学是否已经足够”，对结构方法是有价值的比较。 | [作者代码与生成器](https://github.com/Aiqing-Zhu/CpAE)实存，含训练代码、图像素材和状态输出；未找到预打包的固定单摆数据下载。**较低至中等成本、优先加入**。保留原方法的重建训练差异并报告；只给 LeWM 加同类正则不能称完整 CpAE 复现。 |
| **DFLNN** — *Learning mechanical systems from real-world data using discrete forced Lagrangian dynamics*；2025-05-26 v1。[论文](https://arxiv.org/abs/2505.20370)、[全文](https://arxiv.org/html/2505.20370v1) | 位置观测，以及先经 AE 获得坐标的像素数据；有 Gym 风格阻尼单摆图像实验，使用过去位置隐式求下一位置，能自主 rollout。论文像素设置是 50×30 灰度，不能直接当作当前同一 RGB 数据。 | 离散 Lagrange–d’Alembert / forced DEL，把离散拉格朗日量与非保守力分开学习；去掉力后回到保守变分结构。**学习到非保守力不自动保证力始终耗散**，也不等于任意步长真实机械能严格收支。论文已有关闭学习力的反事实实验，是新颖性重要边界。 | 本次未确认可用作者仓库、固定像素数据下载或完整生成器；匿名投稿 PDF 的代码承诺不算已发布实现。**高成本**：若后续获得官方代码，优先级可以上调；当前作为最接近的离散耗散先行工作讨论。 |
| **LaWM** — *Least Action World Models for Long-Horizon Physical Consistency from Visual Observations*；2026-05-08 v1。[论文](https://arxiv.org/abs/2605.08279)、[全文](https://arxiv.org/html/2605.08279v1) | 视觉编码为潜坐标，学习相邻潜状态的离散拉格朗日量，以 DEL 条件产生未来状态并解码；是真正把动力学放入递推主干的邻居。 | 无外力的离散变分主干；论文把耗散、驱动和接触的充分处理留给 forced / constrained 扩展。有限次残差修正不能直接等同于精确 DEL 解；代码接受 action tensors 也不证明已有正确受迫功率结构。 | [作者方法核心仓库](https://github.com/chloeqxq/LaWM)有实际 RGB 训练接口、编码器/解码器与求解器；完整基准资产需外部提供。**中高成本、第二阶段**：可作结构对照；改成受迫耗散模型后应另名，不能把收益归于原版。 |
| **ALR** — *Same Target, Different Origins: Aligning Latent Rollout for Consistent PDE Emulation*；2026-10-15 v1。[论文](https://arxiv.org/abs/2610.21370)、[全文](https://arxiv.org/html/2610.21370v1) | 物理场自回归预测，在 FNO / F-FNO / LSM 内部表示上对齐不同起点到同一时刻的状态及局部响应；原实验是 Navier–Stokes、浅水和 Kolmogorov 流等，**不是 RGB 物体视频**。 | 零阶潜状态一致性与一阶局部响应一致性的软损失；不保证能量、辛结构或有限步耗散。有强迫的流动也不能按“能量必须单调下降”来判定。 | 本次未确认作者可用仓库。可后续做 **ALR-inspired LeWM loss**，实现成本可能低于换整套架构，但它是迁移实验，不是同数据上的原论文复现；额外展开和 JVP 的训练成本须计入。 |
| **SlotPi** — *Physics-informed Object-centric Reasoning Models*；KDD 2025，arXiv 2025-06-12 v1。[论文](https://arxiv.org/abs/2506.10778)、[会议 DOI](https://doi.org/10.1145/3711896.3737131) | 物体中心视觉 slots，经 Hamiltonian 分支与时空预测分支作未来 slot / 图像预测；主要是被动物理视觉推理。 | Hamiltonian 分支是结构先验，但最后与不受该约束的时空分支相加，完整预测不继承纯 Hamiltonian 保证；不是专门的耗散能量平衡模型。 | [官方仓库](https://github.com/intell-sci-comput/SlotPi)目前只有 README 和两个模型文件，缺训练/数据流程，且 `hamiltonian.py` 引用仓库未包含的 `misc`、`attention`。**高成本、不列为首轮可运行基线**；单摆也不足以体现其多物体分解优势。 |
| **HaM-World** — *Soft-Hamiltonian World Models with Selective Memory for Planning*；2026-05-07 v1 → **2026-09-25 v2**，arXiv 注记 NeurIPS 2026 poster。[论文](https://arxiv.org/abs/2605.05951)、[v2 全文](https://arxiv.org/html/2605.05951v2) | v2 主实验是六个 **state-based DMC** 任务，数值观测经 MLP 编码，带 Mamba 记忆和动作的 latent rollout / 规划；不是已验证的 RGB 基线。 | `(q,p,c)` 中能量导出的场叠加残差和控制，`c` 承载非保守因素；是软 Hamiltonian 先验，不能要求或声称全状态守恒。v2 明确承认长开环区间被基线反超，不能引用为普遍长时稳定。 | [作者仓库](https://github.com/HaoyunT/HaM_World)有较完整训练/评估与结果文件。**高成本**：迁到 RGB 需新视觉训练链路，在线 RL 设定也不同；适合后续受控与记忆研究，当前先保证强 LeWM 历史基线。 |

## 对当前实验的具体影响

**CpAE 数据应按实际生成器版本标记。** 作者仓库的单摆生成脚本输出图像与 `states.npy`，没有动作输入；notebook 默认 1200 条、每条 60 帧。当前源码使用 `L=0.5`、`lam=0.2`、`dt=2/fps`，默认 `fps=60` 对应观测间隔 **1/30 s**，其角加速度为 `−3g/(2L) sin(theta) − lam*omega`。这与论文所写 `dt=1/60`、附录的另一组参数不一致。若使用当前代码，必须把真实配置写进数据 manifest；应使用与这个杆模型匹配的惯量/势能，不能直接套用质点摆的机械能。若修正参数或延长时域，则称“基于 CpAE 生成器的派生数据”。[固定版本生成器](https://github.com/Aiqing-Zhu/CpAE/blob/aff6463ccbde19ca251e77abeb984e99354f8911/datainfo/collect/single_pendulum/make_data.py)、[生成 notebook](https://github.com/Aiqing-Zhu/CpAE/blob/aff6463ccbde19ca251e77abeb984e99354f8911/make_data.ipynb)

**H-JEPA 应比较完整递推而非只比较 pH 公式。** 当前代码第 79–84 行返回 `s + dt*(passive + push)`，不是隐式能量平衡求解器。对它与拟议方法应统一报告真实物理坐标上的长时误差、衰减包络、机械能变化及求解成本；潜空间的 `H` 值不同，不能直接横向比较。对于被动数据，H-JEPA 的端口反演目标所获信息与受控数据不同，需保留这个实验边界。[固定版本预测器](https://github.com/tmz-lab/hjepa/blob/fd4f7de927daa7831a6d51e95e0959a813c13901/hamiltonian_predictor.py)

**加入基线的顺序。** 第一项用 CpAE 回答“连续视觉表征 + 通用潜 ODE 能否解决问题”；第二项用 H-JEPA 回答“已有耗散 JEPA 结构能做到什么”。首轮固定机械参数与阻尼，每种配置独立训练；全部模型使用相同轨迹、划分、观测时间间隔和初始 raw-frame 可见范围。CpAE 的两帧拼接、LeWM 的历史状态和差分相位构造都要折算成实际可见帧数。若换数据、增加历史、重建损失或求解次数，应单独记录，不能将这些变化混称结构收益。当前只有被动数据时，结论限于被动耗散预测；受迫功率平衡和控制恢复需要另外生成含动作的实验。

## 版本与代码核查记录

下表日期为 GitHub API 返回的 UTC 日期。**仓库创建时间、commit 时间、README 自报发布时间不是同一件事**；它们不能证明仓库在该日已经公开。五个确认的作者仓库本次均未返回 GitHub Release。未检测到许可证表示本次未确认授权文本，不能据此假定等同 MIT。

| 作者仓库 | GitHub 创建日期 | 当前最新 commit 日期与快照 | 当前实物/授权 |
|---|---|---|---|
| [H-JEPA](https://api.github.com/repos/tmz-lab/hjepa) | 2026-09-26 | 2026-09-26；[`fd4f7de`](https://github.com/tmz-lab/hjepa/commit/fd4f7de927daa7831a6d51e95e0959a813c13901) | 24 个文件；MIT |
| [CpAE](https://api.github.com/repos/Aiqing-Zhu/CpAE) | 2025-02-07 | 2025-02-25；[`aff6463`](https://github.com/Aiqing-Zhu/CpAE/commit/aff6463ccbde19ca251e77abeb984e99354f8911) | 67 个文件；本次未检测到许可证 |
| [LaWM](https://api.github.com/repos/chloeqxq/LaWM) | 2026-05-27 | 2026-05-27；[`5b50258`](https://github.com/chloeqxq/LaWM/commit/5b50258c75c0ca0254b2f56a5a03b4a3b3ced4c0) | 23 个文件；README 自报 2026-05-14 发布，不能与创建日期混用；本次未检测到许可证 |
| [SlotPi](https://api.github.com/repos/intell-sci-comput/SlotPi) | 2025-05-25 | 2025-07-21；[`6995306`](https://github.com/intell-sci-comput/SlotPi/commit/6995306bc37277363e77a039383beaf6949b1af9) | 3 个文件，两个 `.py`；本次未检测到许可证 |
| [HaM-World](https://api.github.com/repos/HaoyunT/HaM_World) | 2026-05-03 | 2026-09-27；[`6582310`](https://github.com/HaoyunT/HaM_World/commit/65823101116a4ac2bdd8f866ab54e140df9b7470) | 981 个文件，含结果导出；本次未检测到许可证 |

DFLNN 与 ALR 的实现状态为“截至本次核查未确认”，不是证明不存在任何实现。本文只确认论文与仓库证据，不宣称依赖安装成功、训练收敛、数据已下载或论文指标已复现。
