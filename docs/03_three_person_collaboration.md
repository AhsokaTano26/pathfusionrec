# PathFusionRec 三人协作方案

## 协作目标

在同一份 `sports_protocol`、同一候选目录、同一全库评测规则下，完成 ActionPiece、SASRec、VQ-Rec 等基线复现补齐，以及 PathFusionRec 的 Semantic Only、Interaction Only、Fusion 与 Gated Fusion 消融实验。主表只使用完整五个 seed（2024–2028）的测试结果。

## 角色与职责

| 角色 | 负责人 | 核心职责 | 不负责事项 |
|---|--|---|---|
| A：模型研发与结果分析 | 张启睿 | 设计/实现模型，修复问题，决定配置，审核单 seed，汇总多 seed，撰写分析 | 不同时运行多组长期实验 |
| B：GPU 实验执行 | 周瑞璇 | 拉取指定 commit，运行单 seed 与五 seed，监控进程，保存日志/checkpoint/环境信息 | 不修改模型、数据处理、指标或超参数 |
| C：基线复现与质量核验 | 倪榕键 | 补齐 ActionPiece/SASRec/VQ-Rec，核对论文协议、检查结果完整性，复算汇总表与 SHA-256 | 不改 PathFusionRec 主模型，不替代 A 决定实验口径 |

## 分阶段执行清单

每个阶段必须先满足“开始条件”，再分配任务；完成后按“交付物”和“通过条件”验收。未通过时只能由 A 创建新版本后重做，不能在原实验目录补改参数或覆盖结果。

### 阶段 0：项目初始化与环境锁定

**开始条件：** 三人均能访问仓库、GPU 环境和共享结果目录。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 确认 `main` 的 commit；检查 `sports_protocol` 的 `split.json`、`bundles.json`、`subset_masks.json` 和 `statistics.json`；明确主指标为测试 NDCG@10、模型选择指标为验证 NDCG@10。 | `docs/` 中的协议说明、首轮实验编号规则。 |
| B | 在服务器克隆仓库；记录 `python --version`、`pip freeze`、`nvidia-smi`、CUDA 和 PyTorch 版本；检查数据文件存在且可读。 | `environment.txt`、数据目录检查结果。 |
| C | 整理 ActionPiece、SASRec、VQ-Rec 原论文的任务定义、数据集、切分、候选集、指标、seed 数和报告值。 | `baseline_gap_checklist.md`：每个基线的“已满足/缺失/不可比较”清单。 |

**通过条件：** B 能运行 `PYTHONPATH=src python3 -m unittest discover -s tests -v`；A/C 确认所有正式结果都将使用固定 `sports_protocol`，而不是早期 smoke 或采样负例结果。

### 阶段 1：PathFusionRec 代码与冒烟验证

**开始条件：** 阶段 0 通过。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 完成并测试 `Semantic Only`、`Interaction Only`、`Fusion/concat`；确认 `Fusion/gated` 只作为后续消融；为每个模型创建实验卡和固定命令。 | 模型代码、单元测试、`PFR-<模型>-v1/experiment_card.md`。 |
| B | 仅运行 A 给出的极小样本命令，检查“真实数据加载→训练→验证→全库候选评分→四类子集写出”闭环；不报告该数值。 | 冒烟日志、`metrics.json`、报错截图或最后 50 行日志。 |
| C | 审阅 A 的实验卡：是否写明 commit、候选集、模型选择方法、完整参数、输出目录；核对脚本输出是否包含 `all`、`cold_start_target`、`long_tail_target`、`sparse_user`。 | 实验卡核验结论，问题列表。 |

**通过条件：** 单元测试全绿；B 的冒烟任务不出现 NaN、OOM、数据缺失或评测报错；C 确认输出 schema 完整。此阶段不产生可写入论文的指标。

### 阶段 2：基线缺口补齐与正式单 seed

**开始条件：** 阶段 1 通过，A 已发出固定 commit 和正式超参数。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 指定本轮只跑的模型与参数；先选择 seed 2026 作为验收 seed；审阅曲线并判断是否需要修复。 | 已批准的正式实验卡；对每个单 seed 的“继续五 seed/停止修复”结论。 |
| B | 运行 PathFusionRec 的 seed 2026：依次为 Interaction Only、Semantic Only、Fusion/concat；每个完成后保存日志、配置、最佳 checkpoint 和 `metrics.json`。 | 三个 `seed2026` 结果目录；QQ 回传模板中的完整信息。 |
| C | 按缺口清单实现或修复一个基线；在同一协议下运行该基线 seed 2026；核对该基线是否能输出相同 Top-K 和子集指标。 | 一个基线的 seed 2026 结果目录、与原文的协议对照表。 |

**通过条件：** 每个模型都完成一个正式 seed；验证曲线没有异常；测试结果来自全库排序且历史过滤正确；A 批准后才进入五 seed。若某模型要改 lr、早停、负采样、编码器或评测，A 必须创建 `v2`，旧 `v1` 保留。

### 阶段 3：五 seed 批量运行

**开始条件：** 对应模型的 seed 2026 已通过阶段 2。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 制定运行顺序，优先完成 `Interaction Only → Semantic Only → Fusion/concat`；每天审核新完成 seed 的曲线；登记失败实验。 | `run_board.md`：模型、版本、seed、状态、负责人、开始/结束时间。 |
| B | 对已批准的 PathFusionRec 模型运行余下 seed 2024、2025、2027、2028；一个 GPU 任务对应一个 seed；任务结束立即校验文件完整性。 | 每个 seed 的 `metrics.json`、`training_history.csv`、`best_model.pth`、`run.log`、环境记录。 |
| C | 对已批准的基线运行余下 seed；同步抽查 B 的 PathFusionRec 结果：commit、参数、最佳 epoch、文件哈希和指标范围。 | 基线完整五 seed 目录；`integrity_check.csv`。 |

**B 的每个 seed 完成动作：**

1. 检查进程正常退出，保存标准输出到 `run.log`；
2. 确认 `metrics.json` 写有 `best_epoch`、验证指标和四类测试子集；
3. 检查 checkpoint 非零字节、训练曲线 epoch 数与实验卡一致；
4. 记录开始/结束时间、GPU 型号和显存峰值；
5. 将路径和核心数字发给 A/C，不移动或手改文件。

**通过条件：** 同一实验编号下 2024–2028 五个 seed 齐全，且每个 seed 的 commit、数据哈希和参数一致。缺一个 seed 就只能称“进行中”，不能计算主表均值。

### 阶段 4：独立汇总、消融与论文对比

**开始条件：** 至少一个模型完成五 seed，原始文件齐全。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 解释三路模型的整体与子集差异；决定是否启动 `Fusion/gated`；撰写“语义—交互互补性”结论，避免过度归因。 | 主结果表、子集表、`analysis.md` 初稿、下一轮 gated 实验卡。 |
| B | 按 A 给出的 item/user/sample ID 导出 Fusion 注意力：标题/描述/类目权重，以及主体商品与共购配件权重；选取成功、失败各若干案例。 | `attention_cases.json/csv`、案例表和可视化素材。 |
| C | 从五份原始 `metrics.json` 重建汇总 CSV 和图；用 `ddof=1` 计算样本标准差；逐项比对论文数值、当前基线和 PathFusionRec。 | `five_seed_results.csv`、`mean_std.csv`、曲线图、论文对比表、复核结论。 |

**通过条件：** A 与 C 的汇总数值一致；图、表和原始 JSON 能相互追溯；报告明确“协议一致/近似一致/不可直接比较”。只有此时，Fusion 是否优于单路模型才可写入阶段性结论。

### 阶段 5：门控融合与路径生成

**开始条件：** Fusion/concat 已完成五 seed 且有明确的单路对照结果。

| 负责人 | 具体操作 | 交付物 |
|---|---|---|
| A | 启动 `Fusion/gated`，只改变融合结构；定义路径生成的输入、候选约束、路径有效性、多样性、覆盖率和人工案例标准。 | Gated 实验卡；`path_generation_protocol.md`。 |
| B | 运行 gated Fusion 的单 seed、五 seed；在 A 固定路径生成协议后生成指定测试样本的路径结果。 | Gated 五 seed 原始结果；路径生成原始输出。 |
| C | 检查 gated 与 concat 是否使用相同协议和预算；复核路径结果是否遵守候选约束，汇总自动指标和失败案例。 | Gated 对照表；路径生成质量核验报告。 |

**通过条件：** Gated 的优势必须同时对比 concat Fusion 和单路模型；路径生成必须有独立定义的指标，不能拿 next-item Recall/NDCG 代替。

## 每轮实验的固定流程

### 1. A 创建实验卡并固定版本

每一轮须在结果目录写 `experiment_card.md`，至少包含：

```text
实验编号：PFR-Fusion-Concat-v1
代码 commit：<40 位或短 SHA>
模型：fusion / concat
数据协议：sports_protocol
训练/验证/测试：固定 split.json
候选集：bundles.json 中全部映射商品
seed：2024, 2025, 2026, 2027, 2028
超参数：epochs、lr、batch_size、negative_count、hidden_dim
模型选择：验证集 NDCG@10 最佳 checkpoint
输出目录：experiment_results/04_unified_next_item/PFR-Fusion-Concat-v1/
```

代码、数据、早停规则、指标定义或超参数任一项改变，都必须新建实验编号；不得在同一五 seed 批次中混用版本。

### 2. B 或 C 先跑 seed 2026

执行人只能运行实验卡中给出的命令。完成后在任务群/QQ 按以下模板反馈：

```text
实验编号：
commit：
执行人：
seed：2026
状态：完成 / 失败 / 已停止
最佳 epoch：
best validation NDCG@10：
test NDCG@5 / Recall@5：
test NDCG@10 / Recall@10：
日志路径：
metrics.json 路径：
checkpoint 路径：
异常信息：无 / 最后 50 行日志
```

### 3. A 验收并批准五 seed

启动余下 seed 前，A 检查：

- commit、参数和数据哈希与实验卡一致；
- `metrics.json`、逐 epoch 验证曲线和最佳 checkpoint 均存在；
- 没有 NaN、OOM、异常中断或明显早停；
- 结果不是将历史物品、目标或采样候选集错误混入评测；
- 单 seed 数值和训练曲线没有明显异常。

### 4. C 独立核验并汇总

C 以原始 `metrics.json` 重建 `five_seed_results.csv`、`mean_std.csv` 和结果图。主表采用样本标准差（`ddof=1`），不能手填或只复制控制台截图。

## 目录与命名规范

```text
experiment_results/
└── 04_unified_next_item/
    └── PFR-Fusion-Concat-v1/
        ├── experiment_card.md
        ├── seed2024/
        │   ├── metrics.json
        │   ├── training_history.csv
        │   ├── best_model.pth
        │   ├── run.log
        │   └── environment.txt
        ├── seed2025/
        ├── seed2026/
        ├── seed2027/
        ├── seed2028/
        └── summary/
            ├── five_seed_results.csv
            ├── mean_std.csv
            ├── curves.png
            └── analysis.md
```

- `PFR` 表示 PathFusionRec；基线使用 `AP`、`SASRec`、`VQRec` 前缀。
- 输出文件名和实验卡中的模型、版本、seed 必须一致。
- 大型 checkpoint、缓存数据和 TensorBoard 原始事件不提交 Git；结果表、配置、日志摘要、脚本和分析提交 Git。

## 结果交付标准

每个模型至少交付以下内容：

1. 全体测试集 NDCG/Recall@5、@10、@20、@50 的五 seed 均值 ± 样本标准差；
2. `cold_start_target`、`long_tail_target`、`sparse_user` 三个子集的同口径指标；
3. 每个 seed 的最佳验证 epoch、总训练时间、测试时间和 checkpoint；
4. 与论文结果的对比表，明确协议是否完全一致；
5. Fusion 的字段注意力和主体/配件注意力案例；
6. 失败实验也保留日志和失败原因，不删除或以新结果覆盖。

路径生成任务作为独立第三阶段：先定义输入历史、候选约束、路径有效性/多样性/覆盖率指标，再报告结果；不得把 next-item 的 Recall/NDCG 直接解释为路径生成质量。

## 沟通与升级规则

- B/C 发现 OOM、NaN、数据缺失、指标为异常值或训练提前停止时，立即停止该 seed，保留日志，通知 A；不要自行调参重跑。
- A 修复后以新 commit 和新实验编号重新下发；旧结果保留为历史记录。
- C 发现基线与原文协议不一致时，在报告中标注“不可直接比较”，并提出最小补齐方案。
- 每日简报只汇报：运行中的实验、完成 seed、失败原因、下一步阻塞项。避免只报“正在跑”。
