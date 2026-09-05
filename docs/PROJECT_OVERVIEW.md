# PathFusionRec 项目总览

## 1. 项目是什么

PathFusionRec 是一个面向**下一商品推荐**的研究工程。它研究用户下一次会交互的商品，核心问题是：

> 用户历史中的交互行为与商品文本/关联商品语义，能否形成互补，从而提升下一商品排序质量？

项目以 Amazon Reviews 2014 的 `Sports_and_Outdoors` 类目为实验对象，在统一的留一法划分、固定候选目录和 Top-K 指标下，比较三类 PathFusionRec（简称 **PFR**）模型：

| 模型 | 使用的信息 | 要回答的问题 |
|---|---|---|
| Interaction Only | 用户历史商品 ID | 纯交互信号能够达到什么水平？ |
| Semantic Only | 商品文本语义与关联 bundle | 内容语义是否能处理弱交互或冷启动？ |
| Fusion | 交互与语义的联合表示 | 两类信号是否真的互补？ |

这里的 bundle 不是实际订单购物车，而是 Amazon 商品元数据中的 `related.bought_together` 关系构造的“主体商品 + 相关配件”局部集合。

## 2. 仓库与目录

本项目根目录中有两个独立 Git 仓库：

```text
dachuang/
├── pathfusionrec/       # 本项目：统一协议、PFR 模型、实验与分析
└── action_piece/        # 独立维护的上游 ActionPiece 复现代码
```

`pathfusionrec/` 的主要结构如下：

```text
pathfusionrec/
├── data/
│   ├── AmazonReviews2014/Sports_and_Outdoors/processed/
│   │   ├── all_item_seqs.json             # 原始交互序列的固定处理结果
│   │   ├── id_mapping.json                 # 用户/商品数值 ID 映射
│   │   └── sentence-t5-base.sent_emb       # 商品 sentence-T5 语义向量
│   └── processed/sports_protocol/
│       ├── split.json                      # 固定 train/validation/test 样本
│       ├── bundles.json                    # 全部候选商品的关联 bundle
│       ├── subset_masks.json               # cold-start、长尾、稀疏用户掩码
│       └── statistics.json                 # 数据规模、哈希与协议验收信息
├── src/pathfusionrec/
│   ├── models/bundle_encoder.py            # bundle 语义编码器
│   ├── models/next_item.py                 # Semantic/Interaction/Fusion 模型
│   └── evaluation.py                       # 全库排序、历史过滤、NDCG/Recall
├── scripts/
│   ├── prepare_sports_protocol.py           # 构建并核验统一协议
│   └── run_unified_next_item.py             # PFR 三路模型训练与评测
├── experiment_results/                     # 原始实验产物和核验报告
├── tests/                                  # 单元测试
└── docs/                                   # 协议、协作规则和项目文档
```

大体积原始数据、缓存和运行输出通常不进入 Git；正式实验应保存可追溯的配置、日志摘要、指标、哈希和分析。

## 3. 数据与统一实验协议

### 数据范围

唯一交互来源是 ActionPiece 兼容处理后的 Amazon Reviews 2014 `Sports_and_Outdoors` 数据。PFR 不重新过滤用户、不重排时间、不重映射商品 ID，避免不同模型因预处理差异得到不可比结果。

当前协议规模为：

| 项目 | 数量 |
|---|---:|
| 用户 | 35,598 |
| 商品/全库候选 | 18,357 |
| 原始交互 | 296,337 |
| 训练前缀样本 | 189,543 |
| 验证样本 | 35,598 |
| 测试样本 | 35,598 |

### 划分方式

对于一个用户序列 `i1, i2, ..., iL`，采用 leave-one-out：

| 划分 | 历史 | 目标 |
|---|---|---|
| train | 每个不跨入验证/测试目标的前缀 | 下一个商品 |
| validation | `i1 ... i(L-2)` | `i(L-1)` |
| test | `i1 ... i(L-1)` | `iL` |

输入历史最多保留最近 20 个商品。验证集只用于选择 `NDCG@10` 最好的 checkpoint；测试集只在该 checkpoint 上评测。

### 正式评测规则

正式 PFR 结果必须：

1. 对 `bundles.json` 中全部 18,357 个有元数据的商品排序；
2. 从候选中去除用户历史商品，但若真实 target 在历史中则仍保留 target；
3. 按分数降序排序，同分时按商品数值 ID 保证确定性；
4. 报告 `NDCG` 和 `Recall` 的 @5、@10、@20、@50；
5. 除全体测试集外，还报告 `cold_start_target`、`long_tail_target`、`sparse_user` 三个子集。

完整规范见[统一数据与评测协议](01_unified_data_and_evaluation_protocol.md)。

## 4. PFR 模型设计

### 4.1 Interaction Only

每个商品有可训练的 ID embedding。用户查询向量是历史商品 embedding 的平均，候选向量是候选商品 embedding，二者用归一化点积排序。

优点是直接、可扩展，并能学习协同交互关系；局限是当前 v1 不区分历史顺序、位置或最近行为的重要性，本质上更接近“历史兴趣平均”的检索模型。

### 4.2 Semantic Only

历史商品使用预先生成的 sentence-T5 语义向量；候选商品由 `BundleEncoder` 编码：

- 主体商品的 title、description、category 字段；
- 主体商品与最多 7 个 `bought_together` 商品构成的局部 bundle；
- 字段注意力和局部商品注意力。

这一路希望帮助长尾、稀疏交互和冷启动目标，但其效果必须由单独子集指标验证，而不能只凭全体指标推断。

### 4.3 Fusion/concat

Fusion 将交互查询与语义查询拼接后投影；候选侧也将 ID embedding 与 bundle embedding 拼接投影，然后以同一排序接口打分。`concat` 是第一阶段消融；`gated` 是后续候选方案，只有 concat 和单路对照稳定后才应启动。

## 5. 已完成的工程工作

- 已固定 Sports 的数据来源、数值 ID、切分、bundle 和子集掩码；
- 已实现全库候选排序、历史过滤、单目标 NDCG/Recall；
- 已实现 `BundleEncoder` 与字段/局部商品注意力输出；
- 已实现 Semantic、Interaction、Fusion/concat 统一训练脚本；
- 已完成 PFR 三路模型 seed 2026 的正式单 seed 运行；
- 已保存 ActionPiece、SASRec 等已有基线的来源和质量核验材料。

运行模型相关单元测试：

```bash
cd pathfusionrec
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 6. PFR v1 的 seed 2026 结果

实验使用代码提交 `6e0928a840435a55636e808b9b0e68a647807e9b`，三路模型固定训练 100 epoch、batch size 256、hidden dim 128、255 个随机负例、学习率 0.001；仅模型模式不同。

| 实验 | 最佳 epoch | 验证 NDCG@10 | Test NDCG@5 | Test Recall@5 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| PFR-Interaction-v1 | 28 | 0.022435 | 0.012148 | 0.019102 | **0.015959** | **0.030957** |
| PFR-Semantic-v1 | 81 | 0.004933 | 0.002731 | 0.004354 | 0.003636 | 0.007163 |
| PFR-Fusion-Concat-v1 | 10 | 0.004816 | 0.002199 | 0.003596 | 0.003194 | 0.006714 |

原始产物位于：

```text
experiment_results/04_unified_next_item/
├── PFR-Interaction-v1/seed2026/
├── PFR-Semantic-v1/seed2026/
└── PFR-Fusion-Concat-v1/seed2026/
```

每个 seed 目录包含 `metrics.json`、`training_history.csv`、`best_model.pth`、`run.log`、命令、配置和运行环境摘要。

### 结果解读

1. **Interaction Only 是当前最强模型。** 它的 Test NDCG@10 分别约为 Semantic 的 4.39 倍、Fusion/concat 的 5.00 倍，说明当前有效信息主要来自商品 ID 交互。
2. **concat Fusion 没有获得互补收益。** 它整体低于 Semantic，且最佳验证轮次很早（epoch 10），当前实现尚不能说明语义与交互的融合有效。
3. **冷启动没有被解决。** 317 个 cold-start target 上，三种模型的 Recall@20 都是 0；Semantic 和 Fusion 仅在 @50 各命中约一个目标。
4. **这些只是单 seed 结果。** 它们可用于筛选下一轮方向，不能报告为均值、不能做显著性结论，也不能直接成为论文主表。

## 7. 基线与可比性边界

项目保留了 ActionPiece 的五 seed 结果：平均 Test NDCG@10 为 `0.023807 ± 0.000738`。它表面上高于 PFR Interaction v1 的单 seed 数字，但**目前不能把两者直接写成统一主表对比**，原因是：

- PFR 使用统一协议规定的显式全库排序和历史过滤；
- 现有 ActionPiece 交付采用原生生成式候选与指标流程，缺少逐用户预测、统一子集指标，以及可供核验的显式全库候选排序；
- 因此两者的候选集合和过滤规则尚未被证明完全一致。

基线质量核验还指出：目前没有任何基线同时满足“固定 `sports_protocol`、五 seed、显式全库候选、@5/@10/@20/@50、三个子集、闭合哈希记录”这六项要求。详细结果见[基线质量核验报告](../experiment_results/05_baseline_quality_verification/verification_report.md)。

这不表示 ActionPiece 无效，而是表示在完成统一重评前，它应标注为“项目参考/原生生成式评测结果”，不能用于宣称 PFR 相对其提升或下降多少。

## 8. 复现和日常命令

### 建立环境

```bash
cd pathfusionrec
conda env create -f environment.yml
conda activate pathfusionrec
```

也可以使用 Python 虚拟环境安装项目 README 中列出的依赖。

### 构建或核验统一数据协议

```bash
python3 scripts/prepare_sports_protocol.py
```

新环境若缺少 ActionPiece 兼容处理数据，可使用 `--bootstrap-actionpiece`；重新下载原始数据时再增加 `--force-download`。

### 运行单个 PFR 实验

下面以 Interaction 为例；正式实验仅应在固定版本、数据哈希和实验卡后运行。

```bash
PYTHONPATH=src python3 scripts/run_unified_next_item.py \
  --mode interaction \
  --epochs 100 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --negative-count 255 \
  --hidden-dim 128 \
  --max-history-length 20 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --seed 2026 \
  --device cuda \
  --output-dir experiment_results/04_unified_next_item/PFR-Interaction-v1/seed2026
```

### 重建基线质量核验报告

```bash
python scripts/verify_baseline_delivery.py
```

## 9. 当前限制

- PFR v1 仅有 seed 2026；没有五 seed 均值和样本标准差；
- PFR 的三个 seed 目录尚缺 `experiment_card.md`、统一数据文件哈希和 `qq_report.txt`，其中后者被已有 `sha256sum.txt` 引用；
- ActionPiece、SASRec、VQ-Rec 尚未形成可直接进入 PFR 统一主表的完整对照；
- Interaction v1 丢失顺序信息；
- Semantic/Fusion v1 目前未展示冷启动或融合收益；
- 当前结果是 next-item 推荐结果，不能替代路径生成的有效性、多样性或覆盖率评价。

## 10. 下一步路线

### 阶段 A：先解决可比性

1. 新建专用分支 `experiment/ap-unified-reeval-v1`；
2. 不重训 ActionPiece，复用已有五个最佳 checkpoint；
3. 先对 seed 2026 做统一评测适配和小样本全库吞吐测试；
4. 若全库重评成本可接受，再重评余下四个 seed；否则明确保留“不可直接比较”的边界。

### 阶段 B：低成本验证 PFR 改进方向

1. 暂不投入 Semantic/Fusion v1 的剩余四个 seed；
2. 新建 `PFR-Interaction-v2-objective`，先只改变余弦打分的 temperature，其他设置保持不变；
3. 先跑 seed 2026；若不低于 Interaction v1，再单独测试 hard negatives；
4. 不要在同一版本同时修改温度、负采样、序列编码器和融合结构。

### 阶段 C：形成正式证据

当单 seed 改进方向被确认后，运行 Interaction v1 与候选 v2 的五个固定 seed（2024–2028），报告全量和三个子集的均值 ± 样本标准差。之后再引入位置/顺序编码，并在单路模型稳定后重新设计门控或残差融合。

## 11. 读者应如何使用当前结果

- 可以说：PFR v1 已完成统一协议下的单 seed 消融，Interaction 信号目前明显优于 Semantic 和 concat Fusion。
- 不可以说：PFR 已经在正式五 seed 主表上优于或劣于 ActionPiece。
- 不可以把：bundle smoke、随机负采样实验、原生生成式 ActionPiece 结果与 PFR 全库排序结果混为同一张主表。
- 每次数据、模型、训练目标、候选规则或超参数变化，都应新建实验编号和结果目录，保留旧版本作为历史记录。
