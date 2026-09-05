# PathFusionRec

> 面向生成式推荐的结构化语义—交互融合研究工程。

PathFusionRec 的目标是在统一、可复现的 Amazon Reviews 2014 实验协议上，比较并融合三类信息：

- **Interaction Only**：仅使用用户历史交互序列；
- **Semantic Only**：仅使用商品文本语义及关联 bundle；
- **Fusion**：融合序列交互信号、商品全局语义与 bundle 局部结构。

当前仓库已交付统一 Sports_and_Outdoors 数据构建、全库评测工具、bundle 语义编码器和真实数据 smoke test。三类正式模型的统一协议训练与消融实验仍在后续阶段，不能将 smoke test 指标当作正式 next-item 推荐结果。

`../action_piece/` 是独立维护的上游 ActionPiece 复现仓库；本仓库不修改其中的训练代码，而是复用与其一致的 Sports_and_Outdoors 数据口径来开展新方法实验。

## 实验历史与可比性边界

项目中的第一次实验（负责人：杨尚鑫）主要设置按原论文对齐，但运行的是 **PyTorch 复现代码**，不是论文原始的 TensorFlow 实现；测试采用候选集评测，即每位用户使用 **1 个真实物品 + 100 个负样本**。该结果可用于记录复现过程与候选集协议下的表现，但不得与以下结果直接横向比较：

- 本仓库 Sports_and_Outdoors 协议中的全库候选评测；
- 2026-07-30 的 SASRec/ActionPiece 全物品结果；
- 之后的 Semantic Only、Interaction Only、Fusion 正式结果。

需要进行比较时，必须让数据划分、候选目录、负采样规则、代码实现与选模方式完全一致；否则应分别报告，并在表格标题中注明评测协议。

## 目录

```text
pathfusionrec/
├── data/
│   ├── AmazonReviews2014/Sports_and_Outdoors/
│   │   ├── raw/                         # Amazon 原始评论与元数据（本地大文件）
│   │   └── processed/                   # ActionPiece 兼容交互与文本特征（本地大文件）
│   └── processed/sports_protocol/       # 已提交的统一协议产物
│       ├── split.json
│       ├── bundles.json
│       ├── subset_masks.json
│       └── statistics.json
├── docs/
│   └── 01_unified_data_and_evaluation_protocol.md
├── experiment_results/                  # 每次实验的原始指标与分析
├── scripts/
│   ├── prepare_sports_protocol.py       # 构建/验收统一数据协议
│   └── run_real_bundle_smoke.py          # BundleEncoder 真实数据 smoke test
├── src/pathfusionrec/
│   ├── evaluation.py                    # 全库排序、历史过滤、NDCG/Recall
│   └── models/bundle_encoder.py          # 结构化 bundle 语义编码器
└── tests/
    └── test_bundle_encoder.py
```

## 环境要求与安装

- Python 3.10 或更高版本；
- PyTorch 2.0 或更高版本；
- scikit-learn 1.3 或更高版本；
- 正式模型训练是否使用 GPU 由对应模型脚本决定；数据构建和单元测试可在 CPU 上运行。

建议在项目根目录创建虚拟环境并安装依赖：

```bash
cd pathfusionrec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.0' 'scikit-learn>=1.3'
```

当前代码以 `src/` 布局运行，执行脚本和测试时请显式设置 `PYTHONPATH=src`。

## 快速验证

在 `pathfusionrec/` 目录执行：

```bash
# BundleEncoder 的形状、掩码与输入校验测试
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 统一评测的完美分数、随机分数与历史过滤自检
PYTHONPATH=src python3 src/pathfusionrec/evaluation.py
```

预期结果：单元测试全部通过；评测自检中的完美分数应得到 `NDCG@5 = Recall@5 = NDCG@10 = Recall@10 = 1.0`。

如使用 Conda，也可按仓库中的环境定义创建环境：

```bash
conda env create -f environment.yml
conda activate pathfusionrec
```

需要重建独立的基线质量核验报告时，执行：

```bash
python scripts/verify_baseline_delivery.py
```

## 统一数据与评测协议

完整规范见 [统一数据协议与实验评测指南](docs/01_unified_data_and_evaluation_protocol.md)。所有新模型必须读取 `data/processed/sports_protocol/` 的固定产物，不得各自重新筛选用户、重排时间、重映射 ID 或自行定义候选集。

### 数据来源与当前规模

数据集为 **Amazon Reviews 2014 / Sports_and_Outdoors**，与 ActionPiece 的数据口径对齐：

| 项目 | 当前值 |
|---|---:|
| 用户数 | 35,598 |
| 物品数 | 18,357 |
| 交互数 | 296,337 |
| 训练前缀样本 | 189,543 |
| 验证样本 | 35,598 |
| 测试样本 | 35,598 |
| 候选物品 | 18,357（全部有元数据） |

`id_mapping.json` 中的数值用户/物品 ID 是唯一映射来源，padding ID 为 0。PathFusionRec 的 `actionpiece.json` 已与 `../action_piece` 中对应产物做过 SHA-256 一致性核验。

### Leave-one-out 划分

对长度至少为 3 的用户行为序列 `i1, i2, ..., iL`：

| 划分 | 历史 | 目标 |
|---|---|---|
| train | 每个前缀 `i1...it`，`1 ≤ t ≤ L-3` | `i(t+1)` |
| validation | `i1...i(L-2)` | `i(L-1)`，倒数第二项 |
| test | `i1...i(L-1)` | `iL`，最后一项 |

所有历史最多保留最近 20 项。validation 仅用于选模和早停；测试集只用于最终 checkpoint 的评测。

### 全库候选评测

正式结果采用所有有元数据商品的全库候选排序。对每个样本：

1. 从候选中删除历史出现过的商品；
2. 若 target 已在历史中，仍保留 target；
3. 按模型分数降序排序；分数相同时按数值商品 ID 升序，保证可重复；
4. 计算 `NDCG@5`、`NDCG@10`、`Recall@5`、`Recall@10`。

每个样本只有一个真实目标，因而本设置下 Recall 与 HR 数值相同。主表不得混用“1 正样本 + 99 随机负样本”的采样评测结果。

统一实现位于 `src/pathfusionrec/evaluation.py`，主要接口为：

- `filter_history_candidates(candidate_ids, history, target)`；
- `rank_candidates(candidate_ids, scores, history, target, top_k=10)`；
- `single_target_metrics(ranked_item_ids, target)`；
- `evaluate_ranked_lists(ranked_item_ids, targets)`。

## 构建或重建 Sports 协议数据

仓库已提交当前版本的协议产物。若本地 ActionPiece 兼容数据已经存在，直接重建：

```bash
python3 scripts/prepare_sports_protocol.py
```

新克隆且尚无 `data/AmazonReviews2014/Sports_and_Outdoors/processed/` 时，可使用官方 Amazon Reviews 2014 文件启动构建：

```bash
python3 scripts/prepare_sports_protocol.py --bootstrap-actionpiece
```

如需重新下载原始文件：

```bash
python3 scripts/prepare_sports_protocol.py \
  --bootstrap-actionpiece \
  --force-download
```

脚本会生成并验收以下产物：

| 文件 | 用途 |
|---|---|
| `split.json` | 固定 train/validation/test 样本；每个样本含 `sample_id`、`user_id`、`history`、`target` |
| `bundles.json` | 每个候选商品的主体优先关联 bundle 与全局文本字段 |
| `subset_masks.json` | 全量、冷启动目标、长尾目标、稀疏用户的测试掩码 |
| `statistics.json` | 规模、bundle 分布、来源哈希、产物哈希与验收结果 |

构建规则包括：主体商品始终在 `item_ids[0]`；共购商品只保留映射内物品、去重、最多 8 件；没有有效共购品的商品使用单品 bundle。它表示 Amazon `related.bought_together` 关联关系，**不是实际订单购物车**。

重建后请核对 `statistics.json` 的 `artifact_sha256` 和 `validation`，并将数据规则变更、脚本版本、产物哈希记录到对应实验分析中。原始数据和 ActionPiece 处理数据体积较大，保留为本地文件；协议产物、脚本、文档和统计信息应提交版本控制。

## BundleEncoder

`BundleEncoder` 位于 `src/pathfusionrec/models/bundle_encoder.py`，将 bundle 的全局文本语义与局部商品结构融合，而非简单平均商品向量：

1. **全局语义**：对 title、description、category 的预计算向量分别投影，并学习字段注意力；
2. **局部结构**：商品向量在全局语义条件下计算注意力；`item_mask` 排除 padding；
3. **可选角色特征**：可接收数量、价格区间等 `item_role_features`；
4. **融合输出**：拼接全局与局部向量，经投影后输出 bundle 表示。

前向调用示例：

```python
from pathfusionrec import BundleEncoder

encoder = BundleEncoder(
    input_dim=128,
    hidden_dim=128,
    output_dim=128,
    dropout=0.1,
)
output = encoder(global_fields, item_embeddings, item_mask)

bundle_embedding = output.embedding
field_attention = output.field_attention
item_attention = output.item_attention
```

输入形状：

| 输入 | 形状 / 要求 |
|---|---|
| `global_fields[name]` | `[batch, input_dim]`；默认字段为 title、description、category |
| `item_embeddings` | `[batch, max_items, input_dim]` |
| `item_mask` | `[batch, max_items]` 的布尔张量；每个 bundle 至少有一个 `True` |
| `item_role_features` | 可选；`[batch, max_items, role_feature_dim]` |

注意力便于审计模型如何分配字段和包内商品权重，但不应被直接解释为因果重要性。

## 真实数据 Smoke Test

下面的命令用于验证真实 Amazon 元数据读取、变长 bundle padding、语义向量加载和 BundleEncoder 训练链路：

```bash
PYTHONPATH=src python3 scripts/run_real_bundle_smoke.py \
  --max-bundles 128 \
  --max-items 8 \
  --epochs 3 \
  --batch-size 64 \
  --seed 2026 \
  --output experiment_results/04_20260730_bundle_smoke/metrics.json
```

该脚本使用原始 `related.bought_together`，训练 bundle 表示检索其主体商品的缓存 sentence-T5 向量。它是工程 smoke test，**不是**正式 next-item 推荐：主体商品被包含于 bundle 中，且其小规模随机切分不可替代统一 leave-one-out split。

已有实验记录见：

- `experiment_results/01_20260725_real_bundle_smoke_20/`：20 个 bundle 的最小链路验证；
- `experiment_results/02_20260725_real_bundle_smoke/`：128 个 bundle 的真实数据 smoke test；
- `experiment_results/03_sports_protocol/`：统一数据协议构建与验收。

## 实验记录规范

每次实验建立独立目录：

```text
experiment_results/NN_YYYYMMDD_实验简称/
├── metrics.json
└── analysis.md
```

- `NN` 从 `01` 起递增；
- `metrics.json` 保存程序原始输出、配置、随机种子与数据版本；
- `analysis.md` 使用 [分析模板](experiment_results/analysis_template.md)，如实记录目标、设置、结果、限制与下一步；
- 失败和负结果同样必须保留，避免重复试错；
- 正式多种子结果须同时给出逐种子值、均值与样本标准差，不能只保存最优值。

## 当前研究状态与路线图

| 模块 | 状态 | 说明 |
|---|---|---|
| Sports 统一数据构建 | 已完成 | 固定 split、bundle、子集掩码、统计与哈希已生成 |
| 全库评测工具 | 已完成 | 历史过滤、确定性排序、单目标 NDCG/Recall 自检通过 |
| BundleEncoder | 已完成基础实现 | 单元测试和真实数据 smoke test 通过 |
| Interaction Only 正式模型 | 待接入 | 必须使用统一 split 与评测 |
| Semantic Only 正式模型 | 待实现/接入 | 必须使用统一 split 与评测 |
| Fusion 正式模型 | 待实现/接入 | 应与前两者做消融 |
| 正式多种子结果 | 待完成 | 需报告全量及子集指标、均值与样本标准差 |

下一阶段的首要目标不是继续扩大 smoke test，而是将 Interaction Only、Semantic Only 和 Fusion 接入同一份 `split.json`、候选目录及 `evaluation.py`，再完成多种子消融实验。ActionPiece 复现中已观察到后期收敛与早停敏感性，后续比较应保留逐轮验证曲线、最佳 checkpoint、总更新步数及统一计时边界。

## 相关材料

- [统一数据协议与实验评测指南](docs/01_unified_data_and_evaluation_protocol.md)
- [Sports 协议构建与验收记录](experiment_results/03_sports_protocol/analysis.md)
- [实验记录规范](experiment_results/README.md)
- [BundleEncoder 单元测试](tests/test_bundle_encoder.py)
- 上游复现仓库：`../action_piece/`

## 贡献与约束

- 遵循 Python 四空格缩进、`snake_case` 命名与现有代码风格；
- 不提交原始数据、缓存、checkpoint、日志、TensorBoard 或 Weights & Biases 输出；
- 修改数据规则、候选定义、评测指标或 split 时，必须同步更新协议、重建统计与实验记录；
- 提交前至少运行本 README 中的快速验证命令，并在实验分析中写明执行命令和结果。
