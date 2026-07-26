# 统一数据协议与实验评测指南

> 负责人：倪榕键  
> 数据集：AmazonReviews2014 / Sports_and_Outdoors  
> 目标：为 ActionPiece、Semantic Only、Interaction Only 与 Fusion 提供同一份可复现的数据划分、捆绑包定义和评测实现。

## 1. 基本原则

以 `data/AmazonReviews2014/Sports_and_Outdoors/processed/` 中现有的
`all_item_seqs.json` 和 `id_mapping.json` 为唯一交互来源。它们与当前
ActionPiece 使用的数据口径一致；不得重新过滤用户、重排时间或重映射 ID。

首次在空目录中构建时，允许执行：

```bash
python scripts/prepare_sports_protocol.py --bootstrap-actionpiece
```

该选项只下载 ActionPiece 代码指定的 Amazon Reviews 2014
`reviews_Sports_and_Outdoors_5.json.gz` 与
`meta_Sports_and_Outdoors.json.gz`，并按 ActionPiece 的稳定时间排序和首次出现
顺序生成上述两个交互来源文件。来源建立后，日常重建不再解析或过滤原始评论：

```bash
python scripts/prepare_sports_protocol.py
```

所有新模型读取 `data/processed/sports_protocol/` 的产物，不直接各自处理
原始交互。大体积数据文件被 `.gitignore` 忽略；生成脚本、配置、
`statistics.json` 和协议文档必须提交 Git。

## 2. 统一划分

对每个长度至少为 3 的用户序列 `i1, i2, ..., iL`，使用 ActionPiece 的
leave-one-out 规则：

| 划分 | 历史 | 目标 |
|---|---|---|
| train | 每个前缀 `i1...it`，`1 ≤ t ≤ L-3` | `i(t+1)` |
| validation | `i1...i(L-2)` | `i(L-1)`（倒数第二个商品） |
| test | `i1...i(L-1)` | `iL`（最后一个商品） |

原始草案在 validation/test 行存在 off-by-one；上表以 ActionPiece 实现和本文
验收条件为准。训练展开不跨入 validation/test 目标。

- 每条历史只保留最近 20 个商品，与 ActionPiece 的
  `max_item_seq_len=20` 对齐。
- `split.json` 中商品和用户均使用 `id_mapping.json` 已有的数值 ID，不创建
  新映射。
- validation 仅用于早停和超参数选择；测试集只在选定 checkpoint 上评测一次。
- 固定输出 `split.json`，包含 `train`、`validation`、`test` 三部分。每个样本
  保存 `user_id`、`history`、`target`。

## 3. 关联捆绑包构造

从 `raw/meta_Sports_and_Outdoors.json.gz` 读取商品元数据。对候选目录中每个
商品 `i` 构造：

```text
B(i) = [i] + related.bought_together(i)
```

- 只保留出现在 `id_mapping.json` 中的商品；去重，且主体商品 `i` 必须排在首位。
- 最多保留 8 个商品；没有有效共购商品时使用单品 bundle `[i]`。
- bundle 的全局字段为主体商品的 `title`、`description`、`categories`；缺失
  文本字段写空字符串。嵌套的 Amazon categories 按原顺序展平并去重。
- 元数据文件中不存在的映射商品不会进入候选目录，也不会伪造空元数据；其数量
  和 validation/test 目标缺失数写入 `statistics.json`。
- 该定义是“基于 Amazon 共购元数据的关联捆绑包”，并非真实订单购物车；在论文
  和分析中必须如实表述。

`bundles.json` 格式：

```json
{
  "B000000000": {
    "item_ids": [123, 456, 789],
    "title": "...",
    "description": "...",
    "categories": ["Sports & Outdoors", "..."],
    "source": "bought_together"
  }
}
```

`statistics.json` 保存用户数、商品数、交互数、训练/验证/测试样本数、单品
bundle 比例、bundle 长度均值与线性插值分位数、输入/产物 SHA-256 和验收结果。

## 4. 统一评测

所有模型均对同一个候选目录评分：`bundles.json` 的全部键所代表的有元数据
商品。对每个用户：

1. 从候选中移除历史中已经交互的商品；真实 `target` 必须保留；
2. 对剩余商品按分数降序排列，数值商品 ID 作为同分时的确定性次排序键，取
   Top-K；
3. 计算 `NDCG@5`、`NDCG@10`、`Recall@5`、`Recall@10`；每个样本只有一个
   真实目标。

主结果使用**全库候选排序**，不得混用“1 个正样本 + 99 个随机负样本”的数值。
若为了调试使用采样候选，必须在实验分析中显著标注，不得与主结果放在同一表格
比较。统一实现位于 `src/pathfusionrec/evaluation.py`。

## 5. 子集评测

所有子集标签只基于每个合格用户原序列去掉最后两个目标后的训练交互生成，不按
训练前缀样本重复计数：

- 冷启动目标：目标商品训练交互数为 0；
- 长尾目标：目标商品训练交互数不超过 5（包含冷启动目标）；
- 稀疏用户：其未截断训练历史长度不超过 5；
- 普通结果：全部测试样本。

`subset_masks.json` 的 `test` 数组与 `split.json.test` 按 `sample_id` 对齐，
每行保存 `user_id`、`target` 与四个布尔标签；`counts` 保存各子集样本数。若
冷启动样本为 0，仍如实记录，不改用全量交互计数。

## 6. 交付目录与验收

目标目录：

```text
data/processed/sports_protocol/
├── split.json
├── bundles.json
├── subset_masks.json
└── statistics.json
```

另提交：

- `scripts/prepare_sports_protocol.py`：从当前数据副本可重复生成上述文件；
- `src/pathfusionrec/evaluation.py`：统一 Top-K 指标与历史过滤函数；
- `docs/01_unified_data_and_evaluation_protocol.md`：本协议的规范副本；
- `experiment_results/03_sports_protocol/analysis.md`：记录本次数据构建统计、
  异常项和验证结果。

验收条件：重新执行脚本后样本数、产物 SHA-256 和文件内容可复现；验证/测试每个
用户的目标分别为原序列倒数第二、最后一个商品；所有 bundle 主体位于
`item_ids[0]`；统一评测脚本对随机分数和完美分数输出合理的指标。
