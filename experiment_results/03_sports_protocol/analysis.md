# Sports Protocol 数据构建与验收记录

构建日期：2026-07-26  
数据集：AmazonReviews2014 / Sports_and_Outdoors  
协议版本：`sports_protocol` v1

## 来源核实

本任务使用的数据集确实是 Google DeepMind
[ActionPiece](https://github.com/google-deepmind/action_piece) 的公开复现任务
之一，不需要寻找替代数据集。核实时仓库 `main` 提交为
`ae8e61a89ade8d545a16119bcb5b3a43d9da852f`：

- README 的 quick start 与复现实验均使用
  `--category=Sports_and_Outdoors`；
- `genrec/datasets/AmazonReviews2014/dataset.py` 下载
  `reviews_Sports_and_Outdoors_5.json.gz` 和
  `meta_Sports_and_Outdoors.json.gz`；
- 同一文件生成 `all_item_seqs.json` 与 `id_mapping.json`；
- `genrec/datasets/AmazonReviews2014/config.yaml` 使用
  `split: leave_one_out`；
- `genrec/models/ActionPiece/config.yaml` 使用
  `max_item_seq_len: 20`。

原始文件来自 Julian McAuley 发布的
[Amazon Reviews 2014 分类数据](https://snap.stanford.edu/data/amazon/productGraph/)。

## 构建统计

| 项目 | 数量 |
|---|---:|
| 用户 | 35,598 |
| 商品 | 18,357 |
| 完整交互 | 296,337 |
| 训练交互（每序列去掉最后两个目标） | 225,141 |
| 训练前缀样本 | 189,543 |
| 验证样本 | 35,598 |
| 测试样本 | 35,598 |
| 有元数据的候选商品 | 18,357 |
| 缺失元数据的映射商品 | 0 |
| 验证目标缺失候选 | 0 |
| 测试目标缺失候选 | 0 |

全部 35,598 个用户序列长度至少为 3，没有因协议的最短长度要求被跳过。

## Bundle 统计

这里的 bundle 是 **Amazon `related.bought_together` 关联捆绑包**，不是实际订单
购物车。

| 项目 | 数值 |
|---|---:|
| bundle 数 | 18,357 |
| 单品 bundle 数 | 7,198 |
| 单品 bundle 比例 | 39.2112% |
| 平均长度 | 1.825244 |
| P0 / P25 / P50 / P75 | 1 / 1 / 2 / 2 |
| P90 / P95 / P99 / P100 | 3 / 3 / 3 / 4 |

虽然协议允许最大长度 8，本数据经“只保留映射商品”过滤后的实际最大长度为 4。

## 测试子集

子集只按训练交互计算，未使用 validation 或 test 交互：

| 子集 | 样本数 | 占全部测试 |
|---|---:|---:|
| 全部 | 35,598 | 100.00% |
| 冷启动目标（训练计数 = 0） | 317 | 0.89% |
| 长尾目标（训练计数 ≤ 5） | 9,962 | 27.98% |
| 稀疏用户（训练历史 ≤ 5） | 22,639 | 63.60% |

长尾定义包含冷启动目标。冷启动样本不为 0，因此保留协议原定义，无需回退。

## 异常与协议澄清

原协议表格把 validation 目标写成 `i(L-2)`，同时把 test 历史写成
`i1...i(L-2)`；这与文末“验证/测试目标分别为倒数第二/最后一个”、ActionPiece
leave-one-out 实现及常规无泄漏训练划分不一致。已统一修正为：

- validation：历史 `i1...i(L-2)`，目标 `i(L-1)`；
- test：历史 `i1...i(L-1)`，目标 `iL`。

除该 off-by-one 文档问题外，未发现数据异常：映射往返一致、元数据全覆盖、没有
重复的目标元数据记录。

## 验收结果

以下校验全部通过：

- `id_mapping.json` 的正反映射逐项可往返，padding ID 固定为 0；
- 每个 validation/test 目标分别等于原序列倒数第二/最后一个商品；
- 所有 train/validation/test 历史长度不超过 20；
- 每个 bundle 的主体商品位于 `item_ids[0]`，商品去重且长度不超过 8；
- `subset_masks.json.test` 与 `split.json.test` 的 sample/user/target 逐项对齐；
- `src/pathfusionrec/evaluation.py` 的 4 个单元测试通过；
- 完美分数得到全部 `NDCG@5/10 = Recall@5/10 = 1.0`；
- 固定随机分数的指标均处于 `[0, 1]`。

在不下载、不重建 ActionPiece 源交互的情况下第二次执行
`python scripts/prepare_sports_protocol.py`，四个产物的 SHA-256 与首次执行
完全相同：

| 文件 | SHA-256 |
|---|---|
| `split.json` | `5a4e8523dc374f42d394cd64dc1df745002d7f278ba7f85ad5c7e6b25fe0f50a` |
| `bundles.json` | `48f2eceedec38e7d25fa7fa021101c21c6c7f15ec616d20400658465038788a2` |
| `subset_masks.json` | `602456e6657457e46100a7ab259eeb3889e5aeb54de9d0e4c5e5f43bf09eccf3` |
| `statistics.json` | `29a68df2c9d07bba1aa3e6c63da416bb08dfd7a6c03b8ed4a5707b2f409b0ad4` |

