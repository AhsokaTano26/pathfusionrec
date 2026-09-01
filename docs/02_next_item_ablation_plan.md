# PathFusionRec 三路下一物品推荐实验

## 目标

在固定 `sports_protocol` 切分、候选目录与全库评测下比较三类模型：

| 模型 | 查询表示 | 候选表示 | 要验证的问题 |
|---|---|---|---|
| Semantic Only | 历史商品的 sentence-T5 语义均值 | 标题、描述、类目与局部 bundle 注意力编码 | 对冷启动/长尾是否更稳健 |
| Interaction Only | 可训练的历史商品 ID 均值 | 可训练物品 ID 向量 | 个性化交互信号的基线能力 |
| Fusion | 语义查询与交互查询拼接投影 | bundle 语义与交互 ID 拼接投影 | 两种信息是否互补 |

Fusion 默认 `concat`，即先使用简单、稳定的拼接加投影。`gated` 选项会学习每个表示维度中交互与语义的权重，作为第二阶段消融；不应在没有 concat 基线的情况下直接报告门控结果。

## 结构化 bundle 语义

每个候选商品均读取固定 `bundles.json`：主体商品在局部物品集合的第一个位置；`related.bought_together` 内的商品最多保留 7 个。`BundleEncoder` 分别对标题、描述和类目建立字段注意力，再以全局语义为条件为局部物品分配注意力，最后融合全局和局部表示。因此 bundle 不是简单平均，且可导出字段/物品注意力用于检查主体与配件的相对作用。

字段文本向量由稳定哈希生成，且不在 validation/test 文本上拟合；商品语义底座使用已经存在的 sentence-T5 128 维缓存。后续可在不改变协议和模型接口的条件下替换字段文本编码器。

## 运行方式

从 `pathfusionrec/` 运行。先做冒烟检查，再在 GPU 上做正式五种子实验；冒烟输出不能写入正式结果目录。

```bash
PYTHONPATH=src python3 scripts/run_unified_next_item.py \
  --mode fusion --epochs 1 --batch-size 16 --negative-count 31 \
  --hidden-dim 32 --max-train-samples 64 --max-validation-samples 16 \
  --max-test-samples 16 --device cpu \
  --output-dir /private/tmp/pathfusionrec-fusion-smoke
```

正式三路对照应固定全部参数和随机种子，只改变 `--mode`：

```bash
for mode in semantic interaction fusion; do
  PYTHONPATH=src python3 scripts/run_unified_next_item.py \
    --mode "$mode" --epochs 100 --batch-size 256 --eval-batch-size 256 \
    --negative-count 255 --hidden-dim 128 --lr 0.001 --seed 2026 \
    --device cuda \
    --output-dir "experiment_results/04_unified_next_item/${mode}_seed2026"
done
```

随后对 seed 2024–2028 重复。每次运行会保存 `metrics.json`、逐 epoch 验证曲线 `training_history.csv` 和以验证 NDCG@10 选择的 `best_model.pth`。

## 报告规范

主表报告五个 seed 的均值和样本标准差，至少列出 NDCG/Recall@5、@10、@20、@50。测试结果按 `all`、`cold_start_target`、`long_tail_target` 与 `sparse_user` 分开输出。路径生成是后续独立任务：只有在同一用户、历史前缀和候选约束下生成连续路径并定义有效性与多样性指标后，才能与 next-item 指标并列报告。

不将早期 bundle smoke 的主体检索 Recall@10、ActionPiece/SASRec 的不同协议结果，或仅一个 seed 的数值写入该主表。
