# 实验卡：TIGER 语义 ID 生成检索（Stage 3 Retriever）· seed 2026

- **实验编号**：05_tiger_semantic_id / TIGER-v1 / seed2026
- **方法**：TIGER 式生成检索。随机初始化的小型 T5 在 RQ-VAE Semantic ID 上执行下一商品解码，并使用前缀约束的 beam search 在商品目录内检索候选。
- **使用代码提交**：`ff031c0bc79051298192c3c0bbe26eb65bec378d`
- **运行时间**：2026-09-07 13:47 重启，至 2026-09-08 02:11 完成（首次运行中间发生一次 AutoDL 关机；重启后使用完全相同的配置）。
- **GPU / 环境**：RTX 4090；环境 `/root/autodl-tmp/envs/pathfusionrec`（PyTorch 与 Transformers；没有加载预训练检查点，T5 从随机初始化开始训练）。

## 固定协议与数据

- 用户数 35,598；候选商品数 18,357；训练样本数 189,543（约 741 steps / epoch）；验证集与测试集各 35,598 条。
- 测试子集掩码：`cold_start_target`、`long_tail_target`、`sparse_user`。
- 文本输入：处理目录中的 `metadata.sentence.json` 与 `id_mapping.json`。
- 内容向量：`tiger_sentence_t5_768/vectors.npy`，SHA-256 为 `89791f29…0025`，形状 `[18,357, 768]`，`float32`。
- Semantic ID：`semantic_ids.json`，SHA-256 为 `0eae6e9e…4e8`；共有 18,357 个唯一四 token ID。RQ-VAE 三级码本大小为 `[4, 16, 256]`，实际使用 `[4, 16, 226]`；第三级有 30 个死码，重建损失从 1.48 漂移至 1.54，属于需要报告的 RQ-VAE 限制。

## 模型与训练设置

- T5：`d_model=128`、`d_ff=1024`、4 层、6 个注意力头、`d_kv=64`、dropout=0.1。词表包含三级残差码本 token、碰撞消歧 token、2,000 个用户哈希桶，以及 BOS/EOS/PAD。
- Batch size=256；AdamW，学习率 `1e-3`、权重衰减 `1e-4`；seed=2026；训练 100 epochs，每 10 epochs 验证一次；beam size=50，检索 rank `k=50`。
- 检查点选择规则：验证集 NDCG@10 最高。

## 本次实验实际验证的内容

本实验是 TIGER 风格生成式下一商品检索的单 seed 正式基线：先将 Sentence-T5 的 768 维商品文本向量通过 RQ-VAE 压缩并离散化为 Semantic ID；再由 T5 基于用户历史生成下一商品的 Semantic ID，通过约束解码映射回全库商品并计算 Top-K 指标。它验证的是“内容语义向量 → 离散 Semantic ID → 生成式检索”这条完整链路在统一 Sports next-item 协议下是否成立；它不衡量预训练语言模型的收益，因为本次 T5 没有加载预训练权重。

## 结果

- **最佳 epoch**：90；**最佳验证 NDCG@10**：0.02127。
- 验证 NDCG@10：0.01644（epoch 10）→ 0.02127（epoch 90）→ 0.02111（epoch 100）；训练损失：1.8333 → 1.7002。
- **全量测试集（k=10）**：NDCG@10 **0.01478**，Recall@10 **0.02891**。
- 全量测试指标：NDCG@5 0.01114；Recall@5 0.01759；NDCG@10 0.01478；Recall@10 0.02891；NDCG@20 0.01914；Recall@20 0.04621；NDCG@50 0.02582；Recall@50 0.07981。
- 子集测试：`sparse_user` NDCG@10 0.01524 / Recall@10 0.02959；`cold_start_target` 全部指标为 0；`long_tail_target` 接近 0（NDCG@10 为 `3.9e-05`）。

## 与 PFR 主方法及内部消融的单 seed 定位

TIGER 是本项目复现的**外部生成式检索基线**；PFR-Interaction（图中标记为 PathFusionRec）是本项目的**主方法**；PFR-Semantic 与 PFR-Fusion（concat）是主方法的**内部消融变体**。四者均采用 `sports_protocol`、18,357 个全量候选商品以及 seed 2026，所以可作数值上的单 seed 对照；但图表与论述必须保留这一方法角色，不能将 PFR 主方法误写为普通基线。TIGER 的 NDCG@10 为 0.01478，低于 PathFusionRec 的 0.01596（低 7.4%），但高于 semantic-only 消融的 0.00364 与 concat-fusion 消融的 0.00319。完整比较见 `comparison_all_metrics.{png,svg}` 与 `comparison_subgroups.{png,svg}`。

## 结论与限制

- 作为外部生成式检索基线，TIGER 的 Semantic ID 路线在该协议上可行，NDCG@10 已达到 PathFusionRec 主方法的 92.6%；但从验证集到测试集的 NDCG@10 相对下降约 30.5%，稳定性仍需关注。
- 生成的商品 ID 集中在热门、已见目录区域：本 seed 对新商品（`cold_start_target`）与极端长尾（`long_tail_target`）的检索接近 0。PathFusionRec 主方法同样无法命中冷启动目标，但其长尾表现更好。
- 当前仅有单个 seed，不能进入最终的多 seed 主结果表。ActionPiece 和 SASRec 的现有结果尚未满足统一候选集与完整评估语义要求，只能作为独立参考，不能纳入上述直接柱状比较。

## 本目录产物

- `best_model.pth`、`metrics.json`、`config.json`、`delivery_validation.txt`（PASS）、`training_curves.png`、`test_subset.png`、`sha256sums.txt`
- `comparison_all_metrics.{png,svg}`：PathFusionRec 主方法、两个内部消融与 TIGER 外部基线在全量测试集上的 NDCG / Recall@{5,10,20,50}。
- `comparison_subgroups.{png,svg}`：同一组方法在全量与三个测试子集上的 NDCG@10 / Recall@10。
- 原始实验产物的完整 SHA-256 记录在 `sha256sums.txt`；新增对比图由 `scripts/plot_tiger_comparison.py` 可复现生成。
