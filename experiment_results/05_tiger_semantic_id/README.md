# TIGER 语义 ID 实验

该目录用于保存 TIGER 式生成检索实验的说明；每个实际运行应置于
`TIGER-v1/seed<seed>/`，并保留 `metrics.json`、`config.json`、
`delivery_validation.txt` 和最优 checkpoint。不要提交大体积向量缓存或 checkpoint。

方法角色：TIGER 是复现的外部生成式检索基线；PathFusionRec（PFR-Interaction）是
项目主方法，PFR-Semantic 与 PFR-Fusion 是其内部消融。它们可以在统一协议下比较数值，
但图表和论文表述必须保留这一角色区分。

服务器首次运行：

```bash
conda env create -f environment.yml
conda activate pathfusionrec
pip install -e .

PYTHONPATH=src python3 scripts/build_tiger_content_embeddings.py \
  --processed-dir data/AmazonReviews2014/Sports_and_Outdoors/processed \
  --output-dir data/processed/tiger_sentence_t5_768 \
  --device cuda

PYTHONPATH=src python3 scripts/train_tiger_rqvae.py \
  --content-dir data/processed/tiger_sentence_t5_768 \
  --protocol-dir data/processed/sports_protocol \
  --output-dir data/processed/tiger_rqvae_v1 \
  --latent-dim 128 --codebook-sizes 4 16 256 --device cuda
```

先验证 RQ-VAE 的 `delivery_validation.txt`：所有商品必须有唯一四位 Semantic ID，且
三级码本均有至少一个被使用的码。然后运行正式 seed-2026：

```bash
PYTHONPATH=src python3 scripts/train_tiger_retriever.py \
  --protocol-dir data/processed/sports_protocol \
  --semantic-id-dir data/processed/tiger_rqvae_v1 \
  --output-dir experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026 \
  --seed 2026 --batch-size 256 --eval-batch-size 256 \
  --epochs 100 --num-beams 50 --device cuda
```

该结果是单 seed、生成式全目录检索结果；在统一候选与解码语义前，不能与历史
ActionPiece 原生生成式结果作强结论对比。

## 表示输入消融：行为、语义与平衡融合

该消融检验输入 RQ-VAE 的商品表示，而非修改 TIGER decoder：

- **semantic**：冻结的、非 PCA 的 Sentence-T5 768 维向量。
- **behavior**：从 `PFR-Interaction-v1` 最佳检查点提取的 128 维
  `interaction_embedding`。该检查点只能由训练数据训练并依验证集选择，绝不使用测试集。
- **fusion**：语义和行为不直接拼接；二者先独立标准化，再分别经可学习的 128 维分支编码器，
  拼接后由共享编码器压缩、RQ-VAE 量化，并以两个独立 MSE 重建损失等权约束。这避免 768 维语义
  因维数更多而淹没 128 维行为信号。

`latent_dim=32` 是原始 TIGER 的复现点：论文将 768 维 Sentence-T5 向量依次编码为
512、256、128，最后压到 32 维再做三级残差量化。`64`、`128` 只作为 RQ-VAE 容量敏感性检查；
它们不是将行为向量随意升维。

### 1. 导出冻结行为向量

在服务器仓库根目录运行；路径应指向同一 `seed2026` 的 PFR-Interaction 最优检查点：

```bash
PYTHONPATH=src python scripts/build_tiger_behavior_embeddings.py \
  --checkpoint experiment_results/04_unified_next_item/PFR-Interaction-v1/seed2026/best_model.pth \
  --processed-dir data/AmazonReviews2014/Sports_and_Outdoors/processed \
  --output-dir data/processed/tiger_behavior_pfr_interaction_128
```

检查输出 `manifest.json`：`representation` 必须为 `pfr_interaction_embedding`，
`dimension` 必须为 128，且 `item_ids.json` 必须与 `tiger_sentence_t5_768` 完全相同。

### 2. 只运行 RQ-VAE latent 容量筛选

对每个 latent 值分别运行 semantic、behavior、fusion。所有输出只能使用训练商品拟合
标准化统计与 RQ-VAE；这一步不训练 TIGER retriever。

```bash
for LATENT in 32 64 128; do
  PYTHONPATH=src python scripts/train_tiger_rqvae.py \
    --content-dir data/processed/tiger_sentence_t5_768 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/semantic_latent${LATENT} \
    --latent-dim ${LATENT} --codebook-sizes 4 16 256 --seed 2026 --device cuda

  PYTHONPATH=src python scripts/train_tiger_rqvae.py \
    --content-dir data/processed/tiger_behavior_pfr_interaction_128 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/behavior_latent${LATENT} \
    --latent-dim ${LATENT} --codebook-sizes 4 16 256 --seed 2026 --device cuda

  PYTHONPATH=src python scripts/train_tiger_fusion_rqvae.py \
    --semantic-dir data/processed/tiger_sentence_t5_768 \
    --behavior-dir data/processed/tiger_behavior_pfr_interaction_128 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/fusion_latent${LATENT} \
    --branch-dim 128 --latent-dim ${LATENT} --codebook-sizes 4 16 256 --seed 2026 --device cuda
done
```

在三种表示上选用同一个 latent 值：优先选择满足 `delivery_validation.txt`、三级码本均被使用、
死码较少且训练商品重建损失没有明显退化的最小维度。若 32 不满足这些条件，才升至 64，再考虑 128。
不要依据测试集指标选择 latent。

### 3. 使用选定 latent 运行三组正式 TIGER retriever

以下以选择 `LATENT=32` 为例；替换为第 2 步预先锁定的单一值。三组必须使用相同的 seed、
训练轮数、beam 和评测间隔：

```bash
for REPRESENTATION in semantic behavior fusion; do
  PYTHONPATH=src python scripts/train_tiger_retriever.py \
    --protocol-dir data/processed/sports_protocol \
    --semantic-id-dir data/processed/tiger_rqvae_ablation/${REPRESENTATION}_latent32 \
    --output-dir experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026/${REPRESENTATION} \
    --seed 2026 --batch-size 256 --eval-batch-size 256 \
    --epochs 100 --eval-interval 10 --num-beams 50 --device cuda
done
```

报告 `all` 与三个固定子集的 NDCG/Recall@{5,10,20,50}，同时保存每个 RQ-VAE 的
`config.json`、`codebook_diagnostics.json`、`standardization.json` 和每个 retriever 的
`metrics.json`。这三个正式运行形成“Semantic ID 输入表示消融”；不要把它们写成 PFR 主方法的
消融或外部基线比较。
