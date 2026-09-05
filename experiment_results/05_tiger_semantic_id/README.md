# TIGER 语义 ID 实验

该目录用于保存 TIGER 式生成检索实验的说明；每个实际运行应置于
`TIGER-v1/seed<seed>/`，并保留 `metrics.json`、`config.json`、
`delivery_validation.txt` 和最优 checkpoint。不要提交大体积向量缓存或 checkpoint。

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
