# TIGER 语义 ID 检索实施计划

> **给执行代理：** 必须逐任务使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans；全部步骤以复选框跟踪。

**目标：** 以 768 维 Sentence-T5 内容向量训练 RQ-VAE 生成 Semantic ID，再用受约束 T5 自回归生成下一商品 ID。

**架构：** 先生成不可变的 768 维内容向量；RQ-VAE 仅在训练可见商品上拟合，随后冻结并为全目录分配唯一四 token ID。T5 使用历史 ID 和哈希用户 token 预测目标 ID，评测通过合法 ID 前缀树约束，不再走当前双塔打分。

**技术栈：** Python、PyTorch、scikit-learn、sentence-transformers、transformers、现有 Sports 协议与评测器。

**设计依据：** docs/superpowers/specs/2026-09-05-tiger-semantic-id-design.md

## 全局约束

- Sentence-T5 输出必须为 float32 的 768 维向量，禁止 PCA。
- RQ-VAE 只用训练切分商品拟合，T5 训练前必须冻结 RQ-VAE。
- 默认三级码本为 [4, 16, 256]，之后附加碰撞 token。
- T5 从随机权重开始：4 层 encoder、4 层 decoder、d_model=128、d_ff=1024、6 heads、d_kv=64、dropout 0.1。
- 评测是受约束生成式全目录检索，输出 all、cold_start_target、long_tail_target、sparse_user。
- 不改动 04_unified_next_item，不使用 ActionPiece 的 128 维/PQ 产物。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| src/pathfusionrec/tiger/content.py | 读取文本、生成/校验 768 维向量、训练集标准化。 |
| src/pathfusionrec/tiger/rq_vae.py | encoder、残差量化、decoder、损失、码本统计。 |
| src/pathfusionrec/tiger/tokenizer.py | token 偏移、碰撞码、用户哈希、合法 ID trie。 |
| src/pathfusionrec/tiger/retriever.py | T5 训练批次和受约束 beam search。 |
| scripts/build_tiger_content_embeddings.py | 离线 Sentence-T5 编码。 |
| scripts/train_tiger_rqvae.py | RQ-VAE 训练与 Semantic ID 导出。 |
| scripts/train_tiger_retriever.py | T5 正式训练与评测。 |
| tests/test_tiger_*.py | 单元测试与合成协议集成测试。 |

### 任务 1：生成 768 维内容向量产物

**文件：**
- 新建：src/pathfusionrec/tiger/__init__.py、src/pathfusionrec/tiger/content.py、scripts/build_tiger_content_embeddings.py、tests/test_tiger_content.py
- 修改：pyproject.toml、environment.yml

**接口：** load_ordered_item_texts(processed_dir)；save_content_artifact(output_dir, vectors, item_ids, manifest)；standardize_train_items(vectors, train_rows)。

- [ ] **步骤 1：先写失败测试**

~~~
def test_content_artifact_rejects_non_768_vectors(tmp_path):
    with pytest.raises(ValueError, match='768'):
        save_content_artifact(tmp_path, np.zeros((2, 127), np.float32), [1, 2], {})

def test_standardization_uses_only_train_rows():
    _, stats = standardize_train_items(
        np.array([[1., 1.], [3., 3.], [100., 100.]], np.float32),
        np.array([0, 1]),
    )
    assert stats['mean'] == [2.0, 2.0]
~~~

- [ ] **步骤 2：确认 RED**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_content.py -v

预期：因 pathfusionrec.tiger.content 尚不存在而失败。

- [ ] **步骤 3：最小实现**

~~~
if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != 768:
    raise ValueError('Sentence-T5 content vectors must be float32 with dimension 768.')
np.save(output_dir / 'vectors.npy', vectors)
~~~

按数值商品 ID 读取 metadata.sentence.json，使用明确 revision 的 SentenceTransformer 编码；item.sentence.feat 是 ActionPiece token ID，不作为文本输入。manifest 写入文本、映射、模型 revision 的 SHA-256。新增 sentence-transformers 和 transformers 依赖。

- [ ] **步骤 4：确认 GREEN**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_content.py tests/test_next_item.py -v

预期：通过。

- [ ] **步骤 5：提交**

~~~
git add pyproject.toml environment.yml src/pathfusionrec/tiger scripts/build_tiger_content_embeddings.py tests/test_tiger_content.py
git commit -m "feat: add TIGER 768d content artifacts"
~~~

### 任务 2：实现 RQ-VAE 与唯一 Semantic ID

**文件：**
- 新建：src/pathfusionrec/tiger/rq_vae.py、tests/test_tiger_rq_vae.py、scripts/train_tiger_rqvae.py、tests/test_tiger_rqvae_script.py

**接口：** ResidualQuantizedVAE(input_dim, latent_dim, codebook_sizes)；semantic_ids(vectors)；append_collision_codes(item_ids, codes)；run_rqvae(...)。

- [ ] **步骤 1：先写失败测试**

~~~
def test_rqvae_produces_three_residual_codes():
    output = ResidualQuantizedVAE(8, 4, [2, 3, 5])(torch.randn(6, 8))
    assert output.reconstruction.shape == (6, 8)
    assert output.codes.shape == (6, 3)

def test_collision_code_makes_duplicate_tuples_unique():
    ids = append_collision_codes([9, 4], np.array([[1, 2, 3], [1, 2, 3]]))
    assert ids == [(1, 2, 3, 0), (1, 2, 3, 1)]
~~~

- [ ] **步骤 2：确认 RED**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_rq_vae.py -v

预期：因 RQ-VAE 模块尚不存在而失败。

- [ ] **步骤 3：最小实现**

~~~
residual, quantized, codes = latent, torch.zeros_like(latent), []
for codebook in self.codebooks:
    index = torch.cdist(residual, codebook.weight).argmin(dim=1)
    selected = codebook(index)
    quantized, residual = quantized + selected, residual - selected
    codes.append(index)
quantized_st = latent + (quantized - latent).detach()
~~~

实现重建、commitment 和码本损失；用训练商品残差做 k-means 初始化。runner 从 split.train 收集可见商品拟合标准化/RQ-VAE，再为全目录导出四元组 ID、反向索引、码本困惑度和 dead-code 数。

- [ ] **步骤 4：确认 GREEN**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_rq_vae.py tests/test_tiger_rqvae_script.py -v

预期：通过；断言 encoder 获得反向梯度、导出 ID 均唯一且长度为 4。

- [ ] **步骤 5：提交**

~~~
git add src/pathfusionrec/tiger/rq_vae.py scripts/train_tiger_rqvae.py tests/test_tiger_rq_vae.py tests/test_tiger_rqvae_script.py
git commit -m "feat: add TIGER RQ-VAE semantic IDs"
~~~

### 任务 3：实现 ID tokenizer 与合法前缀树

**文件：**
- 新建：src/pathfusionrec/tiger/tokenizer.py、tests/test_tiger_tokenizer.py

**接口：** SemanticIdTokenizer(item_ids, codebook_sizes, collision_size, user_bucket_count)；encode_history(user_id, history)；decode_ids(tokens)；allowed_next(prefix)。

- [ ] **步骤 1：先写失败测试**

~~~
def test_trie_only_allows_catalog_id_continuations():
    tokenizer = SemanticIdTokenizer({1: [0, 3, 7, 0]}, [4, 16, 256], 1, 2000)
    prefix = [tokenizer.user_token(12), tokenizer.level_token(0, 0)]
    assert tokenizer.allowed_next(prefix) == {tokenizer.level_token(1, 3)}
~~~

- [ ] **步骤 2：确认 RED**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_tokenizer.py -v

预期：因 tokenizer 尚不存在而失败。

- [ ] **步骤 3：最小实现**

~~~
def encode_history(self, user_id, history):
    return [self.user_token(user_id), *(token for item in history for token in self.encode_item(item))]
~~~

各 RQ 层使用不重叠 token offset；用户哈希使用固定 BLAKE2b，禁止 Python hash()；trie 对不在目录的完整或部分 ID 都返回空集合。

- [ ] **步骤 4：确认 GREEN 并提交**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_tokenizer.py -v

~~~
git add src/pathfusionrec/tiger/tokenizer.py tests/test_tiger_tokenizer.py
git commit -m "feat: add TIGER token constraints"
~~~

### 任务 4：实现 T5 生成检索与评测 runner

**文件：**
- 新建：src/pathfusionrec/tiger/retriever.py、scripts/train_tiger_retriever.py、tests/test_tiger_retriever.py、tests/test_tiger_training_script.py
- 新建：experiment_results/05_tiger_semantic_id/README.md

**接口：** TigerRetriever(tokenizer, config)；make_batch(samples)；generate_top_k(history_tokens, k, beam_size)；run_training(...)。

- [ ] **步骤 1：先写失败测试**

~~~
def test_teacher_forcing_target_is_four_tokens_and_eos():
    batch = retriever.make_batch([{'user': 7, 'history': [1], 'target': 2}])
    assert batch['labels'].shape[1] == 5
    assert batch['labels'][0, -1] == tokenizer.eos_token_id

def test_runner_reports_all_subgroups(tmp_path):
    metrics = run_training(fixture_protocol(tmp_path), fixture_ids(tmp_path), tmp_path / 'run', epochs=1, device='cpu')
    assert set(metrics['test']) == {'all', 'cold_start_target', 'long_tail_target', 'sparse_user'}
~~~

- [ ] **步骤 2：确认 RED**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest tests/test_tiger_retriever.py tests/test_tiger_training_script.py -v

预期：因 retriever/runner 尚不存在而失败。

- [ ] **步骤 3：最小实现**

~~~
config = T5Config(
    vocab_size=tokenizer.vocab_size, d_model=128, d_ff=1024,
    num_layers=4, num_decoder_layers=4, num_heads=6, d_kv=64,
    dropout_rate=0.1, pad_token_id=tokenizer.pad_token_id,
    decoder_start_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
)
~~~

每一步 beam 扩展时，将 allowed_next(decoded_prefix) 之外的 logits 置为负无穷。仅将完整合法 ID 映射为商品，过滤历史商品；使用现有 rank_candidates 与 evaluate_ranked_lists 计算四组指标，按验证 NDCG@10 保存最优 checkpoint。

- [ ] **步骤 4：确认 GREEN**

运行：cd pathfusionrec && PYTHONPATH=src python -m pytest -q

预期：通过。

- [ ] **步骤 5：提交**

~~~
git add src/pathfusionrec/tiger/retriever.py scripts/train_tiger_retriever.py tests/test_tiger_retriever.py tests/test_tiger_training_script.py experiment_results/05_tiger_semantic_id/README.md
git commit -m "feat: add TIGER generative retrieval"
~~~

### 任务 5：执行 staged seed-2026 验证

**生成产物：**
- data/processed/tiger_sentence_t5_768/
- data/processed/tiger_rqvae_v1/
- experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026/

- [ ] **步骤 1：离线生成内容向量**

~~~
cd pathfusionrec && PYTHONPATH=src python scripts/build_tiger_content_embeddings.py --processed-dir data/AmazonReviews2014/Sports_and_Outdoors/processed --model sentence-transformers/sentence-t5-base --output-dir data/processed/tiger_sentence_t5_768 --device cuda
~~~

- [ ] **步骤 2：训练并冻结 RQ-VAE**

~~~
cd pathfusionrec && PYTHONPATH=src python scripts/train_tiger_rqvae.py --content-dir data/processed/tiger_sentence_t5_768 --protocol-dir data/processed/sports_protocol --latent-dim 128 --codebook-sizes 4 16 256 --output-dir data/processed/tiger_rqvae_v1 --device cuda
~~~

验收：18,357 个候选均有唯一四 token ID，且任一码本不完全空置。

- [ ] **步骤 3：先运行 CPU smoke**

~~~
cd pathfusionrec && PYTHONPATH=src python scripts/train_tiger_retriever.py --protocol-dir data/processed/sports_protocol --semantic-id-dir data/processed/tiger_rqvae_v1 --output-dir /private/tmp/tiger-smoke --epochs 1 --max-train-samples 64 --max-validation-samples 16 --max-test-samples 16 --device cpu
~~~

- [ ] **步骤 4：正式 seed-2026 GPU 实验**

~~~
cd pathfusionrec && PYTHONPATH=src python scripts/train_tiger_retriever.py --protocol-dir data/processed/sports_protocol --semantic-id-dir data/processed/tiger_rqvae_v1 --output-dir experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026 --seed 2026 --batch-size 256 --eval-batch-size 256 --epochs 100 --num-beams 50 --device cuda
~~~

- [ ] **步骤 5：检查交付并汇报边界**

运行：cd pathfusionrec && jq '.best_epoch, .test' experiment_results/05_tiger_semantic_id/TIGER-v1/seed2026/metrics.json

报告时标记为单 seed TIGER 结果；在候选集和评测语义完全一致前，不与历史 ActionPiece 指标作强结论对比。
