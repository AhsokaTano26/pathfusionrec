# Experiment Card — TIGER semantic-ID generation (Stage3 retriever) · seed 2026

- **Experiment #**: 05_tiger_semantic_id / TIGER-v1 / seed2026
- **Method**: TIGER-style generative retrieval — random-init small T5 next-item decoder over RQ-VAE semantic IDs, prefix-constrained beam search over in-catalog IDs.
- **Repo commit (code used)**: `ff031c0bc79051298192c3c0bbe26eb65bec378d`
- **Run window**: relaunched 2026-09-07 13:47 → completed 2026-09-08 02:11 (one AutoDL shutdown interruption mid-first-run; relaunched byte-identical).
- **GPU / env**: RTX 4090; env `/root/autodl-tmp/envs/pathfusionrec` (torch+transformers, no pretrained checkpoint; T5 randomly initialized).

## Protocol / data (fixed)
- users 35,598 · candidate items 18,357 · train samples 189,543 (~741 steps/epoch) · val = test = 35,598.
- Subsets (test masks): `cold_start_target`, `long_tail_target`, `sparse_user`.
- Input text = metadata.sentence.json + id_mapping.json (processed dir).
- Input content vectors `tiger_sentence_t5_768/vectors.npy` sha256 `89791f29…0025` ([18357,768] float32).
- Semantic IDs `semantic_ids.json` sha256 `0eae6e9e…4e8`: 18,357 unique 4-token IDs; RQ-VAE codebooks [4,16,256], used [4,16,226]; 30 dead codes at L2; reconstruction loss drift 1.48→1.54 (reportable RQ-VAE caveat).

## Model / training config
- T5: d_model 128 · d_ff 1024 · layers 4 · heads 6 · d_kv 64 · dropout 0.1. Vocab = 3 residual-code codebooks + collision code + 2,000 user buckets + BOS/EOS/PAD.
- Batch 256, AdamW lr 1e-3 wd 1e-4, seed 2026, 100 epochs, eval every 10 epochs (10 validation points), beam size 50, rank k=50.
- Checkpoint selection: best validation NDCG@10.

## Results
- **Best epoch**: 90 · **best val NDCG@10**: 0.02127.
- Val NDCG@10 trajectory: 0.01644 (ep10) → 0.02127 (ep90) → 0.02111 (ep100); train loss 1.8333 → 1.7002.
- **test (all, k=10)**: NDCG@10 **0.01478**, Recall@10 **0.02891**.
- test all full: NDCG@5 0.01114 · Recall@5 0.01759 · NDCG@10 0.01478 · Recall@10 0.02891 · NDCG@20 0.01914 · Recall@20 0.04621 · NDCG@50 0.02582 · Recall@50 0.07981.
- test subsets: `sparse_user` NDCG@10 0.01524 / Recall@10 0.02959; `cold_start_target` all 0.0; `long_tail_target` ≈0 (NDCG@10 3.9e-05).

## Findings / caveats
- Generatively retrieved IDs land in the popular in-catalog region: new-item (`cold_start_target`) and extreme-long-tail (`long_tail_target`) retrieval are ≈0 for this seed. Baseline ActionPiece shows the same hard cold-start pattern — but do NOT compare numerically against ActionPiece/SASRec/PFR until candidate set + eval semantics are unified.
- Single seed only → not yet eligible for the main results table.

## Artifacts (this dir)
- `best_model.pth` · `metrics.json` · `config.json` · `delivery_validation.txt` (PASS) · `training_curves.png` · `test_subset.png` · `sha256sums.txt`
- Full sha256 for each artifact in `sha256sums.txt`.
