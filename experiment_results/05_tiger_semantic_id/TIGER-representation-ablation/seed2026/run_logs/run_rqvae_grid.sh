#!/bin/bash
cd /root/pathfusionrec
PY=/root/autodl-tmp/envs/pathfusionrec/bin/python
LOG=/root/autodl-tmp/logs/tiger_ablation_rqvae_grid.log
: > "$LOG"
START=$(date +%s)
echo "grid start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

run() {
  local label=$1 out=$2; shift 2
  echo "[$(date +%H:%M:%S)] START $label" >> "$LOG"
  PYTHONPATH=src "$PY" "$@" > /tmp/rq_one.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$out/delivery_validation.txt" ]; then
    echo "[$(date +%H:%M:%S)] PASS  $label  out=$out" >> "$LOG"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $label  rc=$rc  out=$out" >> "$LOG"
    tail -40 /tmp/rq_one.log >> "$LOG"
  fi
}

for LATENT in 32 64 128; do
  run semantic_latent$LATENT data/processed/tiger_rqvae_ablation/semantic_latent$LATENT \
    scripts/train_tiger_rqvae.py \
    --content-dir data/processed/tiger_sentence_t5_768 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/semantic_latent$LATENT \
    --latent-dim $LATENT --codebook-sizes 4 16 256 --seed 2026 --device cuda

  run behavior_latent$LATENT data/processed/tiger_rqvae_ablation/behavior_latent$LATENT \
    scripts/train_tiger_rqvae.py \
    --content-dir data/processed/tiger_behavior_pfr_interaction_128 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/behavior_latent$LATENT \
    --latent-dim $LATENT --codebook-sizes 4 16 256 --seed 2026 --device cuda

  run fusion_latent$LATENT data/processed/tiger_rqvae_ablation/fusion_latent$LATENT \
    scripts/train_tiger_fusion_rqvae.py \
    --semantic-dir data/processed/tiger_sentence_t5_768 \
    --behavior-dir data/processed/tiger_behavior_pfr_interaction_128 \
    --protocol-dir data/processed/sports_protocol \
    --output-dir data/processed/tiger_rqvae_ablation/fusion_latent$LATENT \
    --branch-dim 128 --latent-dim $LATENT --codebook-sizes 4 16 256 --seed 2026 --device cuda
done

END=$(date +%s)
echo "grid done $(date -u +%Y-%m-%dT%H:%M:%SZ)  elapsed=$((END-START))s" >> "$LOG"
echo "GRID_FINISHED" >> "$LOG"
