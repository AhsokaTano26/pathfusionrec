#!/bin/bash
# TIGER representation ablation - behavior input, 200 epochs, eval-interval 20.
# Purpose: confirm/deny undertraining of the 100-epoch behavior run (best_epoch=100, hit the cap).
# Code commit bca2867. Output stays versioned as a NEW directory (never overwrite behavior/).
cd /root/pathfusionrec || exit 1
PY=/root/autodl-tmp/envs/pathfusionrec/bin/python
DL=/root/autodl-tmp/logs/tiger_behavior200_driver.log
RUNLOG=/root/autodl-tmp/logs/tiger_behavior200.log
OUT=experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026/behavior-200ep
BK=/root/autodl-tmp/pathfusionrec/experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026

: > "$DL"
echo "launch $(date -u +%Y-%m-%dT%H:%M:%SZ) epochs=200 eval_interval=20 commit=bca2867" > "$RUNLOG"
echo "START behavior-200ep $(date +%H:%M:%S)" >> "$DL"

PYTHONPATH=src:. "$PY" scripts/train_tiger_retriever.py \
  --protocol-dir data/processed/sports_protocol \
  --semantic-id-dir data/processed/tiger_rqvae_ablation/behavior_latent32 \
  --output-dir "$OUT" \
  --seed 2026 --batch-size 256 --eval-batch-size 256 \
  --epochs 200 --eval-interval 20 --num-beams 50 \
  --num-workers 8 --prefetch-factor 4 --device cuda >> "$RUNLOG" 2>&1
RC=$?
echo "exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUNLOG"

if [ $RC -ne 0 ] || [ ! -f "$OUT/delivery_validation.txt" ]; then
  echo "FAIL behavior-200ep rc=$RC $(date +%H:%M:%S)" >> "$DL"
  exit 1
fi

mkdir -p "$BK"
cp -a "$OUT" "$BK/" && echo "BACKUP_OK $BK/behavior-200ep $(date +%H:%M:%S)" >> "$DL"
echo "DONE behavior-200ep rc=0 $(date +%H:%M:%S)" >> "$DL"
echo "BEHAVIOR200_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DL"
