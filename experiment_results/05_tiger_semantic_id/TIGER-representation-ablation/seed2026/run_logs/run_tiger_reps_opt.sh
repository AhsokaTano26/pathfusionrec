#!/bin/bash
# TIGER representation ablation (latent=32) serial driver - optimized code bca2867.
cd /root/pathfusionrec
PY=/root/autodl-tmp/envs/pathfusionrec/bin/python
DL=/root/autodl-tmp/logs/tiger_rep_ablation_driver.log
: > "$DL"
for REP in semantic behavior fusion; do
  OUT=experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026/$REP
  RUNLOG=/root/autodl-tmp/logs/tiger_ablation_${REP}.log
  echo "launch $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUNLOG"
  echo "START $REP $(date +%H:%M:%S)" >> "$DL"
  PYTHONPATH=src:. "$PY" scripts/train_tiger_retriever.py \
    --protocol-dir data/processed/sports_protocol \
    --semantic-id-dir data/processed/tiger_rqvae_ablation/${REP}_latent32 \
    --output-dir "$OUT" \
    --seed 2026 --batch-size 256 --eval-batch-size 256 \
    --epochs 100 --eval-interval 10 --num-beams 50 \
    --num-workers 8 --prefetch-factor 4 --device cuda >> "$RUNLOG" 2>&1
  RC=$?
  echo "exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUNLOG"
  if [ $RC -ne 0 ] || [ ! -f "$OUT/delivery_validation.txt" ]; then
    echo "FAIL $REP rc=$RC $(date +%H:%M:%S)" >> "$DL"
    exit 1
  fi
  echo "DONE $REP rc=0 $(date +%H:%M:%S)" >> "$DL"
done
echo "ALL_REPS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DL"
