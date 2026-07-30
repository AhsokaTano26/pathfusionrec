#!/usr/bin/env bash
set -uo pipefail

SEED="${1:?请提供随机种子}"
EVAL_BATCH_SIZE="${2:-128}"

case "$SEED" in
  2024|2025|2026|2027|2028) ;;
  *)
    echo "错误：正式种子只能是 2024、2025、2026、2027、2028"
    exit 2
    ;;
esac

PROJECT="/root/autodl-tmp/action_piece"
RECORDS="/root/autodl-tmp/actionpiece_records"
RUN_ID="sports_full_seed${SEED}_eval${EVAL_BATCH_SIZE}"

LOG="${RECORDS}/logs/${RUN_ID}.log"
META="${RECORDS}/results/${RUN_ID}_meta.txt"
SUMMARY="${RECORDS}/results/${RUN_ID}_summary.txt"
GPU_LOG="${RECORDS}/results/${RUN_ID}_gpu.csv"
PATCH="${RECORDS}/patches/${RUN_ID}_working_tree.patch"
STATUS_FILE="${RECORDS}/results/${RUN_ID}_git_status.txt"

mkdir -p \
  "${RECORDS}/logs" \
  "${RECORDS}/results" \
  "${RECORDS}/patches"

cd "$PROJECT"

if pgrep -af '[p]ython main.py' \
  > "${RECORDS}/results/${RUN_ID}_preexisting_process.txt"; then
  echo "检测到已有 main.py 训练进程，拒绝重复启动："
  cat "${RECORDS}/results/${RUN_ID}_preexisting_process.txt"
  exit 3
fi

git status --short > "$STATUS_FILE"
git diff > "$PATCH"

START_TIME="$(date '+%Y-%m-%d %H:%M:%S %z')"
START_SECONDS="$(date +%s)"

{
  echo "run_id=${RUN_ID}"
  echo "start_time=${START_TIME}"
  echo "seed=${SEED}"
  echo "dataset=AmazonReviews2014/Sports_and_Outdoors"
  echo "gpu=NVIDIA A800 80GB PCIe"
  echo "paper_gpu=NVIDIA A100 40GB"
  echo "repo_commit=$(git rev-parse HEAD)"
  echo "lr=0.005"
  echo "weight_decay=0.15"
  echo "dropout_rate=0.1"
  echo "warmup_steps=10000"
  echo "train_batch_size=256"
  echo "eval_batch_size=${EVAL_BATCH_SIZE}"
  echo "vocabulary_size=40000"
  echo "n_hash_buckets=128"
  echo "n_inference_segments=5"
  echo "beam_size=50"
  echo "num_layers=4"
  echo "d_model=128"
  echo "d_ff=1024"
  echo "num_heads=6"
  echo "d_kv=64"
  echo "optimizer=AdamW"
  echo "lr_scheduler=cosine"
  echo "max_epochs=200"
  echo "early_stop_patience=20"
  echo "validation_metric=ndcg@10"
  echo "sentence_model=sentence-transformers/sentence-t5-base"
  echo
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader
} > "$META"

nvidia-smi \
  --query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l 5 > "$GPU_LOG" &

GPU_MONITOR_PID=$!

cleanup_monitor() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}

trap cleanup_monitor EXIT

env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 \
  HF_HOME=/root/autodl-tmp/huggingface \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  python main.py \
    --category=Sports_and_Outdoors \
    --rand_seed="$SEED" \
    --weight_decay=0.15 \
    --lr=0.005 \
    --n_hash_buckets=128 \
    --epochs=200 \
    --eval_batch_size="$EVAL_BATCH_SIZE" \
    --run_id="$RUN_ID" \
  2>&1 | tee "$LOG"

RUN_STATUS=${PIPESTATUS[0]}

END_SECONDS="$(date +%s)"
DURATION=$((END_SECONDS - START_SECONDS))

cleanup_monitor
trap - EXIT

PEAK_GPU_MEMORY="$(
  awk -F',' '
    {
      gsub(/[[:space:]]/, "", $6)
      if (($6 + 0) > maximum) maximum = $6 + 0
    }
    END { print maximum + 0 }
  ' "$GPU_LOG"
)"

{
  echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "duration_seconds=${DURATION}"
  printf 'duration_hms=%02d:%02d:%02d\n' \
    $((DURATION / 3600)) \
    $(((DURATION % 3600) / 60)) \
    $((DURATION % 60))
  echo "peak_gpu_memory_mib=${PEAK_GPU_MEMORY}"
  echo "exit_code=${RUN_STATUS}"
} >> "$META"

grep -E \
  '#Embedding parameters|#Non-embedding parameters|#Total trainable parameters|Train Loss|Val Results|Best epoch|Best val score|Test Results' \
  "$LOG" > "$SUMMARY" || true

if [[ "$RUN_STATUS" -eq 0 ]]; then
  echo "正式实验 ${RUN_ID} 已完成。"
else
  echo "正式实验 ${RUN_ID} 异常结束，退出码：${RUN_STATUS}"
fi

exit "$RUN_STATUS"
