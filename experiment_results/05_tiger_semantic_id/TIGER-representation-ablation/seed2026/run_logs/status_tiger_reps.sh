#!/bin/bash
BASE=/root/pathfusionrec/experiment_results/05_tiger_semantic_id/TIGER-representation-ablation/seed2026
DL=/root/autodl-tmp/logs/tiger_rep_ablation_driver.log
SNAP=/root/autodl-tmp/logs/.rep_snapshot
done_reps=""
for r in semantic behavior fusion; do
  if [ -f "$BASE/$r/delivery_validation.txt" ]; then done_reps="$done_reps $r"; fi
done
last=$(grep -E "^(START|DONE|FAIL|ALL_REPS_DONE)" "$DL" 2>/dev/null | tail -1)
ndone=$(grep -cE "^DONE " "$DL" 2>/dev/null)
cur="done=$ndone last=[$last] completed:[$done_reps]"
prev=""; [ -f "$SNAP" ] && prev=$(cat "$SNAP")
if [ "$cur" != "$prev" ]; then echo "$cur" > "$SNAP"; echo "STATUS: $cur"; else echo "SAME"; fi
if grep -q "watch_done" /root/autodl-tmp/logs/watch_tiger_reps.log 2>/dev/null; then echo "FLAG: WATCH_DONE"; fi
if grep -q "watch_failed\|watch_died" /root/autodl-tmp/logs/watch_tiger_reps.log 2>/dev/null; then echo "FLAG: WATCH_ABNORMAL"; fi
