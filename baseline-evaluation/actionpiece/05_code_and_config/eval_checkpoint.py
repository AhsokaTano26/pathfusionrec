import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from genrec import utils
from genrec.pipeline import Pipeline


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--eval-seed", type=int, default=2024)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# ActionPiece会把sys.argv拼进内部日志文件名。
# 参数包含多个绝对路径时会超过Linux文件名长度限制。
sys.argv = [os.path.basename(sys.argv[0])]

config = {
    "category": "Sports_and_Outdoors",
    "rand_seed": 2024,
    "weight_decay": 0.15,
    "lr": 0.005,
    "n_hash_buckets": 128,
    "epochs": 200,
    "eval_batch_size": args.eval_batch_size,
    "run_id": f"sports_seed2024_checkpoint_eval{args.eval_batch_size}",
}

pipeline = Pipeline(
    model_name="ActionPiece",
    dataset_name="AmazonReviews2014",
    config_dict=config,
)

state_dict = torch.load(args.checkpoint, map_location="cpu")
pipeline.model.load_state_dict(state_dict)

ensemble = pipeline.config["n_inference_ensemble"]
physical_batch_size = (
    args.eval_batch_size
    if ensemble == -1
    else max(args.eval_batch_size // ensemble, 1)
)

test_dataloader = DataLoader(
    pipeline.tokenized_datasets["test"],
    batch_size=physical_batch_size,
    shuffle=False,
    collate_fn=pipeline.tokenizer.collate_fn_test,
)

pipeline.model, test_dataloader = pipeline.accelerator.prepare(
    pipeline.model,
    test_dataloader,
)
pipeline.trainer.model = pipeline.model

# 固定推理阶段的随机分段，便于之后重复比较。
utils.init_seed(args.eval_seed, True)

start = time.time()
results = pipeline.trainer.evaluate(
    test_dataloader,
    split=f"test-eval{args.eval_batch_size}",
)
duration = time.time() - start

record = {
    "checkpoint": os.path.abspath(args.checkpoint),
    "eval_batch_size": args.eval_batch_size,
    "physical_dataloader_batch_size": physical_batch_size,
    "n_inference_ensemble": ensemble,
    "eval_seed": args.eval_seed,
    "duration_seconds": duration,
    "test_results": dict(results),
}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)

print("CHECKPOINT_ONLY_EVALUATION")
print(json.dumps(record, indent=2))
pipeline.trainer.end()
