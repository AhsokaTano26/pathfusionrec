# ActionPiece 最终实验交付包

## 任务

完成 ActionPiece 的设置对齐与充分训练，记录运行时间、GPU、
参数、最佳验证轮次和测试结果，并为统一实验表保存
NDCG@5/10、Recall@5/10、训练时间和推理延迟。

## 实验协议

- Repository: google-deepmind/action_piece
- Commit: ae8e61a89ade8d545a16119bcb5b3a43d9da852f
- Dataset: Amazon Reviews 2014 / Sports_and_Outdoors
- Evaluation: full-item evaluation
- Seeds: 2024, 2025, 2026, 2027, 2028
- Validation metric: NDCG@10
- Maximum epochs: 200
- Early-stop patience: 20
- GPU: NVIDIA A800 80GB PCIe
- Paper GPU reference: NVIDIA A100 40GB
- Vocabulary size: 40000
- Train batch size: 256
- Evaluation batch size: 128
- Beam size: 50
- Inference ensemble segments: 5
- Sentence model: sentence-transformers/sentence-t5-base

## 结果文件

- results/actionpiece_sports_five_seed_results.csv:
  五个随机种子的结构化结果。
- results/actionpiece_sports_five_seed_results.json:
  包含路径和校验信息的完整机器可读结果。
- results/actionpiece_sports_five_seed_mean_std.csv:
  五种子的均值和样本标准差。
- results/actionpiece_sports_five_seed_report.md:
  可直接阅读的结果报告。
- logs/:
  完整原始训练、验证和测试日志。
- checkpoints/:
  每个种子按验证 NDCG@10 选择的最佳模型。
- scripts/:
  训练、独立检查点评估及结果汇总脚本。
- environment/:
  Python、依赖、GPU、Git、模型版本和数据校验信息。
- patches/:
  为使官方代码在当前环境运行而进行的兼容修复。

## 时间定义

- total_run_seconds:
  从脚本启动到最终测试完成的总墙钟时间。
- optimizer_train_seconds:
  所有训练 epoch 进度条的墙钟时间之和。
- validation_seconds:
  所有验证轮次的墙钟时间之和。
- test_eval_seconds:
  最佳检查点最终测试的墙钟时间。
- inference_ms_per_user:
  最终测试墙钟时间除以 35,598 个测试用户。

## 复现资产

主交付压缩包同时包含实际运行的源代码、原始数据、处理后数据、
ActionPiece 词表、语义嵌入、语义 ID 和 item 特征。

Hugging Face 预训练权重缓存未放入主交付包，以避免额外约 1 GB
体积。其精确模型名称、snapshot commit 和缓存信息保存在
environment/sentence_t5_model.txt；当前训练所需的处理后语义特征
已经包含在主交付包中。

## 注意

五个随机种子的结果应全部报告，不应只选择表现最好的种子。
SASRec 的统一比较需要使用相同数据集和全物品评估协议，属于后续
单独的基线对齐任务。
