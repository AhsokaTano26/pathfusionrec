# 计时字段核验

| 字段 | 来源 | 是否用于统一实验表 |
|---|---|---|
| `pure_training_seconds` | 各轮训练计算时间之和 | 否，仅辅助 |
| `selection_wall_seconds` | 训练、逐轮验证、早停选模墙钟时间 | 是，训练时间 |
| `test_metrics.wall_seconds` | 完整测试评估墙钟时间 | 是，测试总时间 |
| `test_metrics.wall_ms_per_user` | 完整测试时间除以用户数 | 是，推理延迟 |
| `test_metrics.ranking_ms_per_user` | GPU核心打分、过滤、Top-K时间除以用户数 | 否，仅辅助 |

当前 `wall_seconds` 在用户评估循环结束后停止，函数末尾的最终除法与结果字典组装未计入；该差异极小。若后续要求SASRec与ActionPiece逐行一致的计时边界，可利用现有checkpoint统一补测，无需重训。

