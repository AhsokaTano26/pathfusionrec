# 配置字段说明

| 字段 | 含义 |
|---|---|
| hidden_units | 物品与位置嵌入维度 |
| num_blocks | SASRec自注意力块数 |
| num_heads | 每个注意力层的头数 |
| maxlen | 输入序列最大长度 |
| dropout_rate | Dropout概率 |
| batch_size | 训练批大小 |
| eval_batch_size | 全物品评估批大小 |
| lr | Adam学习率 |
| l2_emb | 物品嵌入L2正则系数 |
| norm_first | 是否采用pre-norm；本实验为false |
| num_epochs | 最大训练轮数上限 |
| eval_every | 每隔多少轮执行验证 |
| patience | 验证指标连续无提升的容忍轮数 |
| min_delta | 判定提升所需的最小增量 |
| num_workers | 训练采样工作进程数 |
| device | 训练设备 |

`seed_2026_args.json`、`seed_2027_args.json`、`seed_2028_args.json` 是三次正式实验保存的原始参数；`common_config.json` 是本次为便于阅读而生成的共同参数汇总。

