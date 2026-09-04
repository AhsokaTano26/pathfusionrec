# 基线协议缺口清单

> 核验负责人：倪榕键（C）
>
> 核验对象：ActionPiece、SASRec、VQ-Rec 及当前 PathFusionRec seed 2026 交付
>
> 统一口径：`docs/01_unified_data_and_evaluation_protocol.md`

## 结论

| 基线 | 当前状态 | 能否进入统一主表 | 最小补齐项 |
|---|---|---|---|
| ActionPiece | 已有 2024–2028 五 seed、日志、checkpoint 和聚合指标；复算均值与样本标准差一致 | 否 | 在固定 `sports_protocol` 候选目录上输出逐用户排序结果或同口径指标，并补齐三个子集；确认生成式候选/冲突处理与统一全库评分的等价性 |
| SASRec | 已有 2026–2028 三 seed 的项目实测结果 | 否 | 固定 `max_history_length=20`，补跑 2024、2025，输出 @20/@50 和三个子集，并记录统一协议/data hash |
| VQ-Rec | 仓库中没有实现、实验卡或结果 | 否 | A 先确定“原论文跨域迁移版”或“Sports 单域适配版”，再固化代码、文本字段、预训练来源、PQ 配置、实验卡和 seed 2026 命令 |

当前 ActionPiece 与 SASRec 数值只能作为历史/参考结果，不应与 PathFusionRec 的统一主表混排。VQ-Rec 在 A 固定适配方案前不应由 C 自行选定训练口径。

## ActionPiece

原论文：[ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](https://arxiv.org/abs/2502.13581)

| 核验项 | 论文口径 | 当前仓库 | 判定 |
|---|---|---|---|
| 任务 | 按时间序列预测下一物品，使用上下文相关的动作 token | 相同任务 | 已满足 |
| 数据集 | Amazon Reviews 2014 的 Sports、Beauty、CDs | Sports_and_Outdoors | 已满足（Sports） |
| 数据规模 | Sports：35,598 用户、18,357 物品、260,739 个训练/验证动作；最后一次交互另作测试 | 35,598 用户、18,357 物品、296,337 总交互；扣除每用户一个测试目标后为 260,739 | 已满足 |
| 划分 | leave-last-out：最后一个测试、倒数第二个验证 | 相同 | 已满足 |
| 候选与推理 | beam size 50，5 次 SPR 分段集成的生成式全物品评估 | 官方代码路径的生成式评估，记录 `err@K` | 近似一致；与 PathFusionRec 的显式全目录打分并非同一实现 |
| 指标 | Recall/NDCG@5、@10 | 另含 @20、@50 | 已满足 |
| seed | 2024–2028 五次 | 2024–2028 五次 | 已满足 |
| 选模 | 验证 NDCG@10；最多 200 epoch；patience 20 | 相同选模；2024–2027 额外使用延迟早停以避免早期平台期 | 近似一致，不是论文原始训练控制的严格复现 |
| 报告值 | Sports：NDCG@5 0.0205±0.0002、NDCG@10 0.0264±0.0003、Recall@5 0.0316±0.0005、Recall@10 0.0500±0.0007 | 独立复算见 `experiment_results/05_baseline_quality_verification/actionpiece_paper_comparison.csv` | 近似但整体偏低 |
| 统一子集 | 论文不报告本项目三个子集 | 无逐用户预测，不能事后重建 | 缺失 |
| 可追溯性 | 论文公开代码 | 当前五 seed checkpoint 与原交付 SHA-256 一致；日志可重建曲线 | 已满足 |

## SASRec

原论文：[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)

| 核验项 | 原论文口径 | 当前仓库 | 判定 |
|---|---|---|---|
| 任务 | 序列下一物品推荐 | 相同 | 已满足 |
| 数据集 | Amazon Beauty、Games，另含 Steam、ML-1M；5-core | Amazon Sports，沿用项目的 35,598 用户/18,357 物品处理数据 | 不可与原论文数值直接比较 |
| 划分 | 每用户最后一个测试、倒数第二个验证 | 相同 leave-last-out | 已满足 |
| 候选集 | 正样本加 100 个采样负例 | 18,357 个全物品排序 | 与原论文不可直接比较；与项目统一全库方向一致 |
| 最大历史 | Amazon 数据使用 50 | 当前项目交付为 200；统一协议为 20 | 缺失 |
| 指标 | Hit/Recall@10、NDCG@10 | Recall/NDCG@5、@10 | @20、@50 缺失 |
| seed | 原论文未给出本项目所需的固定五 seed | 仅 2026、2027、2028 | 缺 2024、2025 |
| 选模 | 验证集调参，20 epoch 无提升停止 | 验证 NDCG@10，patience 20 | 已满足 |
| 统一子集 | 原论文无本项目子集 | 当前交付无子集结果 | 缺失 |
| 可追溯性 | 原始 TensorFlow 实现 | 当前为 pmixer PyTorch 实现；代码、JSON、checkpoint、环境和 SHA-256 已归档 | 项目实测可追溯，但不是原论文严格复现 |

## VQ-Rec

原论文：[Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders](https://arxiv.org/abs/2210.12316)

| 核验项 | 原论文口径 | 当前仓库 | 判定 |
|---|---|---|---|
| 任务 | 文本→PQ code→表示的可迁移序列推荐；跨域预训练后微调 | 无实现 | 缺失 |
| 数据集 | 预训练：Food、Home、CDs、Kindle、Movies；下游：Scientific、Pantry、Instruments、Arts、Office、Online Retail；5-core | 目标为 Amazon Sports | 原文不含 Sports，必须明确适配实验而非原表复现 |
| 文本字段 | Amazon title、categories、brand，截断 512 | Sports 协议现有 title、description、categories，字段不同 | 缺失，需 A 固定字段口径 |
| 划分 | leave-one-out：最后一个测试、倒数第二个验证 | `sports_protocol` 可提供 | 可满足 |
| 候选集 | 真值对全部其他物品排序 | `sports_protocol` 全映射商品候选目录 | 原则一致，仍需统一历史过滤与 ID 映射 |
| 指标 | Recall/NDCG@10、@50 | 尚无输出 | 缺 @5、@20 和全部结果 |
| 模型配置 | `(M=32)×(D=256)` PQ；预训练 300 epoch；batch 2048；验证 NDCG@10 选模；patience 10 | 无配置 | 缺失 |
| seed | 论文未明确本项目固定 seed 集 | 无结果 | 缺 2024–2028 |
| 统一子集与交付 | 原论文无本项目子集 | 无日志、checkpoint、环境、SHA-256 | 缺失 |

## 当前 PathFusionRec seed 2026 抽查

三个目录均包含 `metrics.json`、100 epoch 曲线、checkpoint、日志、代码 commit、配置和运行环境片段；`metrics.json` 均输出 `all`、`cold_start_target`、`long_tail_target`、`sparse_user` 以及 @5/@10/@20/@50。独立复核也确认 `best_epoch` 对应历史中的最大验证 NDCG@10。

仍有以下问题：

- 三个实验目录都缺 `experiment_card.md`；
- 只有 seed 2026，不能计算五 seed 主表；
- `sha256sum.txt` 均列出未随仓库提交的 `qq_report.txt`，因此清单不是闭合交付；
- 没有 `environment.txt` 单文件，环境信息分散在 `python_version.txt` 与 `nvidia_smi_before.txt`；
- 配置没有固化 `split.json`、`bundles.json`、`subset_masks.json` 的 SHA-256，后续五 seed 无法仅凭结果目录证明数据完全一致；
- Fusion 与 Semantic 的 `config.json` 缺 `experiment_id`、`model_selection` 和 `candidate_count` 字段，Interaction 已包含这些字段。

详细逐项结果见 `experiment_results/05_baseline_quality_verification/integrity_check.csv`。
