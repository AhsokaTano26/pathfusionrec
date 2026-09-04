# C 角色基线质量核验报告

核验源 commit：`a943b4b027c3cdea51a1c9745612f1562692ffef`

生成方式：`python scripts/verify_baseline_delivery.py`

## 自动核验摘要

- PASS：33
- INCOMPLETE：13
- FAIL：4

`FAIL` 表示现有交付内部存在可验证的不闭合项；`INCOMPLETE` 表示尚未达到三人协作文档规定的阶段交付条件。逐项证据见 `integrity_check.csv`。

## ActionPiece

2024–2028 五 seed 齐全。脚本从原始 JSON 独立复算，使用 `statistics.stdev`（`ddof=1`）：

| 指标 | 均值 | 样本标准差 | 论文值 |
|---|---:|---:|---:|
| NDCG@5 | 0.018721009 | 0.000747503 | 0.0205 ± 0.0002 |
| NDCG@10 | 0.023806812 | 0.000738226 | 0.0264 ± 0.0003 |
| Recall@5 | 0.028456654 | 0.001121022 | 0.0316 ± 0.0005 |
| Recall@10 | 0.044244058 | 0.001087796 | 0.0500 ± 0.0007 |

五份日志均重建出 200 个 epoch，日志最佳轮次、最佳验证值及测试指标与结果 JSON 一致；五个 checkpoint 也与原交付清单中的 SHA-256 一致。由于没有逐用户预测和统一子集指标，该结果不能进入 PathFusionRec 统一主表。

## SASRec

当前仅有 3 个 seed（2026, 2027, 2028）。三个逐 seed SHA-256 清单通过；根清单状态为 `FAIL`（checked=86, missing=0, mismatched=1, crlf_normalized=76; mismatched_files=DELIVERY_MANIFEST.csv）。此外缺少 seed 2024/2025、@20/@50、三个子集，且 `maxlen=200` 与统一协议的 20 不一致。因此只能标记为项目实测参考。

## PathFusionRec seed 2026

Interaction、Semantic、Fusion/concat 三个目录的四类子集和 @5/@10/@20/@50 schema 完整，最佳验证轮次复算一致，三个运行使用同一代码 commit。当前仍只有一个 seed，且缺实验卡与统一数据文件哈希。三个 `sha256sum.txt` 都引用缺失的 `qq_report.txt`，所以清单未闭合；这里不修改原目录，只报告问题供 A/B 补交。

## VQ-Rec

仓库中没有 VQ-Rec 代码或结果。原论文是跨域迁移设置且不包含 Sports；在 A 决定 Sports 适配方案、预训练来源、文本字段与实验卡前，本核验不擅自实现或启动训练。

## 主表使用结论

目前没有任何基线同时满足固定 `sports_protocol`、2024–2028 五 seed、统一显式全库候选、@5/@10/@20/@50、三个子集和闭合哈希记录。因此本目录中的数值均不得直接并入最终统一主表。
