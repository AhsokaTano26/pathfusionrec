# TIGER Semantic ID 输入表示消融 · 分析草稿

- 实验编号：05_tiger_semantic_id / TIGER-representation-ablation / seed2026
- 代码提交：`bca28672d5c9019510a008e92cb70ddb4a97bc7a`
- 变量：RQ-VAE 的输入商品表示（semantic / behavior / fusion）；T5 decoder、Semantic ID 词表、
  候选约束、beam 宽度、seed、epoch 数、评测间隔全部固定。
- 选择依据：三组共用 `latent_dim=32`，由 `rqvae_latent_capacity.csv` 的容量筛选决定，**未使用测试指标**。

## 1. 全量测试集（k=5/10/20/50）

| 表示 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | best epoch |
|------|--------|---------|---------|---------|----------|-----------|-----------|-----------|------------|
| semantic | 0.010226 | 0.013622 | 0.018037 | 0.024197 | 0.016237 | 0.026799 | 0.044356 | 0.075397 | 90 |
| behavior | 0.010290 | 0.013597 | 0.017868 | 0.025047 | 0.015956 | 0.026237 | 0.043345 | 0.079639 | 100（触顶） |
| fusion | **0.012868** | **0.016861** | **0.021783** | **0.029303** | **0.020226** | **0.032670** | **0.052306** | **0.090230** | 80 |

相对 semantic 的增减：

| 表示 | ΔNDCG@10 | ΔRecall@10 | ΔNDCG@50 | ΔRecall@50 |
|------|----------|------------|----------|------------|
| behavior | −0.000025（−0.18%） | −0.000562（−2.10%） | +0.000850（+3.51%） | +0.004242（+5.63%） |
| fusion | +0.003239（+23.78%） | +0.005871（+21.91%） | +0.005106（+21.10%） | +0.014833（+19.67%） |

## 2. 测试子集（NDCG@10 / Recall@10）

| 表示 | all | sparse_user | cold_start_target | long_tail_target |
|------|-----|-------------|-------------------|------------------|
| semantic | 0.013622 / 0.026799 | 0.014605 / 0.028270 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| behavior | 0.013597 / 0.026237 | 0.013747 / 0.026945 | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| fusion | 0.016861 / 0.032670 | 0.017618 / 0.034233 | 0.000000 / 0.000000 | 0.000261 / 0.000703 |

`cold_start_target` 在三组上 @10 全为 0（fusion 仅在 @50 非零：NDCG@50 0.000688、Recall@50 0.003155）；
`long_tail_target` 三组都停留在 1e-4 量级（@50 依次为 0.000159 / 0.000757 / 0.001064）。
即：融合输入带来的是**常规样本与稀疏用户样本上的整体抬升**，并未修复生成式路径对冷启动与极端长尾的弱点。

## 3. 与 TIGER-v1 和 PFR-Interaction-v1 的单 seed 对照

| 方法 | 角色 | 输入表示 | latent | NDCG@10 | Recall@10 |
|------|------|----------|--------|---------|-----------|
| TIGER-v1 | 外部生成式基线 | Sentence-T5 768 | 128 | 0.014778 | 0.028906 |
| 本消融 semantic | 输入表示消融 | Sentence-T5 768 | 32 | 0.013622 | 0.026799 |
| 本消融 behavior | 输入表示消融 | PFR interaction 128 | 32 | 0.013597 | 0.026237 |
| 本消融 fusion | 输入表示消融 | 语义 + 行为（平衡融合） | 32 | 0.016861 | 0.032670 |
| PFR-Interaction-v1 | 项目主方法 | 交互式检索 | — | 0.015959 | 0.030957 |
| PFR-Semantic-v1 | 主方法内部消融 | 语义 | — | 0.003636 | 0.007163 |
| PFR-Fusion-Concat-v1 | 主方法内部消融 | 语义 + 行为（直接拼接） | — | 0.003194 | 0.006714 |

- fusion（0.016861）比 semantic（0.013622）高 **23.8%**，比外部基线 TIGER-v1（0.014778）高 **14.1%**，
  比主方法 PFR-Interaction-v1（0.015959）高 **5.7%**。
- baseline semantic 组（0.013622）比 TIGER-v1（0.014778）低 **7.8%**：二者同为 Sentence-T5 输入，
  差别只在 latent 容量（32 vs 128），说明这一差距**不能**单独归因于输入表示。

## 4. 可比性与风险

1. **role 纪律**：本表是"Semantic ID 输入表示消融"，不是 PFR 主方法的消融，也不是外部基线对比。
   fusion 高于 PFR-Interaction 只作数值记录，二者模型族不同（约束解码的生成式检索 vs. 交互式 next-item
   检索），不能据此声称生成式路线更优。
2. **behavior 确实欠训（已由补跑确认）**：100 epoch 版 best epoch 恰好等于轮数上限，未自然收敛。
   诊断性补跑 `behavior-200ep/`（200 epoch / eval-interval 20，同样只按验证集选检查点）结论见第 7 节：
   其验证 NDCG@10 在 **epoch 120** 达到峰值 0.020067 后单调下降，说明 100 epoch 停在了上升段。
   但这一验证集增益**基本没有传导到测试集**（test NDCG@10 仅 +0.29%），因此**不改变第 1–3 节的结论**。
3. **单 seed**：本消融仅有 seed 2026，不能进入最终多 seed 主结果表（主表只由 seed 2024–2028 构成，
   并报告样本标准差 `ddof=1`）。
4. **RQ-VAE 已知限制**：fusion 的第三级码本有 17 个死码（239/256 被使用），semantic/behavior 为 0 死码；
   对比 fusion 时需注意其量化容量略低于另两组。
5. **计算成本**：三组串行总耗时约 32.4 h（semantic 9h34m、behavior 13h25m、fusion 9h26m）。评测阶段为
   单核 CPU 瓶颈（约束解码的 beam search 在 host 侧逐候选取 log-prob），behavior 的候选规模最大故最慢。
   后续若扩到 5 seed，需先优化解码再排期。

## 5. 图表与原始数据

- `representation_ablation_metrics.{png,svg}`：三组全量测试集 NDCG / Recall @ {5,10,20,50}。
- `representation_ablation_subsets.{png,svg}`：三组在 all / sparse_user / cold_start / long_tail 上的 NDCG@10、Recall@10。
- `representation_ablation_training_curves.{png,svg}`：三组验证 NDCG@10 与训练损失随 epoch 的变化。
- `ablation_results.csv`（3 行）、`ablation_subsets.csv`（12 行）：上表的机器可读版本。
- `behavior_epoch_budget.csv`：100 vs 200 epoch 两个 budget 的验证曲线（由两个 `metrics.json` 的
  `history` 直接导出，第 7 节）。
- 图由 `scripts/plot_tiger_representation_ablation.py` 可复现生成；该脚本有对应单测
  `tests/test_plot_tiger_representation_ablation.py`。

## 6. 待补

- 若进入正式投稿，需要 5 seed 扩展与解码性能优化。

## 7. behavior 轮数预算诊断（`behavior-200ep/`）

100 epoch 版的 best epoch 恰好等于轮数上限 100，属于"可能在上升段被截断"。为此在同一 semantic ID、
同一 seed、同一超参下把轮数放宽到 200（`eval_interval=20`，其余完全不变），只作**诊断**，不进入正式消融表。

**可复现性核对**：两个 run 的 seed 与数据顺序完全相同，因此 epoch 20/40/60/80/100 的验证 NDCG@10
应当逐位相同——实测确实逐位相同（见 `behavior_epoch_budget.csv`）。这既确认了两个 run 可比，
也顺带证明了本流程在固定 seed 下的单 seed 可复现性。

| budget | best epoch | 最佳验证 NDCG@10 | test NDCG@10 | test Recall@10 |
|--------|-----------|------------------|--------------|----------------|
| 100 | 100（触顶） | 0.019514 | 0.013597 | 0.026237 |
| 200 | **120** | 0.020067（+2.83%） | 0.013637（+0.29%） | 0.026434（+0.75%） |

200 epoch 版在 **epoch 120** 达到峰值后逐轮下降（140 → 0.019908，160 → 0.019708，180 → 0.019293，
200 → 0.019157），同时训练损失继续从 1.7106 降到 1.6431——典型的峰值后过拟合。

**结论**：

1. **欠训成立**：100 epoch 确实截断在上升段，该表示的合理预算是 **≈120 epoch**（100 epoch 版的
   best epoch = 上限本身就是欠训的直接证据）。
2. **但不改变消融结论**：验证集上 +2.83% 的增益只换来测试集 +0.29% 的 NDCG@10（0.013597 → 0.013637）
   与 +0.75% 的 Recall@10。behavior 与 semantic 在 @10 上仍基本持平（0.013637 vs 0.013622，+0.11%），
   三组的排序与"fusion 显著最优"的结论完全不变。
3. **方法学含义**：验证集上 +2.83% 的增益只换来测试集 +0.29%，衰减近十倍，说明**该表示的验证-测试
   泛化噪声远大于它与其他表示之间的差距**。behavior-200ep 相对 semantic 在验证集上是 +0.61%
   （0.020067 vs 0.019946），在测试集上是 +0.11%（0.013637 vs 0.013622）——两个口径下都只是微弱领先，
   落在单 seed 噪声内。因此**behavior 与 semantic 的差异在单 seed 下无法判定**，这与第 3 节"两者 @10 基本持平"
   一致，也说明 5 seed 扩展对 behavior 这一列尤为必要。
   补充对比：三组中只有 behavior 在 100 epoch 触顶（semantic best epoch 90、fusion 80 都自然收敛），
   即欠训只影响 behavior 一列。
4. 子集上同样没有实质变化：`sparse_user` NDCG@10 0.013899（100ep 为 0.013747），
   `cold_start_target` 仍全为 0，`long_tail_target` 仍在 1e-4～1e-5 量级。
