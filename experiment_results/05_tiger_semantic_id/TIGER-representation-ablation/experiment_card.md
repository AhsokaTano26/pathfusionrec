# 实验卡：TIGER Semantic ID 输入表示消融 · seed 2026

- **实验编号**：05_tiger_semantic_id / TIGER-representation-ablation / seed2026
- **方法**：固定 TIGER decoder（小型随机初始化 T5 + 前缀约束 beam search，商品全目录候选），只替换
  RQ-VAE 的**输入商品表示**，比较 semantic / behavior / fusion 三条输入通道。
- **使用代码提交**：`bca28672d5c9019510a008e92cb70ddb4a97bc7a`（`perf: batch TIGER constrained decoding`）
- **运行时间**：2026-09-09 20:52（semantic）→ 2026-09-10 19:50（behavior）→ 2026-09-11 05:16（fusion），
  三组串行；`run_logs/tiger_rep_ablation_driver.log` 记录每次 start/done 时刻。
- **GPU / 环境**：RTX 4090；`run_logs/../<rep>/environment.txt` 记录 Python/PyTorch/Transformers/CUDA 版本；
  T5 从随机初始化训练，未加载预训练权重。

## 固定协议与数据

- 用户 35,598；候选商品 18,357；训练 189,543；验证 / 测试各 35,598；`sports_protocol` 固定划分。
- 测试子集掩码：`cold_start_target`（317）、`long_tail_target`（9,962）、`sparse_user`（22,639）。
- 指标：NDCG / Recall @ {5, 10, 20, 50}；**模型选择只用验证集 NDCG@10**，测试集不参与任何选择。
- 三组共用同一个 RQ-VAE 容量 `latent_dim=32`、同一个码本 `[4, 16, 256]`、同一个 seed 2026、
  同一套 T5 超参与解码参数；唯一变量是输入表示。

## 三条输入表示的定义

- **semantic**：冻结的 Sentence-T5-base 768 维商品文本向量（`sentence-transformers/sentence-t5-base`，
  revision `fc5d4628`），是本项目对原始 TIGER 的复现输入。
- **behavior**：由 `PFR-Interaction-v1/seed2026/best_model.pth`
  （SHA-256 `1ab1401b…5bd3`）导出的 128 维 `interaction_embedding.weight`。该检查点只由训练数据训练、
  只按验证集选择，测试集未参与导出。
- **fusion**：语义与行为各自标准化后分别进入 128 维可学习分支编码器，拼接后由共享编码器压到 32 维再量化；
  两个分支各自受一个等权 MSE 重建损失约束，避免 768 维语义在数值上淹没 128 维行为信号。

## RQ-VAE 容量筛选（选择依据，非结果）

`rqvae_latent_capacity.csv` 记录 3 表示 × {32, 64, 128} 共 9 次 RQ-VAE 训练。选择规则是：满足 delivery
validation、三级码本均被使用、死码少、训练商品重建损失无明显退化的**最小**维度，与测试指标无关。

| 表示 | latent | 使用码 [L0,L1,L2] | 死码 [L0,L1,L2] | 末轮重建损失 |
|------|--------|-------------------|-----------------|--------------|
| semantic | 32 | [4, 16, 256] | [0, 0, 0] | 1.009 |
| behavior | 32 | [4, 16, 256] | [0, 0, 0] | 1.139 |
| fusion | 32 | [4, 16, 239] | [0, 0, 17] | 2.131（语义 0.589 + 行为 0.743） |
| semantic | 64 | [4, 16, 244] | [0, 0, 12] | 1.252 |
| behavior | 64 | [4, 16, 256] | [0, 0, 0] | 1.531 |
| fusion | 64 | [4, 16, 206] | [0, 0, 50] | 2.375 |
| semantic | 128 | [4, 16, 224] | [0, 0, 32] | 1.546 |
| behavior | 128 | [4, 16, 256] | [0, 0, 0] | 1.886 |
| fusion | 128 | [4, 16, 174] | [0, 0, 82] | 2.539 |

`latent_dim=32` 同时是原始 TIGER 论文的复现点，也是本表唯一一个语义与行为第三级码本零死码的维度。
值得记录的反直觉现象：**升维并没有增加有效容量**，semantic 与 fusion 在 64/128 上第三级死码反而增至
12→32 与 50→82，重建损失同时上升。因此本消融锁定 32，不把 64/128 作为"更强容量"的对照。

## 模型与训练设置（三组完全一致）

- T5：`d_model=128`、`d_ff=1024`、4 层 encoder / 4 层 decoder、6 头、`d_kv=64`、dropout=0.1；
  词表 = 三级残差码本 token + 碰撞消歧 token + 2,000 用户哈希桶 + BOS/EOS/PAD。
- Batch size=256；AdamW，lr `1e-3`、weight decay `1e-4`；epochs=100；`eval_interval=10`；
  beam size=50，检索 `k=50`；`num_workers=8`、`prefetch_factor=4`。
- 检查点选择：验证集 NDCG@10 最高。

## 结果（单 seed，test）

| 表示 | best epoch | NDCG@10 | Recall@10 |
|------|-----------|---------|-----------|
| semantic（Sentence-T5 768） | 90 | 0.013622 | 0.026799 |
| behavior（PFR interaction 128） | 100（触顶） | 0.013597 | 0.026237 |
| fusion（语义+行为，平衡融合） | 80 | **0.016861** | **0.032670** |

完整 @5/10/20/50 数值见 `seed2026/summary/ablation_results.csv` 与 `ablation_subsets.csv`。

## 与 TIGER-v1 及 PFR 主方法的关系

- fusion 相对 semantic 提升 NDCG@10 +23.8%、Recall@10 +21.9%；behavior 相对 semantic 在 @10 基本持平
  （−0.2% / −2.1%），但在 @50 上更高（NDCG@50 +3.5%、Recall@50 +5.6%），说明行为输入改善的是长尾排序位次
  而非头部命中。
- 与外部基线 TIGER-v1（semantic 输入、`latent=128`、NDCG@10 0.014778）相比，本消融的 semantic 组低 7.8%，
  fusion 组高 14.1%。差异同时来自输入表示与 latent 容量，不应单独归因于某一项。
- 与主方法 PFR-Interaction-v1（NDCG@10 0.015959）相比，fusion 组高 5.7%。**但二者属于不同模型族**
  （约束解码的生成式检索 vs. 交互式 next-item 检索），只能作数值对照，不能据此声称生成式路线优于主方法。
- 三组 role 纪律：本实验是 **Semantic ID 输入表示消融**，既不是 PFR 主方法的消融，也不是外部基线对比。

## 结论与限制

- 输入表示对 TIGER 生成式检索的影响显著：平衡融合在 NDCG/Recall@10、@20 上同时取得最好结果。
- `cold_start_target` 在三组上仍接近 0（fusion 仅在 @50 上非零），`long_tail_target` 绝对值仍在 1e-4 量级；
  融合输入改善的是常规与稀疏用户样本，**没有**修复生成式路径对冷启动/长尾的固有弱点。
- behavior 组 best epoch = 100 恰好触顶，存在欠训可能；`seed2026/behavior-200ep/` 是该问题的诊断性补跑，
  **不进入本消融正式表**。
- 目前只有单个 seed（2026），不能进入最终多 seed 主结果表。ActionPiece/SASRec 历史结果不满足统一候选集
  与完整评估语义，不能纳入本消融的柱状比较。

## 本目录产物

- `seed2026/{semantic,behavior,fusion}/`：`metrics.json`、`config.json`、`training_history.csv`、`run.log`、
  `command.txt`、`environment.txt`、`git_commit.txt`、`delivery_validation.txt`、`best_model.pth`、`sha256sums.txt`。
- `seed2026/summary/`：`ablation_results.csv`、`ablation_subsets.csv`、三对 PNG/SVG 图、`analysis.md`。
- `seed2026/run_logs/`：驱动、watch、RQ-VAE 网格、测试等全部编排日志与所用脚本。
- `rqvae_latent32/<rep>/`：三组实际使用的 RQ-VAE 的 `config.json`、`codebook_diagnostics.json`、
  `standardization.json`、`training_history.json`、`delivery_validation.txt`。
- `rqvae_latent_capacity.csv`：latent 容量筛选原始记录。
