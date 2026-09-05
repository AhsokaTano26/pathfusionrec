# TIGER 语义 ID 检索设计

## 目标

在固定 Sports_and_Outdoors 协议上实现可复现的 TIGER 式下一商品生成检索：RQ-VAE 将未压缩的 768 维 Sentence-T5 商品内容向量离散为 Semantic ID，随后由随机初始化的 T5 编码器—解码器生成下一商品的 ID。

## 实验边界

这是新的 05_tiger_semantic_id 实验系列；不得修改 04_unified_next_item 的稠密排序实验，不得复用 128 维 PCA 缓存或 ActionPiece 的 PQ ID。数据切分、历史长度、子集掩码和 NDCG/Recall 评测均复用现有 sports_protocol。

## 数据流

1. 按 id_mapping.json 的数值商品 ID 顺序读取 metadata.sentence.json，使用记录精确版本的 sentence-transformers/sentence-t5-base 编码，得到 [商品数, 768] 的 float32 向量；严禁 PCA。item.sentence.feat 仅含 ActionPiece token ID，不能作为 Sentence-T5 文本输入。
2. 标准化统计量和 RQ-VAE 只用训练历史/训练目标中出现的商品拟合。拟合后冻结 RQ-VAE，并为全目录商品编码，以便仅有内容、没有训练交互的商品也获得 ID。
3. RQ-VAE 使用 768 到 128 的可学习 encoder、大小为 [4, 16, 256] 的三级残差码本，以及 128 到 768 decoder。用重建损失、commitment 损失和码本损失训练，并以残差 k-means 初始化码本。
4. 每个商品的三级码后附加确定性的碰撞码，形成唯一的四 token Semantic ID；保存商品到 ID 和 ID 到商品的双向映射。
5. 用户历史被展平为 Semantic ID token，并在开头加入稳定哈希的用户 token；冻结 RQ-VAE 后训练 T5 自回归生成目标商品的四个 ID token 和 EOS。
6. 推理使用合法 ID 前缀树约束 beam search，只允许生成目录中存在的 ID；将生成 ID 映射回商品、过滤历史商品，并输出全量与三个子集指标。

## 生成模型与验收

T5 不加载语言预训练权重，默认 4 层 encoder、4 层 decoder、d_model=128、d_ff=1024、6 heads、d_kv=64、dropout 0.1。每个 RQ 层、碰撞码、用户码、PAD、BOS、EOS 使用互不重叠的 token 区间；用户 token 桶数为 2,000。

每次运行必须记录内容模型版本、文本/映射 SHA-256、配置、命令、时间、种子和设备。若向量不是 768 维 float32、ID 不是唯一四元组、任一码本完全未被使用，或 T5 训练/评测的冻结 tokenizer 哈希不一致，则判定结果无效。
