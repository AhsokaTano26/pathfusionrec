# 实现定位与适用范围

本实验保留了 SASRec 的自注意力序列建模、学习式位置嵌入、因果掩码、残差连接、LayerNorm、Dropout、逐位置正负样本二元交叉熵训练及共享物品嵌入等核心机制。

本实验不是原论文严格复现，原因包括：

| 项目 | SASRec论文实验 | 当前项目实验 |
|---|---|---|
| 数据集 | Beauty、Games、Steam、MovieLens-1M | Amazon Sports |
| 框架 | TensorFlow | pmixer PyTorch |
| 测试候选 | 真实物品与100个采样负例 | 18,357个目录物品全物品评估 |
| 最大序列长度 | ML-1M为200，论文稀疏集为50 | 200 |
| Dropout | ML-1M为0.2，论文稀疏集为0.5 | 0.2 |
| LayerNorm顺序 | 论文公式描述为pre-norm | `norm_first=false`，为post-norm |

推荐固定名称：

> SASRec-PyTorch（pmixer），Amazon Sports，全物品评估，项目实测版，maxlen=200，dropout=0.2。

