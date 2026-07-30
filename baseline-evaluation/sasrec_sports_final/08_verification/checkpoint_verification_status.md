# Checkpoint核验状态

三个正式checkpoint均已具备：

- `05_runs/seed_2026/best_model.pth`
- `05_runs/seed_2027/best_model.pth`
- `05_runs/seed_2028/best_model.pth`

已确认：

- 文件存在且大小合理；
- 原始交付包和本交付包均用SHA256固化；
- 文件格式静态检查可见SASRec状态字典参数名称，包括物品嵌入、位置嵌入和注意力层；
- 正式实验流程在训练后曾加载最佳checkpoint完成测试。

未执行：

- 本次整理环境没有安装与原实验匹配的PyTorch，因此未再次运行 `torch.load(..., map_location="cpu")`。

该状态不否定checkpoint或正式结果；如需补证，可在原AutoDL环境只读加载并保存键名日志，无需训练。

