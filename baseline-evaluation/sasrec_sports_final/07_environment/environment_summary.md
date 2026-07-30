# 实验环境摘要

- GPU：NVIDIA GeForce RTX 4090，24,564 MiB
- GPU驱动：580.105.08
- `nvidia-smi`显示的最高支持CUDA版本：13.0
- Python：3.10.8
- PyTorch：2.1.2+cu121
- NumPy：1.26.3
- 训练设备：CUDA
- cuDNN deterministic：true
- cuDNN benchmark：false

注意：`nvidia-smi`中的“CUDA Version 13.0”表示驱动支持上限；实际安装的 PyTorch 构建为 CUDA 12.1（`+cu121`）。

完整依赖见 `requirements.txt`，原始环境记录见 `original_environment_records/`。

