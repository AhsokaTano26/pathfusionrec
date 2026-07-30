# 脚本使用说明

在交付包根目录运行：

```bash
python3 09_scripts/extract_sasrec_results.py
python3 09_scripts/summarize_sasrec_results.py
python3 09_scripts/generate_delivery_manifest.py
bash 09_scripts/verify_delivery_sha256.sh
```

- `extract_sasrec_results.py`：从三种子原始JSON与JSONL提取逐次结果；
- `summarize_sasrec_results.py`：计算均值、样本标准差、最小值和最大值；
- `generate_delivery_manifest.py`：生成交付文件清单；
- `verify_delivery_sha256.sh`：核验全包文件完整性。

前两个脚本会重写 `06_results/` 中对应CSV。因此如果要验证原始交付包不变性，先执行SHA256校验；需要重新生成结果表时再运行统计脚本，并重新生成交付清单。
