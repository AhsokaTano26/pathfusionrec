# 实验结果记录

每次完成实验都在本目录新建一个子目录，命名为：

```text
NN_YYYYMMDD_实验简称/
├── metrics.json     # 程序输出的原始指标与配置
└── analysis.md      # 人工撰写的结果分析
```

运行脚本时必须显式指定结果路径，例如：

```bash
PYTHONPATH=src python3 scripts/run_real_bundle_smoke.py \
  --output experiment_results/03_20260726_bundle_smoke_v2/metrics.json
```

`NN` 是从 `01` 开始递增的两位序号。完成运行后立即从 `analysis_template.md` 复制分析模板为该实验的
`analysis.md`，如实记录数据版本、训练设置、核心结果、结论、局限和下一步。
不要只记录最优指标；失败或负结果也必须保留分析，避免重复试错。
