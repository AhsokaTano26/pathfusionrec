# 代码说明

`original_pmixer/` 保存本实验直接依赖的 pmixer Python 基础文件：

- `main.py`：原仓库训练入口，仅作上游实现参照；
- `model.py`：SASRec模型结构；
- `utils.py`：数据划分、采样器和通用工具。

`sports_experiment/` 保存本实验实际新增并运行的文件：

- `main_sports.py`：Sports训练、逐轮验证、早停、最佳模型保存和最终测试；
- `fullsort_eval.py`：全物品打分、历史过滤、Top-K及NDCG/HR/Recall计算。

实际正式运行入口是 `sports_experiment/main_sports.py`。所有代码均从原实验交付包原样复制；本目录未对训练或评估逻辑做后期修改。

