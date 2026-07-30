ActionPiece 任务二交付包说明
============================

本交付包用于“核验 ActionPiece 设置对齐与充分训练”任务。

目录说明
--------

01_report/
  actionpiece_task2_report.txt
    完整核验结论、实验设置、结果解释和可直接用于报告的文字。
  delivery_readme.txt
    本文件。
  setting_alignment_check.xlsx
    论文/官方设置与本次实际设置的逐项核验表。
  training_sufficiency_check.xlsx
    五个种子的停止轮次、最佳轮次、测试指标、耗时与充分性判断。

02_results/
  actionpiece_per_seed_results.csv
    五种子正式结果原始汇总。
  actionpiece_five_seed_summary.csv
    均值、样本标准差、最小值和最大值。
  actionpiece_training_time.csv
    每个种子的停止轮次、最佳轮次、训练/验证/测试耗时及推理速度。
  actionpiece_paper_comparison.csv
    本次五种子均值与论文 Sports 结果对照。
  original_results/seed20xx/
    每个正式种子的原始 summary、meta、GPU 记录、有效配置和 checkpoint 校验值。

03_logs/
  仅包含五个正式种子的完整训练日志。
  冒烟测试、下载失败、重复运行及调试失败日志未收入本包。

04_checkpoints/
  五个种子的最佳 checkpoint。
  checkpoint_manifest.txt 记录文件名、种子、最佳轮次、大小和 SHA256。

05_code_and_config/
  实际运行的 main.py、genrec 源码、运行/评估/汇总脚本、配置说明、命令清单，
  以及相对于官方公开仓库的补丁记录。

06_environment/
  GPU、Python、依赖、Faiss、Git commit、初始 Git 状态和 Sentence-T5 模型信息。

07_data_manifest/
  数据集说明、原始数据与处理后数据 SHA256、缓存生成说明和处理文件清单。
  为控制交付包体积，本目录记录数据文件来源和校验值，不重复打包原始数据与缓存。

SHA256SUMS.txt
  交付目录内全部文件的 SHA256，可用于完整性核验。

重要结论
--------

设置对齐核验通过；五次正式运行均正常完成；充分训练核验部分通过。
当前结果属于阶段性五种子复现结果。前四个种子可能因 patience=20 在后期提升前
停止，只有 seed 2028 表现出较充分的后期收敛。

本包不包含任何报告图表、PNG 或 TensorBoard 图像。
