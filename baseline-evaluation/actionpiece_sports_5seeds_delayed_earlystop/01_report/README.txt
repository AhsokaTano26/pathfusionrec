ActionPiece Amazon Sports_and_Outdoors 五种子统一交付包

Seeds: 2024, 2025, 2026, 2027, 2028
共同参数：lr=0.005, weight_decay=0.15,
n_hash_buckets=128, epochs=200, eval_batch_size=128

seed 2024–2027：延迟早停配置
early_stop_start_epoch=72
min_epochs_before_stop=100
patience=20

seed 2028：保留原始跑满200轮实验。

实际数据规模：35598 users, 18357 items,
296337 interactions。日志模型规模包含编号0，
因此显示35599 users和18358 items。

每个seed均包含正式日志、最佳checkpoint、
hparams.yml和TensorBoard事件文件。

日志未直接记录纯训练时间和逐用户推理延迟；
results中的时间仅为日志时间戳计算的墙钟时间。
