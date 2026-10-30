# 独立自采最终测试集

此目录预留给小组真实拍摄的英文手写答案和教师人工标注，不能用数据集原图或合成图片冒充。当前未采集，不报告自采测试指标。

在本目录放置图片，并创建 UTF-8 CSV `labels.csv`：

```csv
sample_id,prompt,image,reference,score
group_001,3,images/group_001.jpg,"人工逐字转录，保留原有拼写错误",2
```

示例仅展示字段，必须替换为真实采集和标注。题号仅允许 3 或 4，人工分数为 0、1、2。建议两位教师独立标注并记录分歧；保留学生匿名编号，避免重复答案跨集合。reference 可为空，但没有参考转录就不计算该图的 CER/WER。

冻结训练模型和复核阈值后运行：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_self_collected.py --labels data/self_collected/labels.csv
```

该入口只做独立评价，不更新模型。结果写入独立时间戳实验目录。不要将这个最终集用于超参数或阈值调整。
