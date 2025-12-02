import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. 读取混淆矩阵
# -------------------------
df = pd.read_excel("混淆矩阵cifar100.xlsx", index_col=0)
delta = 0.10
coverage_ratio = 1 - delta
num_preds = len(df.columns)

cover_counts = []
true_ratios = []
false_ratios = []
correct_counts = []
# -------------------------
# 2. 计算每个预测类别的 95% 覆盖真实类数 + 正确率
# -------------------------
for i, col in enumerate(df.columns):
    values = df[col].values
    total = values.sum()
    if total == 0:
        cover_counts.append(0)
        true_ratios.append(0)
        correct_counts.append(0)
        continue

    # 按频率排序
    sorted_idx = np.argsort(values)[::-1]
    sorted_vals = values[sorted_idx]
    cumsum = np.cumsum(sorted_vals)

    # 找到覆盖(1-delta)样本的最小类别数
    cutoff = total * coverage_ratio
    freq = df[col].values / total  # 频率而非数量
    sorted_freq = np.sort(freq)[::-1]
    cum_freq = np.cumsum(sorted_freq)
    cover_num = np.argmax(cum_freq >= (1 - delta)) + 1
    cover_counts.append(cover_num)

    # 计算正确部分（排在首位的类别）
    correct_counts.append(1)  # 每个预测类别，其第一名的真实类就是"自己"
    correct_ratio = sorted_vals[0] / total if sorted_idx[0] == i else 0
    true_ratios.append(correct_ratio)
# -------------------------
# 3. 绘图
# -------------------------
plt.figure(figsize=(10, 5))
x = np.arange(num_preds)

# 缝隙更小的柱宽
bar_width = 0.4  # 原来是 0.3，现在几乎贴紧但不重叠

# 红色误判部分
plt.bar(x, cover_counts,   color=(1, 0.6, 0.2, 0.8), width=bar_width, edgecolor='black', label='Misclassified portion')
# 绿色正确部分
plt.bar(x, np.array(cover_counts) * np.array(true_ratios),
        color=(0.1, 0.4, 0.8, 0.8), width=bar_width, edgecolor='black', label='Correct portion')

# 平均线
mean_val = np.mean(cover_counts)
plt.axhline(mean_val, color='blue', linestyle='--', linewidth=1.2, label=f'Average = {mean_val:.2f}')

# -------------------------
# 4. 坐标与美化
# -------------------------
step = 10
ticks_to_show = np.arange(0, num_preds, step)
plt.xticks(ticks_to_show, [str(i) for i in ticks_to_show], fontsize=8)

plt.xlabel("Predicted Class Index")
plt.ylabel("Number of True Classes Required for 90% Sample Coverage")
plt.title("Support Set Size Distribution (δ=10%) on CIFAR-100")

plt.xlim(-0.5, num_preds - 0.5)
plt.grid(alpha=0.3, axis='y', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
