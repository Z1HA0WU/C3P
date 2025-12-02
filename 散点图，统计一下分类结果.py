import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件中指定子表格
file_path = r"H:\CP_cluster_Confusion\class-conditional-conformal-main\utils\per_class_detailed_results.xlsx"
sheet_name = "Clustered_RAPS"

df = pd.read_excel(file_path, sheet_name=sheet_name)

print("列名：", df.columns.tolist())

# 绘制散点图（所有类相同颜色）
plt.figure(figsize=(8, 6))

for class_id, group in df.groupby('class_id'):
    plt.scatter(group['avg_set_size'], group['coverage_rate'], color='royalblue', alpha=0.7, s=40)
 ----------------------------
# 绘图
# ----------------------------
plt.figure(figsize=(8, 6))

# 绘制不同方法的散点图
plt.scatter(apss_c3p,  cov_c3p,   alpha=0.4, label=f'C³P',   marker='o')
plt.scatter(apss_rc3p, cov_rc3p,  alpha=0.9, label=f'RC3P', marker='^')
plt.scatter(apss_cl,   cov_cl,    alpha=0.35, label=f'Cluster CP', marker='s')

# 目标覆盖线
plt.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, label='Target Coverage (90%)')

# 设置坐标轴范围
plt.xlabel('AS (Average Prediction Set Size)', fontsize=12)
plt.ylabel('Class-wise Coverage', fontsize=12)
plt.title('Efficiency–Coverage Trade-off (Places365|RAPS)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)

# 横纵坐标的范围由你指定
plt.xlim(2, 30)  # 横坐标范围
plt.xticks([2,5, 10, 15,20,25,30]) #
plt.ylim(0.85, 0.95)  # 纵坐标范围

plt.legend(ncol=2, fontsize=10)
plt.tight_layout()
plt.show()
plt.xlabel('avg_set_size', fontsize=14)
plt.ylabel('coverage_rate', fontsize=14)
plt.title('Per-class Coverage vs. Avg Set Size (Clustered RAPS)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
