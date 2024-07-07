import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('../processed_train_data.csv', encoding='GBK')

# 计算相关性矩阵
correlation_matrix = data.corr(method='spearman')

# 仅保留与洪水概率的相关性
flood_correlation = correlation_matrix['洪水概率'].drop(['洪水概率','id'])
#排序
flood_correlation = flood_correlation.sort_values(ascending=False)
#输出数值表格
print(flood_correlation)
# 绘制柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名
plt.figure(figsize=(12, 8))
sns.barplot(x=flood_correlation.index, y=flood_correlation.values, palette='viridis')
plt.title('Spearman相关系数')
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便于阅读
plt.ylabel('相关系数')
plt.ylim(0.160, 0.185)  # 设置y轴范围
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

# 显示图形
plt.show();