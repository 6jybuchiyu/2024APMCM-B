import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('../processed_train_data.csv',encoding='GBK')

# 计算斯皮尔曼相关性矩阵
correlation_matrix = data.corr(method='spearman')

# 绘制热图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()
