import pandas as pd
from py_geodetector import GeoDetector
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('../processed_train_data.csv', encoding='GBK')

# 指定因变量和自变量
target_variable = '洪水概率'
independent_variables = [
    '季风强度', '地形排水', '河流管理', '森林砍伐', '城市化',
    '气候变化', '大坝质量', '淤积', '农业实践', '侵蚀',
    '无效防灾', '排水系统', '海岸脆弱性', '滑坡', '流域',
    '基础设施恶化', '人口得分', '湿地损失', '规划不足', '政策因素'
]

# 创建GeoDetector对象
df = data[independent_variables + [target_variable]]
gd = GeoDetector(df, target_variable, independent_variables)

# 因素探测
factor_df = gd.factor_dector()
sorted_columns = factor_df.loc['q statistic'].sort_values(ascending=False)
# 输出数值表格
print(sorted_columns)
# 绘制柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_columns.index, y=sorted_columns.values, palette='viridis')
plt.xticks(rotation=45, ha='right')  # 调整x轴标签的旋转角度和对齐方式
plt.title('地理检测法因素探测分析')  # 添加图表标题
plt.xlabel('自变量')  # 设置x轴标签
plt.ylabel('q统计量')  # 设置y轴标签
plt.ylim(0.03, 0.04)  # 设置y轴范围
plt.tight_layout()  # 调整布局
plt.show()

