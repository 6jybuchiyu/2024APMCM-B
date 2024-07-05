# 导入所需库
import pandas as pd
from py_geodetector import GeoDetector
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('../processed_train_data.csv')

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
print("因素探测结果:\n", factor_df)

# 交互探测
interaction_df = gd.interaction_detector()
print("交互探测结果:\n", interaction_df)

# 同时生成交互关系
interaction_df, interaction_relationship_df = gd.interaction_detector(relationship=True)
print("交互关系探测结果:\n", interaction_relationship_df)

# 生态探测
ecological_df = gd.ecological_detector()
print("生态探测结果:\n", ecological_df)

# 风险探测
risk_result = gd.risk_detector()
print("风险探测结果:\n", risk_result)

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
gd.plot(value_fontsize=14, tick_fontsize=16, colorbar_fontsize=14)
