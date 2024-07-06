import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 熵权法计算权重
def entropy_weight(data):
    # 标准化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # 计算信息熵
    P = data_normalized / data_normalized.sum(axis=0)
    E = -np.nansum(P * np.log(P + 1e-9), axis=0) / np.log(len(data))

    # 计算权重
    d = 1 - E
    w = d / np.sum(d)

    return w


# 选择合适的指标
selected_features = [
        '季风强度', '地形排水', '河流管理', '森林砍伐', '城市化',
        '气候变化', '大坝质量', '淤积', '农业实践', '侵蚀',
        '无效防灾', '排水系统', '海岸脆弱性', '滑坡', '流域',
        '基础设施恶化', '人口得分', '湿地损失', '规划不足', '政策因素'
]

# 读取数据
train = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')

# 提取相关特征数据
data = train[selected_features]

# 计算权重
weights = entropy_weight(data)

# 计算综合评价得分
train['综合评价得分'] = np.dot(data, weights)

# 根据综合评价得分将洪水风险划分为高、中、低
train['风险等级'] = pd.qcut(train['综合评价得分'], q=3, labels=['低风险', '中风险', '高风险'])

# 输出各指标权重
weight_df = pd.DataFrame(weights, index=selected_features, columns=['权重'])
print("\n各指标权重:\n", weight_df)

# 可视化权重
plt.figure(figsize=(14, 8))
sns.barplot(x=weight_df.index, y=weight_df['权重'], palette='bright')
plt.xticks(rotation=90)
plt.title('各指标权重')
plt.show()

# 可视化综合评价得分分布
plt.figure(figsize=(14, 8))
sns.histplot(train['综合评价得分'], bins=30, kde=True)
plt.title('综合评价得分分布')
plt.show()

# 可视化不同风险等级的分布
plt.figure(figsize=(14, 8))
sns.countplot(x='风险等级', data=train, palette='bright')
plt.title('不同风险等级的分布')
plt.show()

# # 将结果保存到新的CSV文件
# train.to_csv('./processed_train_data_with_risk_levels.csv', index=False)
