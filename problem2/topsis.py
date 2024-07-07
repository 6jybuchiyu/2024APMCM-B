# -*- coding: utf-8 -*-
"""
@Auth : Jybuchiyu
@File : topsis.py
@IDE ：PyCharm
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif


def vertical_labels(labels):
    return ['\n'.join(label.get_text()) for label in labels]


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


# ANOVA单因素方差分析
def ANOVA(data, target):
    f, p = f_classif(data, target)
    return f, p


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    train = pd.read_csv('./processed_train_data_kmeans.csv')

    # 单因素方差分析

    # 特征和目标变量
    X = train.drop(columns=['id', 'Cluster', '洪水概率'])
    y = train['Cluster']
    F, p_values = ANOVA(X, y)
    # 创建 DataFrame 显示 p 值
    p_values_df = pd.DataFrame({'Feature': X.columns, 'P-Value': p_values}).sort_values(by='P-Value')
    print(p_values_df)
    # 选择显著性水平小于 0.05 的特征
    significant_features = p_values_df[p_values_df['P-Value'] < 0.05]['Feature']
    print("Significant Features:\n", significant_features)
    # 提取合适的特征
    X_selected = X[significant_features]

    # 熵权法计算权重
    # k-means 0[0.53,0.645],1[0.365,0.465],2[0.47,0.525]

    # 计算权重
    weights = entropy_weight(X_selected)
    # 计算综合评价得分
    train['综合评价得分'] = np.dot(X_selected, weights)
    # 根据综合评价得分将洪水风险划分为高、中、低
    train['风险等级'] = pd.cut(train['综合评价得分'], bins=[-np.inf, 0.465, 0.525, np.inf],
                               labels=['低风险', '中风险', '高风险'])

    # 将风险等级转换为数值
    risk_mapping = {'低风险': 1, '中风险': 2, '高风险': 0}
    train['风险等级数值'] = train['风险等级'].map(risk_mapping)

    # 输出各指标权重
    weight_df = pd.DataFrame(weights, index=significant_features, columns=['权重'])
    print("\n各指标权重:\n", weight_df)

    # 可视化权重
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=weight_df.index, y=weight_df['权重'], palette='bright')
    ax.set_xticklabels(vertical_labels(ax.get_xticklabels()))
    plt.title('各指标权重')
    plt.show()

    # 可视化综合评价得分分布
    plt.figure(figsize=(14, 8))
    sns.histplot(train['综合评价得分'], bins=30, kde=True)
    plt.title('综合评价得分分布')
    plt.show()

    # 打印每个风险等级的样本数量
    risk_counts = train['风险等级'].value_counts()
    print("\n风险等级样本数量:\n", risk_counts)
    # 可视化不同风险等级的分布
    plt.figure(figsize=(20, 8))
    sns.countplot(x='风险等级', data=train, palette='bright')
    plt.title('不同风险等级的分布')
    plt.show()

    # 保存结果到CSV文件
    train.to_csv('./processed_train_data_with_risk_levels.csv', index=False, encoding='gbk')
