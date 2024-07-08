# -*- coding: utf-8 -*-
"""
@Auth : Jybuchiyu
@File : test_process.py
@IDE ：PyCharm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def vertical_labels(labels):
    return ['\n'.join(label.get_text()) for label in labels]


if __name__ == '__main__':
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    data = pd.read_csv('../test.csv', encoding='GBK')
    # 打印每个变量的中位数
    medians = data.median()
    print(medians)

    print(data.info())
    print(data.describe())

    # 用均值处理缺失值，忽略“洪水概率”列
    print(data.isnull().sum())
    data = data.drop(columns=['洪水概率']).fillna(data.mean())

    # 绘制箱线图
    plt.figure(figsize=(12, 8))  # 调整画布大小以适应竖排显示的标签
    ax = sns.boxplot(data=data.drop(columns=['id']))
    # 将X轴标签竖排显示
    ax.set_xticklabels(vertical_labels(ax.get_xticklabels()))
    plt.show()

    # 删除异常值
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 提取所有列，忽略“洪水概率”
    X = data_cleaned.drop(columns=['id'])
    # 提取id列
    ids = data_cleaned['id']

    # 归一化特征列
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 将id列加回数据中，保留“洪水概率”列（全为空值）
    X_scaled.insert(0, 'id', ids.values)
    X_scaled['洪水概率'] = np.nan

    # 保存清洗和归一化后的数据
    X_scaled.to_csv('processed_test_data.csv', index=False,encoding='GBK')

    print("数据处理完成并保存。")
