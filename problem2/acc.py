# -*- coding: utf-8 -*-
"""
@Auth : Jybuchiyu
@File : acc.py
@IDE ：PyCharm
"""
import pandas as pd
from sklearn.metrics import accuracy_score


def calculate_accuracy(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path,encoding='gbk')

    # 获取风险等级数值和聚类标签
    risk_levels = data['风险等级数值']
    clusters = data['Cluster']

    # 计算准确率
    accuracy = accuracy_score(risk_levels, clusters)

    return accuracy


if __name__ == '__main__':
    # 文件路径
    file_path = './processed_train_data_with_risk_levels.csv'

    # 计算准确率
    accuracy = calculate_accuracy(file_path)

    print(f"风险等级数值与聚类标签的对应准确率: {accuracy:.4f}")
