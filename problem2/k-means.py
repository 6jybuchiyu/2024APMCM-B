# -*- coding: utf-8 -*-
"""
@Auth : Jybuchiyu
@File : k-means.py
@IDE ：PyCharm
"""
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def vertical_labels(labels):
    return ['\n'.join(label.get_text()) for label in labels]


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    train = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')
    # train = pd.read_csv('../train.csv', encoding='gbk')

    # 聚类
    flood_probabilities = train[['洪水概率']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(flood_probabilities)
    labels = kmeans.labels_
    train['Cluster'] = labels
    cluster_means = train.groupby('Cluster').mean()

    # 查看高、中、低风险类别的洪水概率均值
    cluster_risk = train.groupby('Cluster')['洪水概率'].mean()

    # 输出聚类中心和特征均值
    print("\nCluster Means:\n", cluster_means)
    print("\nCluster Risk Levels:\n", cluster_risk)

    # 0[0.53,0.645],1[0.365,0.465],2[0.47,0.525]
    # 找到每个簇中洪水概率的最小值
    cluster_min_flood_prob = train.groupby('Cluster')['洪水概率'].min()
    print("\n每个簇中洪水概率的最小值:\n", cluster_min_flood_prob)
    # 找到每个簇中洪水概率的最大值
    cluster_max_flood_prob = train.groupby('Cluster')['洪水概率'].max()
    print("\n每个簇中洪水概率的最大值:\n", cluster_max_flood_prob)

    # 在每个簇中随机抽取500个“洪水概率”点
    sample_data = train.groupby('Cluster').apply(lambda x: x.sample(min(len(x), 500), random_state=42)).reset_index(
        drop=True)

    # 生成一个新的索引用于散点图的 x 轴
    sample_data['Index'] = sample_data.index

    # 设置更显眼的颜色
    palette = sns.color_palette("bright", 3)

    # 可视化每个簇中的“洪水概率”点
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='Index', y='洪水概率', hue='Cluster', palette=palette, data=sample_data, alpha=0.6, s=100)
    plt.xlabel('采样点')
    plt.ylabel('洪水概率')
    plt.title('洪水概率因素的K-Means聚类散点图')
    plt.legend()
    plt.show()

    # 将聚类类别添加到原始数据文件的最后一列并保存
    train.to_csv('./processed_train_data_kmeans.csv', index=False)

    # 可视化每个簇的特征均值（去掉id列）
    cluster_means = cluster_means.drop(columns=['id'])
    features = cluster_means.columns

    fig, ax = plt.subplots(figsize=(14, 8))
    for cluster in cluster_means.index:
        ax.plot(features, cluster_means.loc[cluster, :], marker='o', label=f'Cluster {cluster}')

    ax.set_xlabel('特征')
    ax.set_ylabel('均值')
    ax.set_title('不同簇的特征均值')
    ax.legend()
    ax.grid(True)

    # 设置x轴标签为垂直显示
    ax.set_xticklabels(vertical_labels(ax.get_xticklabels()))

    plt.show()
