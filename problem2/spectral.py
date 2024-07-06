import pandas as pd
from sklearn.cluster import SpectralClustering
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    train = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')

    # 聚类
    flood_probabilities = train[['洪水概率']]
    spectral = SpectralClustering(n_clusters=3, random_state=42, affinity='nearest_neighbors')
    labels = spectral.fit_predict(flood_probabilities)
    train['Cluster'] = labels
    cluster_means = train.groupby('Cluster').mean()

    # 查看高、中、低风险类别的洪水概率均值
    cluster_risk = train.groupby('Cluster')['洪水概率'].mean()

    # 输出聚类中心和特征均值
    print("\nCluster Means:\n", cluster_means)
    print("\nCluster Risk Levels:\n", cluster_risk)

    # 在每个簇中随机抽取500个“洪水概率”点
    sample_data = train.groupby('Cluster').apply(lambda x: x.sample(min(len(x), 500), random_state=42)).reset_index(drop=True)

    # 生成一个新的索引用于散点图的 x 轴
    sample_data['Index'] = sample_data.index

    # 设置更显眼的颜色
    unique_labels = set(labels)
    palette = sns.color_palette("bright", len(unique_labels))

    # 可视化每个簇中的“洪水概率”点
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='Index', y='洪水概率', hue='Cluster', palette=palette, data=sample_data, alpha=0.6, s=100)
    plt.xlabel('采样点')
    plt.ylabel('洪水概率')
    plt.title('洪水概率因素的谱聚类散点图')
    plt.legend()
    plt.show()

    # 将聚类类别添加到原始数据文件的最后一列并保存
    train.to_csv('./processed_train_data_spectral.csv', index=False)
