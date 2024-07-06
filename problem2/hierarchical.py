import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    train = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')

    # 仅使用采样数据来减少内存使用
    sample_data = train.sample(n=10000, random_state=42)  # 调整采样大小以适应内存

    # 聚类
    flood_probabilities = sample_data[['洪水概率']]

    # 使用层次聚类
    Z = linkage(flood_probabilities, method='ward')

    # 获取聚类标签，指定聚为3类
    labels = fcluster(Z, t=3, criterion='maxclust')
    sample_data['Cluster'] = labels
    cluster_means = sample_data.groupby('Cluster').mean()

    # 查看高、中、低风险类别的洪水概率均值
    cluster_risk = sample_data.groupby('Cluster')['洪水概率'].mean()

    # 输出聚类中心和特征均值
    print("\nCluster Means:\n", cluster_means)
    print("\nCluster Risk Levels:\n", cluster_risk)

    # 在每个簇中随机抽取500个“洪水概率”点
    sample_plot_data = sample_data.groupby('Cluster').apply(
        lambda x: x.sample(min(len(x), 500), random_state=42)).reset_index(drop=True)

    # 生成一个新的索引用于散点图的 x 轴
    sample_plot_data['Index'] = sample_plot_data.index

    # 设置更显眼的颜色
    unique_labels = set(labels)
    palette = sns.color_palette("bright", len(unique_labels))

    # 可视化每个簇中的“洪水概率”点
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='Index', y='洪水概率', hue='Cluster', palette=palette, data=sample_plot_data, alpha=0.6, s=100)
    plt.xlabel('采样点')
    plt.ylabel('洪水概率')
    plt.title('洪水概率因素的层次聚类散点图')
    plt.legend()
    plt.show()

    # 将聚类类别添加到原始数据文件的最后一列并保存
    sample_data.to_csv('./processed_train_data_hierarchical.csv', index=False)
