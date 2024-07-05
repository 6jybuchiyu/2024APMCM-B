import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.stats.proportion import proportions_chisquare
# from statsmodels.stats.agreement import kendall_w
from pingouin import intraclass_corr

# 读取数据
df = pd.read_csv('../train.csv')

# 分离y和x
y = df['洪水概率']
X = df.drop(columns=['洪水概率','id'])

# 可视化相关性热力图
def plot_heatmap(corr_matrix, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

# Pearson相关性分析
pearson_corr = X.corr(method='pearson')
plot_heatmap(pearson_corr, 'Pearson Correlation')

# Spearman相关性分析
spearman_corr = X.corr(method='spearman')
plot_heatmap(spearman_corr, 'Spearman Correlation')

# Kendall’s tau-b相关性分析
kendall_corr = X.corr(method='kendall')
plot_heatmap(kendall_corr, 'Kendall Tau-b Correlation')

# Cochran's Q 检验（适用于二分类数据）
# 示例数据，实际需要二分类数据进行分析
# binary_data = np.random.randint(0, 2, size=(20, 3))
# q_stat, p_value = proportions_chisquare(binary_data.sum(axis=0), binary_data.shape[0])
# print(f"Cochran's Q test: Q-stat={q_stat}, p-value={p_value}")

# Kappa一致性检验
# 示例数据，实际需要评级数据进行分析
# ratings1 = np.random.randint(0, 3, size=20)
# ratings2 = np.random.randint(0, 3, size=20)
# kappa = cohens_kappa(ratings1, ratings2)
# print(f"Cohen's Kappa: {kappa[0]}, p-value={kappa[1]}")

# Kendall一致性检验
# 示例数据，实际需要评级数据进行分析
# ratings_matrix = np.random.randint(0, 3, size=(20, 3))
# kendall_w_result = kendall_w(ratings_matrix)
# print(f"Kendall's W: {kendall_w_result}")

# 组内相关系数
# 示例数据，实际需要多评估者数据进行分析
icc_data = pd.DataFrame({
    'targets': np.tile(np.arange(10), 3),
    'raters': np.repeat(['rater1', 'rater2', 'rater3'], 10),
    'ratings': np.random.rand(30)
})
icc_result = intraclass_corr(data=icc_data, targets='targets', raters='raters', ratings='ratings')
print(icc_result)
