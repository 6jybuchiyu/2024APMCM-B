import pandas as pd
from scipy.stats import shapiro, pearsonr

# 加载数据
data = pd.read_csv('../processed_train_data.csv',encoding='GBK')

# 检查每个变量是否符合正态分布
normality_results = {}
for column in data.columns:
    stat, p_value = shapiro(data[column].dropna())  # 使用Shapiro-Wilk测试
    normality_results[column] = (stat, p_value, p_value > 0.05)  # p值大于0.05通常认为数据符合正态分布

# 输出正态分布检验结果
for column, result in normality_results.items():
    print(f"{column} - Test Statistic: {result[0]}, P-Value: {result[1]}, Normal Distribution: {result[2]}")

# 计算y与每个x的Pearson相关性
correlation_results = {}
y_column = '洪水概率'
for x_column in data.columns:
    if x_column != y_column:
        correlation, p_value = pearsonr(data[y_column].dropna(), data[x_column].dropna())
        correlation_results[x_column] = (correlation, p_value)

# 输出相关性分析结果
for x_column, result in correlation_results.items():
    print(f"Correlation between {y_column} and {x_column}: {result[0]}, P-Value: {result[1]}")
