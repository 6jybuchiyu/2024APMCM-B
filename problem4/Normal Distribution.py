import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
# 读取数据
data = pd.read_csv('../submit.csv')
plt.rcParams['font.sans-serif'] = ['SimSun']
# 绘制直方图和正态分布曲线
plt.figure(figsize=(10, 6))
sns.histplot(data['洪水概率'], bins=30, kde=True, color="blue")
# 添加正态分布曲线
mean, std = data['洪水概率'].mean(), data['洪水概率'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std)
plt.plot(x, p * len(data['洪水概率']) * (xmax - xmin) / 30, 'k', linewidth=2)
title = "平均值 = %.2f,  标准差 = %.2f" % (mean, std)
plt.title(title)
plt.xlabel('洪水概率')
plt.ylabel('频数')
plt.show()

# 绘制Q-Q图
plt.figure(figsize=(6, 6))
stats.probplot(data['洪水概率'], dist="norm", plot=plt)
plt.title('Q-Q图正态性检验')
plt.xlabel('理论量化值')
plt.ylabel('样本量化值')
plt.show()

# Shapiro-Wilk 测试
shapiro_test = stats.shapiro(data['洪水概率'])
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Kolmogorov-Smirnov 测试
ks_test = stats.kstest(data['洪水概率'], 'norm', args=(mean, std))
print(f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue}")

# Anderson-Darling 测试
anderson_test = stats.anderson(data['洪水概率'], dist='norm')
print(f"Anderson-Darling Test: Statistic={anderson_test.statistic}, "
      f"Significance Levels={anderson_test.significance_level}, "
      f"Critical Values={anderson_test.critical_values}")
