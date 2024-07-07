import pandas as pd
from scipy.stats import anderson
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 加载宋体字体
font = FontProperties(fname='C:\Windows\Fonts\simsun.ttc')  # 请替换为你的宋体字体文件的路径

# 加载数据
data = pd.read_csv('../processed_train_data.csv', encoding='GBK')

# 移除id列
data = data.drop(columns=['id'])

# 存储检验结果和可视化数据
results = []
plots = []

# 对每一列进行Anderson-Darling测试
for column in data.columns:
    result = anderson(data[column].dropna())
    crit = {f'crit_val_{p}%': v for p, v in zip(result.significance_level, result.critical_values)}
    crit.update({'statistic': result.statistic, 'column': column})
    results.append(crit)

    # 为了可视化，收集数据
    plots.append((column, data[column].dropna()))

# 转换为DataFrame并保存到Excel
results_df = pd.DataFrame(results)
results_df.to_excel('anderson_darling_results.xlsx', index=False)

# 可视化
fig, axs = plt.subplots(len(plots), 1, figsize=(10, 5 * len(plots)))

for ax, (col, col_data) in zip(axs, plots):
    sns.histplot(col_data, kde=True, ax=ax)
    ax.set_title(f'{col} 的分布与安德森统计量: {results_df.loc[results_df["column"] == col, "statistic"].values[0]}',
                 fontproperties=font)
    ax.axvline(x=col_data.mean(), color='red', linestyle='--', label='均值')  # 画出均值线
    ax.axvline(x=col_data.median(), color='green', linestyle=':', label='中位数')  # 画出中位数线
    ax.legend(prop=font)

plt.tight_layout()
plt.savefig('distribution_plots.png')  # 保存图表
plt.show()
