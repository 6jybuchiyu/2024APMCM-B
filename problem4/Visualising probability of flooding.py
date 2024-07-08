import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('../submit.csv')

# 绘制洪水概率的直方图
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(10, 6))
sns.histplot(data['洪水概率'], bins=30, kde=True)
plt.title('洪水概率的直方图')
plt.xlabel('洪水概率')
plt.ylabel('频数')
plt.grid(True)
plt.show()

# 绘制洪水概率的折线图
plt.figure(figsize=(10, 6))
sns.lineplot(x=data['id'], y=data['洪水概率'])
plt.title('洪水概率的折线图')
plt.xlabel('ID')
plt.ylabel('洪水概率')
plt.grid(True)
plt.show()
