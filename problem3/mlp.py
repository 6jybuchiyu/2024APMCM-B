import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
train_data = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')
selected_columns = ["地形排水", "基础设施恶化", "季风强度", "淤积", "滑坡", "人口得分", "气候变化", "无效防灾",
                    "农业实践", "流域", "政策因素", "规划不足", "洪水概率"]
train_data = train_data[selected_columns]

# 数据预处理
def preprocess_data(data, target_column):
    input_data = data.drop(columns=[target_column])
    output_data = data[target_column]

    scaler_input = StandardScaler()
    scaler_output = StandardScaler()

    input_data_scaled = scaler_input.fit_transform(input_data)
    output_data_scaled = scaler_output.fit_transform(output_data.values.reshape(-1, 1))

    return input_data_scaled, output_data_scaled, scaler_input, scaler_output

# 对训练数据进行预处理
X, y, scaler_input, scaler_output = preprocess_data(train_data, '洪水概率')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为Tensor并移动到GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)
        return x

input_size = X_train.shape[1]
model = MLP(input_size).to(device)  # 移动模型到GPU

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
train_losses = []
val_losses = []
val_mses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_outputs_rescaled = scaler_output.inverse_transform(val_outputs.cpu().numpy())
        y_val_rescaled = scaler_output.inverse_transform(y_val_tensor.cpu().numpy())
        val_mse = mean_squared_error(y_val_rescaled, val_outputs_rescaled)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    val_mses.append(val_mse)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation MSE: {val_mse:.4f}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.legend()
plt.title('训练和验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 绘制验证MSE曲线
plt.figure(figsize=(10, 5))
plt.plot(val_mses, label='验证集上的MSE')
plt.legend()
plt.title('验证集上的MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# 模型验证
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_tensor)
    y_pred_val_rescaled = scaler_output.inverse_transform(y_pred_val.cpu().numpy())
    y_val_rescaled = scaler_output.inverse_transform(y_val_tensor.cpu().numpy())

# 计算验证指标
mse = mean_squared_error(y_val_rescaled, y_pred_val_rescaled)
mae = mean_absolute_error(y_val_rescaled, y_pred_val_rescaled)
r2 = r2_score(y_val_rescaled, y_pred_val_rescaled)

print(f"Validation R²: {r2}")
print(f"Validation Mean Squared Error: {mse}")
print(f"Validation Mean Absolute Error: {mae}")

# 预测test数据
test_data1 = pd.read_csv('../DATA_PROCESS/processed_test_data.csv', encoding='gbk')
test_data = test_data1[selected_columns]
# 确保从测试数据中删除洪水概率列
X_test = test_data.drop(columns=['洪水概率']).values
X_test_scaled = scaler_input.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred_rescaled = scaler_output.inverse_transform(y_test_pred.cpu().numpy())

# 保存预测结果
submit = pd.DataFrame({'id': test_data1['id'], '洪水概率': y_test_pred_rescaled.flatten()})
submit.to_csv('submit.csv', index=False)
