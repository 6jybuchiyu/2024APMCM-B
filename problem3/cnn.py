import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
train_data = pd.read_csv('../DATA_PROCESS/processed_train_data.csv')
selected_columns = ["地形排水", "基础设施恶化", "季风强度", "淤积", "滑坡", "人口得分", "气候变化", "洪水概率"]
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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# 定义批处理大小
batch_size = 256

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor),
                                         batch_size=batch_size, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2, padding=0)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=0)

        # 计算展平后的张量大小
        self.flatten_input_size = self._get_flatten_size(input_size)

        self.fc1 = nn.Linear(self.flatten_input_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def _get_flatten_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_size)
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            return x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = X_train.shape[1]
model = CNN(input_size).to(device)  # 移动模型到GPU

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

# 绘制训练和验证损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 绘制单独的验证MSE曲线
plt.figure()
plt.plot(val_losses, label='Validation MSE')
plt.legend()
plt.title('Validation Mean Squared Error')
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
X_test = test_data.drop(columns=['洪水概率'])
X_test_scaled = scaler_input.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device).unsqueeze(1)

with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred_rescaled = scaler_output.inverse_transform(y_test_pred.cpu().numpy())

# 保存预测结果
submit = pd.DataFrame({'id': test_data1['id'], '洪水概率': y_test_pred_rescaled.flatten()})
submit.to_csv('submit.csv', index=False)
