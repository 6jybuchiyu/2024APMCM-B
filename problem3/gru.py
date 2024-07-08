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

# 定义BiGRU模型
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        out, _ = self.gru(x, h0)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

input_size = X_train_tensor.shape[2]
hidden_size = 64
num_layers = 2
model = BiGRU(input_size, hidden_size, num_layers).to(device)  # 移动模型到GPU

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 256
train_losses = []
val_losses = []
val_mses = []

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor),
                                         batch_size=batch_size, shuffle=False)

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
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_preds_rescaled = scaler_output.inverse_transform(val_preds)
    val_targets_rescaled = scaler_output.inverse_transform(val_targets)

    mse = mean_squared_error(val_targets_rescaled, val_preds_rescaled)
    val_mses.append(mse)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
        print(f'Validation MSE: {mse:.4f}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制验证MSE曲线
plt.subplot(1, 2, 2)
plt.plot(val_mses, label='Validation MSE')
plt.legend()
plt.title('Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.show()

# 打印最终验证指标
best_epoch = np.argmin(val_losses)
print(f'Best Epoch: {best_epoch + 1}')
print(f'Best Validation MSE: {val_mses[best_epoch]:.4f}')

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
