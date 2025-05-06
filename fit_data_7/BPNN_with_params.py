import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class CustomLoss(nn.Module):
    def __init__(self, mse_weight=1.0, negative_penalty=10.0):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.negative_penalty = negative_penalty

    def forward(self, predictions, targets):
        # 计算标准的MSE损失
        mse = self.mse_loss(predictions, targets)

        # 创建一个掩码，标记哪些预测值为负数
        negative_mask = (predictions < 0).float()

        # 计算负输出的惩罚项
        # 这里使用线性惩罚，可以根据需要调整为其他形式
        negative_penalty_term = self.negative_penalty * torch.sum(negative_mask * torch.abs(predictions)**2)

        # 总损失为MSE加上惩罚项
        total_loss = self.mse_weight * mse + negative_penalty_term

        return total_loss


class RegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, dropout_rate):
        super(RegressionNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        i = 0
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim[i], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        print(f'<{epoch+1}>:', end='\t')
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'loss:{loss}')

    # 验证集评估
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)
    # 假设 model 是你的模型对象
    torch.save(model.state_dict(), "bpnn-model.pth")
    return val_loss


if __name__ == '__main__':
    # 检查是否有可用的 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open('best-params.txt', 'r') as file:
        best_params = eval(file.read())
    hidden_layers = best_params['hidden_layers']
    hidden_dim = [best_params[f'hidden_dim_{i}'] for i in range(hidden_layers)]
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    epochs = 100  # best_params['epochs']
    # 导入数据
    data1 = pd.read_csv('raw_data/data_NNGA.csv')
    data2 = pd.read_csv('raw_data/data_GA.csv')
    data = pd.concat([data1, data2], axis=0)
    X_data = data.iloc[:, 1:13].values  # 获取所有特征列
    Y_data = data.iloc[:, 13:-1].values  # 获取所有标签列
    # 找到 Y_data 中最后一列大于 60 的行
    last_col = Y_data[:, -1]  # 获取最后一列
    L = [(10, 20), (20, 74), (74, 100)]
    m = [5, 20, 40]
    for v in range(len(L)):
        condition = (last_col >= L[v][0]) & (last_col < L[v][1])  # 条件：最后一列大于 60

        # 筛选出符合条件的行
        rows_to_copy = np.where(condition)[0]  # 获取符合条件的行索引
        rows_to_copy = np.repeat(rows_to_copy, m[v])  # 将这些行复制 2 遍

        # 将复制的行加入到原始数据中
        X_data = np.vstack([X_data, X_data[rows_to_copy]])
        Y_data = np.vstack([Y_data, Y_data[rows_to_copy]])

    X_data = X_data.astype(np.float64)
    Y_data = Y_data.astype(np.float64)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_data_scaled = scaler_X.fit_transform(X_data)
    Y_data_scaled = scaler_Y.fit_transform(Y_data)
    dump(scaler_X, 'scaler_X.pkl')
    dump(scaler_Y, 'scaler_Y.pkl')
    X_train, X_val, y_train, y_val = train_test_split(X_data_scaled, Y_data_scaled, test_size=0.01, random_state=32)

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    model = RegressionNet(input_dim=12, output_dim=46, hidden_layers=hidden_layers, hidden_dim=hidden_dim,
                          dropout_rate=dropout_rate).to(device)
    # 使用自定义损失函数
    criterion = CustomLoss(mse_weight=1.0, negative_penalty=0.005)  # 根据需要调整参数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练并评估
    val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    print(val_loss)
