import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# 检查是否有可用的 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义神经网络模型
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
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 验证集评估
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)
    return val_loss


# Optuna 优化目标函数
def objective(trial):
    # 超参数空间
    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
    hidden_dim = [trial.suggest_int(f"hidden_dim_{i}", 32, 512) for i in range(hidden_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 30, 100)

    # 导入数据
    data1 = pd.read_csv('raw_data/data_NNGA.csv')
    data2 = pd.read_csv('raw_data/data_GA.csv')
    data = pd.concat([data1, data2], axis=0)
    X_data = data.iloc[:, 1:13].values  # 获取所有特征列
    Y_data = data.iloc[:, 13:-1].values  # 获取所有标签列
    X_data = X_data.astype(np.float64)
    Y_data = Y_data.astype(np.float64)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_data_scaled = scaler_X.fit_transform(X_data)
    Y_data_scaled = scaler_Y.fit_transform(Y_data)
    dump(scaler_X, 'scaler_X.pkl')
    dump(scaler_Y, 'scaler_Y.pkl')
    X_train, X_val, y_train, y_val = train_test_split(X_data_scaled, Y_data_scaled, test_size=0.1, random_state=32)

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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练并评估
    val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    return val_loss


# Optuna 优化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # 运行50次试验

# 输出最佳超参数
best_params = study.best_params
best_loss = study.best_value
with open('best-params.txt', 'w+') as file:
    file.write(str(best_params))
print("Best Validation Loss:", best_loss)
print("Best Parameters:", best_params)
