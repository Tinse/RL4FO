import torch
import numpy as np
from joblib import load
import torch.nn as nn
from time import perf_counter
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

scaler_X = load(os.path.join(current_dir, 'scaler_X.pkl'))
scaler_Y = load(os.path.join(current_dir, 'scaler_Y.pkl'))
GPU = (torch.cuda.is_available() and False)  # 是否启用GPU，需要设备支持
device = 'cpu'
# 将模型迁移到 GPU（如果需要）,需要设备支持
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, dropout_rate):
        super(RegressionNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        i = 0
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim[i], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


with open(os.path.join(current_dir, 'best-params.txt'), 'r') as file:
    best_params = eval(file.read())
hidden_layers = best_params['hidden_layers']
hidden_dim = [best_params[f'hidden_dim_{i}'] for i in range(hidden_layers)]
dropout_rate = best_params['dropout_rate']
model = RegressionNet(input_dim=12, output_dim=46, hidden_layers=hidden_layers,
                      hidden_dim=hidden_dim, dropout_rate=dropout_rate)
model.load_state_dict(torch.load(os.path.join(current_dir, "bpnn-model.pth")))
model.to(device)
model.eval()


def predict(input_x):
    # start_time = perf_counter()
    x_scaled = scaler_X.transform(np.array(input_x).reshape(1, -1))
    x = torch.tensor(x_scaled, dtype=torch.float32)
    if GPU:
        x = x.to(device)
    y_pre = model(x).detach().numpy()
    # end_time = perf_counter()
    # print(f'used time: {(end_time - start_time) * 1000:.2f}ms')
    return scaler_Y.inverse_transform(y_pre)[0]


def predicts(input_x):
    start_time = perf_counter()
    x_scaled = scaler_X.transform(np.array(input_x))
    x = torch.tensor(x_scaled, dtype=torch.float32)
    if GPU:
        x = x.to(device)
    y_pre = model(x).detach().numpy()
    end_time = perf_counter()
    print(f'used time: {(end_time - start_time) * 1000:.2f}ms')
    return scaler_Y.inverse_transform(y_pre)


if __name__ == '__main__':
    input_x1 = [[0.1 * i for i in range(1, 13)] for i in range(10)]
    input_x2 = [0.1 * i for i in range(1, 13)]
    print(predict(input_x2))
