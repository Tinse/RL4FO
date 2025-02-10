import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from joblib import dump



# 设置随机种子
seed = 20040614
# 设置 Python 内置 random 库的种子
random.seed(seed)
# 设置 NumPy 随机数种子
np.random.seed(seed)
# 设置 TensorFlow 随机数种子
tf.random.set_seed(seed)

def BPNNModel(input_size, hidden_size1, hidden_size2,output_size):
    model = models.Sequential()
    model.add(layers.Dense(hidden_size1, input_dim=input_size, activation='relu'))
    model.add(layers.Dense(hidden_size2, activation='relu'))
    model.add(layers.Dense(output_size, activation='relu')) 
    return model

# 导入数据
data = pd.read_csv('data_NNGA.csv')
X_train = data.iloc[:, 1:13].values  # 获取所有特征列
Y_train = data.iloc[:, 13:-1].values   # 获取所有标签列

X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)



# 形状调整
print(X_train.shape)  # 打印原始数据的形状
print(Y_train.shape)  # 打印原始数据的形状
Y_train = Y_train.reshape(-1, 46)  # 将标签调整为 (样本数, 46)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
Y_train = np.log10(Y_train+10)  # 对数变换
print(X_train.shape)  # 打印原始数据的形状
print(Y_train.shape)  # 打印原始数据的形状
print(X_test.shape)  # 打印原始数据的形状
print(Y_test.shape)  # 打印原始数据的形状
# 归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_y.fit_transform(Y_train)
dump(scaler_X, 'scaler_X.pkl')
dump(scaler_y, 'scaler_y.pkl')

# BPNN模型参数
input_size = X_train_scaled.shape[1]  # 输入的特征维度
hidden_size1 = 32
hidden_size2 = 64  # 第二层隐藏层大小
output_size = Y_train_scaled.shape[1]
model = BPNNModel(input_size, hidden_size1, hidden_size2, output_size)


# 训练模型, callbacks=[early_stopping]
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
model.fit(X_train_scaled, Y_train_scaled, epochs=160, batch_size=10, verbose=1)

# 进行预测并反归一化
Y_pred_scaled = model.predict(X_train_scaled)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

# 对预测结果进行反log变换
Y_train = np.power(10, Y_train) - 10  # 对预测值进行反对数变换并去掉平移
Y_pred = np.power(10, Y_pred) - 10  # 对预测值进行反对数变换并去掉平移

""" # 打印反归一化后的预测值
print(f"Predicted values (scaled back): {Y_pred[1000:1021,:]}")
print(f"Actual values (scaled back): {Y_train[1000:1021,:]}") """

loss = model.evaluate(X_train_scaled, Y_train_scaled)
print(f"Model Loss (MSE): {loss}")
r2 = r2_score(Y_train[:, -1], Y_pred[:, -1])
print(f"pd_energy in train r2 is {r2}\n")


model.save('my_model.h5')
model.export('tfmodel')


# 创建一个 DataFrame 保存测试集中的预测值与实际值
results_df = pd.DataFrame(Y_pred, columns=[f"Predicted_{i+1}" for i in range(Y_pred.shape[1])])
actual_df = pd.DataFrame(Y_train, columns=[f"Actual_{i+1}" for i in range(Y_train.shape[1])])
# 合并预测值与实际值
final_df = pd.concat([results_df, actual_df], axis=1)
# 保存到 CSV 文件
final_df.to_csv("predictions_vs_actuals.csv", index=False)

X_test_scaled = scaler_X.fit_transform(X_test)
Y_pred_test_scaled = model.predict(X_test_scaled)
# 对预测结果进行反标准化
Y_pred_test = scaler_y.inverse_transform(Y_pred_test_scaled)
Y_pred_test = np.power(10, Y_pred_test) - 10  # 对预测值进行反对数变换并去掉平移

mse = mean_squared_error(Y_test, Y_pred_test)
pd_energy_mse = mean_squared_error(Y_test[:, -1], Y_pred_test[:, -1])
mae = mean_absolute_error(Y_test, Y_pred_test)
r2 = r2_score(Y_test[:, -1], Y_pred_test[:, -1])



print(Y_test.shape)  # 打印原始数据的形状
print(Y_pred_test.shape)  # 打印原始数据的形状
print('\n')
print(f"mse is {mse}\n")
print(f"pd_energy mse is {pd_energy_mse}\n")
print(f"mae is {mae}\n")
print(f"pd_energy r2 is {r2}\n")