import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import load

# 加载已保存的归一化
scaler_X = load('scaler_X.pkl')
scaler_Y = load('scaler_y.pkl')
""" 假设模型输入形状为 (batch_size, 12)，每个样本有12个输入，46个输出
创建一个包含12个特征的输入数据，可以用pandas导入一整个csv文件，请参考NN_for_DCLSFEL
需要保证input_data.shape = (x, 12)，x为样本数
那么output.shape = (x, 46) """


# 加载模型
model = tf.keras.models.load_model('my_model.keras')
X_new = [1.013212857,0.582868039,-0.896577023,0.300449353,1.187644863,0.894654847,0.18921743,0.173152047,0.860531377,-0.716794012,-0.320151538,-0.752025613]
X_new = np.array(X_new).reshape(1, -1)

# 步骤1：特征归一化
X_new_scaled = scaler_X.transform(X_new)
# 使用模型进行预测（这里不是预测的y！！！！）
Y_pred_scaled = model.predict(X_new_scaled)
# 步骤3：反归一化
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
# 步骤4：反对数变换
Y_pred = np.power(10, Y_pred) - 10  # 反对数变换
print("预测输出：", Y_pred)

