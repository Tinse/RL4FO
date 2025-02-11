import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
model = tf.keras.models.load_model('my_model2.keras')

def predict(x):
    x = np.array(x).reshape(1, -1)
    # 步骤1：特征归一化
    X_arr_scaled = scaler_X.transform(x)
    # 使用模型进行预测（这里不是预测的y！！！！）
    Y_pred_scaled = model.predict(X_arr_scaled)
    # 步骤3：反归一化
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    # 步骤4：反对数变换
    Y_pred = np.power(10, Y_pred) - 10  # 反对数变换
    # 返回结果中的最后一个预测值
    return Y_pred[0][-1]

if __name__ == '__main__':
    # 创建一个新的输入数据
    # X_new = [0.995341739, 0.786010426, -0.886444809, 0.182665509, 1.73261416, 1.087535277, -0.051123515, 0.407043123, 0.597274363, -0.618234349, -0.19764553, -0.25507233]
    X_new = [1.16637251, 0.57247359, 1.29420623, -0.76916659, -0.16960175, -1.07479055, -0.08722904, 0.5976718, -0.60964133, -0.74038666, 0.16908688, 0.75509292]
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

    # 使用自定义函数
    Y_pred = predict(X_new)
    print("预测输出：", Y_pred)

