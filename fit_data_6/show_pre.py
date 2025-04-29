import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 导入数据
data1 = pd.read_csv('raw_data/data_NNGA.csv')
data2 = pd.read_csv('raw_data/data_GA.csv')
data = pd.concat([data1, data2], axis=0)
Y_data = data.iloc[:, -2].values
Y_actual = Y_data.astype(np.float64)

with open('bpnn-test-y.txt', 'r') as file:
    Y_pred = np.array(eval(file.readline()))

# 通过 argsort 排序 Y_actual，并保持对应关系
sorted_indices = np.argsort(Y_actual)  # 获取按 Y_actual 排序的索引

# 使用排序的索引对Y_pred 进行排序
Y_actual_sorted = Y_actual[sorted_indices]
Y_pred_sorted = Y_pred[sorted_indices]  # 获取预测值的最后一列

# 计算排序后的差值
difference_sorted = Y_actual_sorted - Y_pred_sorted

# 可视化排序后的 Y_actual 和 Y_pred
plt.figure(figsize=(10, 6))
plt.plot(Y_pred_sorted, label='Predicted', color='red', linestyle='-', markersize=3)
plt.plot(Y_actual_sorted, label='Actual', color='blue', linestyle='-', markersize=5)
plt.title('Sorted Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)
plt.show()