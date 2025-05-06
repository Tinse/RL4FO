import sys
import os

# 将父目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit_data_7.BPNN_predict import predict
import numpy as np

def model_predict(x):
    bound=[-1.5, 1.5]
    output = predict(x)
        # 计算平滑边界惩罚
    penalty = 0.0
    margin = 0.1  # 定义离边界多少以内开始惩罚
    k = 20     # 惩罚系数，可根据需要调整
    for i in range(12):
        lower_bound = bound[0]
        upper_bound = bound[1]
        print(f'lower_bound: {lower_bound}, upper_bound: {upper_bound}')
        d_lower = x[i] - lower_bound
        d_upper = upper_bound - x[i]
        d_min = min(d_lower, d_upper)
        print(f'd_lower: {d_lower}, d_upper: {d_upper}, d_min: {d_min}')
        if d_min < margin:
            # 当接近边界时，采用二次函数施加惩罚，越近惩罚越大
            penalty += - k * ((margin - d_min) / margin) ** 2
    print(f'penalty: {penalty}')
    # 如果状态实际超出边界，也可以设定一个极大惩罚(例如 -1000)，这里我们主要关注平滑惩罚
    if np.any(x[0:12] <= bound[0]) or np.any(x[0:12] >= bound[1]):
        penalty = -200
    print(f'penalty: {penalty}')
    # 将平滑惩罚加入奖励中
    # output += penalty
    return output

# 创建一个新的输入数据
# X_new = np.array([0.995341739, 0.786010426, -0.886444809, 0.182665509, 1.73261416, 1.087535277, -0.051123515, 0.407043123, 0.597274363, -0.618234349, -0.19764553, -0.25507233])
# X_new = np.array(
#     [
#         -0.8451857297013837,
#         -1.2725517499116727,
#         0.3604348169422873,
#         -1.3108941665677087,
#         0.24912691058287728,
#         -1.270428966536534,
#         1.36779115900974,
#         0.5068387357303039,
#         -1.1809464866284458,
#         0.017036937676423225,
#         1.3743642594443104,
#         1.2372916077683138,
#     ]
# )
# X_new = np.array([-1.196, -1.265, 1.107, -1.339, 1.144, -0.955, 1.034, 1.083, -0.5951, -0.003296, 0.9803, 1.385], dtype=np.float32)
X_new = np.array([-1.5, -1.5,  1.5, -1.139, 1.144, -1.5,
  1.4340001, 0.48299995, 1.4, -1.4, 1.5, 0.50000006], dtype=np.float32)

Y_pred = model_predict(X_new)
print("预测输出：", Y_pred)
