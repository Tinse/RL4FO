import numpy as np
import matplotlib.pyplot as plt

def levy(X):
    X = np.array(X)
    w = 1 + (X - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

# 生成 x1 在 [-10, 10] 之间的值，其余维度固定为 1
x1_values = np.linspace(-10, 10, 1000)

# 修复 X 初始化为二维数组
X = np.ones((1000, 12))*7  # 创建一个二维数组，每行是一个12维的数组，初始值为1
X[:, 0] = x1_values  # 仅改变第一维

y_values = np.array([levy(x) for x in X])

# 绘制函数曲线
plt.figure(figsize=(8, 5))
plt.plot(x1_values, y_values, label='Levy Function w.r.t. $x_1$', color='b')
plt.xlabel('$x_1$')
plt.ylabel('$f(x_1)$')
plt.title('Levy Function with 12 Dimensions (Only $x_1$ Varying)')
plt.grid(True)
plt.legend()
plt.show()