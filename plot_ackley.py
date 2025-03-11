import numpy as np
import matplotlib.pyplot as plt

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)  # 获取维度
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    part2 = -np.exp(np.sum(np.cos(c * x)) / d)
    result = part1 + part2 + a + np.exp(1)
    return result

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):        
        Z[i, j] = ackley_function(np.array([X[i, j], Y[i, j]]))
        

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_title('Ackley Function Surface Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title('Ackley Function Surface Plot')
plt.show()
