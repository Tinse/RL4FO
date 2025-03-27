import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def sphere(x):
    return -np.sum(x ** 2)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)  # 获取维度
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    part2 = -np.exp(np.sum(np.cos(c * x)) / d)
    result = part1 + part2 + a + np.exp(1)
    return -result

def rastrigin(x):
    """
    Rastrigin函数实现，支持任意维度
    
    f(x) = A*n + sum(x_i^2 - A*cos(2π*x_i))
    其中A=10, n是维度数
    """
    A = 10
    n = len(x)
    result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return result

def griewank(x):
    """
    Griewank函数实现，支持任意维度
    
    f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i/sqrt(i+1)))
    
    特点：
    - 有无数个规则分布的局部极小值点
    - 全局最小值在 f(0,...,0) = 0
    - 典型搜索空间: x_i ∈ [-600, 600]

    """
    # 计算第一部分: sum(x_i^2)/4000
    sum_part = np.sum(x**2) / 4000
    
    # 计算第二部分: prod(cos(x_i/sqrt(i)))
    # 注意：numpy的索引从0开始，而Griewank函数定义中i从1开始
    indices = np.arange(1, len(x) + 1)
    prod_part = np.prod(np.cos(x / np.sqrt(indices)))
    
    # 计算Griewank函数值
    result = 1 + sum_part - prod_part

    result = np.clip(result, -1e6, 1e6)
    # 返回负值（因为RL是最大化奖励）
    return -result

def levy(x):
    """
    Levy函数实现，支持任意维度
    
    f(x) = sin²(πw₁) + Σᵢ₌₁ᵏ⁻¹[(wᵢ-1)² · (1+10sin²(πwᵢ₊₁))] + (wₙ-1)² · (1+sin²(2πwₙ))
    其中 wᵢ = 1 + (xᵢ-1)/4
    
    特点：
    - 多峰函数，有多个局部最小值
    - 全局最小值在 f(1,...,1) = 0
    - 典型搜索空间: xᵢ ∈ [-10, 10]
    """
    w = 1.0 + (x - 1.0) / 4.0
    
    term1 = np.sin(np.pi * w[0]) ** 2
    
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[1:]) ** 2))
    
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    
    result = term1 + term2 + term3
    
    # 返回负值（因为RL是最大化奖励）
    return -result

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):        
        Z[i, j] = levy(np.array([X[i, j], Y[i, j]]))
        

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('Levy Function Surface Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title('Levy Function Surface Plot')
plt.show()