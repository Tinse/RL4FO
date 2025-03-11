import numpy as np
import matplotlib.pyplot as plt

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
