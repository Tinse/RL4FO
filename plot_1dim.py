import numpy as np
import matplotlib.pyplot as plt

def ackley_function(x):
    x = np.array(x)  # 添加这行来转换输入
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)  # 获取维度
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    part2 = -np.exp(np.sum(np.cos(c * x)) / d)
    result = part1 + part2 + a + np.exp(1)
    return result

def levy(x):
    x = np.array(x)  # 添加这行来转换输入
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

def levy_1d(x,input=None):
    x = np.array(x)  # 添加这行来转换输入
    if input is None:
        input = np.array([-7.0]*12, dtype=float)  # 初始化为1维数组，默认值为-7.0

    # input = np.random.uniform(-10, 10, 12)  # 随机生成12个元素

    input[0] = x  # 只修改第一个元素

    w = 1.0 + (input - 1.0) / 4.0
    
    term1 = np.sin(np.pi * w[0]) ** 2
    
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[1:]) ** 2))
    
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    
    result = term1 + term2 + term3
    
    # 返回负值（因为RL是最大化奖励）
    return -result
    



# 定义绘图函数
def plot_1dim(func, x_min, x_max, num_points=100):
    input = np.random.uniform(-10, 10, 12)
    print(input)
    x = np.linspace(x_min, x_max, num_points)
    y = np.array([func([xi]) for xi in x])
    
    plt.plot(x, y)
    plt.title(func.__name__)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()

# 绘制Ackley函数
plot_1dim(levy_1d, -10, 10)
