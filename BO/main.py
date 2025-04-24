from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import time # 导入时间模块

def sphere(x):
    return -np.sum(x ** 2)

def ackley(x):
    """
    Ackley函数实现，支持任意维度

    取值范围: [-32.768, 32.768]
    最小值在f(0,...,0) = 0
    
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)  # 获取维度
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    part2 = -np.exp(np.sum(np.cos(c * x)) / d)
    result = part1 + part2 + a + np.exp(1)
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
    x = np.array(x, dtype=float)  # 添加这行来转换输入
    w = 1.0 + (x - 1.0) / 4.0
    
    term1 = np.sin(np.pi * w[0]) ** 2
    
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[1:]) ** 2))
    
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    
    result = term1 + term2 + term3
    
    # 返回负值（因为RL是最大化奖励）
    return -result

def black_box_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    """
    12维的黑箱函数示例
    """
    # 将输入参数组合成一个数组
    X = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], dtype=np.float32)
    return ackley(X)

# 初始化一个字典来存储参数范围
pbounds = {}
for i in range(1, 13):
    pbounds[f'x{i}'] = (-32.0, 32.0)
print(pbounds)

acquisition_function = acquisition.UpperConfidenceBound(kappa=10)  # 选择UCB作为采集函数
# acquisition_function = acquisition.ProbabilityOfImprovement(xi=0.1)  # 选择PI作为采集函数
# acquisition_function = acquisition.ExpectedImprovement(xi=0.1) # 选择EI作为采集函数

optimizer = BayesianOptimization(
    f=black_box_function,
    acquisition_function=acquisition_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)

optimizer.probe(
    params=np.array([-30.0]*12, dtype=np.float32),
    lazy=False,
)

optimizer.maximize(
    init_points=100,
    n_iter=0,
)

# 计算每次迭代的时间
for i in range(1, 11):
    start_time = time.time()
    optimizer.maximize(
        init_points=0,
        n_iter=10,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {i} took {elapsed_time:.4f} seconds")

print(optimizer.max)
