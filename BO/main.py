import sys
import os

# 将父目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import time # 导入时间模块
import matplotlib.pyplot as plt
from fit_data_6.BPNN_predict import predict

# matplotlib的中文支持
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
rcParams['axes.unicode_minus'] = False  # 显示负号

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
    # return predict(X)[45]
    return griewank(X)  # 使用Griewank函数作为目标函数

# 初始化一个字典来存储参数范围
pbounds = {}
for i in range(1, 13):
    pbounds[f'x{i}'] = (-600, 600)  # 修改参数范围为[-10, 10]
print(pbounds)

acquisition_function = acquisition.UpperConfidenceBound(kappa=10.0)  # 选择UCB作为采集函数
# acquisition_function = acquisition.ProbabilityOfImprovement(xi=0.1)  # 选择PI作为采集函数
# acquisition_function = acquisition.ExpectedImprovement(xi=0.1) # 选择EI作为采集函数

optimizer = BayesianOptimization(
    f=black_box_function,
    acquisition_function=acquisition_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)

# optimizer.probe(
#     params=np.array([0.6]*12, dtype=np.float32),
#     lazy=False,
# )

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
# print(optimizer.res)

# 将每次迭代的目标函数值绘制成图
# 提取优化过程中的目标函数值
targets = [res['target'] for res in optimizer.res]
iterations = list(range(1, len(targets) + 1))

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(iterations, targets, 'b-o', linewidth=2, markersize=8)

# 添加标题和标签
plt.title('Levy目标函数值优化过程', fontsize=16)
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('目标函数值', fontsize=14)

# 添加网格线增强可读性
plt.grid(True, linestyle='--', alpha=0.7)

# 标记最佳值
best_iteration = targets.index(max(targets)) + 1
plt.scatter(best_iteration, max(targets), color='red', s=100, zorder=5)
plt.annotate(f'最优值: {max(targets):.4f}', 
             xy=(best_iteration, max(targets)),
             xytext=(best_iteration+0.5, max(targets)),
             fontsize=12)

# 美化图表
plt.tight_layout()

# 显示图表
# plt.savefig('optimization_process.png', dpi=300)
plt.show()