import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 创建环境
env = FunctionEnv(
    function=levy,
    dim=12,
    step_size=1,
    bound=[-10, 10],
    max_steps=100,
    reset_state=np.array([-7.0]*12)
)

# 加载模型
# model = PPO.load("./models/PPO_12dim_ackley_step0.1")
model = PPO.load("./logs/PPO_12dim_levy_step10_max100_mixreward_10000000_steps")

# 测试模型
obs, info = env.reset()
init_obs = obs
init_val = info['value']
while True:
    action, _states = model.predict(obs, deterministic=True)
    # action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminal, truncated, info = env.step(action)
    print(f'state: {obs}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
    print('----------------------------------')

    if terminal or truncated:
        break
print(f'init_obs: {init_obs}, init_val: {init_val}, \nbest: {info["best"]}, best_value: {info["best_value"]}')
env.close()

# # 创建网格点
# num = 20
# x = np.linspace(-10, 10, num)
# y = np.linspace(-10, 10, num)
# X, Y = np.meshgrid(x, y)

# # 计算每个网格点的动作值
# Z = np.zeros((num, num))
# for i in range(num):
#     for j in range(num):
#         state = [X[i,j], Y[i,j]]
#         action, _ = model.predict(state, deterministic=True)
#         Z[i,j] = action[0]  # 假设动作是标量

# # 创建3D图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 绘制曲面
# surface = ax.plot_surface(X, Y, Z, cmap='viridis')

# # 添加颜色条
# fig.colorbar(surface)

# # 设置标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Action')
# ax.set_title('State-Action Surface')

# plt.show()