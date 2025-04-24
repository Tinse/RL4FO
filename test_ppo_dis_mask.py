import gymnasium as gym
from function_env_dis_mask import FunctionDisMaskEnv
from stable_baselines3 import PPO
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
env = FunctionDisMaskEnv(
    function=rastrigin,
    dim=12,
    step_size=0.1,
    bound=[-5.12, 5.12],
    # reset_state=np.array([-7.0]*12, dtype=np.float32),
    reset_state=np.random.uniform(-5, 5, 12),
    action_dim = 6,
    is_eval=True,
    eval_steps=300
)
 
# 加载模型
# model = PPO.load("./models/PPO_12dim_ackley_step0.1")
model = PPO.load("./logs/0328PPO_dis_12dim_rastrigin_step01_max100000_reward_reset_failure_3510000_steps")

# 测试模型
info_list = []
step_list = []
count = 0

obs, info = env.reset()
init_obs = obs
init_val = info["value"]

while True:
    # action, _states = model.predict(obs)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminal, truncated, info = env.step(action)
    info_list.append(info)
    print(
        f'state: {obs}, action: {action[0]}, selected: {min(int(action[1] * info["dim"]), info["dim"] - 1)}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}, dim: {info["dim"]}'
    )
    print("----------------------------------")
    if terminal or truncated:
        break

steps, val = zip(*[(step["current_steps"], step["value"]) for step in info_list])
plt.plot(steps, val)
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Value per Step")


print(
    f'init_obs: {init_obs}, init_val: {init_val}, \nfinal_state: {obs}, final_val: {info["value"]}, \nbest: {info["best"]}, best_value: {info["best_value"]}'
)
env.close()
plt.show()




# obs, info = env.reset()
# init_obs = obs
# init_val = info['value']
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     # action, _states = model.predict(obs, deterministic=False)
#     obs, reward, terminal, truncated, info = env.step(action)
#     print(f'state: {obs}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
#     print('----------------------------------')

#     if terminal or truncated:
#         break
# print(f'init_obs: {init_obs}, init_val: {init_val}, \nbest: {info["best"]}, best_value: {info["best_value"]}')
# env.close()
