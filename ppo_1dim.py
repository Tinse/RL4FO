'''
使用stable-baselines3库中的PPO算法训练模型，解决function optimization问题
'''
import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np

# 定义目标函数
def objective(x):
    return (x)**2

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(x ** 2))
    part2 = -np.exp(np.cos(c * x))
    result = part1 + part2 + a + np.exp(1)
    return result

# 创建环境
env = FunctionEnv(
    function=objective,
    dim=1,
    bound=[-10, 10],
    step_size=0.1,
    max_steps=200
)

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose = 1, device='cpu')
model.learn(total_timesteps=300_000)

# 保存模型
model.save("ppo_function")
del model

# 加载模型
model = PPO.load("ppo_function")

# 测试模型
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminal, truncated, info = env.step(action)
    print(f'state: {obs}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
    print('----------------------------------')

    if terminal or truncated:
        break

env.close()