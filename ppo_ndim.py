'''
使用stable-baselines3库中的PPO算法训练模型，解决function optimization问题
'''
import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np

# 定义目标函数
def objective(x):
    return np.sum(x ** 2)

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

# 创建环境
env = FunctionEnv(
    function=ackley_function,
    dim=12,
    bound=[-10, 10],
    max_steps=20
)

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=100_000)

# 保存模型
model.save("ppo_function")
del model

# 加载模型
model = PPO.load("ppo_function")

# 测试模型
obs, info = env.reset()
init_obs = obs
init_val = info['value']
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminal, truncated, info = env.step(action)
    print(f'state: {obs}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
    print('----------------------------------')

    if terminal or truncated:
        break
print(f'init_obs: {init_obs}, init_val: {init_val}, \nbest: {info["best"]}, best_value: {info["best_value"]}')
env.close()