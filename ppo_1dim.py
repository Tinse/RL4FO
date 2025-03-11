'''
使用stable-baselines3库中的PPO算法训练模型，解决function optimization问题
'''
import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback

# 定义目标函数
def objective(x):
    return -(x)**2

def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    # 使用 numpy 的向量化操作，对 x 中的每个元素进行计算
    part1 = -a * np.exp(-b * np.sqrt(x ** 2))
    part2 = -np.exp(np.cos(c * x))
    result = part1 + part2 + a + np.exp(1)
    return -result

# 创建环境
env = FunctionEnv(
    function=objective,
    dim=1,
    bound=[-10, 10],
    step_size=0.1,
    max_steps=200
)

# 创建PPO模型
model_name = "PPO_03_06_dis"
checkpoint_cb = CheckpointCallback(save_freq=10_000, save_path='./logs/', name_prefix=model_name)

model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=r'./tensorboard_logs/', device='cpu')
model.learn(total_timesteps=1000_000, log_interval=1, callback=checkpoint_cb)

# 保存模型
model.save("./models/ppo_function")
del model

# 加载模型
model = PPO.load("./models/ppo_function")

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