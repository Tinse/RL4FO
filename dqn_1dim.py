'''
DQN算法优化1维函数
'''
import numpy as np
from stable_baselines3 import DQN
from function_env_discrete import FunctionDisEnv

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

# 定义环境
env = FunctionDisEnv(
    function=ackley_function,
    dim=1,
    bound=[-10, 10],
    max_steps=20
)

model = DQN.load("dqn_1dim", env)

# 定义模型
# model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, log_interval=100)
model.save("dqn_1dim")
del model

# 加载模型
model = DQN.load("dqn_1dim")

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
print(f'init_obs: {init_obs}, init_val: {init_val}, \nfinal_state: {obs}, final_val: {info["value"]}, \nbest: {info["best"]}, best_value: {info["best_value"]}')
env.close()
