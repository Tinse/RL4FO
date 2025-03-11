import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def sphere(x):
    return -np.sum(x ** 2)

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
    function=sphere,
    dim=1,
    step_size=0.1,
    bound=[-10, 10],
    max_steps=100,
    reset_state=np.array([6.0])
)

# 加载模型
# model = PPO.load("./logs/PPO_03_06_dis_1000000_steps")
model = PPO.load("./logs/PPO_12dim_sphere_step0.1_140000_steps")

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

# 绘制动作预测图
# x = np.linspace(-10, 10, 1000, dtype=np.float32)
state_list = []
action_list = []
action_perfect_list = []
num = 1000
for i in range(num):
    # x = np.random.uniform(-10, 10, 1)
    x = [i/num*20-10]
    # print(x)
    y, _states = model.predict(x, deterministic=True)
    if x[0] < -1:
        y_perfect = 1.0
    elif x[0] > 1:
        y_perfect = -1.0
    else:
        y_perfect = -x[0]
    state_list.append(x)
    action_list.append(y)
    action_perfect_list.append(y_perfect)
    # plt.plot(x, y, 'r,')
plt.plot(state_list, action_list)
plt.plot(state_list, action_perfect_list)
plt.legend(['predict', 'perfect'])
plt.xlabel('state')
plt.ylabel('action')
plt.title('state-action')
plt.grid()
plt.show()