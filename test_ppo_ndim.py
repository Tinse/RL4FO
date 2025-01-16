import gymnasium as gym
from function_env import FunctionEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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