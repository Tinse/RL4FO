'''
使用stable-baselines3库中的PPO算法训练模型，解决function optimization问题
'''
import gymnasium as gym
from function_env_dis_mask_step import FunctionDisMaskStepEnv
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
# from fit_data_6.BPNN_predict import predict
from fit_data_7.BPNN_predict import predict

# 定义目标函数
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

def rastrigin(x):
    """
    Rastrigin函数实现，支持任意维度
    
    f(x) = A*n + sum(x_i^2 - A*cos(2π*x_i))
    其中A=10, n是维度数

    取值范围: [-5.12, 5.12]
    最小值在f(0,...,0) = 0
    最大值在f(±4.52299366, ±4.52299366, ±4.52299366, ±4.52299366) = 40.35329019*n
    12维度时，最大值为484.2394823
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

def model_predict(x):
    bound=[-1.5, 1.5]
    output = predict(x)
        # 计算平滑边界惩罚
    penalty = 0.0
    margin = 0.1  # 定义离边界多少以内开始惩罚
    k = 20     # 惩罚系数，可根据需要调整
    for i in range(12):
        lower_bound = bound[0]
        upper_bound = bound[1]
        d_lower = x[i] - lower_bound
        d_upper = upper_bound - x[i]
        d_min = min(d_lower, d_upper)
        if d_min < margin:
            # 当接近边界时，采用二次函数施加惩罚，越近惩罚越大
            penalty += - k * ((margin - d_min) / margin) ** 2
    
    # 如果状态实际超出边界，也可以设定一个极大惩罚(例如 -1000)，这里我们主要关注平滑惩罚
    if np.any(x[0:12] <= bound[0]) or np.any(x[0:12] >= bound[1]):
        penalty = -1000
    # print(f'penalty: {penalty}')
    # 将平滑惩罚加入奖励中
    output += penalty
    return output

# 创建环境
env = FunctionDisMaskStepEnv(
    function=model_predict,
    dim=12,
    step_size=0.01,
    bound=[1.5, 1.5],
    max_steps_explore=20.0,
    # reset_state=np.array([-1.0]*12, dtype=np.float32),
    # reset_state=np.array(
    #     [
    #         1.3050455489629131,
    #         -0.9303592139937444,
    #         1.1838638460491353,
    #         -1.4131328932334999,
    #         1.1189238903776646,
    #         1.4383262491054345,
    #         0.09580881515837869,
    #         -1.2162059962407716,
    #         1.047388737137548,
    #         0.07394510192515447,
    #         -0.44473003134873057,
    #         0.4537798552896315,
    #     ],
    #     dtype=np.float32,
    # ),
    reset_state=np.array(
        [-1.196, -1.265, 1.107, -1.339, 1.144, -0.955, 1.034, 1.083, -0.5951, -0.003296, 0.9803, 1.385],
        dtype=np.float32,
    ),
    # reset_state=np.array([73.1468512345271, -45.26593997179781, 0.8761172772384085, -4.789395276694479, -36.134215660196084, 55.91877571113968, 9.036882677023637, -100.0403131667313, -52.750683573791115, 52.0634218546809, -68.13752178868114, -32.27691062735976], dtype=np.float32),
    action_dim=12,
    failure_times_max1=1000,  # 局部最优解最大失败次数
    failure_times_max2=10,  # 探索模式最大失败次数
)

# 创建PPO模型
model_name = "0424PPO_dis_mask_step_12dim_predict_step001_reward1"
# 加载模型
# model = PPO.load(f"./logs/{model_name}", learning_rate=1e-4, env=env)
checkpoint_cb = CheckpointCallback(save_freq=10_000, save_path='./logs/', name_prefix=model_name)


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=r'./tensorboard_logs/', device='cpu', learning_rate=1e-3)
model.learn(total_timesteps=100_000_000, log_interval=1, callback=checkpoint_cb)

# 保存模型
model.save(f"./models/{model_name}")
del model

# 加载模型
model = PPO.load(f"./models/{model_name}")

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
