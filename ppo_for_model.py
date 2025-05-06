from fit_data.user import predict
from stable_baselines3 import PPO
from function_env import FunctionEnv
import numpy as np
from fit_data_6.BPNN_predict import predict

# 创建一个新的输入数据
# X_new = np.array([0.995341739, 0.786010426, -0.886444809, 0.182665509, 1.73261416, 1.087535277, -0.051123515, 0.407043123, 0.597274363, -0.618234349, -0.19764553, -0.25507233])
X_new = np.array([1.039911351, 0.810837188, -1.05117746, 0.289422608, 1.277286447, 0.74878728, 0.00329239, 0.210143669, 0.92078762, -0.557522222, -0.320331774, -0.585318485])
Y_pred = predict(X_new) 
print("预测输出：", Y_pred)

# 定义环境
env = FunctionEnv(
    function=predict,
    dim=12,
    bound=[-1.5, 1.5],
    step_size=0.1,
    max_steps=200
)

# model = PPO.load("PPO_for_model", env)

# 定义模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=20)
model.save("PPO_for_model")
del model

# 加载模型
model = PPO.load("PPO_for_model")

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
