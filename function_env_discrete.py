'''
一个适用于单目标优化问题的强化学习环境
基于
'''
import gymnasium as gym
import numpy as np
from collections import deque

class FunctionDisEnv(gym.Env):
    def __init__(self, function, dim, bound, step_size = 0.1, max_steps=100, reset_state=None, action_dim=1):
        self.function = function
        self.dim = dim
        self.action_dim = action_dim
        self.bound = bound
        self.action_space = gym.spaces.MultiDiscrete([2] * self.action_dim)  # 每个维度的动作空间为0代表-1, 1代表+1
        self.observation_space = gym.spaces.Box(low=bound[0], high=bound[1], shape=(dim,), dtype=np.float32)
        self.step_size = step_size
        self.state = None
        self.last_val = None
        self.val = None
        self.best = None
        self.last_best_val = None
        self.best_value = None
        self.max_steps = max_steps
        self.current_steps = 0
        assert reset_state is None or len(reset_state) == dim
        self.reset_state = reset_state
        self.best_list = deque(maxlen=10)  # 记录最优解的队列
        self.v = np.random.uniform(0, 0.1, self.dim)  # 初始状态的移动速度，目标是向最优解列表靠近
        self.select = None
        self.reset()


    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {
            'state': self.state, 
            'value': self.val, 
            'best': self.best, 
            'best_value': self.best_value,
            'current_steps': self.current_steps,
            'dim': self.dim,
            'action_dim': self.action_dim
            }

    def reset(self, seed=None, reset_state=None):
        self.current_steps = 0
        if self.reset_state is not None:
            self.state = self.reset_state.copy()
            self.val = self.function(self.state)
        else:
            self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
            self.val = self.function(self.state)

        # self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
        # self.val = self.function(self.state)

        print(f'reset: {self.state}, val: {self.val}')


        self.best = self.state.copy()
        self.best_value = self.function(self.state)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        self.current_steps += 1
        self.last_val = self.val
        self.last_best_val = self.best_value
        action = np.array(action)  # 将动作转换为numpy数组
        self.state[0:self.action_dim] = np.clip(self.state[0:self.action_dim] + (action * 2 - 1) * self.step_size, self.bound[0], self.bound[1])
        self.val = self.function(self.state)
        if self.val > self.best_value:
            self.best = self.state.copy()
            self.best_value = self.val
            self.reset_state = self.best.copy()
            print(f'+++++++++++++  best: {self.best}, best_value: {self.best_value}')
        terminal = False
        # 判断是否结束
        truncated = (self.current_steps >= self.max_steps)  # 直接使用布尔数组比较
        reward = self.val  # 新状态函数值的绝对大小
        # reward = self.val - self.last_val   # 当前动作的改进幅度
        # reward = self.best_value - self.last_best_val   # 最优值的改进幅度
        # reward = self.best_value - self.last_val   # 最优值的改进幅度
        # reward = (self.last_val - self.val) + (-self.val) + (self.best_value - self.last_best_val)  # 三者的综合
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminal, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass

def main():
    def function(x):
        return np.sum(x**2)
    # 创建环境
    env = FunctionDisEnv(
        function=function,
        dim=12,
        step_size=0.1,
        bound=[-10, 10],
        max_steps=100,
        reset_state=np.array([-7]*12, dtype=np.float32),
        action_dim = 12
    )
    observation, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminal, truncated, info = env.step(action)
        print(f'state: {observation}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
        print('----------------------------------')
        if terminal or truncated:
            break
    env.close()


if __name__ == '__main__':
    main()
