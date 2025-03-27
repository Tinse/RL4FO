'''
一个适用于单目标优化问题的强化学习环境
基于
'''
import gymnasium as gym
import numpy as np
from collections import deque

class FunctionEnv(gym.Env):
    def __init__(self, function, dim, bound, step_size = 1, max_steps=100, reset_state=None):
        self.function = function
        self.dim = dim
        self.bound = bound
        self.action_space = gym.spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=bound[0],high=bound[1], shape=(dim,), dtype=np.float32)
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
        # print('init')


    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {
            'state': self.state, 
            'value': self.val, 
            'best': self.best, 
            'best_value': self.best_value,
            'current_steps': self.current_steps,
            'dim': self.dim
            }
    
    def reset(self, seed=None):
        self.select = np.random.randint(0, self.dim-1)
        self.current_steps = 0
        if self.reset_state is not None:
            self.state = self.reset_state
            self.val = self.function(self.state)
        else:
            self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
            self.val = self.function(self.state)
        
        # 每次重置为最优值
        # if self.best is not None:
        #     self.state = self.best
        #     self.val = self.function(self.state)
        # elif self.reset_state is not None:
        #     self.state = self.reset_state
        #     self.val = self.function(self.state)
        # else:
        #     self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
        #     self.val = self.function(self.state)

        # while True:
        #     self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
        #     self.val = self.function(self.state)
        #     if self.val != 2.872849464416504:
        #         break
        # self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
        # self.state = np.array([1.07917519, 0.80312352, -1.18543571, 0.31763992, 1.24870056, 0.75935469, 0.05456781, 0.32651591, 0.89882907, -0.56687066, -0.25786348, -0.74467846])
        # self.val = self.function(self.state)
        self.best = self.state
        self.best_value = self.function(self.state)
        # if len(self.best_list) == 0:
        #     self.best_list.append((self.best, self.best_value))
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        # print(f'step_size: {self.step_size}')   
        selected_index = min(int(action[1] * self.dim), self.dim - 1)
        self.current_steps += 1
        self.last_val = self.val
        self.last_best_val = self.best_value
        self.state[selected_index] = np.clip(self.state[selected_index] + action[0] * self.step_size, self.bound[0], self.bound[1])
        self.val = self.function(self.state)
        # if self.val > self.best_list[-1][1]:
        if self.val > self.best_value:
            self.best = self.state
            self.best_value = self.val
            print(f'best: {self.best}, best_value: {self.best_value}')
            # self.best_list.append((self.best, self.best_value))  # 记录最优解
        terminal = False
        # 判断是否结束
        truncated = (self.current_steps >= self.max_steps)  # 直接使用布尔数组比较
        reward = self.val  # 新状态函数值的绝对大小
        # reward = self.val - self.last_val  # 当前动作的改进幅度·
        # reward = self.best_value - self.last_best_val    # 最优值的改进幅度
        # reward = (self.val - self.last_val ) + (self.val) + (self.best_value - self.last_best_val)  # 三者的综合
        observation = self._get_obs()
        info = self._get_info()

        terminal = (self.current_steps >= self.max_steps)
        truncated = (self.current_steps >= self.max_steps)

        return observation, reward, terminal, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass

class Ackley(FunctionEnv):
    def __init__(self, dim, bound=[-32.768, 32.768], step_size=0.1, max_steps=100, reset_state=None):
        super().__init__(self.ackley, dim, bound, step_size, max_steps, reset_state)
        

def main():
    def function(x):
        return np.sum(x**2)
    env = FunctionEnv(function, 12, [-10, 10], 0.1, reset_state=np.array([1.07917519]*12, dtype=np.float32))
    observation, info = env.reset()
    print(f'Initial state: {observation}, Initial best: {info["best"]}, Initial best_value: {info["best_value"]}, Initial current_steps: {info["current_steps"]}')

    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminal, truncated, info = env.step(action)
        print(f'state: {observation}, action: {action[0]}, selected: {min(int(action[1] * info["dim"]), info["dim"] - 1)}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
        print('----------------------------------')
        if terminal or truncated:
            break
    

    print(f'Final state: {observation}, Final best: {info["best"]}, Final best_value: {info["best_value"]}, Final current_steps: {info["current_steps"]}')
    
    env.close()


if __name__ == '__main__':
    main()
