'''
一个适用于单目标优化问题的强化学习环境
基于
'''
import gymnasium as gym
import numpy as np

class FunctionEnv(gym.Env):
    def __init__(self, function, dim, bound, step_size = 1, max_steps=100):
        self.function = function
        self.dim = dim
        self.bound = bound
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=bound[0],high=bound[1], shape=(dim,), dtype=np.float32)
        self.step_size = step_size
        self.state = None
        self.last_val = None
        self.val = None
        self.best = None
        self.best_value = None
        self.max_steps = max_steps
        self.current_steps = 0
        self.reset()


    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {
            'state': self.state, 
            'value': self.val, 
            'best': self.best, 
            'best_value': self.best_value,
            'current_steps': self.current_steps
            }
    
    def reset(self, seed=None):
        self.current_steps = 0
        self.state = np.random.uniform(self.bound[0], self.bound[1], self.dim)
        self.val = self.function(self.state)
        self.best = self.state
        self.best_value = self.function(self.state)
        obsevation = self._get_obs()
        info = self._get_info()
        return obsevation, info
    
    def step(self, action):
        self.current_steps += 1
        self.last_val = self.val
        self.state = np.clip(self.state + action, self.bound[0], self.bound[1])
        self.val = self.function(self.state)
        if self.val < self.best_value:
            self.best = self.state
            self.best_value = self.val
        terminal = False
        # 判断是否结束
        truncated = (self.current_steps >= self.max_steps)  # 直接使用布尔数组比较
        reward = -self.val
        # reward = self.last_val - self.val
        # reward = self.best_value - self.val
        # reward = (self.last_val - self.val) + (-self.val) + (self.best_value - self.val)
        obsevation = self._get_obs()
        info = self._get_info()

        return obsevation, reward, terminal, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass

def main():
    def function(x):
        return np.sum(x**2)
    env = FunctionEnv(function, 1, [-10, 10])
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
