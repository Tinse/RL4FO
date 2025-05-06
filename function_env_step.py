'''
一个适用于单目标优化问题的强化学习环境
基于
'''
import gymnasium as gym
import numpy as np
from collections import deque

class FunctionDisStepEnv(gym.Env):
    def __init__(self, function, dim, bound, step_size = 0.1, max_steps_explore=100, reset_state=None, action_dim=1, failure_times_max1=1000, failure_times_max2=5, is_eval = False, eval_steps=100):
        self.function = function
        self.dim = dim
        self.action_dim = action_dim
        self.bound = bound
        self.action_space = gym.spaces.MultiDiscrete([2] * self.action_dim+[10])  # 每个维度的动作空间为0代表-1, 1代表+1, 最后一个维度0代表1, 1代表10
        # self.action_space = gym.spaces.MultiDiscrete([3] * self.action_dim)  # 每个维度的动作空间为0代表-1, 1代表0, 2代表+1
        self.observation_space = gym.spaces.Box(low=bound[0], high=bound[1], shape=(dim,), dtype=np.float32)
        self.is_eval = is_eval
        self.eval_steps = eval_steps
        self.step_size = step_size
        self.state = None
        self.last_val = None
        self.val = None
        self.best = None
        self.last_best_val = None
        self.best_value = None
        self.max_steps = 1.0  # 当前最大步数
        self.max_steps_explore = max_steps_explore  # 跳出局部最优解的探索步数
        self.current_steps = 0
        assert reset_state is None or len(reset_state) == dim
        self.reset_state = reset_state
        self.reset_val = None
        self.best_list = deque(maxlen=10)  # 记录最优解的队列
        self.v = np.random.uniform(0, 0.1, self.dim)  # 初始状态的移动速度，目标是向最优解列表靠近
        self.select = None
        self.failure_times = 0  # 当前失败次数
        self.failure_times_max1 = failure_times_max1  # 最大失败步数
        self.failure_times_max2 = failure_times_max2  # 最大失败次数
        self.explore_mode = False  # 是否处于探索模式
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

        # print(f'reset: {self.state}, val: {self.val}')

        self.reset_val = self.val
        self.best = self.state.copy()
        self.best_value = self.function(self.state)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        truncated = False  # 是否改进而结束
        terminal = False  # 是否达到最大步数而中止

        self.current_steps += 1
        self.last_val = self.val
        self.last_best_val = self.best_value
        action = np.array(action)  # 将动作转换为numpy数组
        self.state[0:self.action_dim] = np.clip(self.state[0:self.action_dim] + (action[0:self.action_dim] * 2 - 1) * self.step_size *action[-1], self.bound[0], self.bound[1])
        # self.state[0:self.action_dim] = np.clip(self.state[0:self.action_dim] + (action - 1) * self.step_size, self.bound[0], self.bound[1])
        self.val = self.function(self.state)
        if self.val > self.best_value:
            self.best = self.state.copy()
            self.best_value = self.val
            self.reset_state = self.best.copy()
            self.reset_val = self.val
            print(f'+++++++++++++  best: {self.best}, best_value: {self.best_value}')
            terminal = True  # 立即结束,以免剩下的步数不足以跳出局部最优解,造成浪费
            self.max_steps = 1.0  # 设置最大步数为1,以便立即结束
            self.failure_times = 0  # 重置失败次数
            self.explore_mode = False  # 重置探索模式
            
        # terminal = False
        # 判断是否结束
        truncated = (self.current_steps >= self.max_steps)  # 直接使用布尔数组比较
        if not terminal and truncated:  # 如果没有改进，且达到最大步数，则失败次数加1
            self.failure_times += 1
        if self.explore_mode:  # 如果处于探索模式，则失败次数加1
            if (self.failure_times >= self.failure_times_max2) :  # 如果达到最大失败次数, 增加最大步数
                print(f'failure!!!!! self.max_steps:{self.max_steps}, reset: {self.reset_state}, val: {self.best_value}')
                # self.max_steps *= 1.05  # 最大步数增加
                # self.max_steps = min(self.max_steps, self.max_steps_explore * 100)  # 最大步数不超过探索步数的100倍    
                self.failure_times = 0   
        elif (self.failure_times >= self.failure_times_max1) :  # 如果不是探索模式，且达到最大失败次数,则修改认为是局部最优解
            print(f'failure!!!!! self.max_steps:{self.max_steps}, reset: {self.reset_state}, val: {self.best_value}')
            if self.max_steps >= self.max_steps_explore:  # 如果当前最大步数小于探索步数，则增加最大步数
                self.max_steps *= 1.1  # 最大步数翻倍
                print(f"### new max_steps: {self.max_steps}")
            else:   
                self.max_steps = self.max_steps_explore  # 设置最大步数为探索步数
            self.max_steps = self.max_steps_explore  # 设置最大步数为探索步数
            self.explore_mode = True  # 进入探索模式
            self.failure_times = 0

        # if truncated: # 如果达到最大步数,判断是否找到更优的解
        #     if self.best_value == self.reset_val: # 如果没有找到更优的解，则继续查找
        #         truncated = False

        # 计算奖励
        reward = self.val  # 新状态函数值的绝对大小
        # reward = self.val - self.last_val   # 当前动作的改进幅度
        # reward = self.best_value - self.last_best_val   # 最优值的改进幅度
        # reward = self.best_value - self.last_val   # 最优值的改进幅度
        # reward = (self.last_val - self.val) + (-self.val) + (self.best_value - self.last_best_val)  # 三者的综合
        observation = self._get_obs()
        info = self._get_info()
        
        if self.is_eval:
            terminal = False
            truncated = False
            terminal = (self.current_steps >= self.eval_steps)  # 直接使用布尔数组比较
        return observation, reward, terminal, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass

def main():
    def function(x):
        return np.sum(x**2)
    # 创建环境
    env = FunctionDisStepEnv(
        function=function,
        dim=12,
        step_size=0.1,
        bound=[-10, 10],
        max_steps_explore=100,
        reset_state=np.array([-7]*12, dtype=np.float32),
        action_dim = 12,
        is_eval = True
    )
    observation, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        observation, reward, terminal, truncated, info = env.step(action)
        print(f'state: {observation}, action: {action}, reward: {reward}, \nval: {info["value"]}, best: {info["best"]}, best_value: {info["best_value"]}, current_steps: {info["current_steps"]}')
        print('----------------------------------')
        if terminal or truncated:
            break
    env.close()


if __name__ == '__main__':
    main()
