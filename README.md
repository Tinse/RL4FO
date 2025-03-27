reward1 = -self.val  # 新状态函数值的绝对大小
reward2 = self.last_val - self.val  # 当前动作的改进幅度
reward3 = self.last_best_val - self.best_value   # 最优值的改进幅度
reward = (self.last_val - self.val) + (-self.val) + (self.last_best_val - self.best_value)  # 三者的综合

reward1 对简单(低维度/少局部最优解)的函数效果好,对复杂函数效果差,缺乏动作信息,奖励尺度不稳定,但利用了最根本的信息
reward2 对复杂函数效果好,可能导致贪婪短视,但局部探索能力强
reward3 对复杂函数效果好,特点是奖励稀疏,但全局探索能力强

下一步短步长下结果的对比

PPO算法
对sphered函数,1-2-6-12维函数都能逐渐趋于收敛，奖励为函数值,固定起始【6.0】，步长0.1，步数100，其他参数默认，12维接近收敛训练步数72w
对ackley函数,1-4-12维函数都能逐渐趋于收敛，奖励为函数值,固定起始【6.0】，步长0.1，步数100，其他参数默认，12维接近收敛训练步数50w
对rastrigin函数，12维固定起始点，经历100w步趋于近优，仍有3个维度还未接近最优解，将奖励修改为综合后，100w步完全
收敛
对于levyh函数，12维固定点无法找到最优解，1000w都没有收敛

SAC算法相对PPO效果更好，但依然无法使levy函数收敛到最优解

尝试mask降低动作空间

需要实验步长对收敛的影响

2025年3月27日21:59:08
50w步找到12维levy函数最优解,在离散动作-1和+1，步长0.1，步数限制1000，每次初始位置修改为当前最优解，学习率1e-4，甚至可以在任意初始状态寻找到最优解
0327PPO_dis_12dim_levy_step01_max100000_reward_reset_690000_steps.zip
tensorboard --logdir="./PPO_102"

