reward1 = -self.val  # 新状态函数值的绝对大小
reward2 = self.last_val - self.val  # 当前动作的改进幅度
reward3 = self.last_best_val - self.best_value   # 最优值的改进幅度
reward = (self.last_val - self.val) + (-self.val) + (self.last_best_val - self.best_value)  # 三者的综合

reward1 对简单(低维度/少局部最优解)的函数效果好,对复杂函数效果差,缺乏动作信息,奖励尺度不稳定,但利用了最根本的信息
reward2 对复杂函数效果好,可能导致贪婪短视,但局部探索能力强
reward3 对复杂函数效果好,特点是奖励稀疏,但全局探索能力强

下一步短步长下结果的对比