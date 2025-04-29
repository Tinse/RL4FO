import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from BPNN_predict import predicts
import matplotlib.pyplot as plt


# 定义目标函数（最大化问题转换为最小化问题）
def objective_function(x):
    return -predicts(x)[:, -1]


# 设置参数的上下界
lb = -1.5 * np.ones(12)  # 下界
ub = 1.5 * np.ones(12)   # 上界

# 设置优化器参数
options = {
    'c1': 0.5,  # 认知因子
    'c2': 0.3,  # 社会因子
    'w': 0.9   # 惯性权重
}

# 初始化优化器
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=12, options=options, bounds=(lb, ub))

# 运行优化
cost, pos = optimizer.optimize(objective_function, iters=1000)
print(f'cost:{-cost}, \npos:{pos}')
# 绘制优化曲线
plot_cost_history(cost_history=optimizer.cost_history)
plt.title("Cost History")  # 自定义标题
plt.show()
