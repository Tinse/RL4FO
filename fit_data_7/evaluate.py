import pandas as pd

data1 = pd.read_csv('raw_data/data_NNGA.csv')
data2 = pd.read_csv('raw_data/data_GA.csv')
data = pd.concat([data1, data2], axis=0)
X = data.iloc[:, 1:13].values  # 获取所有特征列
real_y = data.iloc[:, -2].values  # 获取所有标签列


def R_2(y1,y2):
    """
    计算R2的值
    :param y1: 真实值
    :param y2: 预测值
    :return: R^2
    """
    real_y_mean = y1.mean()
    v1 = sum([(yi-real_y_mean)**2 for yi in y1])
    v2 = sum([(y1[i]-y2[i])**2 for i in range(len(y1))])
    return 1-v2/v1


def bpnn_eval():
    from BPNN_predict import predicts
    test_y = predicts(X)[:, -1]
    with open('bpnn-test-y.txt', 'w+') as file:
        file.write(str(test_y.tolist()))
    print(R_2(real_y, test_y))


def xgboost_eval():
    from BPNN_predict import predicts
    test_y = predicts(X)[:, -1]
    with open('xgboost-test-y.txt', 'w+') as file:
        file.write(str(test_y.tolist()))
    print(R_2(real_y, test_y))


bpnn_eval()

