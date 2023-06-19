import mph
import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os
from os import listdir
import shutil


class Setting:
    project_path = r'F:\神奇的作品存档（临时）\仿真\毕业设计\mphtest.mph'
    edge_path = r'F:\神奇的作品存档（临时）\仿真\毕业设计\\'
    save1_path = r'F:\神奇的作品存档（临时）\仿真\毕业设计\mphtest(save).mph'
    save2_path = r'F:\神奇的作品存档（临时）\仿真\毕业设计\mphtest(best-save)'
    recoveries_path = r'C:\Users\Lenovo\.comsol\v60\recoveries\\'

    a = 1  # 长
    b = 2  # 宽
    c = 5  # 高


# 加载项目并导出名字版本信息
def load():
    global client

    model = client.load(setting.project_path)
    print(model.name())
    print(model.version())
    return model


# 计算接口
def model_solve(model):
    global best_makabaka
    # 计算机械振动和光声耦合系数
    model.solve('研究 1')
    solid_freq = model.evaluate(expression='solid.freq', unit='GHz', dataset='研究 1//解 1')
    solid_Q_eig = model.evaluate(expression='solid.Q_eig', dataset='研究 1//解 1')
    print('声学一阶模式频率：%.2f GHz' % np.real(solid_freq[0]))
    print('声学一阶模式品质因子：%d' % solid_Q_eig[0])
    model.save(setting.save1_path)

    # 计算特征函数（目前还只是最大的光声耦合，改进目标是构建一个以光学模式在194THz附近为前提，考察g的公式）
    makabaka = solid_freq[0]
    if makabaka > best_makabaka:
        model.save(setting.save2_path)

    return makabaka


# 模型修改
def model_revise(pymodel):

    pymodel.parameter('a', setting.a)
    pymodel.parameter('b', setting.b)
    pymodel.parameter('c', setting.c)

    pymodel.mesh()

    pymodel.save(setting.save1_path)

    return 0


def plot_bo(bo):
    x = np.linspace(-180, 180, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean, label="GaussianProcessRegressor")
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1, label="latent function")
    plt.scatter(bo.space.params.flatten(), bo.space.target, label="simulation", c="red", s=50, zorder=10)
    plt.legend()
    plt.xlabel('a', fontsize=14)
    plt.ylabel('g', fontsize=14)
    plt.show()


def makabaka(a, b, c):
    global setting

    setting.a = a
    setting.b = b
    setting.c = c

    print('a = ' + str(a))
    print('b = ' + str(b))
    print('c = ' + str(c))
    model_revise(pymodel)
    makabaka = model_solve(pymodel)
    return makabaka


def initialize_bo():
    bo = BayesianOptimization(
        f=makabaka,
        pbounds={"a": (1, 180),"b": (1, 180), "c": (1, 180)},  # 范围
        verbose=2,  # 显示模式
        random_state=114514,  # 随机种子
    )
    bo.maximize(
        init_points=2,  # 随机步数
        n_iter=3,  # 贝叶斯优化步数
    )
    return bo


def clear():
    # 清空恢复文件
    for dir_name in listdir(setting.recoveries_path):
        if dir_name.startswith('MPHRecovery') and dir_name.endswith('_PM.mph'):
            shutil.rmtree(setting.recoveries_path + dir_name)

    # 清空防爆存档
    os.remove(setting.save1_path)
    return 0


best_makabaka = 0
setting = Setting
client = mph.start()
pymodel = load()
bo = initialize_bo()
# plot_bo(bo)
client.clear()
clear()
