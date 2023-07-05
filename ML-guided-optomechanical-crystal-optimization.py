import mph
import numpy as np
from scipy.optimize import curve_fit
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import os
from os import listdir
import shutil
import pandas as pd
from datetime import datetime


class Setting:
    path = r'F:\神奇的作品存档（临时）\仿真\毕业设计\\'  # 项目工作地址（把comsol模型文件放在这里面）
    model_name = '腔体(对接mph).mph'
    recoveries_path = r'C:\Users\Lenovo\.comsol\v60\recoveries\\'  # comsol默认恢复文件夹存档位置，需要自己去设置找，释放临时文件空间用

    project_path = path + model_name  # comsol模型文件地址
    edge_path = path + 'edge\\'  # 边界点集存档位置
    last_model_save_path = path + 'last-model.mph'  # 防爆查错模型存档位置（完成后自动清理）
    best_model_save_path = path + 'best-model.mph'  # 最佳结果模型存档位置
    data_save_path = path + 'data.csv'  # 数据存储
    logs_save_path = path + 'log.json'  # BayesianOptimization日志保存
    logs_load_path = path + 'load_log.json'  # 要读取的日志

    num_periodic = 15  # 周期区每侧包含晶格数
    a_periodic = 330  # 周期区中心椭圆孔a半轴（沿梁方向）[nm]
    b_periodic = 859  # 周期区中心椭圆孔b半轴（垂直梁方向）[nm]
    w1_periodic = 1247  # 周期区正弦边缘峰间距[nm]
    w2_periodic = 919  # 周期区正弦边缘谷间距[nm]
    wide_periodic = 558  # 周期区晶胞宽[nm]

    num_defect = 27  # 缺陷区包含晶格数
    a_defect = 389  # 缺陷区中心椭圆孔a半轴（沿梁方向）[nm]
    b_defect = 778  # 缺陷区中心椭圆孔b半轴（垂直梁方向）[nm]
    w1_defect = 1247  # 缺陷区正弦边缘峰间距[nm]
    w2_defect = 1124  # 缺陷区正弦边缘谷间距[nm]
    wide_defect = 450  # 缺陷区晶胞宽[nm]

    beta_ang = -128  # β角度
    alpha_ang = 0  # α角度

    sigma_square = 24.5  # 正弦拟合尺度参数

    wide_pml = 2000  # 完美匹配层宽度[nm]
    w_add = 80.38  # 样本蚀刻倾斜导致的底层增宽（边缘）[nm]
    a_add = 80.38  # 样本蚀刻倾斜导致的底层增宽（孔内a轴半径）[nm]
    b_add = 139.89  # 样本蚀刻倾斜导致的底层增宽（孔内b轴半径）[nm]
    high = 300  # 纳米梁厚度[nm]
    w_air = 2 * wide_pml  # 空气宽度（依赖于梁+匹配层长）
    l_air = 3000  # 空气长度
    h_air = 5000  # 空气厚度


# 加载项目，导出名字版本信息，建立项目目录
def load():
    global client

    model = client.load(setting.project_path)
    if not os.path.exists(setting.edge_path):
        os.mkdir(setting.edge_path)
    print("Model name:", model.name())
    print("Version:", model.version())
    return model


# 计算接口
def model_solve(model):
    global best_makabaka, df

    # 计算光学模式
    model.solve('研究 2')
    ewfd_freq = model.evaluate(expression='ewfd.freq', unit='THz', dataset='研究 2//解 2')
    ewfd_Qfactor = model.evaluate(expression='ewfd.Qfactor', dataset='研究 2//解 2')
    ewfd_Qfactor_max = ewfd_Qfactor.max()
    imax1 = ewfd_Qfactor.argmax()
    print('光学一阶模式频率：%.2f THz' % ewfd_freq[imax1])
    print('光学一阶模式品质因子：%d' % ewfd_Qfactor_max)

    # 计算机械振动和光声耦合系数
    model.property(model/'studies/研究 1/特征频率', 'notsolnum', imax1 + 1)
    model.solve('研究 1')
    g = model.evaluate(expression='g', unit='Hz', dataset='研究 1//解 1')
    gom = model.evaluate(expression='gom', unit='Hz', dataset='研究 1//解 1')
    gpe = model.evaluate(expression='gpe', unit='Hz', dataset='研究 1//解 1')
    solid_freq = model.evaluate(expression='solid.freq', unit='GHz', dataset='研究 1//解 1')
    solid_Q_eig = model.evaluate(expression='solid.Q_eig', dataset='研究 1//解 1')
    g_max = g.max()
    imax2 = g.argmax()
    print('声学一阶模式频率：%.2f GHz' % np.real(solid_freq[imax2]))
    print('声学一阶模式品质因子：%d' % solid_Q_eig[imax2])
    print('光声耦合系数：%d' % g_max)
    print('移动边界光声耦合系数：%d' % gpe[imax2])
    print('光弹效应光声耦合系数：%d' % gom[imax2])
    model.save(setting.last_model_save_path)

    # 打印数据到文件
    df.loc[len(df)] = [np.real(ewfd_freq[imax1]), np.real(ewfd_Qfactor_max),
                       np.real(solid_freq[imax2]), np.real(solid_Q_eig[imax2]),
                       np.real(gom[imax2]), np.real(gpe[imax2]), np.real(g_max)]
    df.to_csv(setting.data_save_path, mode='w+', header=True, index=False)

    # 计算特征函数（目前还只是最大的光声耦合，改进目标是构建一个以光学模式在194THz附近为前提，考察g的公式）
    makabaka = g_max

    # 如果这是最好结果就存档
    if makabaka > best_makabaka:
        model.save(setting.best_model_save_path)
        best_makabaka = makabaka

    return makabaka


# 模型修改
def model_revise(pymodel):
    a, b, w1, w2, wide = fit()
    x_list = []
    model = pymodel.java

    x = setting.wide_periodic * setting.num_periodic
    for i in range(setting.num_defect):
        x += wide[i] / 2
        x_list += [x]
        x += wide[i] / 2
    x_final = x + setting.wide_periodic * (setting.num_periodic - 0.5)

    # 更改常数，用于插值的edge文件，建模
    pymodel.parameter('wide_periodic', setting.wide_periodic)
    pymodel.parameter('w_add', setting.w_add)
    pymodel.parameter('a_add', setting.a_add)
    pymodel.parameter('b_add', setting.b_add)
    pymodel.parameter('w1_periodic', setting.w1_periodic)
    pymodel.parameter('wide_pml', setting.wide_pml)
    pymodel.parameter('x_final', x_final)
    pymodel.parameter('num_periodic', setting.num_periodic + 1)
    pymodel.parameter('high', setting.high)
    pymodel.parameter('w_air', setting.w_air)
    pymodel.parameter('l_air', setting.l_air)
    pymodel.parameter('h_air', setting.h_air)
    pymodel.parameter('alpha_ang', setting.alpha_ang)
    pymodel.parameter('beta_ang', setting.beta_ang)
    for i in range(setting.num_defect):
        pymodel.parameter('a%d' % i, a[i])
        pymodel.parameter('b%d' % i, b[i])
        pymodel.parameter('x%d' % i, x_list[i])
    edge(w1, w2, wide)
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ic1").set("source", "file");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("ic1").set("filename", setting.edge_path + 'edge%d.out' % rank);
    model.component("comp1").geom("geom1").feature("wp3").geom().feature("ic1").set("source", "file");
    model.component("comp1").geom("geom1").feature("wp3").geom().feature("ic1").set("filename", setting.edge_path + 'edge%d.out' % rank);
    pymodel.build()

    # 更改网格参数并建立网格
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", float(900));  # 900
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", float(300));  # 100
    model.component("comp1").mesh("mesh1").feature("size").set("hgrad", float(2));
    model.component("comp1").mesh("mesh1").feature("size").set("hcurve", float(1));
    model.component("comp1").mesh("mesh1").feature("size").set("hnarrow", float(0.1));
    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", float(900));  # 300
    model.component("comp1").mesh("mesh1").feature("size1").set("hmin", float(300));  # 100
    model.component("comp1").mesh("mesh1").feature("size1").set("hgrad", float(1.8));
    model.component("comp1").mesh("mesh1").feature("size1").set("hcurve", float(0.9));
    model.component("comp1").mesh("mesh1").feature("size1").set("hnarrow", float(0.2));
    model.component("comp1").mesh("mesh1").feature("size1").selection().set(2, 3, 4);
    pymodel.mesh()

    pymodel.save(setting.last_model_save_path)

    return 0


# 用于描述周期区到缺陷区的尺寸变化，采用无需归一化的类高斯函数+常数拟合，sigma平方在setting中给出
def size_function(x, c1, c2):
    return c1 * np.exp(-(x - setting.num_defect // 2) ** 2 / (2 * setting.sigma_square)) + c2


# 拟合尺寸并给出尺寸参数表
def fit():
    x = [0, setting.num_defect // 2, setting.num_defect - 1]
    a_origin = [setting.a_periodic, setting.a_defect, setting.a_periodic]
    b_origin = [setting.b_periodic, setting.b_defect, setting.b_periodic]
    w1_origin = [setting.w1_periodic, setting.w1_defect, setting.w1_periodic]
    w2_origin = [setting.w2_periodic, setting.w2_defect, setting.w2_periodic]
    wide_origin = [setting.wide_periodic, setting.wide_defect, setting.wide_periodic]
    a, b, w1, w2, wide = [], [], [], [], []

    c_a, cov = curve_fit(size_function, x, a_origin)
    c_b, cov = curve_fit(size_function, x, b_origin)
    c_w1, cov = curve_fit(size_function, x, w1_origin)
    c_w2, cov = curve_fit(size_function, x, w2_origin)
    c_wide, cov = curve_fit(size_function, x, wide_origin)

    for i in range(setting.num_defect):
        a += [size_function(i, c_a[0], c_a[1])]
        b += [size_function(i, c_b[0], c_b[1])]
        w1 += [size_function(i, c_w1[0], c_w1[1])]
        w2 += [size_function(i, c_w2[0], c_w2[1])]
        wide += [size_function(i, c_wide[0], c_wide[1])]

    return a, b, w1, w2, wide


# 产生边界曲线并导出
def edge(w1, w2, wide):
    x = 0
    edge = []

    for num in range(setting.num_periodic):
        for i in range(100):
            y = setting.w2_periodic + (setting.w1_periodic - setting.w2_periodic) / 2 * (1 - np.cos(2 * np.pi * i / 100))
            edge.append([x, y])
            x += setting.wide_periodic / 100

    for num in range(setting.num_defect):
        for i in range(100):
            y = w2[num] + (w1[num] - w2[num]) / 2 * (1 - np.cos(2 * np.pi * i / 100))
            edge.append([x, y])
            x += wide[num] / 100

    for num in range(setting.num_periodic):
        for i in range(100):
            y = setting.w2_periodic + (setting.w1_periodic - setting.w2_periodic) / 2 * (1 - np.cos(2 * np.pi * i / 100))
            edge.append([x, y])
            x += setting.wide_periodic / 100

    with open(setting.edge_path + 'edge%d.out' % rank, 'w') as file:
        for e in edge:
            x, y = e[0], e[1]
            file.write(f'{x} {y / 2}\n')

    return 0


def plot_bo():
    global optimizer, gp

    x = np.linspace(-180, 180, 10000)
    mean, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean, label="GaussianProcessRegressor")
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1, label="latent function")
    plt.scatter(optimizer.space.params.flatten(), optimizer.space.target, label="simulation", c="red", s=50, zorder=10)
    plt.legend()
    plt.xlabel('a', fontsize=14)
    plt.ylabel('g', fontsize=14)
    plt.show()


def makabaka(**kwargs):
    global rank, setting, start_time

    step_start_time = datetime.now()
    rank += 1
    print("Iteration:", rank)
    for key, value in kwargs.items():
        setattr(setting, key, value)
        print(f"{key}: {value}")

    a, b, w1, w2, wide = fit()
    edge(w1, w2, wide)
    model_revise(pymodel)
    makabaka = model_solve(pymodel)
    print("本步用时：", datetime.now() - step_start_time)
    print("累计运行用时：", datetime.now() - start_time, '\n')

    return makabaka


def initialize_optimizer():
    _optimizer = BayesianOptimization(
        f=makabaka,
        pbounds={"alpha_ang": (-180, 180)},  # 范围
        verbose=2,  # 显示模式
        random_state=114514,  # 随机种子
    )

    logger = JSONLogger(path=setting.logs_save_path)
    _optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    if os.path.exists(setting.logs_load_path):
        load_logs(_optimizer, logs=[setting.logs_load_path])

    _optimizer.maximize(
        init_points=5,  # 随机步数
        n_iter=10,  # 贝叶斯优化步数
    )
    return _optimizer


def clear():
    # 清空恢复文件
    for dir_name in listdir(setting.recoveries_path):
        if dir_name.startswith('MPHRecovery') and dir_name.endswith('_PM.mph'):
            shutil.rmtree(setting.recoveries_path + dir_name)

    # 清空防爆存档
    os.remove(setting.last_model_save_path)
    return 0


# 设置初始值和打开文件
start_time = datetime.now()
print(start_time.strftime("%Y-%m-%d %H:%M:%S"))
rank = 0
best_makabaka = 0
df = pd.DataFrame(columns=['ewfd_freq', 'ewfd_Qfactor', 'solid_freq', 'solid_Qfactor', 'gom', 'gpe', 'g'])
setting = Setting
client = mph.start()
pymodel = load()
now_time = datetime.now()
print("模型导入用时：", now_time - start_time, "\n")

# 初始化优化器，拟合器，打印
optimizer = initialize_optimizer()
with open(setting.data_save_path, 'a') as file:
    file.write(optimizer.max)
gp = GaussianProcessRegressor(  # 最终拟合器（这个地方代码逻辑不好，不能直接获取运行拟合器，是手动把拟合器参数抄下来写的新对象，待修改）
    kernel=Matern(nu=2.5),
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=5,
)
gp.fit(optimizer._space.params, optimizer._space.target)
plot_bo()

# 清理空间
client.clear()
clear()
