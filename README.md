### 简介
本项目为基于机器学习（高斯过程）的一维光声晶体腔结构自动优化。其中一维光声晶体腔的结构整体上参考文献Jiang W, Patel R N, Mayor F M, et al. Lithium niobate piezo-optomechanical crystals[J]. Optica, 2019, 6(7): 845-853.

目前为自用，很多接口没有调好，甚至path都没有改，泛用性很差，会慢慢改，有问题issue。

### 环境依赖
项目使用的python版本为3.11

该项目目前使用的环境中比较可能影响兼容性的：
- mph  1.2.3
- comsol  6.0
- bayesian-optimization  1.4.3

次要的：
- numpy  1.24.2
- matplotlib  3.7.1  

不保证其他环境下的兼容性。

### 目前存在的主要问题
对节点的修改不是自适应的，而是基于我的具体模型，可移植性极差。模型修改的全部内容都放在了model_revise里面，换模型应该要对应改。我的模型已经上传。

### 更新目录已上传！
