# ICLHIF

一种基于对比学习的增量轴承故障诊断技术

文件说明：
oneD：存放西储大学基准轴承数据
models：存放对比学习对应的step的权重
picture：存放每次特征提取后，特征可视化图
Features：存放CL降维后的特征，以及后面的异常检测、新类检测操作
data_util：对西储大学数据进行处理
model_CL：特征提取器网络设计
train：训练网络
experience：验证网络
envs：环境配置和所需要的库包
