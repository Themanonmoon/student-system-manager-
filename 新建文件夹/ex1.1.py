#多变量线性回归的拟合
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path =  'D:/Desktop/python-test/machine learing/ex1data2.txt'
#1、 数据读取 加表头
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

#*2、 特征归一化 (根据实际考虑) (x-平均数)/标准差
data2 = (data2 - data2.mean()) / data2.std()#mean求平均值 std求标准差  data2统一量级
data2.head()

data2.insert(0, 'Ones', 1)

# 初始化X和y 把数据提取出来变成向量
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]#第一个冒号 从 n行 到 m 行，第二个冒号 从 n 列到 m 列
y2 = data2.iloc[:,cols-1:cols]

# 转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0])) #对应θ0 θ1 θ2 两个变量的权重参数+一个额外参数

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X)) #计算J(θ)

# 运行梯度下降算法   
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))  #
    parameters = int(theta.ravel().shape[1]) #把θ向量展成一行 返回列数
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y  #矩阵运算
        
        for j in range(parameters):#parameters是参数θ个数 这里即为要求的偏导次数
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))#这一步完成了对θ的更新
            #temp[0][j]=theta[0][j]-......
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

alpha=0.01
iters=1500
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)