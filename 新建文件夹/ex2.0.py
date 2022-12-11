import pandas as pd #数据处理
import numpy as np #计算
import matplotlib.pyplot as plt #绘图
import scipy.optimize as opt
#读取数据
path = 'D:/Desktop/神经网络/Andrew-NG-Meachine-Learning/machine-learning-ex2/ex2/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()


positive = data[data['Admitted'].isin([1])]   #把通过的 即数据为1的提取出来
negative = data[data['Admitted'].isin([0])]   #把数据为0的提取出来
'''
fig, ax = plt.subplots(figsize=(12,8))       #绘图函数 画布
#参数 s 指定了散点的大小，c 指定了散点的颜色 marker 指定了散点的形状 
#label 指定了这类散点的标签
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score') #贴上x轴的标签
ax.set_ylabel('Exam 2 Score') #贴上y轴的标签
plt.show()
'''
# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 实现代价函数
#不记得J(θ)回去查笔记
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# 加一列常数列  
data.insert(0, 'Ones', 1)
# 初始化X，y，θ   
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)
# 转换X，y的类型 
X = np.array(X.values)
y = np.array(y.values)

#检查矩阵的维度
#X.shape, theta.shape, y.shape
cost(theta, X, y)
# 实现梯度计算的函数（并没有更新θ）即J(θ)的偏导
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad
#利用已有的库函数计算代价函数的最优解 X0指定初始值 即不断变化的θ 
#fprime：提供优化函数func的梯度函数
#即 需要优化的函数 初始数据 梯度方法
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
result
print(result)
#生成等差数列 它接受三个参数：起始值，结束值，元素个数。
# 数组中的元素依次为 30、31、32……100
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (-result[0][0] - result[0][1] * plotting_x1) / result[0][2]
#上面两步是在生成x y 用来拟合曲线

#画图基本操作
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')#画出拟合曲线
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

def hfunc1(theta, X):
    #dot完成矩阵相乘的运算（包括相乘后相加）
    return sigmoid(np.dot(theta.T, X))
   
hfunc1(result[0],[1,45,85])

