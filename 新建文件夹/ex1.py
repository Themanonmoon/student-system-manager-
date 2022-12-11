import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#A = np.eye(5)
#print(A)

path = 'D:/Desktop/python-test/machine learing/ex1data1.txt'
data = pd.read_csv(path, header=None,names=["Population", "Profit"])
data.head()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)
cols = data.shape[1]  #shape返回矩阵的维度 0是行 1是列
#print(data)
X = data.iloc[:,:-1]#X是data里的除最后列
y = data.iloc[:,cols-1:cols]#y是data最后一列
X.head()
y.head()
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))   #这一步后续需要改成随机数 theta 高度对称会使提取的特征减少

print(computeCost(X,y,theta))


'''
上面这一部分完成了对J(θ)的初始化
'''


'''
下面进行Gradient descent
'''

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) #把θ向量展成一行 返回列数
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y  #矩阵运算
        
        for j in range(parameters):#parameters是参数θ个数 这里即为要求的偏导次数
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))#这一步完成了对θ的更新
            #temp[0][j]=theta[0][j]-......
        #theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

'''
在这段代码中，函数 gradientDescent() 接收五个参数：X、y、theta、alpha 和 iters。
X 和 y 分别是训练数据集的特征和目标变量。
theta 是模型的参数，alpha 是学习率，iters 是迭代次数。
返回两个值：theta 和 cost。
theta 是模型的训练后的参数，cost 是每次迭代后的代价。
该函数首先使用一个临时变量 temp 来存储更新后的参数。然后，它使用一个循环来进行迭代。
在每次迭代中，它会计算模型的预测值和实际值之间的误差，然后对每个参数进行更新。
最后，它会计算更新后的代价。
 '''

alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
predict1 = [1,3.5]*g.T      #1是初始化参数X0
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)
#预测35000和70000城市规模的小吃摊利润