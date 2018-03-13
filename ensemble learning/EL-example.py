from numpy import * 
def loadData():  #  生成训练样本和标签
    datMat = matrix([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels
    
    
    import matplotlib
import matplotlib.pyplot as plt # 相当于画笔
def DrawData(data, labels):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(len(labels)):
        if (labels[i] == 0):
            x0.append(data[i,0]) # 点的 x y坐标
            y0.append(data[i,1])
        else:
            x1.append(data[i,0]) # 点的 x y坐标
            y1.append(data[i,1])
    fig = plt.figure() # 创建画笔
    ax = fig.add_subplot(111)  # 创建画布
    ax.scatter(x0, y0, marker='s', s=90)  # 画散点图， 参数： x,y坐标，以及形状 颜色 大小等
    ax.scatter(x1, y1, marker='o', s=90, c='red')
    plt.title('Original Data') # 标题
    plt.show()  # 显示
    
    #######
    ########
    ######
    def stumpClassify(data, dimen, threshval, threshIneq): 
    # 通过比较阈值对数据进行分类，以阈值为界 分为｛+1 -1｝两类 相当于上一节课的 剪枝分类器。
    # 输入参数 dimen 是哪个特征， threshIneq：有两种模式，在大于和小于之间切换不等式 
    retArray = ones((shape(data)[0], 1))
    if (threshIneq== 1):
        retArray[data[:,dimen] <= threshval] = -1.0
    else:
        retArray[data[:,dimen] > threshval] = -1.0
    return retArray

def buildStump(dataArray, classLabels, D):
    # 输入 样本数组 标签 D为样本初始权重向量
    dataMat = mat(dataArray)
    labelMat = mat(classLabels).T  # 转置
    m, n = shape(dataMat) # 矩阵行列
    bestStump = {} # 用字典来存放最后的单层决策树
    bestclassEst = mat(zeros((m,1)))  # 类别估计值（预测标签）
    minerror = inf # 最小错误率初始值设置为 无限大
    for i in range(n):  #对数据中的每一个特征
        minVal = dataMat[:,i].min() # 得到某一个特征中最小最大值
        maxVal = dataMat[:,i].max() 
        stepsize = (maxVal-minVal)/10   #步长
        for j in range(-1, 11):  # 从-1开始 到 10
            for inequal in [1,2]:  # 在大于和小于之间切换不等式？？？
                threshVal = (minVal+float(j)*stepsize)  #  不同的阈值  ???
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal) # 预测的标签值。
                errArr = mat(ones((m,1))) #  ones((m,n))
                errArr[predictedVals == labelMat] = 0 # 预测正确的为0 预测错误的为1
                weightedError = D.T * errArr  # 得到加权分类误差
                if weightedError < minerror:
                    minerror = weightedError
                    bestclassEst = predictedVals.copy()
                    bestStump['dim'] = i # 字典赋值
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minerror, bestclassEst 
      # 返回 字典、分类器估计错误率、类别估计值(这事实上就是训练出来的弱分类器)
      
    ######链接：http://pan.baidu.com/s/1kVntHzh 密码：ss1x
