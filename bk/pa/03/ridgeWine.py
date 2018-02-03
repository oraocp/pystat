# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示通过岭回归算法来控制过拟合
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from bk.pa.common import open_url

"""

"""

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# ----------------------------------------------------------------------------
# ~ Main
if __name__ == '__main__':
    # 存放标签和数据到列表中
    xList = []
    labels = []  # 判别标志
    names = []  # 字段标题

    # 首行存储字段标题
    firstLine = True

    # 读入Winequality数据集，并且将其解析为包含标签与属性的记录
    with open_url(target_url) as data:
        for line in data:
            if firstLine:
                names = line.strip().split(";")
                firstLine = False
            else:
                row = line.strip().split(";")
                labels.append(float(row[-1]))
                row.pop()
                floatRow = [float(num) for num in row]
                xList.append(floatRow)

    # 将数据集分为2个子集：
    indices = range(len(xList))  # 总行数
    # 测试集xListTest包含1/3的数据
    xListTest = [xList[i] for i in indices if i % 3 == 0]
    # 训练集xListTrain包含2/3的数据
    xListTrain = [xList[i] for i in indices if i % 3 != 0]
    labelsTest = [labels[i] for i in indices if i % 3 == 0]
    labelsTrain = [labels[i] for i in indices if i % 3 != 0]

    # 检查下各数据形态
    xTrain = np.array(xListTrain);
    yTrain = np.array(labelsTrain)
    xTest = np.array(xListTest);
    yTest = np.array(labelsTest)

    print("Shape of xTrain array", xTrain.shape)
    print("Shape of yTrain array", yTrain.shape)
    print("Shape of xTest array", xTest.shape)
    print("Shape of yTest array", yTest.shape)

    # 惩罚系数
    alphaList = [0.1 ** i for i in [0, 1, 2, 3, 4, 5, 6]]

    # 输出错误率
    rmsError = []
    for alph in alphaList:
        # 使用scikit-learn的岭回归算法
        wineRidgeModel = linear_model.Ridge(alpha=alph)
        wineRidgeModel.fit(xTrain, yTrain)
        # 错误率 =
        rmsError.append(np.linalg.norm((yTest - wineRidgeModel.predict(xTest)), 2) / sqrt(len(yTest)))
        sqrt(len(yTest))

    print("RMS Error             alpha")
    for i in range(len(rmsError)):
        print(rmsError[i], alphaList[i])

    # 输出错误率与alph取值间的联系
    x = range(len(rmsError))
    plt.plot(x, rmsError, 'k')
    plt.xlabel('-log(alpha)')
    plt.ylabel('Error (RMS)')
    plt.show()

    # 输出使用岭回归的实际口感得分与预测得分的散点图
    indexBest = rmsError.index(min(rmsError))
    alph = alphaList[indexBest]
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain, yTrain)
    errorVector = yTest - wineRidgeModel.predict(xTest)
    plt.hist(errorVector)
    plt.xlabel("Bin Boundaries")
    plt.ylabel("Counts")
    plt.show()

    # 输出预测错误（错误率）的直方图
    plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
    plt.xlabel('Predicted Taste Score')
    plt.ylabel('Actual Taste Score')
    plt.show()
