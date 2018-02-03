# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示通过前向逐步回归算法来控制过拟合
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import numpy as np
from bk.pa.common import open_url
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

"""
前向逐步回归算法
"""

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def xattrSelect(x, idxSet):
    """

    :param x:
    :param idxSet:
    :return:
    """
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return xOut


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

    # 构建属性集
    attributeList = []
    index = range(len(xList[1]))
    indexSet = set(index)
    indexSeq = []
    oosError = []

    for i in index:
        attSet = set(attributeList)

        attTrySet = indexSet - attSet

        attTry = [ii for ii in attTrySet]
        errorList = []
        attTemp = []

        for iTry in attTry:
            attTemp = [] + attributeList
            attTemp.append(iTry)
