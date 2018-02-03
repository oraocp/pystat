# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示使用岭回归算法分析"岩石&水雷"数据集
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import numpy
import pylab as plt
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

from bk.pa.common import open_url

target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

if __name__ == '__main__':
    xList = []
    labels = []

    with open_url(target_url) as data:
        for line in data:
            row = line.strip().split(",")
            if (row[-1] == 'M'):
                labels.append(1.0)
            else:
                labels.append(0.0)
            row.pop()
            floatRow = [float(num) for num in row]
            xList.append(floatRow)

    indices = range(len(xList))
    xListTest = [xList[i] for i in indices if i % 3 == 0]
    xListTrain = [xList[i] for i in indices if i % 3 != 0]
    labelsTest = [labels[i] for i in indices if i % 3 == 0]
    labelsTrain = [labels[i] for i in indices if i % 3 != 0]

    xTrain = numpy.array(xListTrain)
    yTrain = numpy.array(labelsTrain)
    xTest = numpy.array(xListTest)
    yTest = numpy.array(labelsTest)

    alphaList = [0.1 ** i for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]

    aucList = []
    for alph in alphaList:
        rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
        rocksVMinesRidgeModel.fit(xTrain, yTrain)
        fpr, tpr, thresholds = roc_curve(yTest, rocksVMinesRidgeModel.predict(xTest))
        roc_auc = auc(fpr, tpr)
        aucList.append(roc_auc)

    print("AUC             alpha")
    for i in range(len(aucList)):
        print(aucList[i], alphaList[i])

    # plot auc values versus alpha values
    x = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
    plt.plot(x, aucList)
    plt.xlabel('-log(alpha)')
    plt.ylabel('AUC')
    plt.show()

    # visualize the performance of the best classifier
    indexBest = aucList.index(max(aucList))
    alph = alphaList[indexBest]
    rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
    rocksVMinesRidgeModel.fit(xTrain, yTrain)

    # scatter plot of actual vs predicted
    plt.scatter(rocksVMinesRidgeModel.predict(xTest), yTest, s=100, alpha=0.25)
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.show()
