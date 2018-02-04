# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：用随机森林来分类岩石和水雷数据集
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import matplotlib.pyplot as plot
import numpy
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from bk.pa.common import open_url

target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

"""
使用RandomForestClassifer算法

"""

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    xList = []

    # 读入数据集
    with open_url(target_url, cache_dir="../data") as data:
        for line in data:
            row = line.strip().split(",")
            xList.append(row)

    xNum = []
    labels = []

    for row in xList:
        lastCol = row.pop()
        # 分类问题， 标签从M和R转换成了0和1
        if lastCol == 'M':
            labels.append(1)
        else:
            labels.append(0)
        attrRow = [float(elt) for elt in row]
        xNum.append(attrRow)

    nrows = len(xNum) # 行数
    ncols = len(xNum[1]) #列数

    X = numpy.array(xNum)
    y = numpy.array(labels)
    rocksVMinesNames = numpy.array(['V' + str(i) for i in range(ncols)]) # 列名称V0...Vn-1

    # 训练集和测试集的划分， 其中测试集占1/3
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

    auc = []
    nTreeList = range(50, 2000, 50)
    for iTrees in nTreeList:
        depth = None
        maxFeat = 8  # try tweaking
        rocksVMinesRFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth,
                                                                 max_features=maxFeat,
                                                                 oob_score=False, random_state=531)

        rocksVMinesRFModel.fit(xTrain, yTrain)

        # Accumulate auc on test set
        prediction = rocksVMinesRFModel.predict_proba(xTest)
        aucCalc = roc_auc_score(yTest, prediction[:, 1:2])
        auc.append(aucCalc)

    print("AUC")
    print(auc[-1])

    plot.plot(nTreeList, auc)
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Area Under ROC Curve - AUC')
    plot.show()

    featureImportance = rocksVMinesRFModel.feature_importances_
    featureImportance = featureImportance / featureImportance.max()

    idxSorted = numpy.argsort(featureImportance)[30:60]
    idxTemp = numpy.argsort(featureImportance)[::-1]
    print(idxTemp)
    barPos = numpy.arange(idxSorted.shape[0]) + .5
    plot.barh(barPos, featureImportance[idxSorted], align='center')
    plot.yticks(barPos, rocksVMinesNames[idxSorted])
    plot.xlabel('Variable Importance')
    plot.show()

    fpr, tpr, thresh = roc_curve(yTest, list(prediction[:, 1:2]))
    ctClass = [i * 0.01 for i in range(101)]

    plot.plot(fpr, tpr, linewidth=2)
    plot.plot(ctClass, ctClass, linestyle=':')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.show()

    idx25 = int(len(thresh) * 0.25)
    idx50 = int(len(thresh) * 0.50)
    idx75 = int(len(thresh) * 0.75)

    totalPts = len(yTest)
    P = sum(yTest)
    N = totalPts - P

    print('')
    print('Confusion Matrices for Different Threshold Values')

    # 25th
    TP = tpr[idx25] * P;
    FN = P - TP;
    FP = fpr[idx25] * N;
    TN = N - FP
    print('')
    print('Threshold Value =   ', thresh[idx25])
    print('TP = ', TP / totalPts, 'FP = ', FP / totalPts)
    print('FN = ', FN / totalPts, 'TN = ', TN / totalPts)

    # 50th
    TP = tpr[idx50] * P;
    FN = P - TP;
    FP = fpr[idx50] * N;
    TN = N - FP
    print('')
    print('Threshold Value =   ', thresh[idx50])
    print('TP = ', TP / totalPts, 'FP = ', FP / totalPts)
    print('FN = ', FN / totalPts, 'TN = ', TN / totalPts)

    # 75th
    TP = tpr[idx75] * P;
    FN = P - TP;
    FP = fpr[idx75] * N;
    TN = N - FP
    print('')
    print('Threshold Value =   ', thresh[idx75])
    print('TP = ', TP / totalPts, 'FP = ', FP / totalPts)
    print('FN = ', FN / totalPts, 'TN = ', TN / totalPts)
