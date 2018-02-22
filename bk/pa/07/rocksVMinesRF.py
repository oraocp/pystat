# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：用随机森林来分类岩石和水雷数据集
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import matplotlib.pyplot as plot
import numpy
from sklearn import ensemble  # 引入集成方法工具包
from sklearn.model_selection import train_test_split
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

    nrows = len(xNum)  # 行数
    ncols = len(xNum[1])  # 列数

    X = numpy.array(xNum)
    y = numpy.array(labels)
    rocksVMinesNames = numpy.array(['V' + str(i) for i in range(ncols)])  # 列名称V0...Vn-1

    # 训练集和测试集的随机划分， 其中测试集占1/3
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

    auc = []
    nTreeList = range(50, 2000, 50)
    for iTrees in nTreeList:
        depth = None
        maxFeat = 8  # try tweaking

        """
        构建随机森林模型
        参数说明如下：
            -- n_estimators, 整形， 可选（缺省值为10）　
               此参数指定集成方法中决策树的数目。通常缺省值就可以工作得很好。
               如果想获得最佳的性能， 就需要多于10个决策树。
               可以通过做实验尝试确定最佳的决策树数目。
               合适的模型复杂度（决策树的深度和决策树的数目）取决于问题的复杂度和可获得的数据规模。
               比较好的尝试是100-500
            -- max_depth 整形或者none, 可选
               如果这个参数设置为None，决策树就会持续增长，直到叶子节点为空或者所含数据实例小于min_samples_split
               除了指定决策树的深度，可以用参数max_leaf_nodes来指定决策树的叶子节点数。
               如果指定了max_leaf_nodes, max_depth参数就会被忽略。
               不设置max_depth， 让决策树自由生长 ， 形成一个满席的决策树可能可以获得性能上的好处， 
               当然与之相伴的代价就是训练时间。
               在模型训练中要尝试不同深度的决策树。
            -- max_features 整型、浮点型或字符串型，可选（缺省值为None, max_features = nFeatures)
               当查找最佳分割点时，需要考虑多少个属性是max_features参数和问题中一共有多少个属性共同决定的。
               假设问题数据集中共有nFeatures个属性，则：
                  。如果max_features为整形， 则在每次分割中考虑max_features个属性
                  o 如果max_features为浮点型，max_features表示需考虑的属性占全部属性的百分比，即int(max_features*nFeatures)
                  o 可选择的字符串值如下：
                       auto   max_features = nFeatures
                       aqrt   max_features = sqrt(nFeatures)
                       log2   max_features = log2(nFeatures)
               Brieman和Cutle建议对回归问题使用sqrt(nFeatures)个属性。
               模型通常对max_features不是很敏感，但是这个参数还是有一些影响 ，因此可以根据需要尝试一些不同的值  
            -- random_state 整形或RandomState实例或者None(缺省值为None)
               。 如果类型是整形，则此整数作为随机数生成器的种子
               。 如果是RandomState的一个实例，则此实例用来作为随机数生成器
               。 如果是None, 则随机数生成器是numpy.random用的RandomState的一个实例
                       
            -- oob_score 布尔值 样品外样品估算泛化精度               
        """
        rocksVMinesRFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth,
                                                             max_features=maxFeat,
                                                             oob_score=False, random_state=531)

        """
        """
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
