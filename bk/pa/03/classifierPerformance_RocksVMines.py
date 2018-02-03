# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：检测分类算法的性能
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import numpy as np
import requests
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl


# 从uci数据仓库中读取数据
# 数据集文件由逗号分割，一次实验数据占据一行
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")


def confusionMatrix(predicted, actual, threshold):
    """
    计算数据的混淆矩阵表数值

    :param predicted: 实验数据集
    :param actual: 真实数据集
    :param threshold: 阀值
    :return: 数值形式[tp, fn, tp, tn, rate]
    """
    # 检查数据长度，避免无效输入
    if len(predicted) != len(actual):
        return -1

    # 定义混淆矩阵的四项指标
    tp = 0.0  # 真实例
    fp = 0.0  # 假实例
    tn = 0.0  # 真负例
    fn = 0.0  # 假负率

    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    # 误分率
    rate =(fn + fp) / (tp + fn + tn + fp)
    rtn = [tp, fn, fp, tn, rate]
    return rtn

def print_confusionMatrix(title, cm):
    print(title)
    print("tp = " + str(cm[0]) + "\tfn = " + str(cm[1]) + "\n" + "fp = " + str(cm[2]) + "\ttn = " + str(cm[3]) + '\n')
    print("wrong rate:", cm[4])
    print("")

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # 超时60秒， 加载文件到内存中
    r = requests.get(target_url, timeout=60)

    # 存放标签和数据到列表中
    xList = []
    labels = []

    # 读入Irvine数据集，并且将其解析为包含标签与属性的记录
    pos = r.text.index('\n')
    start = 0
    while pos != -1:
        try:
            line = r.text[start:pos]
            print(line)

            # 读入一行，对数据按逗号进行分割，将结果列表存入输出列表
            row = line.strip().split(",")

            # 设置行标签, 1.0表示 "M", 0.0表示"R"
            if (row[-1] == 'M'):
                labels.append(1.0)
            else:
                labels.append(0.0)

            # 移除标签行
            row.pop()

            floatRow = [float(num) for num in row]
            xList.append(floatRow)
            start = pos + 1
            pos = r.text.index('\n', start)
        except ValueError as ve:
            break
    nrow = len(xList)  # 列数
    ncol = len(xList[1])  # 行数

    # 将数据集分为2个子集：
    indices = range(len(xList))  # 总行数
    # 测试集xListTest包含1/3的数据
    xListTest = [xList[i] for i in indices if i % 3 == 0]
    # 训练集xListTrain包含2/3的数据
    xListTrain = [xList[i] for i in indices if i % 3 != 0]
    labelsTest = [labels[i] for i in indices if i % 3 == 0]
    labelsTrain = [labels[i] for i in indices if i % 3 != 0]

    # 注：测试集不能用于训练分类，但会被保留用于评估训练得到的分类器性能。
    # 这一步模拟分类器在新数据样本上的行为

    xTrain = np.array(xListTrain);
    yTrain = np.array(labelsTrain)
    xTest = np.array(xListTest);
    yTest = np.array(labelsTest)

    # 检查下各数据形态
    print("Shape of xTrain array", xTrain.shape)
    print("Shape of yTrain array", yTrain.shape)
    print("Shape of xTest array", xTest.shape)
    print("Shape of yTest array", yTest.shape)

    # 分类器训练使用线性回归模型
    # 训练通过将标签M以及标签转换为2个数值：1.0对应于水雷，0.0对应于岩石
    # 然后使用最小二乘法来拟合一个线性模型
    # 本例使用scikit-learn中的线性回归包来训练普通的最小均方模型
    # 训练的模型用于在训练集和测试上生成预测
    rocksVMinesModel = linear_model.LinearRegression()
    rocksVMinesModel.fit(xTrain, yTrain)

    # 生成训练集预测数据
    trainingPredictions = rocksVMinesModel.predict(xTrain)
    # 打印一些预测值的样例， 大部分集中在0.0到1.0， 但也不是全部
    print("\nSome values predicted by model")
    print(trainingPredictions[0:5])
    # [-0.10240253  0.42090698  0.38593034  0.36094537  0.31520494]
    print(trainingPredictions[-6:-1])
    # [ 1.11094176  1.12242751  0.77626699  1.02016858  0.66338081]
    print("")

    # 以上生成的预测不只是概率，仍然可以将它们与决策阈值进行比较生成分类标签。

    # 生成训练集的混淆矩阵
    confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)
    tp = confusionMatTrain[0]
    fn = confusionMatTrain[1]
    fp = confusionMatTrain[2]
    tn = confusionMatTrain[3]

    print_confusionMatrix("MatTrain:", confusionMatTrain)
    # rate=9.42%

    # 用训练集的模型预测测试集数据
    testPredictions = rocksVMinesModel.predict(xTest)

    # 生成测试集的训练数据
    conMatTest = confusionMatrix(testPredictions, yTest, 0.5)
    tp = conMatTest[0]
    fn = conMatTest[1]
    fp = conMatTest[2]
    tn = conMatTest[3]

    print_confusionMatrix("MatTest:", conMatTest)
    # rate = 25.71%

    # 在训练集上识分率9.42%， 在测试集上误分率25.71%
    # 一般来说， 测试集上的性能要差于训练集上的性能， 在测试集上更能代表错误率

    # 生成训练集的ROC曲线

    fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
    roc_auc = auc(fpr, tpr)
    print('AUC for Training ROC curve: %f' % roc_auc)
    # 0.98

    # 绘制ROC曲线
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('In sample ROC rocks versus mines')
    pl.legend(loc="lower right")
    pl.show()

    # 生成测试集的ROC曲线
    fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
    roc_auc = auc(fpr, tpr)
    print('AUC for Test ROC curve: %f' % roc_auc)

    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Out-of-sample ROC rocks versus mines')
    pl.legend(loc="lower right")
    pl.show()


