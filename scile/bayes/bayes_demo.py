# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的： 演示scikit-learn中朴素贝叶斯算法
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

"""
参见scikit-learn中文文档：
http://cwiki.apachecn.org/pages/viewpage.action?pageId=10814109
https://www.cnblogs.com/pinard/p/6074222.html
http://blog.csdn.net/lsldd/article/details/41542107
"""

if __name__ == '__main__':
    iris = datasets.load_iris()
    # 使用高斯朴素贝叶斯分类法
    print("高斯朴素贝叶斯分类法,适用于正态分布的数据集 ")
    gnb = GaussianNB()

    # 训练集和测试集的随机划分， 其中测试集占1/3
    xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size=0.30, random_state=31)

    y_pred = gnb.fit(xTrain, yTrain).predict(xTest)
    # Number of mislabeled points out of a total 150 points : 6
    print("Number of mislabeled points out of a total %d points : %d"
          % (yTest.shape[0], (yTest != y_pred).sum()))
    # 准确度评估 评估正确/总数
    print("accuracy:" , gnb.score(xTest, yTest))
    print("accuracy_score:" , accuracy_score(y_pred, yTest))

    # 使用多项式贝叶斯分类法
    print("\n多项式贝叶斯分类法: ")
    mnb = MultinomialNB()

    y_pred = mnb.fit(xTrain, yTrain).predict(xTest)
    # Number of mislabeled points out of a total 150 points : 6
    print("Number of mislabeled points out of a total %d points : %d"
          % (yTest.shape[0], (yTest != y_pred).sum()))
    # 准确度评估 评估正确/总数
    print("accuracy:", mnb.score(xTest, yTest))
    print("accuracy_score:", accuracy_score(y_pred, yTest))

    # 使用伯努利贝叶斯分类法
    print("\n伯努利贝叶斯分类法，适用于伯努利分布（二值分布）的特征 ")
    bnb = BernoulliNB()

    y_pred = bnb.fit(xTrain, yTrain).predict(xTest)
    # Number of mislabeled points out of a total 150 points : 6
    print("Number of mislabeled points out of a total %d points : %d"
          % (yTest.shape[0], (yTest != y_pred).sum()))
    # 准确度评估 评估正确/总数
    print("accuracy:", bnb.score(xTest, yTest))
    print("accuracy_score:", accuracy_score(y_pred, yTest))