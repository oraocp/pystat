# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的： 演示iris数据集
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

"""
加载并查看iris数据集
"""

from sklearn import datasets

if __name__ == '__main__':
    # 查看IRIS数据集
    iris = datasets.load_iris()

    # data对应了样本的4个特征，150行4列,
    # data不包含结果, 结果在Target
    print(iris.data.shape)  # (150,4)

    # 显示样本特征的前5行
    print("iris.data[:5]:")
    print(iris.data[:5])

    # target对应了样本的类别（目标属性），150行1列
    print(iris.target.shape)  # (150,)

    # 显示所有样本的目标属性
    print("iris target:")
    print(iris.target)
