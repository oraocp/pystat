# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示最小二乘法
# 创建日期：2018/1/30
# -------------------------------------------------------------------------

"""
最小二乘法（又称最小平方法）是一种数学优化技术，它通过最小化误差的平方和寻找数据的最佳函数匹配。
作用：
利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。
原理：
以”残差平方和最小”确定直线位置(在数理统计中，残差是指实际观察值与估计值之间的差
http://blog.csdn.net/yauphy1/article/details/43735763

基本思路：
对于一元线性回归模型, 假设从总体中获取了n组观察值（X1，Y1），（X2，Y2）， …，（Xn，Yn），
对于平面中的这n个点，可以使用无数条曲线来拟合。
而线性回归就是要求样本回归函数尽可能好地拟合这组值，也就是说，
这条直线应该尽可能的处于样本数据的中心位置。
因此，选择最佳拟合曲线的标准可以确定为：使总的拟合误差（即总残差）达到最小。

S(B1,B2)=[6-(B1+1*B2)^2]+[5-(B1+2*B2)]^2+[7-(B1+3*B2)]^2+[10-(B1+4*B2)]^2
最小值 可以通过对S(B1,B2)分别求B1和B2的偏导数，
偏（B1） = 0  偏（B2） = 0

高斯和勒让德的方法是，假设测量误差的平均值为0。令每一个测量误差对应一个变量并与其它测量误差不相关（随机无关）。人们假设，在测量误差中绝对不含系统误差，它们应该是纯偶然误差(有固定的变异数)，围绕真值波动。
除此之外，测量误差符合正态分布，这保证了偏差值在最后的结果y上忽略不计。

min(sum((ym-yi)^2)) 使函数曲线与观测值之差的平方和最小
典型的一类函数是线性函数模型，最简单的线性式是y = b0+b1*t
公式为：
        b1 = sum((ti-Pt)(yi-Py)) / sum((ti-Pt)^2)

"""

import numpy as np
import matplotlib.pyplot as plt  #
from scipy.optimize import leastsq

# ----------------------------------------------------------------------------
# ~ 最小二乘法实现

# 样本数据(Xi,Yi)，需要转换成数组(列表)形式
Xi = np.array([1, 2, 3, 4])
Yi = np.array([6, 5, 7, 10])


def simple_leastsq(x, y):
    """
    最小二乘法算法的简单实现，依赖原理为残差平方和最小。

    :param x: 要拟合数据的自变量列表 
    :param y: 要拟合数据的因变量列表
    :return: 拟合的两个参数值
    """

    meanx = sum(x) / len(x)  # 求x的平均值
    meany = sum(y) / len(y)  # 求y的平均值

    xsum = 0.0
    ysum = 0.0

    for i in range(len(x)):
        xsum += (x[i] - meanx) * (y[i] - meany)
        ysum += (x[i] - meanx) ** 2

    k = xsum / ysum
    b = meany - k * meanx

    return k, b


# ----------------------------------------------------------------------------
# ~ scipy库提供的最小二乘法
# ~ 实现一元一次函数

# 采样点数据
X1 = np.array([6.19, 2.51, 7.29, 7.01, 5.7, 2.66, 3.98, 2.5, 9.1, 4.2])
Y1 = np.array([5.25, 2.83, 6.41, 6.71, 5.1, 4.23, 5.05, 1.98, 10.5, 6.3])


def func(p, x):
    """
    需要拟合的函数， 本例为直线
    :param p:  斜率值
    :param x:  X坐标
    :return:
    """
    k, b = p
    return k * x + b


def error(p, x, y, s):
    """
    拟合的误差函数
    :param p:
    :param x:
    :param y:
    :param s:
    :return:
    """
    print(s)
    return func(p, x) - y


def sci_leastsq(x, y):
    """
    使用SCI提供的最小二乘法
    :param x:
    :param y:
    :return:
    """
    p0 = [100, 2]
    # 提示，试验最小二乘法函数leastsq得调用几次error函数才能找到使得均方误差之和最小的k、b
    s = "调用一次ERROR函数"
    para = leastsq(error, p0, args=(x, y, s))
    (k, b) = para[0]
    return k, b


# ----------------------------------------------------------------------------
# ~ scipy库提供的最小二乘法
# ~ 实现二元一次函数

X2 = np.array([0, 1, 2, 3, -1, -2, -3])
Y2 = np.array([-1.21, 1.9, 3.2, 10.3, 2.2, 3.71, 8.7])


def func2(p, x):
    """
    定义二元一次方程需要拟合的函数
    :param p:
    :param x:
    :return:
    """
    a, b, c = p
    return a * x ** 2 + b * x + c


def error2(p, x, y, s):
    """
     定义二元一次方程误差函数
    :param p:
    :param x:
    :param y:
    :param s:
    :return:
    """
    print(s)
    return func2(p, x) - y


def sci_leastsq2(x, y):
    p0 = [5, 2, 10]
    # 提示，试验最小二乘法函数leastsq得调用几次error函数才能找到使得均方误差之和最小的k、b
    s = "调用一次ERROR2函数"
    para = leastsq(error2, p0, args=(x, y, s))
    a, b, c = para[0]
    return a, b, c


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print(simple_leastsq(Xi, Yi))
    print(sci_leastsq(Xi, Yi))

    # 一元一次方程
    (k, b) = sci_leastsq(X1, Y1)
    # 绘图查看拟合效果

    plt.figure(figsize=(8, 6))
    # 画样本点
    plt.scatter(X1, Y1, color="red", label="Sample Point", linewidths=3)
    x = np.linspace(0, 10, 1000)
    y = k * x + b
    # 画拟合直线
    plt.plot(x, y, color="orange", label="Fitting Line", linewidth=2)
    # 显示图形
    plt.legend()
    plt.show()

    # 二元一次方程
    plt.figure(figsize=(8, 6))
    # 画样本点
    plt.scatter(X2, Y2, color="red", label="Sample Point",linewidth=3)
    y = np.linspace(-5, 5, 1000)
