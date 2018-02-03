# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：检验回归算法的性能
# 创建日期：2018/1/30
# -------------------------------------------------------------------------

"""
在回归问题中，真实目标以及预测值都是实数。
错误很自然被定义为目标值与预测值的差异。
生成错误的统计摘要对性能比较以及问题诊断都非常有用。
最常用的错误摘要是均方误差MSE以及平均绝对错误MAE, 根MSE=RMSE

"""

from math import sqrt

# ----------------------------------------------------------------------------
if __name__ == '__main__':

    # 测试数据
    target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
    prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

    error = []
    # 计算误差
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    # 打印误差
    print("Errors :")
    print(error)

    # 计算误差的平方和误差的绝对值
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    # 打印误差的平方和误差的绝对值
    print("Squared Error")
    print(squaredError)
    print("Absolute Value of Error")
    print(absError)

    # 计算均方误差MSE
    print("")
    print("MSE=", sum(squaredError) / len(squaredError)) # 2.72875

    # 计算根均方误差RMSE
    print("")
    print("RMSE=", sqrt(sum(squaredError) / len(squaredError))) # 1.651892853668179

    # 计算平均绝对错误MAE
    print("")
    print("MAE=", sum(absError) / len(absError)) # 1.4916666666666665

    targetDeviation = []
    targetMean = sum(target) / len(target)
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))

    # 计算目标变量 - 平均值，
    print("")
    print("Target Variance = ", sum(targetDeviation) / len(targetDeviation)) # 7.570347222222222

    print("")
    print("Target Standard Deviation=", sqrt(sum(targetDeviation) / len(targetDeviation))) # 2.7514263977475797

    # RMSE 1.65 大约是目标平均差 2.75 的一半， 这已经是相当不错的性能

    # 评估性能
    print("")
    print("Target Standard Deviation - MSE = ",
          sqrt(sum(targetDeviation) / len(targetDeviation)) - sum(squaredError) / len(squaredError)) # 0.0226763977475799
