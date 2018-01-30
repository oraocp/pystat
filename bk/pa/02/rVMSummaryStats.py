# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：确定"岩石&水雷"数据集的规模，该数据集来自UC Irvine数据仓库
# 创建日期：2018/1/30
# -------------------------------------------------------------------------

import requests
import sys
import numpy as np

# 从uci数据仓库中读取数据
# 数据集文件由逗号分割，一次实验数据占据一行
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

if __name__ == '__main__':
    # 超时60秒， 加载文件到内存中
    r = requests.get(target_url, timeout=60)

    # 存放数据到列表中
    xList = []
    labels = []

    pos = r.text.index('\n')
    start = 0
    while pos != -1:
        try:
            line = r.text[start:pos]
            print(line)

            # 读入一行，对数据按逗号进行分割，将结果列表存入输出列表
            row = line.strip().split(",")
            xList.append(row)
            start = pos+1
            pos = r.text.index('\n', start)
        except ValueError as ve:
            break

    nrow = len(xList) #列数
    ncol = len(xList[1]) #行数

    type = [0]*3
    colCount = []

    # 生成第3行数据的统计信息（e.g) - 数值
    col = 3
    colData = []
    for row in xList:
        colData.append(float(row[col]))

    colArray = np.array(colData)
    colMean = np.mean(colArray)
    colsd = np.std(colArray)
    sys.stdout.write("\nCalculate row 3 statics:  \n")
    sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                     "Standard Deviation = " + '\t ' + str(colsd) + "\n")

    # 计算四分位数(quantile)的边界
    ntiles = 4
    percentBdry = []
    for i in range(ntiles+1):
        percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
    print(percentBdry)
    sys.stdout.write(" \n")

    # 计算十分位数(quantile)的边界
    ntiles = 10
    percentBdry = []
    for i in range(ntiles + 1):
        percentBdry.append(np.percentile(colArray, i * (100) / ntiles))
    sys.stdout.write("\nBoundaries for 10 Equal Percentiles \n")
    print(percentBdry)
    sys.stdout.write(" \n")

    # 生成第60行数据的统计信息（e.g) - 标签， R或M
    col = 60
    colData = []
    for row in xList:
        colData.append(row[col])
    print(colData)

    # 取得标签唯一值， R或M
    unique = set(colData)
    sys.stdout.write("Unique Label Values \n")
    print(unique)

    # 计算每个值的分类标签？
    catDict = dict(zip(list(unique), range(len(unique))))

    catCount = [0] * 2

    for elt in colData:
        catCount[catDict[elt]] += 1

    sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
    print(list(unique))
    print(catCount)



