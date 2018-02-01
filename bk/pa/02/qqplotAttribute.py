# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：确定"岩石&水雷"数据集的字段类型，该数据集来自UC Irvine数据仓库
# 创建日期：2018/1/30
# -------------------------------------------------------------------------

"""
用分位数图展示异常点

以下代码展示，如何使用probplot函数来帮助确认数据中是否含有异常点。
分布图展示了数据的百分位边界与高斯分布的同样百分位的边界对比。
如果此数服从高斯分布，则画出来的点应该是一条直线。
"""

import numpy as np
import pylab # matplotlib绘图工具包中的一个子包
import scipy.stats as stats
import requests
import sys

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
            start = pos + 1
            pos = r.text.index('\n', start)
        except ValueError as ve:
            break
    nrow = len(xList)  # 列数
    ncol = len(xList[1])  # 行数

    type = [0] * 3
    colCounts = []

    # 生成第4列的统计信息
    col = 3
    colData = []
    for row in xList:
        colData.append(float(row[col]))

    # 显示第4属性的分位数图
    stats.probplot(colData, dist="norm", plot=pylab)
    pylab.show()

