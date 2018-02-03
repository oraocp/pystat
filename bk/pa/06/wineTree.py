# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示使用"红酒口感"数据集构建二元决策树
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

import requests
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

if __name__ == '__main__':

    # 超时60秒， 加载文件到内存中
    r = requests.get(target_url, timeout=60)

    # 存放数据到列表中
    xList = []
    labels = []
    names = []
    firstLine = True

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

    names = xList[0]
    xList.pop()

    wineTree = DecisionTreeRegressor(max_depth=3)
    wineTree.fit(xList, names)

    with open("wineTree.dot",'w') as f:
        f = tree.export_graphviz(wineTree, out_file = f)