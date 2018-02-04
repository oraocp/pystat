# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：演示使用"红酒口感"数据集构建二元决策树
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

from bk.pa.common import open_url
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

if __name__ == '__main__':

    # 读取数据集并进行解析， 包括标签和数据
    xList = []  # 数据集
    labels = [] # 标签集
    names = []
    firstLine = True
    with open_url(target_url, cache_dir="../data") as data:
        for line in data:
            if firstLine:
                names = line.strip().split(";")
                firstLine = False
            else:
                row = line.strip().split(";")
                labels.append(float(row[-1]))
                # 移除标签列
                row.pop()
                # 转换列值为浮点数
                floatRow = [float(num) for num in row]
                xList.append(row)

    nrows = len(xList)
    ncols = len(xList[0])

    # 使用二元决策树模型
    wineTree = DecisionTreeRegressor(max_depth=3)
    # 传入数据集及字段名称
    wineTree.fit(xList, labels)

    # 输出决策树的决策图
    with open("wineTree.dot",'w') as f:
        f = tree.export_graphviz(wineTree, out_file = f)

    # Note: The code above exports the trained tree info to a Graphviz "dot" file.
    # Drawing the graph requires installing GraphViz and the running the following on the command line
    # dot -Tpng wineTree.dot -o wineTree.png