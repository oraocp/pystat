# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：确定"岩石&水雷"数据集的字段类型，该数据集来自UC Irvine数据仓库
# 创建日期：2018/1/30
# -------------------------------------------------------------------------

import sys

import requests

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
            start = pos + 1
            pos = r.text.index('\n', start)
        except ValueError as ve:
            break
    nrow = len(xList)  # 列数
    ncol = len(xList[1])  # 行数

    type = [0] * 3
    colCounts = []

    for col in range(ncol):
        for row in xList:
            # 检查字段是否为数值类型 -1
            try:
                a = float(row[col])
                if isinstance(a, float):
                    type[0] += 1
            except ValueError:
                # 检查是否为字符串类型 -2
                if len(row[col]) > 0:
                    type[1] += 1
                else:
                    # 空值 -3
                    type[2] += 1

        colCounts.append(type)
        type = [0] * 3

    # 统计每一行中数值类型、字符串类型、空值类型的字段数目
    sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
                     "Strings" + '\t ' + "Other\n")
    iCol = 0
    for types in colCounts:
        sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                         str(types[1]) + '\t\t' + str(types[2]) + "\n")
        iCol += 1
