# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：确定"岩石&水雷"数据集的规模，该数据集来自UC Irvine数据仓库
# 创建日期：2018/1/30
# 说明： 代码输出所示，此数据集为208行，61列（每行61个字段）
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

    sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
    sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))
