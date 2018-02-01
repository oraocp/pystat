# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：确定"岩石&水雷"数据集的字段类型，该数据集来自UC Irvine数据仓库
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

import pandas as pd

# 从uci数据仓库中读取数据
# 数据集文件由逗号分割，一次实验数据占据一行
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

if __name__ == '__main__':
    # 读取网络数据到Pandas data frame
    rocksVMines = pd.read_csv(target_url, header=None, prefix="V")

    # 打印数据
    print(rocksVMines.head())  # 自动生成标题V0, V1...
    print(rocksVMines.tail())

    # 打印数据的统计信息 , Pandas会计算数据的均值、方差、分位数
    summary = rocksVMines.describe()
    print('')
    print('---------------------------------------')
    print('summary info:')
    print(summary)
