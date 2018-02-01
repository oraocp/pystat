# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plot # matplotlib绘图工具包中的一个子包

# 从uci数据仓库中读取数据
# 数据集文件由逗号分割，一次实验数据占据一行
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

if __name__ == '__main__':
    # 读取网络数据到Pandas data frame
    rocksVMines = pd.read_csv(target_url, header=None, prefix="V")

    for i in range(208):

        if rocksVMines.iat[i, 60] == 'M':
            pcolor = 'red'
        else:
            pcolor = 'blue'

        dataRow = rocksVMines.iloc[i, 0:60]
        dataRow.plot(color=pcolor, alpha = 0.5)

    plot.xlabel("Attribute Index")
    plot.ylabel("Attribute Values")
    plot.show()