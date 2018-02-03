# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的：通用函数
# 创建日期：2018/2/3
# -------------------------------------------------------------------------

import os
import requests


def open_url(surl, params=None):
    pos = surl.rindex("/")
    filename = surl[pos + 1:]
    if not os.path.exists(filename):
        with open(filename, mode="wb") as w:
            # 超时60秒， 加载文件到内存中
            r = requests.get(surl, params=params, timeout=60)
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    w.write(chunk)
    # 返回的读写模式为文本
    return open(filename, mode="rt")


if __name__ == '__main__':
    target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    with open_url(target_url) as data:
        for line in data:
            print(line)
