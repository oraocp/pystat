# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# 文件目的： 演示如何通过决策树对安德森鸢尾花卉数据集进行分类
# 创建日期：2018/2/1
# -------------------------------------------------------------------------

import pydotplus
from sklearn import tree
from sklearn.datasets import load_iris

"""
参见scikit-learn中文文档：
http://datahref.com/archives/169
"""

if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    # 构建决策树分类模型
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)

    # 图形化显示决策树结果
    try:
        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        #graph.set_graphviz_executables
        graph.write_pdf("iris_data.pdf")
    except pydotplus.graphviz.InvocationException as e:
        print(e)
        print("Graphviz工具未安装或者未将它的bin目录加入到环境变量PATH中!.")
        raise  e
