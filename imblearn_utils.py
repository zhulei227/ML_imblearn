import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
import pandas as pd

"""
粗略查看数据的分布，包括：
1.样本绝对数量
2.重叠样本量
3.异常样本量
4.类内不平衡度量
"""


class ImbLearnUtils(object):
    def __init__(self, k=3):
        # 计算重叠区域和异常点所需的k
        self.k = k
        # 每个label包含的样本数
        self.label_map_num = {}
        # 正样本标签
        self.pos_label = None
        # 负样本标签
        self.neg_label = None
        # 类间不平衡率
        self.ir = None
        # 所有标签集合
        self.labels = None
        # 各样本最近k个样本的类别分布
        self.k_neighbors_dist = []
        # 记录该点是否在重叠区域
        self.overlap_marks = []
        # 记录该点是否是异常点
        self.abnormal_marks = []
        # 正样本密度聚类类别数
        self.pos_db_scan_classes = None
        # 负样本密度聚类类别数
        self.neg_db_scan_classes = None
        # 正样本中重叠量
        self.pos_and_overlap_nums = None
        # 负样本中重叠量
        self.neg_and_overlap_nums = None
        # 正样本中异常量
        self.pos_and_abn_nums = None
        # 负样本中异常量
        self.neg_and_abn_nums = None

    def fit(self, X, y):
        # 确定正类和负类
        tmp_labels = []
        tmp_nums = []
        self.labels = set(y)
        if (len(self.labels)) != 2:
            raise Exception("标签类别数:", len(self.labels))
        for y_label in self.labels:
            self.label_map_num[y_label] = np.sum(y == y_label)
            tmp_labels.append(y_label)
            tmp_nums.append(np.sum(y == y_label))
        if tmp_nums[0] < tmp_nums[1]:
            self.pos_label = tmp_labels[0]
            self.neg_label = tmp_labels[1]
        else:
            self.pos_label = tmp_labels[1]
            self.neg_label = tmp_labels[0]

        # 计算类间不平衡度
        self.ir = self.label_map_num[self.neg_label] / self.label_map_num[self.pos_label]
        # 计算每个样本的最近k个点的类别分布
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X, y)
        distances = knn.kneighbors_graph().todense()
        for index in range(len(y)):
            indices = np.argwhere(np.asarray(distances[index]).reshape(-1) == 1).reshape(-1)
            self.k_neighbors_dist.append(y[indices].tolist())
        # 计算重叠区域和噪声
        for index in range(len(y)):
            label_set = set(self.k_neighbors_dist[index])
            # 该点附近既有正样本又有负样本
            if len(label_set) == 2:
                self.overlap_marks.append(1)
            else:
                self.overlap_marks.append(0)
            if len(label_set) == 1 and label_set.pop() != y[index]:
                self.abnormal_marks.append(1)
            else:
                self.abnormal_marks.append(0)
        self.overlap_marks = np.asarray(self.overlap_marks)
        self.abnormal_marks = np.asarray(self.abnormal_marks)

        self.pos_and_overlap_nums = np.sum((y == self.pos_label) * (self.overlap_marks == 1))
        self.neg_and_overlap_nums = np.sum((y == self.neg_label) * (self.overlap_marks == 1))

        self.pos_and_abn_nums = np.sum((y == self.pos_label) * (self.abnormal_marks == 1))
        self.neg_and_abn_nums = np.sum((y == self.neg_label) * (self.abnormal_marks == 1))

        # 计算类内的不平衡度
        self.pos_db_scan_classes = len(set(DBSCAN().fit(X[np.argwhere(y == self.pos_label).reshape(-1)]).labels_)) - 1
        self.neg_db_scan_classes = len(set(DBSCAN().fit(X[np.argwhere(y == self.neg_label).reshape(-1)]).labels_)) - 1

    def show(self):
        """
        展示各统计指标
        :return:
        """
        show_list = []
        show_list.append(['样本数量', self.label_map_num[self.pos_label], self.label_map_num[self.neg_label]])
        show_list.append(['重叠样本数', self.pos_and_overlap_nums, self.neg_and_overlap_nums])
        show_list.append(['异常样本数', self.pos_and_abn_nums, self.neg_and_abn_nums])
        show_list.append(['类内不平衡度', self.pos_db_scan_classes, self.neg_db_scan_classes])
        return pd.DataFrame(show_list, columns=['index_name', 'pos', 'neg'])


"""
计算G_{mean}
"""


def g_score(y_true, y_pre):
    y_set = set(y_true)
    accs = []
    for y_label in y_set:
        accs.append(np.sum((y_true == y_label) * (y_pre == y_label)) / np.sum(y_true == y_label))
    g_score = 1.0
    for acc in accs:
        g_score *= acc
    return np.power(g_score, 1.0 / len(accs))

"""
绘制决策边界
"""

def plot_decision_function(X, y, clf, plt):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')