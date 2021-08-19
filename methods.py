# coding: utf-8

# -------------------------------------------------
#    Description:  基于最大熵模型算法的词汇挖掘程序，支持挖掘指定长度的词汇和指定长度组合的词组
#    Author:  sir.housir@gmail.com
#    Date: 2020/05/30
#    LastEditTime: 2020/09/10
# -------------------------------------------------

import os
import json
import math
import numpy as np
import pandas as pd
from queue import Queue
from itertools import chain
# from pathlib import PurePath

# WORK_PATH = str(PurePath(__file__).parent)

# ## 内置的bert字向量路径位置，为节约加载时间，已将其分离
# embeddings = f'{WORK_PATH}/params/embedding.txt'
# id2char = f'{WORK_PATH}/params/id2char.json'
# char2id = f'{WORK_PATH}/params/char2id.json'

def normalize(x):
    '''x为一个/组向量，返回归一化后的结果'''
    if len(x.shape) > 1:
        return x / np.clip(
            x**2, 1e-12, None).sum(axis=1).reshape((-1, 1) + x.shape[2:])**0.5
    else:
        return x / np.clip(x**2, 1e-12, None).sum()**0.5


def topK(arr, n):
    '''求数组的前top个值及其索引
    # x = np.array([1,0,3,9])
    # xs = np.sin(np.arange(9)).reshape((3, 3))
    # print(xs)
    # print(topK(x,3))
    # print(topK(xs,3))

    # print(xs[topK(xs,3)])
    # print(x[topK(x,3)[0]])
    '''
    # 解索引
    flat = arr.flatten()
    # 求前k个最大值的索引
    indices = np.argpartition(flat, -n)[-n:]
    # 索引排序
    indices = indices[np.argsort(-flat[indices])]
    # 求每个索引在原数组中的对应位置
    return np.unravel_index(indices, arr.shape)


def matrixD(mat_a, mat_b, similarity=True):
    # -------------------------------------------------
    #    description: 快速计算矩阵行与行之间的距离及相似度算法(待修正)
    #    param mat_a array A*N的矩阵
    #    param mat_b array B*N的矩阵
    #    param similarity boolean similairty为True时返回余弦相似度，否则为欧式距离
    #    return: A*B的矩阵，表示A的每一行和B的每一行之间的距离或相似度
    # -------------------------------------------------
    la = mat_a.shape[0]
    lb = mat_b.shape[0]
    dists = np.zeros((la, lb))
    dists = np.sqrt(-2 * np.dot(mat_a, mat_b.T) +
                    np.sum(np.square(mat_b), axis=1) +
                    np.transpose([np.sum(np.square(mat_a), axis=1)]))
    if similarity:
        dists = 1 - dists * dists / 2
        return dists
    return dists

# 信息熵函数
def calEnt(sl):
    sum_ent = np.sum(-((sl / sl.sum()).apply(np.log) * sl / sl.sum()))
    return sum_ent


class ExtractSeeds(object):
    """词汇挖掘算法"""
    def __init__(self, min_count=100, min_support=50, min_s=2.0, max_sep=3):
        super(ExtractSeeds, self).__init__()
        # 录取词语最小出现次数
        self.min_count = min_count
        # 录取词语最低支持度，1代表着随机组合
        self.min_support = min_support
        # 录取词语最低信息熵，越大说明越有可能独立成词
        self.min_s = min_s
        # 候选词语的最大字数,最大支持7个字符查找
        self.max_sep = max_sep

    def do(self, contents):
        # -------------------------------------------------
        #    param contents str，文本
        #    return:
        # -------------------------------------------------
        rs = []  # 存放最终结果
        rt = []  # 存放临时结果
        # 统计每个词出现的频率
        rs.append(pd.Series(list(contents)).value_counts())
        # 统计输入文本的总长度
        tsum = rs[0].sum()

        for m in range(2, self.max_sep + 1):
            print(f'正在挖掘长度为{m}的短语...')
            rs.append([])
            for i in range(m):  # 生成所有可能的m字词，构造n元组
                for j in range(len(contents) - m + 1):
                    rs[m - 1].append(','.join(contents[j:j + m]))
            rs[m - 1] = pd.Series(rs[m - 1]).value_counts()  # 逐词统计
            rs[m - 1] = rs[m - 1][rs[m - 1] > self.min_count]  # 最小次数筛选
            tt = rs[m - 1][:]

            for k in range(m - 1):
                try:
                    qq = np.array(
                        list(
                            map(
                                lambda index: tsum * rs[m - 1][index] /
                                int(rs[m - 2 - k][','.join(
                                    index.split(',')[:m - 1 - k])]) / int(rs[
                                        k][','.join(
                                            index.split(',')[m - 1 - k:])]),
                                tt.index))) > self.min_support  # 最小支持度
                    tt = tt[qq]
                except Exception:
                    continue
            rt.append(tt.index)

        for i in range(2, self.max_sep + 1):
            print(f'正在筛选长度为{i}的短语({len(rt[i - 2])})...')
            pp = []  # 保存所有的左右邻结果
            for j in range(len(contents) - i - 1):
                sp = ([
                    contents[j], ','.join(contents[j + 1:j + i + 1]),
                    contents[j + i + 1]
                ])
                pp.append(sp)
            pp = pd.DataFrame(pp).set_index(1).sort_index()  # 先排序，加快检索速度
            index = np.sort(np.intersect1d(rt[i - 2], pp.index))  # 作交集
            # 分别计算左邻和右邻信息熵
            index = index[np.array(
                list(
                    map(lambda s: calEnt(pd.Series(pp[0][s]).value_counts()),
                        index))) >= self.min_s]
            rt[i - 2] = index[np.array(
                list(
                    map(lambda s: calEnt(pd.Series(pp[2][s]).value_counts()),
                        index))) >= self.min_s]
        # 下面都是输出前处理
        for i in range(len(rt)):
            rs[i + 1] = rs[i + 1][rt[i]]
            rs[i + 1] = rs[i + 1].sort_values(ascending=False)

        # 返回词频字典
        key = list(
            chain(*[[''.join(ix.split(',')) for ix in list(elem.index)]
                    for elem in rs[1:]]))
        val = list(chain(*[[ix for ix in list(elem)] for elem in rs[1:]]))
        # value_dict =removeRedundant(dict(zip(key, val)))
        return dict(zip(key, val))


class ExpandSeeds(object):
    """基于词向量的标签传播算法,主要用于挖掘部分长尾词汇"""
    def __init__(self, model_path=None, topn=20):
        super(ExpandSeeds, self).__init__()
        from .backend import WordVector
        self.model = WordVector().load(model_path)
        self.topn = topn

    def do(self, seeds, max_sim=1.0, min_sim=0.6, alpha=0.25):
        # 获取词表大小
        word_size = self.model.wv.vector_size
        queue_count = 1
        task_count = 0
        cluster = []
        queue = Queue()  # 建立队列,用idx控制衰减系数

        for w in seeds:
            queue.put((0, w))
            if w not in cluster:
                cluster.append(w)

        while not queue.empty():
            idx, word = queue.get()
            queue_count -= 1
            task_count += 1
            if word not in self.model.wv:
                continue

            sim_words = self.model.most_similar(word, topn=self.topn)
            threshold = min_sim + (max_sim - min_sim)*(1 - math.exp(-alpha*idx))

            if task_count % 10 == 0:
                log = "%s in cluster,%s in queue,%s tasks done,%s min_sim" % (
                    len(cluster), queue_count, task_count, threshold)
                print(log)

            for i, j in sim_words:
                if j >= threshold:
                    # 只将字数长度大于等于2
                    if i not in cluster and len(i) > 1:
                        cluster.append(i)
                        queue.put((idx+1, i))
                        queue_count += 1
        return cluster

# 效果一般，还是建议训练词向量再使用
# class ExpandSeeds(object):
#     """基于字向量的标签传播算法,主要用于挖掘部分长尾词汇"""
#     def __init__(self, candidates):
#         super(ExpandSeeds, self).__init__()
#         print("加载字向量...")
#         self.embeddings = np.loadtxt(embeddings)
#         with open(id2char, 'r', encoding='utf-8') as fp:
#             self.id2char = json.load(fp)
#         with open(char2id, 'r', encoding='utf-8') as fp:
#             self.char2id = json.load(fp)
#         print("计算词向量...")
#         self.id2cand = dict(enumerate(candidates))
#         self.cand2id = {v:k for k,v in self.id2cand.items()}
#         self.sentemb = np.zeros(shape=(len(candidates),768)) # 将所有的词向量写入一张大表中
#         for i,can in enumerate(candidates):
#             self.sentemb[i] = self.sent2vec(can) ## 这一步用到来上面的相似度计算
#         self.normalized_embeddings = normalize(self.sentemb) ## 归一化以计算余弦相似度

#     def do(self, seeds, topn=10, max_sim=1.0, min_sim=0.6, alpha=0.25):

#         queue_count = 1
#         task_count = 0
#         cluster = []
#         queue = Queue()  # 建立队列,用idx控制衰减系数

#         for w in seeds:
#             queue.put((0, w))
#             if w not in cluster:
#                 cluster.append(w)

#         while not queue.empty():
#             idx, word = queue.get()
#             queue_count -= 1
#             task_count += 1

#             sim_words = self.most_similar(word, topn)
#             threshold = min_sim + (max_sim - min_sim)*(1 - np.exp(-alpha*idx))

#             if task_count % 10 == 0:
#                 log = "%s in cluster,%s in queue,%s tasks done,%s min_sim" % (
#                     len(cluster), queue_count, task_count, threshold)
#                 print(log)

#             for i, j in sim_words:
#                 if j >= threshold:
#                     # 只将字数长度大于等于2
#                     if i not in cluster and len(i) > 1:
#                         cluster.append(i)
#                         queue.put((idx+1, i))
#                         queue_count += 1
#         return cluster
    
#     def most_similar(self, word, topn=10, nb_context_words=None, with_sim=True):
#         # -------------------------------------------------
#         #    description: 计算相似词语
#         #    param word str，输入词语
#         #    param topn int，返回前topn个词语
#         #    param nb_context_words int，默认为100000，限制词语范围
#         #    return:
#         # -------------------------------------------------

#         if nb_context_words is not None:
#             embeddings_ = self.sentemb[:nb_context_words]
#             embeddings_ = embeddings_ - embeddings_.mean(axis=0)
#             U = np.dot(embeddings_.T, embeddings_)
#             U = np.linalg.cholesky(U)
#             embeds = np.dot(self.embeddings, U)
#             self.normalized_embeddings = embeds/(embeds**2).sum(axis=1).reshape((-1,1))**0.5
    
#         word_vec = normalize(self.sent2vec(word))[0].T ## 可以处理任意输入向量,要将其转置
#         word_sim = np.dot(self.normalized_embeddings, word_vec)
#         word_sim_sort = word_sim.argsort()[::-1]
#         if with_sim:
#             return [(self.id2cand[i], word_sim[i])
#                     for i in word_sim_sort[:topn]]
#         return [self.id2cand[i] for i in word_sim_sort[:topn]]
        
#     def sent2vec(self, sent):
#         # -------------------------------------------------
#         #    description: 使用字/词向量生成句子/短语向量
#         #    param sent str/list 一个/一组句子/短语
#         #    return:
#         # -------------------------------------------------
#         Z = []
#         if isinstance(sent, list):
#             for s in sent:
#                 idxs = [self.char2id[w] for w in s if w in self.char2id]
#                 sv = self.embeddings[idxs].sum(axis=0)
#                 Z.append(sv)
#         else:
#             sent = sent
#             idxs = [self.char2id[w] for w in sent if w in self.char2id]
#             sv = self.embeddings[idxs].sum(axis=0)
#             Z.append(sv)
#         return normalize(np.array(Z))