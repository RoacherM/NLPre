# coding: utf-8

# -------------------------------------------------
#    Description:  后端程序，包含基础的过滤规则和一些便捷工具
#    Author:  sir.housir@gmail.com
#    Date: 2020/05/30
#    LastEditTime: 2020/09/10
# -------------------------------------------------

import os
import re
import pkuseg
import gensim
import chardet
import logging
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 常用的正则表达式
# url网址
Urls = '''[a-zA-z]+://[^\s]*'''
# IP地址
IPs = '''\d+\.\d+\.\d+\.\d+'''
Banks = '''/^([1-9]{1})(\d{14}|\d{18})$/'''
# 中国大陆固定电话号码
Phones = '''(\d{4}-|\d{3}-)?(\d{8}|\d{7})'''
# 中国大陆手机号码
Mphones = '''1\d{10}'''
# 中国大陆邮编
Mails = '''[1-9]\d{5}'''
# 电子邮箱
Emails = '''\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*'''
# 中国大陆身份证号(18位或者15位)
IDs = '''\d{15}(\d\d[0-9xX])?'''
# 中国车牌号码
Plates = '''([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}
        (([0-9]{5}[DF])|(DF[0-9]{4})))|([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼
        使领A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]{1})'''
# QQ号码
QQs = '''/^[1-9]\d{4,9}$/'''
# 微信号码
Wechats = '''/^[a-zA-Z]{1}[-_a-zA-Z0-9]{5,19}$/'''


def xre(text):
    """中文常见的过滤规则"""
    text = re.sub('\W+', ' ', text).replace('_', ' ')    # 只保留数字字母及下划线
    text = re.sub('[A-Za-z]+', ' ', text)
    text = re.sub('[\d]+', ' ', text)
    return text


def processbar(p_now, p_end):
    """简单封装的进度条程序，打印进度条: 50.00% [=====     ]
    """
    # -------------------------------------------------
    #    param p_now: 当前位置
    #    param p_end: 终止位置
    #    return:
    # -------------------------------------------------
    percent = min(p_now * 10000 // p_end, 10000)
    done = "=" * (percent // 1000)
    half = "-" if percent // 100 % 10 > 5 else ""
    tobe = " " * (10 - percent // 1000 - len(half))

    processbar_fmt = "\r{:.2f}% [{}{}{}]"
    bar_content = processbar_fmt.format(
        percent / 100,
        done,
        half,
        tobe
    )
    print(bar_content, end="")
    if percent == 10000:
        print("\33[?25h\n", end="")  # 显示光标


class Cutter(object):
    """切词器基类
    """
    def __init__(self, re=None, postag=False, userwords=None, stopwords=None, mustwords=None):
        super(Cutter, self).__init__()

        self.re = re
        self.postag = postag
        self.userwords = userwords  # 文件路径
        # 读取指定路径下文件内容
        self.stopwords = [line.strip().split(',')[0] for line in open(
            stopwords, encoding='utf-8').readlines()] if stopwords is not None else ['']  # ...
        self.mustwords = [line.strip().split(',')[0] for line in open(
            stopwords, encoding='utf-8').readlines()] if mustwords is not None else ['']  # ...
        ## 加载切词器
        self.poseg = pkuseg.pkuseg(user_dict=userwords,postag=postag).cut
        print("切词器初始化...")

    def piece(self, sent):
        # 统一切词的输出格式
        sentences = []
        words = self.poseg(
            self.re(sent)) if self.re is not None else self.poseg(sent)
        for word in words:
            if self.postag:
                word = list(word)
                if word[0] not in self.stopwords or word[0] in self.mustwords:
                    if word[0]!=' ':
                        sentences.append(word)
            else:
                if word not in self.stopwords or word in self.mustwords:
                    if word != ' ':
                        sentences.append(word)
        return sentences


class SentenceCutter(Cutter):
    """句子级别的切词器，支持词性标注"""
    def __init__(self, re=None, postag=False, userwords=None, stopwords=None, mustwords=None):
        super(SentenceCutter, self).__init__(re=re,postag=postag,userwords=userwords,stopwords=stopwords,mustwords=mustwords)

    def cut(self, sents):
        # -------------------------------------------------
        #    param sents 输入为一则字符串/字符串列表
        #    return:
        # -------------------------------------------------
        sentences = []
        if isinstance(sents, list):
            for sent in sents:
                sentences.append(self.piece(sent))
        else:
            sentences = self.piece(sents)
        return sentences


class CorpusCutter(Cutter):
    """基于多进程实现的文本切词器，主要用于切分大型文本"""
    def __init__(self, re=None, postag=False, userwords=None, stopwords=None, mustwords=None):
        super(CorpusCutter, self).__init__(re=re,postag=postag,userwords=userwords,stopwords=stopwords,mustwords=mustwords)

    def cut(self, fpath, tpath=None, workers=10):

        if not os.path.isfile(fpath):
            raise Exception(fpath + "file not exists!")
        with open(fpath, 'rb') as fp:
            coding = chardet.detect(fp.read(10000))["encoding"]  # 获取文件编码格式
        fsize = os.path.getsize(fpath)
        workers = workers if workers is not None else 2

        rss = [] # 存放最后的结果
        res = []
        pool = Pool(workers)
        for i in range(workers):
            p_start = fsize * i // workers
            p_end = fsize * (i + 1) // workers
            args = [fpath, p_start, p_end, coding]
            # args = [self, fpath, p_start, p_end]   
            rs = pool.apply_async(func=self.chunksize, args=args)
            res.append(rs)
        pool.close()
        pool.join()
        print("聚合结果...")
        for rs in res:
            rss.extend(rs.get())
        if tpath:
            print("写入结果...")
            with open(tpath, 'w', encoding='utf-8') as fp:
                for i,s in enumerate(rss):
                    r = ' '.join(s) + '\n'
                    fp.writelines(r)
                    processbar(i+1, len(rss))

    def chunksize(self, fpath, p_start, p_end, coding):
        # -------------------------------------------------
        #    param fpath 原始语料路径
        #    param p_start  开始读取的位置
        #    param p_end 终止读取的位置
        #    return:
        # -------------------------------------------------
        contents = []
        with open(fpath, 'rb') as fp:
            if p_start:
                fp.seek(p_start - 1)
                while b"\n" not in fp.read(1):
                    pass
            while True:
                line = fp.readline()
                if line:
                    line = self.piece(line.decode(coding).strip().split('\n')[0])
                    contents.append(line)
                pos = fp.tell()
                if p_start == 0:
                    processbar(pos, p_end)
                if pos >= p_end:
                    break
        return contents


class WordVector(object):

    def __init__(self, size=128, window=5, iters=100, min_count=10, sg=1, hs=1, workers=10):
        super(WordVector, self).__init__()
        self.size = size
        self.window = window
        self.iter = iters
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.workers = workers

    def train(self, path):
        # -------------------------------------------------
        #    description: Word2Vec生成词向量算法
        #    return:
        # -------------------------------------------------
        print('初始化Word2Vec模型...')
        model = gensim.models.Word2Vec(corpus_file=path,
                        size=self.size,
                        window=self.window,
                        min_count=self.min_count,
                        iter=self.iter,
                        sg=self.sg,
                        hs=self.hs,
                        workers=self.workers)
        print('训练Word2Vec词向量...')
        model.save('Word2Vec.model')
        model.wv.save_word2vec_format('Word2Vec.vector',binary=False)
        print('模型参数保存完毕！')

    def load(self, path):
        print('加载Word2Vec模型...')
        model = gensim.models.Word2Vec.load(path)
        return model
