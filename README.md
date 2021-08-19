# 一个包含常见无监督算法的自然语言处理工具包

unlp中的u是“Unsupervised(无监督)”的意思，旨在用无监督学习的方法解决NLP任务中的“冷启动”问题。

# 包含的功能如下:

## 词语切分

目前默认使用pkuseg分词器作为后端，相比jieba分词器，pkuseg的性能要更高一些

分词器性能对比如下](https://github.com/lancopku/pkuseg-python/blob/master/readme/comparison.md)

| Default | MSRA      | CTB8      | PKU       | WEIBO     | All Average |
| ------- | --------- | --------- | --------- | --------- | ----------- |
| jieba   | 81.45     | 79.58     | 81.83     | 83.56     | 81.61       |
| pkuseg  | **87.29** | **91.77** | **92.68** | **93.43** | **91.29**   |

```python

import unlp
from unlp.backend import xre
from unlp.backend import SentenceCutter, CorpusCutter

## 初始化分词器
spliter = SentenceCutter(re=xre)
spliter.cut(sents='unlp是一个包含常见无监督学习算法的自然语言处理(NLP)工具包') ## re为过滤规则函数,默认为None,unlp中的xre默认只保留汉字认只保留汉字
>>>['是', '一个', '包含', '常见', '无', '监督', '学习', '算法', '的', '自然语言', '处理', '工具包']
## 切分列表
spliter.cut(sents=['unlp是一个自然语言处理(NLP)工具包','包含了常见的无监督学习算法'])
>>>[['unlp', '是', '一个', '自然', '语言', '处理', '(', 'NLP', ')', '工具包'],['包含', '了', '常见', '的', '无', '监督', '学习', '算法']]
## 开多进程切分大型文本，需要使用main函数
if __name__ == "__main__":
    spliter = CorpusCut(re=xre)
    spliter.cut(fpath='./unlp/corpus/xiyou.txt',tpath='./unlp/corpus/xiyou.corpus',workers=16) ## fpath为原始文件路径,tpath为生成的语料,workers为进程数目
切词器初始化...
100.00% [==========]
聚合结果...
写入结果...
100.00% [==========]
```

## 词向量生成

```python
from unlp.backend import WordVector

WordVector(iters=60).train('./unlp/corpus/xiyou.corpus')
## 初始化部分可选参数
## size=128 词向量维度
## window=5 窗口的大小
## iters=100 迭代的次数
## min_count=10 录取的最低频数 
## sg=1 使用skip-gram方式训练词向量 
## hs=1 使用霍夫曼树编码
## workers=10 开启多进程训练
```

## 词汇挖掘

基于最大熵算法的词汇挖掘算法,支持**字粒度**(用于挖掘词汇)和**词粒度**(用于挖掘短语)两种挖掘方式

```python

from unlp.extractor import Seeds

contents = []
with open('./unlp/corpus/xiyou.corpus',encoding='utf-8') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        line = line.strip().split('\n')
        contents += line
## 初始化挖掘程序
sd = Seeds(min_count=50,min_support=5,min_s=2.5,max_sep=5)
sd.do(contents)
>>>{'行者道': 138, '一个': 2218, '八戒': 129, '两个': 1194, '不知': 1192, '八戒道': 1158,...}
```

## 词汇扩充

基于软聚类的词汇挖掘方法

```python

from unlp.extractor import ExpandSeeds

ExpandSeeds(model_path='./Word2Vec.model').do(['沙僧'],min_sim=0.35)
>>>
加载Word2Vec模型...
2020-10-20 17:48:03,216 : INFO : loading Word2Vec object from ./Word2Vec.model
2020-10-20 17:48:03,300 : INFO : loading wv recursively from ./Word2Vec.model.wv.* with mmap=None
2020-10-20 17:48:03,300 : INFO : setting ignored attribute vectors_norm to None
2020-10-20 17:48:03,300 : INFO : loading vocabulary recursively from ./Word2Vec.model.vocabulary.* with mmap=None
2020-10-20 17:48:03,300 : INFO : loading trainables recursively from ./Word2Vec.model.trainables.* with mmap=None
2020-10-20 17:48:03,300 : INFO : setting ignored attribute cum_table to None
2020-10-20 17:48:03,300 : INFO : loaded ./Word2Vec.model
2020-10-20 17:48:03,305 : INFO : precomputing L2-norms of word weight vectors
17 in cluster,7 in queue,10 tasks done,0.4937794910035868 min_sim
['沙僧', '行李', '担子', '行囊', '沙和尚', '哥哥', '行者', '呆子', '紧随', '盼望', '唐僧', '三藏', '戒道', '白马', '师父', '马匹', '看守', '长老']
```

# 环境依赖:

numpy>=1.1.0
pandas>=1.19.1
pkuseg>=0.0.25
gensim>=3.8.3

# 安装方法(暂不开放)
pip install unlp


