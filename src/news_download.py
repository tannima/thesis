# coding = utf-8
import numpy as np
import pandas as pd
import jieba
import os
from importlib import reload
import sklearn
"""
选取2016-07-31至2016-12-31这期间数据
"""

# 写一个接口，传入一段文本和词汇表，返回tf-idf值（已经确定是这个更好）
# 怎么把新的转tf-idf? 一种方式是，将每个词在多少篇文章中有这个量找出来，其他的都可以自己来
# 先按照词典生成词频向量。再把这个向量和训练TF-IDF的一些东西联系起来。
# 下面做试验

# 其实吧，将来有了更多的语料，可以把TF-IDF的建立方式改了，不只是那1500篇，因此这是非监督的




"""
下面是为了试验转成tf-idf的代码
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
counts = np.array([[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]])
tfidf = transformer.fit_transform(counts)
"""

ele_doc_count = np.sum(train_counts>0,axis = 0)

def trans_tf_tfidf(x,train_counts):
    tfidf_pre = x*(np.log((train_counts.shape[0]+1)/(ele_doc_count+1))+1)
    tfidf = tfidf_pre/np.sqrt(np.sum(tfidf_pre**2))
    return tfidf