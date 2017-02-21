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
transformer = transword2vec(corpus,label)
transformer.feature_selection(1.0)
train_counts = transformer.trans_tf()  # 训练样本的词频
vocab = transformer.ordered_vocab   # 训练样本的词典


# 先写词语组变为词频的向量的方法
def trans_words_tf(words,vocab):
    vec=[0]*len(vocab) #可以直接乘积
    for word in words:
        if word in vocab:
            vec[list(vocab).index(word)]+=1
            #else: print("%s is not in dictionary"%word)
    return vec

def trans_words_tf(words,vocab):
    unique, counts = np.unique(words, return_counts=True)
    words_df = pd.DataFrame(counts,columns=["counts"])
    words_df.index = unique
    vocab_df = pd.DataFrame(data=np.zeros(len(vocab)),index=vocab,columns=["counts"])
    mask = np.in1d(unique, vocab)
    # vocab_df.loc[words_df.index[mask]] += words_df.loc[mask]
    # vocab_df += words_df.loc[mask]
    #vocab_df.add(words_df.loc[mask], fill_value=0)
    return np.array((words_df.loc[mask].reindex_like(vocab_df).fillna(0) + vocab_df).fillna(0)).ravel()
    # return np.array(vocab_df).ravel()

"""
下面为验证转为词频的方法
i=0
file_name=os.listdir(path+classes[i])
file = file_name[0]
news=open(path+classes[i]+'/'+file).read()
news_cut=parse(news)
tf = trans_words_tf(news_cut,vocab)
(train_counts.toarray()[0,]-tf).sum() # 为0
tfidf = trans_tf_tfidf(tf,ele_doc_count,sample_size)
vec_tfidf = transformer.trans_tfidf(toarray=True)
(tfidf - vec_tfidf[0,]).sum() # 也为0，都正确了。
"""


ele_doc_count = np.sum(train_counts.toarray()>0,axis = 0)
sample_size = train_counts.shape[0]

def trans_tf_tfidf(x,ele_doc_count,sample_size):
    tfidf_pre = x*(np.log((sample_size+1)/(ele_doc_count+1))+1)
    tfidf = tfidf_pre/np.sqrt(np.sum(tfidf_pre**2))
    return tfidf


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