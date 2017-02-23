# coding = utf-8
import numpy as np
import pandas as pd
import jieba
import os
from importlib import reload
import sklearn
from sklearn import svm
import datetime
import pickle
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
vec_tfidf = transformer.trans_tfidf(toarray=False)
clf = svm.SVC(kernel='linear', C=1) # 采用默认参数结果会出错，用线性核的准确率很高
clf.fit(vec_tfidf, label)


train_counts = transformer.trans_tf()  # 训练样本的词频
vocab = transformer.ordered_vocab   # 训练样本的词典


# 先写词语组变为词频的向量的方法
"""
淘汰掉的很慢的方法
def trans_words_tf(words,vocab):
    vec=[0]*len(vocab) #可以直接乘积
    for word in words:
        if word in vocab:
            vec[list(vocab).index(word)]+=1
            #else: print("%s is not in dictionary"%word)
    return vec
"""


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

def trans_tf_tfidf(x):
    if np.sum(x)==0:
        tfidf = np.zeros(x.shape[0])  # 如果是tf全是0，即一个词典上的词都没匹配上的话
    else:
        tfidf_pre = x*(np.log((sample_size+1)/(ele_doc_count+1))+1)
        tfidf = tfidf_pre/np.sqrt(np.sum(tfidf_pre**2)) # 标准化
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

# 下面开始检验，如果一段文字进来了，会判别如何

"""
text = "环保公司中报亮眼：行业分化明显 净利润最高增16.5倍"
news_cut = parse(text)
tf = trans_words_tf(news_cut,vocab)
tfidf = trans_tf_tfidf(tf,ele_doc_count,sample_size)
y_pred = clf.predict(tfidf.reshape(1,29789))
print(y_pred)

df_temp = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
df_temp["d"] = df_temp.apply(lambda x:x['a']+10,axis=1)
"""

# 现在先用标题数据来做。已经在sublime里面写了如何download数据，直接把结果读进来，分词
total_df = pd.read_pickle('C:/Users/TanXiao/Desktop/thesis/data/total_df.pkl')

# 按照选出的50只股票的信息来做。选出属于这50个股票的新闻数据
price_df = pd.read_pickle("C:/Users/TanXiao/Desktop/thesis/data/total50_price.pkl")
price_df.Date = pd.to_datetime(price_df.Date)
chosed_news_df = total_df.loc[np.in1d(total_df.stock,price_df.stock.unique()),]

chosed_news_df["headline_cut"] = chosed_news_df.apply(lambda row:parse(row['headline']),axis=1)

# 将文本转为向量
chosed_tf_list = chosed_news_df.apply(lambda row:list(trans_words_tf(row['headline_cut'],vocab)),axis=1)
chosed_tf_array = np.array(list(chosed_tf_list))
chosed_tfidf_array = np.apply_along_axis(trans_tf_tfidf,1,chosed_tf_array)
# 有部分文本因为标题太短所以最后的向量元素全为0
pickle.dump(chosed_tfidf_array, open("C:/Users/TanXiao/Desktop/thesis/data/chosed_tfidf_array.txt", "wb"))
obj2 = pickle.load(open("C:/Users/TanXiao/Desktop/thesis/data/chosed_tfidf_array.txt", "rb"))
# 之后最好是能用sparse的操作，光是存储下来就1个G

# 再将tf-idf矩阵放入svm里做情感判别
emotion = clf.predict(chosed_tfidf_array)
np.unique(emotion,return_counts=True)  # 发现中性的偏多，占了一半。

# 情感要转为-1,0,1放回到矩阵中去
emotion_num = (emotion == 'negtive')*(-1) + (emotion == 'neutral')*0 + (emotion == 'positive')*1
chosed_news_df["emotion_num"] = np.array(emotion_num)

# 先修正日期，每天下午15点后的算作后一天；然后情感按照code和日期汇总求平均
chosed_news_df.date = pd.to_datetime(chosed_news_df.date)

def date_modify(x):
    hour = int(x["time"][:2])
    if hour >= 15:
        date_modified = x['date'] + datetime.timedelta(days=1)
    else:
        date_modified = x['date']
    return  date_modified

chosed_news_df['date_modified'] = chosed_news_df.apply(date_modify,axis=1)
# 发现计算倒是很快，但是给它内部赋值太慢了


"""
grouped = chosed_news_df.head().groupby(["stock", "date_modified"])
temp = grouped['emotion_num'].agg(np.mean)
b=[]
for a,group in grouped:
    print(np.mean(group["emotion_num"]))
"""

# 现在按照股票代码和日期去匹配找到属于哪一天的情感。（暂时停掉）


# 先来个快点的吧，直接按股票和情感
# 用pd.merge，可以像sql那样用，但是变量名字必须一致
price_df["date_modified"] = price_df.Date
# 发现有的时候用"."就行，比如上面pd.to_datetime时，但是有时候用就不行，比如上面price_df.date_modified = price_df.Date
# 是不是已存在的才能用"."？
data_merge = pd.merge(price_df, chosed_news_df, how='left', on=['stock', 'date_modified'])
data_merge.emotion_num = data_merge.emotion_num.fillna(0)
data_merge.to_pickle("C:/Users/TanXiao/Desktop/thesis/data/data_merge.pkl")

grouped = data_merge.groupby(['stock', 'date_modified'])
data_grouped = grouped.agg({"return_rate":np.mean,"sh_return":np.mean,"if_large":np.mean,"emotion_num":np.mean})
data_grouped2 = data_grouped.ix[~np.isnan(data_grouped.return_rate),]

# 建回归模型
from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(data_grouped2[["sh_return","emotion_num"]],data_grouped2["return_rate"])
reg.coef_

import statsmodels.api as sm
model = sm.OLS(data_grouped2["return_rate"], data_grouped2[["sh_return","emotion_num"]])
results = model.fit()
print(results.summary())

# 分成大小盘股两组
model0 = sm.OLS(data_grouped2.ix[data_grouped2.if_large ==0,"return_rate"],
                data_grouped2.ix[data_grouped2.if_large ==0,["sh_return","emotion_num"]])
results0 = model0.fit()
print(results0.summary()) # 0.056的系数

model1 = sm.OLS(data_grouped2.ix[data_grouped2.if_large ==1,"return_rate"],
                data_grouped2.ix[data_grouped2.if_large ==1,["sh_return","emotion_num"]])
results1 = model1.fit()
print(results1.summary()) # 0.037的系数

# 符合预期，基本上就是有关的

# 再试一下logistic回归，结果不显著不说，系数值也不太大。
data_grouped2['if_rise'] = (data_grouped2.return_rate >=0)*1
logit0 = sm.Logit(data_grouped2.ix[data_grouped2.if_large ==0,"if_rise"],
                data_grouped2.ix[data_grouped2.if_large ==0,["sh_return","emotion_num"]])
results0 = logit0.fit()
print(results0.summary())


logit1 = sm.Logit(data_grouped2.ix[data_grouped2.if_large ==1,"if_rise"],
                  data_grouped2.ix[data_grouped2.if_large ==1,["sh_return","emotion_num"]])
results1 = logit1.fit()
print(results1.summary())