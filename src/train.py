# coding = utf-8
import numpy as np
import pandas as pd
import jieba
import os
from importlib import reload
import sklearn

# 读入数据，分词，转为向量
path = 'D:/my_projects/text_mining/stock_project/TrainingData/'
stopwords = [line.rstrip() for line in open(path+'stopwords.txt')]
stopwords.extend([line.rstrip() for line in open(path+'company_stop.txt',encoding='utf-8')])

def parse(text):
    words=[]
    seg_list=jieba.cut(text,cut_all=False)  # 精确模式
    for word in seg_list:
        if len(word)>1 and (word not in stopwords) and (not word[0].isdigit()) \
                and (not word[1].isdigit()):  # 去掉停用词、一个字符的词和所有数字
            words.append(word)
    return words
# 思考是不是最好还要去掉公司名称，媒体名称等，要不用jieba的词性标注将所有名词都去掉？

classes=['negtive','neutral','positive']
dic=set([]) #词典
alltxt=[]  #存放解析后每个文本分好的词
label=[]  #每个自向量的label

for i in range(3):
    file_name=os.listdir(path+classes[i])
    for file in file_name:
        try:
            news=open(path+classes[i]+'/'+file).read()
            news_to_words=parse(news)  # 将文本进行分词
            alltxt.append(news_to_words)
            label.append(classes[i])
            dic=dic | set(news_to_words)
        except:
            print(classes[i]+file)
len(dic)


# 先分成训练和验证，之后再用那个CV
from sklearn.model_selection import train_test_split

# 重新写转为向量的方法
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = []
label = []
for i in range(3):
    file_name=os.listdir(path+classes[i])
    for file in file_name:
        try:
            news=open(path+classes[i]+'/'+file).read()
            news_cut=parse(news)
            corpus.append(" ".join(news_cut))
            # 用空格分开，使用CountVectorizer需要这种格式
            label.append(classes[i])
        except:
            print(classes[i]+file)

class transword2vec(object):
    def __init__(self,data):
        self.data = data
        self.vec_tf = np.zeros(1)
        self.vec_tfidf = np.zeros(1)
    def trans_tf(self,toarray = False): # 转为词频的向量,以稀疏形式存储，虽然也可以转为矩阵.toarray()即可
        vectorizer = CountVectorizer()
        self.vec_tf = vectorizer.fit_transform(self.data)
        self.vocab = vectorizer.get_feature_names()
        if toarray == True:
            return self.vec_tf.toarray()
        else:
            return self.vec_tf
    def trans_tfidf(self,toarray = False): # 转为tf-idf值的向量
        tfidf_transformer=TfidfTransformer()
        if self.vec_tf.shape[0] == 1:
            vectorizer = CountVectorizer()
            self.vec_tf = vectorizer.fit_transform(self.data)
        self.vec_tfidf=tfidf_transformer.fit_transform(self.vec_tf)
        if toarray == True:
            return self.vec_tfidf.toarray()
        else:
            return self.vec_tfidf

transformer = transword2vec(corpus)
vec_tf = transformer.trans_tf(toarray=False)
tfidf = transformer.trans_tfidf(toarray=False)

# 模型训练
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(vec_tf,label, test_size=0.3, random_state=0)
# tf向量,NB模型
MNB = MultinomialNB()
y_pred = MNB.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test,y_pred,labels=['negtive','neutral','positive'])
f1_score(y_test,y_pred,labels=['negtive','neutral','positive'], average=None)
# F1-score :array([ 0.81978799,  0.78767123,  0.83692308])

# tf向量,svm模型
from sklearn import svm
clf = svm.SVC() # 全部采用默认参数，尝试改变decision_function_shape参数但结果不变
# 默认的核是rbf
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred,labels=['negtive','neutral','positive'])
f1_score(y_test,y_pred,labels=['negtive','neutral','positive'], average=None)
# F1-score :array([ 0.3375    ,  0.59726027,  0.68266667])
# 为什么这个这么差？是不是要进行特征选择后结果才会好一点？


# tf-idf向量，NB模型
X_train, X_test, y_train, y_test = train_test_split(tfidf,label, test_size=0.3, random_state=0)
MNB = MultinomialNB()
y_pred = MNB.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test,y_pred,labels=['negtive','neutral','positive'])
f1_score(y_test,y_pred,labels=['negtive','neutral','positive'], average=None)
# F1-score:array([ 0.82909091,  0.80412371,  0.8502994 ])
# 这个效果不错。让我惊讶的是中性的居然也可以很高，不敢相信

# tf-idf向量，SVM模型
clf = svm.SVC(kernel='linear', C=1) # 采用默认参数结果会出错，用线性核的准确率很高
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred,labels=['negtive','neutral','positive'])
f1_score(y_test,y_pred,labels=['negtive','neutral','positive'], average=None)
# F1-score:array([ 0.85818182,  0.81208054,  0.87461774])

# 可能需要做一下cv，下面是示例
from sklearn.model_selection import cross_val_score
from sklearn import metrics
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
scores

# 还是要搞清楚f1的计算方式，毕竟这里有好几种f1出现了。这里的scores有5个值，意思是5次
# 每次的三个label都求了平均？

