#数据输入
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
#运行之前，需要确认，电脑中的python有certificate
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#数据输入
df = pd .read_csv('/Users/apple/Downloads/toxic_comments-2.csv')
int(df.describe().iloc[0,0])
#数据分类
train=df.iloc[range(100000),range(1,8)]
validation = df.iloc[range(100001,len(df['comment_text'])),range(1,8)]
print(len(train))
print(len(validation))
#

#用空格替换各种符号
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
#删除各种符号
BAD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
STOPWORDS = set(stopwords.words('english'))
#定义数据清洗函数
def text_prepare(text):
    text = text.lower() # 字母小写化
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)
    text = ' '.join([w for w in text.split() if w not in STOPWORDS]) # 删除停用词
    return text


X_train = train.comment_text
X_val= validation.comment_text

# 开始进行数据清洗
X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]

#生成词频
cv = CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = cv.fit_transform(X_train)
print(feature.shape)
print()
print(feature)

#词频权重
tfidf = TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = tfidf.fit_transform(X_train)
print(feature.shape)
print()
print(feature)

#取y_train
toxi=train['toxic']
toxi=list(toxi)
severe_toxic=train['severe_toxic']
severe_toxic=list(severe_toxic)
obscen=train['obscene']
obscen=list(obscen)
threat=train['threat']
threat=list(threat)
insult=train['insult']
insult=list(insult)
identity_hate=train['identity_hate']
identity_hate=list(identity_hate)

#取词与手动评
jishu = []
toxic = []
sever = []
obsce = []
threa = []
insul = []
ident = []

#toxic/words 创建
for item in range(len(toxi)):
    a=[toxi[item]]*1
    toxic.extend(a)
#severe_toxic/words 创建
for item in range(len(severe_toxic)):
    a=[severe_toxic[item]]*1
    sever.extend(a)
#obscene/words 创建
for item in range(len(obscen)):
    a=[obscen[item]]*1
    obsce.extend(a)
#threat/words 创建
for item in range(len(threat)):
    a=[threat[item]]*1
    threa.extend(a)
#insult/words 创建
for item in range(len(insult)):
    a=[insult[item]]*1
    insul.extend(a)
#identity_hate/words 创建
for item in range(len(identity_hate)):
    a=[identity_hate[item]]*1
    ident.extend(a)
print(len(toxic)); print(len(sever));  print(len(obsce));  print(len(threa));  print(len(insul));   print(len(ident))
a=[toxic,sever,obsce,threa,insul,ident]

#新的数据表，检索单词
y_train = []
for i in range(len(a[0])):
    n = []
    for j in range(len(a)):
        n.append(a[j][i])
    y_train.append(n)

#取y_val
toxi=validation['toxic']
toxi=list(toxi)
severe_toxic=validation['severe_toxic']
severe_toxic=list(severe_toxic)
obscen=validation['obscene']
obscen=list(obscen)
threat=validation['threat']
threat=list(threat)
insult=validation['insult']
insult=list(insult)
identity_hate=validation['identity_hate']
identity_hate=list(identity_hate)

#取词与手动评
jishu = []
toxic = []
sever = []
obsce = []
threa = []
insul = []
ident = []

#toxic/words 创建
for item in range(len(toxi)):
    a=[toxi[item]]*1
    toxic.extend(a)
#severe_toxic/words 创建
for item in range(len(severe_toxic)):
    a=[severe_toxic[item]]*1
    sever.extend(a)
#obscene/words 创建
for item in range(len(obscen)):
    a=[obscen[item]]*1
    obsce.extend(a)
#threat/words 创建
for item in range(len(threat)):
    a=[threat[item]]*1
    threa.extend(a)
#insult/words 创建
for item in range(len(insult)):
    a=[insult[item]]*1
    insul.extend(a)
#identity_hate/words 创建
for item in range(len(identity_hate)):
    a=[identity_hate[item]]*1
    ident.extend(a)
print(len(toxic)); print(len(sever));  print(len(obsce));  print(len(threa));  print(len(insul));   print(len(ident))
a=[toxic,sever,obsce,threa,insul,ident]

#新的数据表，检索单词
y_val = []
for i in range(len(a[0])):
    n = []
    for j in range(len(a)):
        n.append(a[j][i])
    y_val.append(n)

#数据类型转换
y_train=np.array(y_train)
y_val=np.array(y_val)

#分析
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# 朴素贝叶斯模型
NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(MultinomialNB())),
])

NB_pipeline.fit(X_train, y_train)
prob=NB_pipeline.predict_proba(X_val)
predicted = NB_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

#TF-IDF+线性支持向量机
SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])

SVC_pipeline.fit(X_train, y_train)
predicted = SVC_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

#TF-IDF+逻辑回归
LogReg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs',max_iter=10000), n_jobs=1)),
])

LogReg_pipeline.fit(X_train, y_train)
predicted = LogReg_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)


#CountVectorizer+朴素贝叶斯
NB_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),
])

NB_pipeline.fit(X_train, y_train)
predicted = NB_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

# CountVectorizer+线性支持向量机
SVC_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])

SVC_pipeline.fit(X_train, y_train)
predicted = SVC_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

#CountVectorizer+逻辑回归
LogReg_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs'), n_jobs=1)),
])

LogReg_pipeline.fit(X_train, y_train)
predicted = LogReg_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

# 返回每个测试评论为toxic的概率
LogReg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs'), n_jobs=1)),
])

LogReg_pipeline.fit(X_train, y_train)
predicted = LogReg_pipeline.predict(X_val)
score=LogReg_pipeline.predict_proba(X_val)
print_evaluation_scores(y_val, predicted)

finl=[]
final=[]
for i in range(len(y_val)):
    finl=[y_val[i],score[i]]
    final.extend(finl)

final=np.array(final)
y_val=np.array(y_val)

