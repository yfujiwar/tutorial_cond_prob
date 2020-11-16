#!/usr/bin/env python
# coding: utf-8

# # <center>付録: 日本語文章の形態素解析：素人による入門</center>
# Author: 藤原 義久 <yoshi.fujiwara@gmail.com>    
# Data: ディレクトリ"data"以下  
# - 短いサンプル文章
# - 日本国憲法

# ## 形態素解析のツールmecab
# - mecab 本家: https://taku910.github.io/mecab/
# - mecab 自体のインストール
#     - Windows: https://github.com/ikegami-yukino/mecab/releases/tag/v0.996  
#     「MeCab 0.996 64bit version (旧)」にあるインストーラ"mecab-0.996-64.exe"を使う  
#     注：環境変数 PATH にmecab をインストールしたパスを追加(例: C:\w10\mecab\bin)  
#     注：環境変数 MECABRC を新たに追加して設定(例: C:\w10\mecab\etc\mecabrc)
#     - Linux: "mecab linux"でググるとUbuntu, CentOS などでのインストール方法が分かる
#     - Mac: "mecab mac"でググる(未確認)
# - python からmecab を使うパッケージのインストール  
# jupyter notebook を新規に開いて以下を実行する(先頭の"!"に注意)
#     - Windows:  
#     > !pip install mecab-python-windows
#     - Linux:  
#     > !pip install mecab-python3
#     - Mac: Linuxと同じ(?)

# In[1]:


# !pip install mecab-python-windows  # Windows の場合のインストール
# !pip install mecab-python3  # Linux/Mac の場合のインストール


# ### パッケージの読み込み

# In[1]:


import MeCab


# ### 使い方

# In[2]:


# tagger = MeCab.Tagger()  # Windows の場合
tagger = MeCab.Tagger("-r /etc/mecabrc")  # Linux の場合


# In[3]:


print(tagger.parse("すもももももももものうち"))


# In[4]:


# 文章：「今日はきれいな虹が出た。」
with open("data/sample1/01.txt", encoding="utf_8") as fin:
    s = fin.read()
    print(tagger.parse(s))


# In[5]:


# 文章：日本国憲法前文(現代語)
with open("data/sample1/02.txt", encoding="utf_8") as fin:
    s = fin.read()
    print(tagger.parse(s))


# In[6]:


# 結果をくわしく見るには

with open("data/sample1/02.txt", encoding="utf-8") as fin:
    s = fin.read()
    node = tagger.parseToNode(s)
    
while node:
    print("%s\t%s" % (node.surface, node.feature))
    node = node.next


# ### 応用例：名詞だけを取り出して、それぞれの頻度を調べる

# In[7]:


from collections import Counter


# In[8]:


# 名詞だけを取り出して、それぞれの頻度を調べる
with open("data/sample1/02.txt", encoding="utf-8") as fin:
    s = fin.read()
    node = tagger.parseToNode(s)

l = []
while node:
    if node.feature.split(',')[0] == "名詞":
        l.append(node.surface)
    node = node.next
    
print(l)

freq = Counter(l) 
for k,v in sorted(freq.items(), key=lambda x:x[1], reverse=True):
    print("%s\t%d" % (k,v))


# ### 応用例：名詞、動詞、形容詞、助詞だけを選んで文書の「ダイジェスト」を作る

# In[9]:


# 名詞、動詞、形容詞、助詞だけを選んで文書の「ダイジェスト」を作る関数を定義

def digest_doc(filename):
    with open(filename, encoding="utf-8") as fin:
        s = fin.read()
        node = tagger.parseToNode(s)

    l = []
    while node:
        x = node.feature.split(',')[0]
        if x == "名詞" or x == "動詞" or x == "形容詞" or x == "助詞":
            l.append(node.feature.split(',')[6])  # 原形を用いる
        node = node.next

    return " ".join(l)


# In[10]:


d = digest_doc("data/sample1/02.txt")
d


# ### 応用例：ディレクトリ以下のすべての文書について処理する

# In[11]:


import glob


# In[12]:


docs = []
for fn in glob.glob("data/sample2/*"):
    print(fn)
    d = digest_doc(fn)
    docs.append(d)


# In[13]:


docs


# #### ディレクトリ以下のすべての文書について語とその頻度の表を作る
# メモ：テキスト解析で、語とその頻度の表はterm-frequency matrix と呼ばれている  
# 以下では機械学習の学習用パッケージ scikit-learn からテキスト解析のツールを用いる

# In[14]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


count_vec = CountVectorizer()

x = count_vec.fit_transform(np.array(docs))
# 疎な行列として扱われている
# print(type(X))

td = x.toarray()  # term-document matrix

# 出現したすべての語のリスト
terms = count_vec.get_feature_names()
print(terms)

# term-frequency matrix の次元 = 文書数 * 全語数
print(td.shape)

# term-frequency の中身
print(td)


# In[16]:


# pandas のデータフレームに変換する

df_td = pd.DataFrame(data=td, columns=terms)
df_td


# In[17]:


# 1番目の文書について、出現頻度によって語をソート
i = 0
df_td[i:i+1].sort_values(by=i, axis=1, ascending=False)


# In[18]:


# 2番目の文書について、出現頻度によって語をソート
i = 1
df_td[i:i+1].sort_values(by=i, axis=1, ascending=False)


# In[19]:


# 各文書について、頻度の合計を計算
df_td.sum(axis=1)


# In[20]:


# 各文書について、語ごとの出現確率

freqs = np.array(td, np.float)
freq_sums = np.array(df_td.sum(axis=1), np.float).reshape(2,1)  # For numpy's broadcast

probs = freqs / freq_sums

for i in range(probs.shape[1]):
    print("%s\t%f\t%f" % (terms[i],probs[0,i],probs[1,i]))


# In[21]:


# 各文書について，出現確率の合計は1になるはず
probs.sum(axis=1)

