#!/usr/bin/env python
# coding: utf-8

# # <center>ベイズ推論の例: 簡単なテキスト解析「森鴎外か夏目漱石か」</center>
# Author: 藤原 義久 <yoshi.fujiwara@gmail.com>    
# Data: ディレクトリ"data"以下  
# - [青空文庫](https://www.aozora.gr.jp/)にある森鴎外と夏目漱石の作品抜粋  
#     > data/docs/01.txt | 森鴎外『雁』  
#     > data/docs/02.txt | 森鴎外『かのように』  
#     > data/docs/03.txt | 森鴎外『鶏』  
#     > data/docs/04.txt | 森鴎外『ヰタ・セクスアリス』  
#     > data/docs/05.txt | 夏目漱石『永日小品』  
#     > data/docs/06.txt | 夏目漱石『硝子戸の中』  
#     > data/docs/07.txt | 夏目漱石『思い出す事など』  
#     > data/docs/08.txt | 夏目漱石『夢十夜』  

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

# ### パッケージの読み込み

# In[1]:


import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import glob


# ### 準備：形態素解析

# In[2]:


# tagger = MeCab.Tagger()  # Windows の場合
tagger = MeCab.Tagger("-r /etc/mecabrc")  # Linux の場合


# ### 準備：名詞、動詞、形容詞、助詞だけを選んで文書の「ダイジェスト」を作る

# In[3]:


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


# ### すべての文書について処理

# In[4]:


docs = []
for fn in sorted(glob.glob("data/docs/*")):
    print(fn)
    d = digest_doc(fn)
    docs.append(d)


# #### すべての文書について語とその頻度の表を作る
# メモ：テキスト解析で、語とその頻度の表はterm-frequency matrix と呼ばれている  
# 以下では機械学習の学習用パッケージ scikit-learn からテキスト解析のツールを用いる

# In[5]:


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


# In[6]:


# pandas のデータフレームに変換する

df_td = pd.DataFrame(data=td, columns=terms)
df_td


# In[7]:


# 1番目の文書(森鴎外)について、出現頻度によって語をソート
i = 0
df_td[i:i+1].sort_values(by=i, axis=1, ascending=False)


# In[8]:


# 5番目の文書(夏目漱石)について、出現頻度によって語をソート
i = 4
df_td[i:i+1].sort_values(by=i, axis=1, ascending=False)


# In[9]:


# 各文書について、頻度の合計を計算
df_td.sum(axis=1)


# ### 「学習用」データを作る

# In[10]:


df_td_train = df_td.iloc[[0,1,4,5]]  # 0,1(森鴎外); 4,5(夏目漱石)
df_td_train


# #### 各作家について，語ごとの出現頻度を合計する

# In[11]:


# 森鴎外
x = df_td_train[0:2].sum()
freq_ogai = pd.DataFrame(x,columns=["ogai"]).transpose()
freq_ogai


# In[12]:


# 夏目漱石
x = df_td_train[2:4].sum()
freq_soseki = pd.DataFrame(x,columns=["soseki"]).transpose()
freq_soseki


# In[13]:


# データフレームを連結する
freqs_df = pd.concat([freq_ogai, freq_soseki])
freqs_df


# In[14]:


# 各文書について、語ごとの出現確率を計算
# (1) 頻度が0の場合は確率も0とするナイーブな方法

freqs = np.array(freqs_df, np.float)
freq_sums = np.array(freqs_df.sum(axis=1), np.float).reshape(2,1)  # For numpy's broadcast

probs = freqs / freq_sums

for i in range(probs.shape[1]):
    print("%s\t%f\t%f" % (terms[i],probs[0,i],probs[1,i]))


# In[15]:


# 各文書について、語ごとの出現確率を計算
# (2) 頻度が0の場合は1として扱って確率は0にならないようにする方法

freqs = np.array(freqs_df, np.float)
freq_sums = np.array(freqs_df.sum(axis=1), np.float).reshape(2,1)  # For numpy's broadcast

probs = (freqs + 1.0) / (freq_sums + len(terms))

for i in range(probs.shape[1]):
    print("%s\t%f\t%f" % (terms[i],probs[0,i],probs[1,i]))


# In[16]:


# 各文書について，出現確率の合計は1になるはず
probs.sum(axis=1)


# ### 「テスト用」データを作る

# In[17]:


df_td_test = df_td.iloc[[2,3,6,7]]
df_td_test


# In[18]:


freqs = np.array(df_td_test, np.float)
freqs.shape


# ### 尤度(likelihood)の対数をそれぞれのモデル(作家)の場合に計算する

# In[19]:


ll = np.dot(freqs, np.log(probs.T))
ll


# #### テスト用の各文書について，どちらのモデル(作家)があてはまるか

# In[20]:


writers = ["森鴎外","夏目漱石"]
for k in np.argmax(ll,axis=1):
    print(writers[k])

