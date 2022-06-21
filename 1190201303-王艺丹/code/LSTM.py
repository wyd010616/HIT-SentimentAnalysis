#!/usr/bin/env python
# coding: utf-8


# In[1]:


import pandas as pd
import numpy as np
import jieba
from utils import processing
import pdb
from gensim.models import word2vec, keyedvectors
from gensim.corpora.dictionary import Dictionary
from keras.utils import np_utils


# In[2]:


'''
读入训练好的词向量模型
建立｛word：索引｝映射 w2id
建立｛索引：词向量｝映射 embedding_weights
则根据 word 查询 索引，根据索引查询词向量
'''
def word2idx(model):    
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.key_to_index.keys(), allow_update=True)

    #  freqxiao10->0 所以k+1
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号 ｛word：索引｝
    w2vec = {word: model[word] for word in w2id.keys()}  # 词语的词向量 ｛word：词向量｝
    
    # 获取词的长度
    n_vocabs = len(w2id) + 1
    
    # 初始化一个空白词向量
    embedding_weights = np.zeros((n_vocabs, 300))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id, embedding_weights


# In[3]:


# 将文本转为索引数字模式
def text_to_array(w2id, sents): 
    sents_array = []
    for sent in sents:
        new_sent = [w2id.get(word,0) for word in sent.split(' ')]   # 词转索引数字
        # pdb.set_trace()
        sents_array.append(new_sent)
    return np.array(sents_array)



# ## 构造并切分数据集与测试集

# In[4]:


from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


# In[5]:


def prepare_data(w2id, df ,max_len):
    # 切分训练集和测试集样本
    sents = df['content']
    labels = df['label']
    X_train, X_val, y_train, y_val = train_test_split(sents, labels, test_size=0.2)

    # 按照词典将文本转化为索引文本，每个索引对应词向量
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
 
    # 填充文本至统一长度max_len处理变长序列
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    
    # to_categorical()独热编码
    return np.array(X_train) ,np.array(X_val), np_utils.to_categorical(y_train), np_utils.to_categorical(y_val)


# # Step3 搭建LSTM模型

# In[6]:


from keras import Sequential, callbacks
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Activation, Softmax


# In[7]:


class Sentiment:
    # 初始化
    def __init__(self, w2id, embedding_weights, Embedding_dim, max_len, labels_category, batch_size, units, drop_out, monitor):
        self.units = units
        self.monitor = monitor
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = max_len
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.model = self.build_model()
        self.callback = callbacks.EarlyStopping(monitor=self.monitor, patience=10, verbose=2, mode='auto', restore_best_weights=True)
        
      
    # 模型搭建，返回一个model
    def build_model(self):
        model = Sequential()
        model.add(Embedding(output_dim = self.Embedding_dim, input_dim=len(self.vocab)+1, weights=[self.embedding_weights], input_length=self.maxlen))      
        model.add(Bidirectional(LSTM(self.units),merge_mode='concat'))
        model.add(Dropout(self.drop_out))
        model.add(Dense(self.labels_category, activation='softmax'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self,X_train, y_train,X_test, y_test, n_epoch, model_load=0, model_path=None):
        if model_load:
            model = self.model
            model.load_weights(model_path)
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=n_epoch, validation_data=(X_test, y_test))
            model.save(model_path)
        else:
            self.model.fit(X_train, y_train, self.batch_size, epochs=n_epoch, validation_data=(X_test, y_test),callbacks=self.callback)
            self.model.save(model_path)
    
 
    def predict(self, model_path, df_test):
        model = self.model
        model.load_weights(model_path)
        pre_list = []
        for i in range(len(df_test)):
            sent = processing(df_test.iloc[i]['content']) # 获得分词去停用词处理后的空格分隔的字符串
            sen2id =[self.vocab.get(word, 0) for word in sent.split(' ')]
            sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
            res = model.predict(sen_input)[0]
            pre = np.argmax(res)
            pre_list.append((df_test.iloc[i]['id'], pre))
        return pd.DataFrame(pre_list, columns=['id', 'label'])

