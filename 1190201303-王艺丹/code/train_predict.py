#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from utils import clean_file, show_len, read_json, label2num
from LSTM import Sentiment, word2idx, prepare_data, text_to_array
from gensim.models import word2vec, keyedvectors
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[2]:


if __name__ == '__main__':
    # 准备训练数据
    df_train = clean_file('../data/train_data/train_data.json')
    df_train.head()
    
    # 统计分词后句子长度，选取合适的max_len作为lstm模型统一的输入维度
    lens = [len(w.split(' ')) for w in df_train['content']]
    show_len(lens)
    
    # 加载词向量模型
    w2v=keyedvectors.load_word2vec_format('../data/embedding/sgns.weibo.bin')
    # 建立映射
    w2id, embedding_weights = word2idx(w2v)

    # 根据直方图以及分位数长度选取的最终max_len
    max_len = 100
    # 根据训练集大小选取的最终batch_size
    batch_size = 128
    # 切分训练集测试集
    x_train, x_val, y_train, y_val = prepare_data(w2id, df_train, max_len)

    # 模型初始化
    ss = Sentiment(w2id=w2id, embedding_weights=embedding_weights, 
                   Embedding_dim = 300, 
                   max_len = max_len, 
                   labels_category = 3, 
                   batch_size = batch_size, 
                   units = 50, 
                   drop_out = 0.5, 
                   monitor = 'val_loss')

    # 模型训练
    ss.train(x_train,y_train, x_val ,y_val, n_epoch=100, model_path='../model/sentiment.h5')

    # 读入测试数据转为DataFrame格式用于模型预测
    predict_data = read_json('./data/test_data/test.json')
    predict_data = pd.DataFrame(predict_data,columns=['id','content'])
    predict_data.head()
    
    # 用模型对test数据进行分类预测并将结果写入csv文件
    label_pre = ss.predict('../model/sentiment.h5', predict_data)
    label_pre.to_csv('../result/1190201303-王艺丹.csv', index=0, header=0)

