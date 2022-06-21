#!/usr/bin/env python
# coding: utf-8

# In[34]:


import json
import jieba
import re
import pandas as pd
from harvesttext import HarvestText


# In[36]:


'''
将标签转为数字
neutral->0
positive->1
negative->2
'''
label2num = {'neutral':0,'positive':1,'negative':2, 0:0, 1:1, 2:2}


'''字典树结构，构建停用词词典'''
class Trie:
    def __init__(self):
        self.root = {}  # 用字典存储
        self.end_of_word = '#'   # 用#标志一个单词的结束
        
    def insert(self, word: str):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word

    # 查找一个单词是否完整的存在于字典树里，判断node是否等于#
    def search(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node

# In[37]:

'''基于路径文件构建停用词字典树'''
def get_stop_dic(file_path):
    stop_dic = Trie()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_dic.insert(line.strip())
    return stop_dic

# 构建停用词字典树
stop_dic = get_stop_dic('./data/stopwords.txt')


'''
数据处理
观察到语料库中为微博评论的形式，对于content中的内容去掉用户名、url等不必要的噪声词
'''
def read_json(file_dir):
    with open(file_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_dir, indent=1):
    with open(file_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=1, ensure_ascii=False))
        
def processing(content:str):
    # 数据清洗部分
    ht = HarvestText()
    
#     content = re.sub('\#.+?\#', ' ', content)           # 去除 #xxx# (微博话题等)
    content = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b/?', ' ', content)  # 去除url
    content = ht.clean_text(content, emoji=False) # 去除用户名等，繁体等转换处理
    
    # 分词
    words = [w for w in jieba.lcut(content) if w.isalpha() and not stop_dic.search(w)] # 去掉符号和停用词
    
    while '不' in words: # 对否定词`不`做特殊处理: 与其后面的词进行拼接
        index = words.index('不')
        if index == len(words) - 1:
            break
        words[index: index+2] = [''.join(words[index: index+2])]  # 合并为一个词语（列表切片赋值）
        
    return ' '.join(words) # 返回空格分隔的字符串

def clean_file(file_dir, clean_dir=None):
    data = read_json(file_dir)
    cleaned_data = []
    for i in range(len(data)):
        content = processing(data[i]['content'])
        label = label2num[data[i]['label']]
#         label = data[i]['label']
        cleaned_data.append({'id':data[i]['id'], 'content':content, 'label': label})
    if clean_dir is not None:
        filename = 'clean_' + file_dir.split('/')[-1]
        save_json(cleaned_data, clean_dir)
    final = pd.DataFrame(cleaned_data, columns=['id','content','label'])
    final['id'] = [i+1 for i in range(len(final))]
    return final

import matplotlib.pyplot as plt


'''
统计并可视化分词后切分长度分布直方图，以便选取合适的max_len
'''
def show_len(len_data):  
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制直方图
    plt.hist(len_data, 20, (0,100))
    plt.title('分词后长度直方图')
    plt.show()