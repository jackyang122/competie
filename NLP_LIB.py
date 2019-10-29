import matplotlib.pyplot as plt
from string import punctuation
import seaborn as sns
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pandas as pd
import jieba
import logging
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,Softmax
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
import pandas as pd
import jieba
import  re

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.expand_frame_repr', False)
sns.set(style='white', context='notebook', palette='deep')

train=pd.read_csv('nlp_test/Train/Train_DataSet.csv')
lablel=pd.read_csv('nlp_test/Train/Train_DataSet_Label.csv')
test=pd.read_csv('nlp_test/Test/Test_DataSet.csv')
test_title=test['title']
#将文本和标签合并到一个数据集
train=pd.merge(train,lablel,on='id')

train['title']=train['title'].apply(lambda x:str(x))
train['words']=train['title'].apply(lambda x:jieba.lcut(x))
#构建特征工程
x=train['words']
y=train['label']

w2v=Word2Vec(size=100,min_count=5,window=5)
w2v.build_vocab(x)
w2v.train(x,total_examples=w2v.corpus_count,epochs=w2v.iter)
#
# #获取一个句子的向量
#
# def total_vec(words):
#     vec=np.zeros(300).reshape(1,300)
#     for word in words:
#         try:
#             vec+=w2v.wv[word].reshape(1,300)
#         except KeyError:
#             continue
#     return vec
#
# train_vec=np.concatenate(total_vec(words) for words in x)

def train_word2vec(x,save_path):

    print("开始训练词向量")
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v = Word2Vec(size=100, min_count=5, window=5)
    w2v.build_vocab(x)
    w2v.train(x, total_examples=w2v.corpus_count, epochs=w2v.iter)
    w2v.save(save_path)
    return w2v

model =  train_word2vec(x,'word2vec.model')

def generate_id2wec(word2vec_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2id.keys()}  # 词语的词向量
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights
def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)

def prepare_data(w2id,sentences,labels,max_len=200):
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)

w2id,embedding_weights = generate_id2wec(model)
x_train,y_trian, x_val , y_val = prepare_data(w2id,x,y,200)


class Sentiment:
    def __init__(self, w2id, embedding_weights, Embedding_dim, maxlen, labels_category):
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # input dim(140,100)
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_test, y_test, n_epoch=5):
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                       validation_data=(X_test, y_test))
        self.model.save('sentiment.h5')

    def predict(self, model_path, new_sen):
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)

senti = Sentiment(w2id,embedding_weights,100,200,3)

senti.train(x_train,y_trian, x_val ,y_val,1)

# label_dic = {2:"消极的",1:"中性的",0:"积极的"}
# for x in test_title:
#     pre = senti.predict("./sentiment.h5",x)
#     print("'{}'的情感是:\n{}".format(x,label_dic.get(pre)))