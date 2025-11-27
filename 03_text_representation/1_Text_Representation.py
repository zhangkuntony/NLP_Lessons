# 🔧 基础工具
import os          # 操作系统接口
import random      # 随机数生成器
import re          # 正则表达式，用于文本清理

# 📊 数据处理和分析的"瑞士军刀"
import numpy as np           # 数值计算库
import pandas as pd          # 数据分析神器

# 🎨 让数据"现形"的可视化工具
import matplotlib.pyplot as plt  # 基础画图工具
import seaborn as sns           # 更美观的统计图表

# 🔤 文本处理专业工具
import nltk                     # 自然语言工具包
from paddlex.inference.models.common.tokenizer import vocab
from sklearn.feature_extraction.text import CountVectorizer  # 词袋模型工具

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🎉 工具箱准备完毕！让我们开始文本魔法之旅吧！")
#
# clean_data = pd.read_csv("Tweets.csv")
# print(clean_data.head())
# print(clean_data.info())
#
# sns.countplot(x="airline_sentiment", data=clean_data)
# # plt.title('航空情感分布')
# # plt.xlabel('airline_sentiment')
# # plt.ylabel('count')
# # plt.show()
#
# # First of all let's drop the columns which we don't required
#
# waste_col = [
#     "tweet_id",
#     "airline_sentiment_confidence",
#     "negativereason",
#     "negativereason_confidence",
#     "airline",
#     "airline_sentiment_gold",
#     "name",
#     "negativereason_gold",
#     "retweet_count",
#     "tweet_coord",
#     "tweet_created",
#     "tweet_location",
#     "user_timezone",
# ]
#
# data = clean_data.drop(waste_col, axis=1)
#
# print(data.head())
#
# def sentiment(x):
#     if x == "positive":
#         return 1
#     elif x == "negative":
#         return -1
#     else:
#         return 0
#
# nltk.download('stopwords')
#
# stopwords = stopwords.words('english')
# stemmer = SnowballStemmer('english')
# tokenizer = RegexpTokenizer(r'\w+')
# # As this dataset is fetched from twitter so it has lots of people tag in tweets
# # we will remove them
# tags = r"@\w*"
#
# def preprocess_text(sentence, stem=False):
#     sentence = [re.sub(tags, "", sentence)]
#     text = []
#     for word in sentence:
#         if word not in stopwords:
#             if stem:
#                 # 启用词干提取，例如running -> run
#                 text.append(stemmer.stem(word).lower())
#             else:
#                 text.append(word.lower())
#
#     return tokenizer.tokenize(" ".join(text))
#
# print(f"Orignal Text : {data.text[11]}")
# print()
# print(f"Preprocessed Text : {preprocess_text(data.text[11])}")
#
# data.text = data.text.map(preprocess_text)
# print(data.head())

# # 第一关：One-Hot 编码
# # this is an example vocabulary just to make concept clear
# sample_vocab = ["the", "cat", "sat", "on", "mat", "dog", "run", "green", "tree"]
# # data_vocab = set(sample_vocab)
#
# # vocabulary of words present in dataset
# data_vocab = []
# for text in data.text:
#     for word in text:
#         if word not in data_vocab:
#             data_vocab.append(word)
#
# # function to return one-hot representation of passed text
# def get_onehot_representation(text_to_onehot, vocab_for_onehot=None):
#     if vocab_for_onehot is None:
#         vocab_for_onehot = data_vocab
#     onehot_encoded = []
#     for word_to_onehot in text_to_onehot:
#         temp = [0] * len(vocab_for_onehot)
#         temp[vocab_for_onehot.index(word_to_onehot)] = 1
#         onehot_encoded.append(temp)
#     return onehot_encoded
#
# print('One Hot Representation for sentence "the cat sat on the mat" :')
# print(get_onehot_representation(["the", "cat", "sat", "on", "the", "cat"], sample_vocab))
#
# print(f"Length of Vocabulary : {len(data_vocab)}")
# print(f"Sample of Vocabulary : {data_vocab[302 : 312]}")
#
# sample_one_hot_rep = get_onehot_representation(data.text[7], data_vocab)
# print(f"Shapes of a single sentence : {np.array(sample_one_hot_rep).shape}")
#
# # 句子的 one-hot 表示
#
# # data.loc[:, 'one_hot_rep'] = data.loc[:, 'text'].map(get_onehot_representation)
#
# # 如果您运行此单元，它将给您一个内存错误
#
# print(data.head())


# 第二关：词袋模型（BOW）

from sklearn.feature_extraction.text import CountVectorizer

sample_bow = CountVectorizer()

sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]

sample_bow.fit(sample_corpus)

def get_bow_representation(text):
    return sample_bow.transform(text)

print(f"Vocabulary mapping for given sample corpus : \n {sample_bow.vocabulary_}")
print(f"Sorted vocabulary (by index): \n{sorted(sample_bow.vocabulary_.items(), key=lambda x: x[1])}")
print("\nBag of word Representation of sentence 'the cat cat sat in the hat'")
print(get_bow_representation(["the cat cat sat in the hat"]).toarray())