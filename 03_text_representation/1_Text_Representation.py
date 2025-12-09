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
from fastai.metrics import perplexity
from paddlex.inference.models.common.tokenizer import vocab
from sklearn.feature_extraction.text import CountVectorizer  # 词袋模型工具
from gensim.models import Word2Vec  # Word2Vec模型

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🎉 工具箱准备完毕！让我们开始文本魔法之旅吧！")

clean_data = pd.read_csv("Tweets.csv")
print(clean_data.head())
print(clean_data.info())

sns.countplot(x="airline_sentiment", data=clean_data)
plt.title('航空情感分布')
plt.xlabel('airline_sentiment')
plt.ylabel('count')
plt.show()

# First of all let's drop the columns which we don't required

waste_col = [
    "tweet_id",
    "airline_sentiment_confidence",
    "negativereason",
    "negativereason_confidence",
    "airline",
    "airline_sentiment_gold",
    "name",
    "negativereason_gold",
    "retweet_count",
    "tweet_coord",
    "tweet_created",
    "tweet_location",
    "user_timezone",
]

data = clean_data.drop(waste_col, axis=1)

print(data.head())

def sentiment(x):
    if x == "positive":
        return 1
    elif x == "negative":
        return -1
    else:
        return 0

nltk.download('stopwords')

stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
# As this dataset is fetched from twitter so it has lots of people tag in tweets
# we will remove them
tags = r"@\w*"

def preprocess_text(sentence, stem=False):
    sentence = [re.sub(tags, "", sentence)]
    text = []
    for word in sentence:
        if word not in stopwords:
            if stem:
                # 启用词干提取，例如running -> run
                text.append(stemmer.stem(word).lower())
            else:
                text.append(word.lower())

    return tokenizer.tokenize(" ".join(text))

print(f"Orignal Text : {data.text[11]}")
print()
print(f"Preprocessed Text : {preprocess_text(data.text[11])}")

data.text = data.text.map(preprocess_text)
print(data.head())

# 第一关：One-Hot 编码
# this is an example vocabulary just to make concept clear
sample_vocab = ["the", "cat", "sat", "on", "mat", "dog", "run", "green", "tree"]
# data_vocab = set(sample_vocab)

# vocabulary of words present in dataset
data_vocab = []
for text in data.text:
    for word in text:
        if word not in data_vocab:
            data_vocab.append(word)

# function to return one-hot representation of passed text
def get_onehot_representation(text_to_onehot, vocab_for_onehot=None):
    if vocab_for_onehot is None:
        vocab_for_onehot = data_vocab
    onehot_encoded = []
    for word_to_onehot in text_to_onehot:
        temp = [0] * len(vocab_for_onehot)
        temp[vocab_for_onehot.index(word_to_onehot)] = 1
        onehot_encoded.append(temp)
    return onehot_encoded

print('One Hot Representation for sentence "the cat sat on the mat" :')
print(get_onehot_representation(["the", "cat", "sat", "on", "the", "cat"], sample_vocab))

print(f"Length of Vocabulary : {len(data_vocab)}")
print(f"Sample of Vocabulary : {data_vocab[302 : 312]}")

sample_one_hot_rep = get_onehot_representation(data.text[7], data_vocab)
print(f"Shapes of a single sentence : {np.array(sample_one_hot_rep).shape}")

# 句子的 one-hot 表示

# data.loc[:, 'one_hot_rep'] = data.loc[:, 'text'].map(get_onehot_representation)

# 如果您运行此单元，它将给您一个内存错误

print(data.head())


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

sample_bow = CountVectorizer(binary=True)

sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]

sample_bow.fit(sample_corpus)

def get_bow_representation(text):
    return sample_bow.transform(text)

print(f"Vacabulary mapping for given sample corpus : \n {sample_bow.vocabulary_}")
print(
    "\nBag of word Representation of sentence 'the the the the cat cat sat in the hat'"
)
print(get_bow_representation(["the the the the cat cat sat in the hat"]).toarray())

# generate bag of word representation for given dataset

bow = CountVectorizer()
bow_rep = bow.fit_transform(data.loc[:, "text"].astype("str"))

# intrested one can see vocabulary of given corpus by uncommenting below code line
# bow.vocabulary_
print(f"Shape of Bag of word representaion matrix : {bow_rep.toarray().shape}")


# 第三关：N-Grams词袋

# Bag of 1-gram (unigram)
from sklearn.feature_extraction.text import CountVectorizer

sample_boN = CountVectorizer(ngram_range=(1, 1))
sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]
sample_boN.fit(sample_corpus)

def get_bo_n_representation(text):
    return sample_boN.transform(text)

print(f"Unigram Vocabulary mapping for given sample corpus : \n {sample_boN.vocabulary_}")
print("\nBag of 1-gram (unigram) Representation of sentence 'the cat cat sat in the hat'")
print(get_bo_n_representation(["the cat cat sat in the hat"]).toarray())

# Bag of 2-gram (bigram)
sample_boN = CountVectorizer(ngram_range=(2, 2))
sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]
sample_boN.fit(sample_corpus)

def get_bo_n_representation(text):
    return sample_boN.transform(text)

print(f"Bigram Vocabulary mapping for given sample corpus : \n {sample_boN.vocabulary_}")
print("\nBag of 2-gram (bigram) Representation of sentence 'the cat cat sat in the hat'")
print(get_bo_n_representation(["the cat cat sat in the hat"]).toarray())


# Bag of 3-gram (trigram)
sample_boN = CountVectorizer(ngram_range=(3, 3))
sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]
sample_boN.fit(sample_corpus)

def get_bo_n_representation(text):
    return sample_boN.transform(text)


print(f"Trigram Vocabulary mapping for given sample corpus : \n {sample_boN.vocabulary_}")
print("\nBag of 3-gram (trigram) Representation of sentence 'the cat cat sat in the hat'")
print(get_bo_n_representation(["the cat cat sat in the hat"]).toarray())


# 第四关：TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
sample_corpus = ["the cat sat", "the cat sat in the hat", "the cat with the hat"]
tfidf_rep = tfidf.fit_transform(sample_corpus)
print(f"IDF Values for sample corpus : {tfidf.idf_}")

print("TF-IDF Representation for sentence 'the cat sat in the hat' :")
print(tfidf.transform(["the cat sat in the hat"]).toarray())


# 第五关：Word2vec

# 创建一些示例数据来训练Word2Vec模型
sample_sentences = [
    ['good', 'movie', 'great', 'acting'],
    ['bad', 'movie', 'terrible', 'acting'],
    ['computer', 'science', 'programming', 'python'],
    ['happy', 'feeling', 'good', 'great'],
    ['sad', 'feeling', 'bad', 'terrible'],
    ['love', 'romance', 'great', 'story'],
    ['hate', 'dislike', 'bad', 'terrible'],
    ['python', 'programming', 'computer', 'good'],
    ['excellent', 'great', 'amazing', 'good'],
    ['awful', 'terrible', 'horrible', 'bad'],
    ['technology', 'computer', 'science', 'innovative'],
    ['art', 'beautiful', 'creative', 'great']
]

# 训练Word2Vec模型
print("🚀 训练Word2Vec模型...")
Word2VecModel = Word2Vec(
    sentences=sample_sentences,
    vector_size=100,    # 词向量维度
    window=5,           # 上下文窗口大小
    min_count=1,        # 最小词频
    workers=4,          # 并行数
    sg=1                # 使用Skip-gram算法
)

print(f"✅ 模型训练完成！词汇表大小: {len(Word2VecModel.wv)}")

# 方法1: 使用Plotly进行交互式可视化
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

plotly_available = True

if plotly_available:
    def plot_embeddings_plotly(embeddings, words, title="交互式词嵌入可视化"):
        """使用Plotly创建交互式的embedding可视化"""

        # 将列表转换为numpy数组
        embeddings = np.array(embeddings)
        print(f"词向量矩阵形状: {embeddings.shape}")

        # 调整perplexity参数，确保小于样本数
        perplexity = min(15, embeddings.shape[0] - 1)
        if perplexity < 1:
            perplexity = 1

        # 使用t-SNE降为到2D
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne_2d.fit_transform(embeddings)

        # 创建DataFrame用于Plotly
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'word': words,
            'cluster': ['cluster_' + str(i // 30) for i in range(len(words))]
        })

        # 创建交互式散点图
        fig = px.scatter(df, x="x", y="y", color="cluster",
                         hover_name='word', title=title,
                         width=800, height=600)

        # 自定义悬停信息
        fig.update_traces(
            hovertemplate = '<b>%{hovertext}</b><br>' +
                          'X: %{x:.2f}<br>' +
                          'Y: %{y:.2f}<br>' +
                          '<extra></extra>',
            hovertext = df['word']
        )

        # 美化图表
        fig.update_layout(
            title_font_size=16,
            xaxis_title='t-SNE 维度 1',
            yaxis_title='t-SNE 维度 2',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    # 创建示例数据进行可视化
    if 'Word2VecModel' in locals():
        # 选择一些关键词
        sample_words = ["good", "bad", "great", "terrible", "computer", "science",
                       "python", "programming", "happy", "sad", "love", "hate"]

        # 获取对应的词向量
        sample_embeddings = []
        available_words = []

        for word in sample_words:
            try:
                embedding = Word2VecModel.wv[word]
                sample_embeddings.append(embedding)
                available_words.append(word)
            except KeyError:
                print(f"词 '{word}' 不在词汇表中")

        if sample_embeddings:
            # 创建交互式可视化
            fig = plot_embeddings_plotly(sample_embeddings, available_words)
            fig.show()
        else:
            print("没有找到可用的词汇")

    else:
        print("Word2VecModel 不可用，跳过Plotly可视化")
else:
    print("Plotly不可用，跳过交互式可视化")

# 方法2：使用TensorBoard Embedding Projector (官方方法)
import os
import tensorflow as tf
from tensorboard.plugins.projector import ProjectorConfig


def create_tensorboard_embeddings(embeddings, labels, log_dir="./embedding_logs"):
    """
    创建TensorBoard embedding projector可视化

    Args:
        embeddings: 词嵌入矩阵 (n_words, embedding_dim)
        labels: 词汇列表
        log_dir: 日志目录
    """

    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建metadata文件（词汇标签）
    metadata_path = os.path.join(log_dir, "metadata.tsv")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("Word\n")  # 列标题
        for label in labels:
            f.write(f"{label}\n")

    # 保存词向量到文件
    embeddings_path = os.path.join(log_dir, "embeddings.tsv")
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        for embedding in embeddings:
            f.write("\t".join(map(str, embedding)) + "\n")

    # 创建配置文件
    config = {
        "embeddings": [
            {
                "tensorName": "word_embeddings",
                "tensorShape": list(embeddings.shape),
                "tensorPath": embeddings_path,
                "metadataPath": metadata_path
            }
        ]
    }

    import json
    config_path = os.path.join(log_dir, "projector_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"TensorBoard embedding 文件已保存到: {log_dir}")
    print("运行以下命令启动TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    print("然后在浏览器中打开 http://localhost:6006 查看交互式embedding可视化")


# 如果有词向量模型，创建TensorBoard可视化
if 'Word2VecModel' in locals():
    try:
        # 选择前1000个最常用的词
        vocab_size = min(1000, len(Word2VecModel.wv.key_to_index))
        selected_words = list(Word2VecModel.wv.key_to_index.keys())[:vocab_size]
        selected_embeddings = np.array([Word2VecModel.wv[word] for word in selected_words])

        # 创建TensorBoard可视化
        create_tensorboard_embeddings(selected_embeddings, selected_words)

    except Exception as e:
        print(f"创建TensorBoard可视化时出错: {e}")
else:
    print("Word2VecModel 不可用，跳过TensorBoard可视化")