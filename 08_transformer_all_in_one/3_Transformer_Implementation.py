# 1. 包的导入
import re
import math
import importlib

import nltk
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 2. 使用Spacy构建分词器
# # 1. 安装最新版 spaCy（通常有预编译 wheel，无需编译）
# pip install spacy
#
# # 2. 下载中文模型（自动匹配版本）
# python -m spacy download zh_core_web_sm
class Tokenize(object):
    def __init__(self, lang):
        self.nlp = importlib.import_module(lang).load()

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[*\"“”\n\\…+\-/=()‘•:\[\]|’!;]", " ", str(sentence))
        sentence = re.sub(r" +", " ", sentence)
        sentence = re.sub(r"!+", "!", sentence)
        sentence = re.sub(r",+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

tokenize = Tokenize('zh_core_web_sm')
print(tokenize.tokenizer('你好，这里是中国。'))


# 3. Input Embedding
# 3.1. Token Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# 3.2. Positional Encoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # 根据pos和i创建一个常量pe矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 让embeddings vector相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到embedding中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x


# 4. Transformer Block
# ['我', '爱', '自然', '语言', '处理', <PAD>, <PAD>]
# [1, 1, 1, 1, 1, 0, 0 ]
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output



