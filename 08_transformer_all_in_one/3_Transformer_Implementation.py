# 1. 包的导入
import copy
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
# 4.1. Attention
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


# 4.2. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        # d_model = dk * head
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


# 4.3. Norm Layer
class NormLayer(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# 4.4. Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# 5. Encoder
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layer = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layer[i](x, mask)
        return self.norm(x)


# 6. Decoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """
        初始化解码器层

        参数:
            d_model: 模型的维度/特征数量
            heads: 多头注意力机制中的头数
            dropout: Dropout比率，用于防止过拟合
        """
        super().__init__()

        # 规范化层
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)

        # dropout层
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # 第一个子层：带掩码的多头自注意力
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)
        # 第二个子层：多头编码器-解码器注意力
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)
        # 第三个子层：前馈神经网络
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, encoder_outputs, src_mask, trg_mask):
        """
        前向传播函数

        参数:
            x: 目标序列的嵌入表示，形状为(batch_size, tgt_seq_len, d_model)
            encoder_outputs: 编码器的输出，形状为(batch_size, src_seq_len, d_model)
            src_mask: 源序列的掩码，用于忽略填充位置
            trg_mask: 目标序列的掩码，防止解码器注意到后续位置

        返回:
            解码器层的输出
        """
        # 第一个子层：自注意力层
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))

        # 第二个子层：编码器-解码器注意力层
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_outputs, encoder_outputs, src_mask))

        # 第三个子层：前馈网络
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        return x

class Decoder(nn.Module):
    """
    Transformer解码器类
    包含多个解码器层的堆叠,实现目标序列的解码过程

    参数:
        vocab_size: 目标语言词表大小
        d_model: 模型维度,即词嵌入和各层的特征维度
        N: 解码器层的数量
        heads: 多头注意力机制中的头数
        dropout: Dropout比率,用于防止过拟合
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N                                          # 解码器层数量
        self.embed = Embedding(vocab_size, d_model)         # 词嵌入层
        self.pe = PositionalEncoder(d_model)                # 位置编码器
        # 克隆N个解码器层
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = NormLayer(d_model)                      # 最后的层归一化

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """
        解码器前向传播

        参数:
            trg: 目标序列输入张量
            e_outputs: 编码器的输出
            src_mask: 源序列的掩码,用于处理填充位置
            trg_mask: 目标序列的掩码,用于防止解码时看到未来位置

        返回:
            经过所有解码器层处理后的输出
        """
        # 1. 词嵌入
        x = self.embed(trg)
        # 2. 位置编码
        x = self.pe(x)
        # 3. 依次通过N个解码器层
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        # 4. 最后进行层归一化
        return self.norm(x)


# 7. Transformer
class Transformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

# 创建用于训练和评估的辅助函数
def create_masks(src, trg):
    # 创建源序列填充掩码
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # 创建目标序列掩码
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(1).unsqueeze(2)

        # 创建目标序列前瞻掩码（look-ahead mask）
        size = trg.size()
        nopeak_mask = torch.triu(torch.ones(1, size, size), diagonal=1) == 0
        nopeak_mask = nopeak_mask.to(trg.device)

        # 组合填充掩码和前瞻掩码
        trg_mask = trg_mask & nopeak_mask
        return src_mask, trg_mask

    return src_mask, None


# 简单的训练函数
def train_model(model, iterator, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        # 创建掩码
        src_mask, trg_mask = create_masks(src, trg[:, :-1])

        optimizer.zero_grad()

        # 前向传播
        output = model(src, trg[:, :-1], src_mask, trg_mask)

        # 计算损失
        loss = criterion(output.contiguous().view(-1, output.size(-1)),
                         trg[:, 1:].contiguous().view(-1))

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 更新参数
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 创建掩码
            src_mask, trg_mask = create_masks(src, trg[:, :-1])

            # 前向传播
            output = model(src, trg[:, :-1], src_mask, trg_mask)

            # 计算损失
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                             trg[:, 1:].contiguous().view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 示例：简单的中英文翻译演示
def translate_sentence(model, sentence, src_field, trg_field, device, max_len=50):
    """
    使用训练好的模型翻译一个句子

    参数:
        model: 训练好的Transformer模型
        sentence: 要翻译的句子(字符串或者标记列表)
        src_field: 源语言的Field对象，提供词汇表
        trg_field: 目标语言的Field对象，提供词汇表
        device: 计算设备(CPU或GPU)
        max_len: 生成的最大长度

    返回:
        翻译后的句子(标记列表)
    """
    model.eval()

    if isinstance(sentence, str):
        tokens = [token.lower() for token in tokenize.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 添加首尾标记
    tokens = ['<sos>'] + tokens + ['<eos>']

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # 创建源句子的掩码
    src_mask = (src_tensor != src_field.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 开始生成翻译，从<sos>标记开始
    trg_indexes = [trg_field.vocab.stoi['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 创建目标句子的掩码
        trg_mask = create_masks(src_tensor, trg_tensor)[1]

        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            output = model.out(output)

        # 获取下一个词的预测
        pred_token = output.argmax(2)[:, -1].item()

        # 添加到预测序列
        trg_indexes.append(pred_token)

        # 如果预测到了<eos>，终止生成
        if pred_token == trg_field.vocab.stoi['<eos>']:
            break

    # 转换回标记
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 返回不包括<sos>和<eos>的结果
    return trg_tokens[1:-1]


# 示例使用
def translate_example():
    """
    演示如何使用Transformer模型进行简单的中英翻译
    """
    # 这里只是展示使用方法，实际使用时需要先构建和训练模型
    # 假设我们已经有了训练好的模型和相关的Field对象

    # 模型参数
    SRC_VOCAB_SIZE = 10000
    TRG_VOCAB_SIZE = 10000
    D_MODEL = 512
    N_LAYERS = 6
    HEADS = 8
    DROPOUT = 0.1

    # 初始化模型
    model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, N_LAYERS, HEADS, DROPOUT)

    # 将模型加载到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 加载预训练权重
    # model.load_state_dict(torch.load('transformer_model.pt'))

    # 准备示例句子
    src_sentence = "我喜欢自然语言处理"

    # 翻译句子
    # translated_tokens = translate_sentence(model, src_sentence, SRC, TRG, device)

    # 显示翻译结果
    # print(f"源句子: {src_sentence}")
    # print(f"翻译结果: {' '.join(translated_tokens)}")

    # 由于我们没有实际的训练数据和模型，这里只是展示流程
    print("这是Transformer模型翻译的演示函数")
    print("实际使用需要先构建词表和训练模型")

# 调用示例函数
# translate_example()


# 简单的实际演示
import io
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 定义一些简单的中英平行语料
TRAIN_DATA = [
    ("我喜欢自然语言处理", "I like natural language processing"),
    ("深度学习很有趣", "Deep learning is interesting"),
    ("Transformer模型很强大", "Transformer model is powerful"),
    ("机器翻译是自然语言处理的重要应用", "Machine translation is an important application of NLP"),
    ("我们正在学习Transformer", "We are learning about Transformer"),
    ("人工智能正在改变世界", "Artificial intelligence is changing the world"),
    ("神经网络需要大量数据", "Neural networks need a lot of data"),
    ("注意力机制是核心创新", "Attention mechanism is a core innovation"),
    ("自监督学习是重要趋势", "Self-supervised learning is an important trend")
]

# 定义特殊标记
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# 使用我们之前定义的分词器
chinese_tokenizer = tokenize.tokenizer
english_tokenizer = lambda x: x.lower().split()

# 构建词表
def yield_tokens(data_iter, tokenizer, index):
    for data_sample in data_iter:
        yield tokenizer(data_sample[index])

def build_vocabulary(data_iter, tokenizer, index):
    vocab = build_vocab_from_iterator(
        yield_tokens(data_iter, tokenizer, index),
        special_first=True
    )
    # Manually add special symbols
    for symbol in special_symbols:
        vocab.append_token(symbol)
    vocab.set_default_index(UNK_IDX)
    return vocab

# 构建中文和英文词表
cn_vocab = build_vocabulary(TRAIN_DATA, chinese_tokenizer, 0)
en_vocab = build_vocabulary(TRAIN_DATA, english_tokenizer, 0)

# 将文本转换为数字索引
def text_transform(text, tokenizer, vocab):
    tokens = tokenizer(text)
    # 处理未知词汇，如果词汇不在词表中，使用UNK_IDX
    token_indices = []
    for token in tokens:
        try:
            idx = vocab[token]
        except KeyError:
            idx = UNK_IDX
        token_indices.append(idx)
    tokens = [SOS_IDX] + token_indices + [EOS_IDX]
    return torch.tensor(tokens)

# 定义数据加载和批处理函数
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_text, tgt_text in batch:
        src_tensor = text_transform(src_text, chinese_tokenizer, cn_vocab)
        tgt_tensor = text_transform(tgt_text, english_tokenizer, en_vocab)
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return src_batch, tgt_batch

# 创建数据加载器
train_dataloader = DataLoader(TRAIN_DATA, batch_size=3, collate_fn=collate_fn)

# 初始化模型
SRC_VOCAB_SIZE = len(cn_vocab)
TRG_VOCAB_SIZE = len(en_vocab)
D_MODEL = 128  # 小一点更快训练
N_LAYERS = 2
HEADS = 4
DROPOUT = 0.1

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, N_LAYERS, HEADS, DROPOUT).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 训练函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    losses = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 创建掩码
        src_mask, tgt_mask = create_masks(src, tgt[:, :-1])

        # 前向传播
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)

        # 计算损失
        loss = criterion(output.reshape(-1, TRG_VOCAB_SIZE), tgt[:, 1:].reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(dataloader)

# 简单的评估和翻译示例
def evaluate_sample(model, src_text):
    model.eval()
    src_tensor = text_transform(src_text, chinese_tokenizer, cn_vocab).unsqueeze(0).to(device)
    src_mask = (src_tensor != PAD_IDX).unsqueeze(1).unsqueeze(2).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)

    trg_idx = [SOS_IDX]
    max_len = 50

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx]).to(device)
        trg_mask = torch.ones((1, trg_tensor.size(1), trg_tensor.size(1))).to(device)

        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_output, src_mask, trg_mask)
            prediction = model.out(output)

        next_word = prediction[:, -1].argmax(dim=1).item()
        trg_idx.append(next_word)

        if next_word == EOS_IDX:
            break

    # 将索引转换回文本
    output_text = ' '.join([en_vocab.get_itos()[i] for i in trg_idx[1:-1]])
    return output_text


# 进行一次简单的训练和测试（实际训练需要多个周期）
def demo():
    print("开始简单训练示例...")
    try:
        # 简单训练几轮
        for epoch in range(5):
            loss = train_epoch(model, train_dataloader, optimizer, criterion)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        # 测试几个例子
        test_sentences = ["我喜欢自然语言处理", "深度学习很有趣", "注意力机制是核心创新"]

        print("\n翻译测试:")
        for sentence in test_sentences:
            translation = evaluate_sample(model, sentence)
            print(f"源文本: {sentence}")
            print(f"翻译结果: {translation}")
            print()
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("这是一个简化的示例，可能需要更多调整才能正常运行")

print("开始运行演示...")
# 仅作为示例，实际运行可能需要更多配置
demo()
