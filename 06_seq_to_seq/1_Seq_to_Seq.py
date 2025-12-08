# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import time

# 设置中文字体以便matplotlib显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("✅ 所有库导入成功！")
print(f"🔥 PyTorch版本: {torch.__version__}")
print(f"🎯 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# 动手实现：构建一个Seq2Seq模型
# 4.1 准备数据：使用英中翻译数据集
# 从cmn.txt文件读取英中翻译数据集
def load_cmn_data(file_path, max_samples=1000):
    """
    从cmn.txt文件加载英中翻译数据
    Args:
        file_path: cmn.txt文件路径
        max_samples: 最大样本数量（为了演示，限制数据量）
    Returns:
        list: 英中句子对列表
    """
    data_pairs = []

    print(f"📁 正在加载数据文件: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"📊 文件总行数: {len(lines):,}")

        for i, line in enumerate(lines[:max_samples]):
            if i % 10000 == 0:
                print(f"   处理进度: {i:,}/{min(max_samples, len(lines)):,}")

            # 解析每一行：英文\t中文\t版权信息
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                en_sentence = parts[0].strip().lower()              # 英文句子，转小写
                zh_sentence = parts[1].strip()                      # 中文句子

                # 简单过滤：只保留长度适中的句子
                if len(en_sentence.split()) <= 10 and len(zh_sentence.split()) <= 20:
                    data_pairs.append((en_sentence, zh_sentence))

    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return []

    print(f"✅ 数据加载完成！共加载 {len(data_pairs):,} 个句子对")
    return data_pairs

# 加载cmn.txt数据
cmn_file_path = "cmn.txt"
raw_data = load_cmn_data(cmn_file_path, max_samples=2000)

print(f"\n📝 数据样例:")
if raw_data:
    for i, (en, zh) in enumerate(raw_data[:10]):
        print(f"{i + 1:2d}. 英文: '{en}' → 中文: '{zh}'")

    print(f"\n📈 数据统计:")
    en_lengths = [len(sent.split()) for sent, _ in raw_data]
    zh_lengths = [len(sent) for _, sent in raw_data]

    print(f"   英文句子长度: 最短 {min(en_lengths)}, 最长 {max(en_lengths)}, 平均 {sum(en_lengths)/len(en_lengths):.1f}")
    print(f"   中文句子长度: 最短 {min(zh_lengths)}, 最长 {max(zh_lengths)}, 平均 {sum(zh_lengths)/len(zh_lengths):.1f}")
else:
    print("❌ 未能加载数据，使用备用数据集")
    # 备用数据集：基础英中翻译句子
    raw_data = [
        ("hello", "你好"),
        ("goodbye", "再见"),
        ("thank you", "谢谢"),
        ("how are you", "你好吗"),
        ("good morning", "早上好"),
        ("good night", "晚安"),
        ("i love you", "我爱你"),
        ("what is your name", "你叫什么名字"),
        ("nice to meet you", "很高兴见到你"),
        ("see you later", "再见"),
        ("excuse me", "不好意思"),
        ("i am sorry", "对不起"),
        ("yes", "是的"),
        ("no", "不是"),
        ("please", "请"),
        ("where are you from", "你来自哪里"),
        ("i am from china", "我来自中国"),
        ("do you speak english", "你会说英语吗"),
        ("i don't understand", "我不明白"),
        ("can you help me", "你能帮助我吗"),
        ("i am hungry", "我饿了"),
        ("i am thirsty", "我渴了"),
        ("how much is it", "多少钱"),
        ("where is the bathroom", "洗手间在哪里"),
    ]
    print(f"📊 备用数据集大小: {len(raw_data)} 个句子对")

# 构建词汇表类 - 这是NLP任务的基础工具
class Vocabulary:
    def __init__(self):
        self.PAD_TOKEN = 'PAD'                  # 填充符号
        self.SOS_TOKEN = 'SOS'                  # 句子开始符号
        self.EOS_TOKEN = 'EOS'                  # 句子结束符号
        self.UNK_TOKEN = 'UNK'                  # 未知词符号

        # 词汇表字典： word -> index
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }

        # 反向字典： index -> word
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add_word(self, word):
        """向词汇表添加新词"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def add_sentence(self, sentence):
        """向词汇表添加整个句子的词汇"""
        for word in sentence.split():
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def encode_sentence(self, sentence, add_eos=True):
        """将句子转换为索引序列"""
        indices = []
        for word in sentence.split():
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])

        if add_eos:
            indices.append(self.word2idx[self.EOS_TOKEN])

        return indices

    def decode_sentence(self, indices):
        """将索引序列转换回句子"""
        words = []
        for idx in indices:
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            if idx == self.word2idx[self.PAD_TOKEN]:
                continue
            words.append(self.idx2word[idx])
        return ' '.join(words)

# 创建英文和中文词汇表
en_vocab = Vocabulary()
zh_vocab = Vocabulary()

# 构建词汇表
for en_sentence, zh_sentence in raw_data:
    en_vocab.add_sentence(en_sentence)
    # 中文按字符分割（每个汉字作为一个词）
    zh_words = ' '.join(list(zh_sentence))
    zh_vocab.add_sentence(zh_words)

print(f"📚 英文词汇表大小: {len(en_vocab)}")
print(f"📚 中文词汇表大小: {len(zh_vocab)}")

# 展示一些词汇
print(f"\n🔤 英文词汇示例: {list(en_vocab.word2idx.keys())[:15]}")
print(f"🔤 中文词汇示例: {list(zh_vocab.word2idx.keys())[:15]}")

# 特殊处理：为中文词汇表添加字符级别的编码解码方法
class ChineseVocabulary(Vocabulary):
    def add_sentence(self, sentence):
        """中文句子按字符添加到词汇表"""
        for char in sentence:
            if char.strip():                # 忽略空格
                self.add_word(char)

    def encode_sentence(self, sentence, add_eos=True):
        """将中文句子转换为字符索引序列"""
        indices = []
        for char in sentence:
            if char.strip():                # 忽略空格
                if char in self.word2idx:
                    indices.append(self.word2idx[char])
                else:
                    indices.append(self.word2idx[self.UNK_TOKEN])

        if add_eos:
            indices.append(self.word2idx[self.EOS_TOKEN])

        return indices

    def decode_sentence(self, indices):
        """将字符索引序列转换回中文句子"""
        chars = []
        for idx in indices:
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            if idx == self.word2idx[self.PAD_TOKEN]:
                continue
            chars.append(self.idx2word[idx])

        return ''.join(chars)

# 重新创建中文词汇表
zh_vocab = ChineseVocabulary()
for en_sentence, zh_sentence in raw_data:
    zh_vocab.add_sentence(zh_sentence)

print(f"\n📚 更新后的中文词汇表大小: {len(zh_vocab)}")
print(f"🔤 中文字符示例: {list(zh_vocab.word2idx.keys())[:20]}")


# tokenize (分词)
# 创建数据集类
class TranslationDataset(Dataset):
    def __init__(self, data_pairs, src_vocab, tgt_vocab, max_length=20):
        """
        翻译数据集
        Args:
            data_pairs: 句子对列表 [(src_sentence, tgt_sentence), ...]
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_length: 最大序列长度
        """
        self.data_pairs = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data_pairs[idx]

        # 编码源句子
        src_indices = self.src_vocab.encode_sentence(src_sentence, add_eos=True)

        # 编码目标句子（用于训练的输入，需要添加SOS）
        tgt_input_indices = [self.tgt_vocab.word2idx[self.tgt_vocab.SOS_TOKEN]] + \
            self.tgt_vocab.encode_sentence(tgt_sentence, add_eos=False)

        # 编码目标句子（用于计算损失的标签，需要添加EOS）
        tgt_output_indices = self.tgt_vocab.encode_sentence(tgt_sentence, add_eos=True)

        return {
            'src': src_indices,
            'tgt_input': tgt_input_indices,
            'tgt_output': tgt_output_indices,
            'src_text': src_sentence,
            'tgt_text': tgt_sentence
        }

def collate_fn(batch):
    """自定义的批处理函数，用于处理不同长度的序列"""

    # 获取批次中每个样本的数据
    src_sequences = [item['src'] for item in batch]
    tgt_input_sequences = [item['tgt_input'] for item in batch]
    tgt_output_sequences = [item['tgt_output'] for item in batch]

    # 填充序列到相同长度
    src_padded = pad_sequences(src_sequences, en_vocab.word2idx[en_vocab.PAD_TOKEN])
    tgt_input_padded = pad_sequences(tgt_input_sequences, zh_vocab.word2idx[zh_vocab.PAD_TOKEN])
    tgt_output_padded = pad_sequences(tgt_output_sequences, zh_vocab.word2idx[zh_vocab.PAD_TOKEN])

    return {
        'src': torch.tensor(src_padded, dtype=torch.long),
        'tgt_input': torch.tensor(tgt_input_padded, dtype=torch.long),
        'tgt_output': torch.tensor(tgt_output_padded, dtype=torch.long),
        'src_text': [item['src_text'] for item in batch],
        'tgt_text': [item['tgt_text'] for item in batch]
    }

def pad_sequences(sequences, pad_token):
    """将序列填充到相同长度"""
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        padded_seq = seq + [pad_token] * (max_length - len(seq))
        padded_sequences.append(padded_seq)

    return padded_sequences

# 创建数据集
dataset = TranslationDataset(raw_data, en_vocab, zh_vocab)

# 创建数据加载器
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print(f"📦 数据集创建完成！")
print(f"📊 数据集大小: {len(dataset)}")
print(f"🔄 批次大小: {batch_size}")

# 测试一个批次
sample_batch = next(iter(dataloader))
print(f"\n🔍 样本批次形状:")
print(f"   源序列: {sample_batch['src'].shape}")
print(f"   源序列: {sample_batch['src']}")
print(f"   目标输入: {sample_batch['tgt_input'].shape}")
print(f"   目标输入: {sample_batch['tgt_input']}")
print(f"   目标输出: {sample_batch['tgt_output'].shape}")
print(f"   目标输出: {sample_batch['tgt_output']}")
print(f"     英文: '{sample_batch['src_text']}'")
print(f"     中文: '{sample_batch['tgt_text']}'")

# 显示一个样本的详细信息
print(f"\n📝 样本详情:")
for i in range(min(2, len(sample_batch['src_text']))):
    print(f"   样本 {i+1}:")
    print(f"     英文: '{sample_batch['src_text'][i]}'")
    print(f"     中文: '{sample_batch['tgt_text'][i]}'")
    print(f"     英文编码: {sample_batch['src'][i].tolist()}")
    print(f"     中文输入编码: {sample_batch['tgt_input'][i].tolist()}")
    print(f"     中文输出编码: {sample_batch['tgt_output'][i].tolist()}")


# 4.2 编码器实现：理解输入序列
# 词汇表构建过程可视化和数据流演示

