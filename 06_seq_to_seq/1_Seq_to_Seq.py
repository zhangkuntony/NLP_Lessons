# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
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

print("🔍 词汇表构建过程详解")
print("=" * 60)

# 演示词汇表构建过程
print("\n📚 词汇表构建步骤演示:")
sample_sentences = ["hello world", "good morning", "thank you very much"]

demo_vocab = Vocabulary()
print(f"1. 初始词汇表：{list(demo_vocab.idx2word.keys())}")

for i, sentence in enumerate(sample_sentences):
    print(f"\n2.{i+1} 添加句子：'{sentence}'")
    demo_vocab.add_sentence(sentence)
    print(f"     当前词汇表: {list(demo_vocab.word2idx.keys())}")
    print(f"     词汇表大小: {len(demo_vocab)}")

print(f"\n📊 最终词汇表统计:")
print(f"   总词汇数: {len(demo_vocab)}")
print(f"   特殊符号数: 4 (PAD, SOS, EOS, UNK)")
print(f"   实际单词数: {len(demo_vocab) - 4}")

# 演示编码和解码过程
print(f"\n🔄 编码解码过程演示:")
test_sentence = "hello world"
print(f"原始句子: '{test_sentence}'")

# 编码过程
encoded = demo_vocab.encode_sentence(test_sentence)
print(f"编码结果：{encoded}")
print(f"对应词汇：{[demo_vocab.idx2word[idx] for idx in encoded]}")

# 解码过程
decoded = demo_vocab.decode_sentence(encoded)
print(f"解码结果：{decoded}")

# 展示实际数据集的词汇分布
print(f"\n📈 数据集词汇分布分析:")
en_words = []
zh_chars = []

for en_sent, zh_sent in raw_data:
    en_words.extend(en_sent.split())
    zh_chars.extend(list(zh_sent))

en_word_freq = Counter(en_words)
zh_char_freq = Counter(zh_chars)

print(f"\n🇬🇧 英文词汇统计:")
print(f"   总词汇数: {len(en_words)} (包含重复)")
print(f"   唯一词汇数: {len(en_word_freq)}")
print(f"   最高频词汇: {en_word_freq.most_common(5)}")

print(f"\n🇨🇳 中文字符统计:")
print(f"   总字符数: {len(zh_chars)} (包含重复)")
print(f"   唯一字符数: {len(zh_char_freq)}")
print(f"   最高频字符: {zh_char_freq.most_common(5)}")

# 检查数据集的序列长度分布
en_lengths = [len(sent.split()) for sent, _ in raw_data]
zh_lengths = [len(sent) for _, sent in raw_data]

print(f"\n📏 序列长度分析:")
print(f"   英文句子长度: 最短 {min(en_lengths)}, 最长 {max(en_lengths)}, 平均 {sum(en_lengths)/len(en_lengths):.1f} 个单词")
print(f"   中文句子长度: 最短 {min(zh_lengths)}, 最长 {max(zh_lengths)}, 平均 {sum(zh_lengths)/len(zh_lengths):.1f} 个字符")

# 找出最长和最短的句子
max_en_idx = en_lengths.index(max(en_lengths))
min_en_idx = en_lengths.index(min(en_lengths))

print(f"\n📝 长度示例:")
print(f"   最长英文句子: '{raw_data[max_en_idx][0]}' (长度: {max(en_lengths)} 个单词)")
print(f"   最短英文句子: '{raw_data[min_en_idx][0]}' (长度: {min(en_lengths)} 个单词)")

max_zh_idx = zh_lengths.index(max(zh_lengths))
min_zh_idx = zh_lengths.index(min(zh_lengths))

print(f"   最长中文句子: '{raw_data[max_zh_idx][1]}' (长度: {max(zh_lengths)} 个字符)")
print(f"   最短中文句子: '{raw_data[min_zh_idx][1]}' (长度: {min(zh_lengths)} 个字符)")


# 编码器实现
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        """
        编码器
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 词嵌入层：将词索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层：处理序列信息
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        """
        前向传播
        Args:
            input_seq: 输入序列 [batch_size, seq_len]
        Returns:
            outputs: 所有时间步的输出 [batch_size, seq_len, hidden_dim]
            (hidden, cell): 最终的隐状态和细胞状态
        """
        # 1. 词嵌入：[batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_seq)

        # 2. LSTM处理：获取所有时间步的输出和最终隐状态
        outputs, (hidden, cell) = self.lstm(embedded)

        # 返回最后一个时间步的隐状态作为句子表示
        return outputs, (hidden, cell)

# 测试编码器
vocab_size = len(en_vocab)
embedding_dim = 64
hidden_dim = 128

encoder = Encoder(vocab_size, embedding_dim, hidden_dim)

print(f"🏗️ 编码器创建完成！")
print(f"📏 参数数量: {sum(p.numel() for p in encoder.parameters()):,}")

# 测试编码器
test_input = sample_batch['src'][:2]            # 取前2个样本测试
print(f"\n🧪 测试输入形状: {test_input.shape}")

with torch.no_grad():
    outputs, (hidden, cell) = encoder(test_input)
    print(f"✅ 编码器输出形状: {outputs.shape}")
    print(f"✅ 最终隐状态形状: {hidden.shape}")
    print(f"✅ 最终细胞状态形状: {cell.shape}")


# 数据流动可视化：从原始数据到模型输入
print("🌊 数据流动全过程可视化")
print("=" * 70)

# 选择一个样本进行详细演示
sample_en, sample_zh = raw_data[0]
print(f"📝 演示样本: '{sample_en}' → '{sample_zh}'")
print("-" * 50)

# 步骤1: 原始数据
print("🏁 步骤1: 原始数据")
print(f"   英文: '{sample_en}'")
print(f"   中文: '{sample_zh}'")

# 步骤2: 词汇表编码
print(f"\n🔤 步骤2: 词汇表编码")
en_encoded = en_vocab.encode_sentence(sample_en, add_eos=True)
zh_encoded_input =[zh_vocab.word2idx[zh_vocab.SOS_TOKEN]] + zh_vocab.encode_sentence(sample_zh, add_eos=False)
zh_encoded_target = zh_vocab.encode_sentence(sample_zh, add_eos=True)

print(f"    英文编码：{en_encoded}")
print(f"      -> 对应词汇：{[en_vocab.idx2word[idx] for idx in en_encoded]}")
print(f"    中文输入编码：{zh_encoded_input}")
print(f"      -> 对应字符：{[zh_vocab.idx2word[idx] for idx in zh_encoded_input]}")
print(f"    中文目标编码：{zh_encoded_target}")
print(f"      -> 对应字符：{[zh_vocab.idx2word[idx] for idx in zh_encoded_target]}")

# 步骤3: 批处理和填充
print(f"\n📦 步骤3: 批处理和填充演示")
# 模拟一个小批次
mini_batch_indices = [0, 1, 2]
mini_batch_data = [raw_data[i] for i in mini_batch_indices]

print(f"    小批次原始数据：")
for i, (en, zh) in enumerate(mini_batch_data):
    print(f"    样本{i}: '{en}' -> ‘{zh}'")

# 编码所有样本
batch_en_encoded = []
batch_zh_input_encoded = []
batch_zh_target_encoded = []

for en, zh in mini_batch_data:
    batch_en_encoded.append(en_vocab.encode_sentence(en, add_eos=True))
    batch_zh_input_encoded.append([zh_vocab.word2idx[zh_vocab.SOS_TOKEN]] + zh_vocab.encode_sentence(zh, add_eos=False))
    batch_zh_target_encoded.append(zh_vocab.encode_sentence(zh, add_eos=True))

print(f"\n    编码后长度：")
for i, (en, zh_inp, zh_tgt) in enumerate(zip(batch_en_encoded, batch_zh_input_encoded, batch_zh_target_encoded)):
    print(f"    样本{i}: en={len(en)}, zh_input={len(zh_inp)}, zh_target={len(zh_tgt)}")

# 填充到相同长度
batch_en_padded = pad_sequences(batch_en_encoded, en_vocab.word2idx[en_vocab.PAD_TOKEN])
batch_zh_input_padded = pad_sequences(batch_zh_input_encoded, zh_vocab.word2idx[zh_vocab.PAD_TOKEN])
batch_zh_target_padded = pad_sequences(batch_zh_target_encoded, zh_vocab.word2idx[zh_vocab.PAD_TOKEN])

print(f"\n    填充后：")
for i, (en, zh_inp, zh_tgt) in enumerate(zip(batch_en_padded, batch_zh_input_padded, batch_zh_target_padded)):
    print(f"    样本{i}: {en}")
    print(f"      -> {[en_vocab.idx2word[idx] for idx in en]}")
    print(f"    样本{i}: {zh_inp}")
    print(f"      -> {[zh_vocab.idx2word[idx] for idx in zh_inp]}")
    print(f"    样本{i}: {zh_tgt}")
    print(f"      -> {[zh_vocab.idx2word[idx] for idx in zh_tgt]}")

# 步骤4: 转换为张量
print(f"\n🔢 步骤4: 转换为PyTorch张量")
batch_en_tensor = torch.tensor(batch_en_padded, dtype=torch.long)
batch_zh_input_tensor = torch.tensor(batch_zh_input_padded, dtype=torch.long)
batch_zh_target_tensor = torch.tensor(batch_zh_target_padded, dtype=torch.long)

print(f"   英文张量形状: {batch_en_tensor.shape}")
print(f"   中文输入张量形状: {batch_zh_input_tensor.shape}")
print(f"   中文目标张量形状: {batch_zh_target_tensor.shape}")

print(f"\n  第一个样本的张量值：")
print(f"    英文：{batch_en_tensor[0]}")
print(f"    中文输入：{batch_zh_input_tensor[0]}")
print(f"    中文目标：{batch_zh_target_tensor[0]}")

# 步骤5: 损失计算的解释
print(f"\n💡 步骤5: 训练时的损失计算")
print(f"   模型预测: 基于英文和中文输入，预测中文的下一个字符")
print(f"   损失计算: 预测结果与中文目标比较")
print(f"   💡 为什么中文输入和目标不同？")
print(f"      - 输入: [SOS, 你] → 模型看到开始标记和前面的字符")
print(f"      - 目标: [你, EOS] → 模型应该预测的下一个字符")
print(f"      - 这样在每个时间步，模型都知道应该预测什么！")

print(f"\n🎯 数据流动总结:")
print(f"   原始文本 → 分词 → 编码 → 填充 → 张量 → 模型 → 损失 → 梯度 → 更新")


# 4.3 解码器实现：生成输出序列
# 解码器实现
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        """
        解码器
        Args:
            vocab_size: 目标语言词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
        """
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层：用于生成序列
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)

        # 输出层：将隐状态映射到词汇表大小
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, hidden_state):
        """
        前向传播
        Args:
            input_seq: 输入序列 [batch_size, seq_len]
            hidden_state: 编码器传来的隐状态 (hidden, cell)
        Returns:
            outputs: 输出序列的词汇分布 [batch_size, seq_len, vocab_size]
            hidden_state: 更新后的隐状态
        """
        # 1. 词嵌入
        embedded = self.embedding(input_seq)

        # 2. LSTM处理
        outputs, hidden_state = self.lstm(embedded, hidden_state)

        # 3. 投影到词汇表
        outputs = self.output_projection(outputs)

        return outputs, hidden_state

    def generate(self, hidden_state, max_length=20, start_token=1, end_token=2):
        """
        生成序列（推理时使用）
        Args:
            hidden_state: 编码器的隐状态
            max_length: 生成的最大长度
            start_token: 开始标记的索引
            end_token: 结束标记的索引
        Returns:
            generated_sequence: 生成的词汇索引序列
        """
        batch_size = hidden_state[0].size(1)

        # 初始化输入为开始标记
        current_input = torch.tensor([[start_token]] * batch_size)

        generated_sequence = []

        for _ in range(max_length):
            # 获取当前次的输出
            output, hidden_state = self.forward(current_input, hidden_state)

            # 贪心选择概率最高的词
            predicted_word = output.argmax(dim=-1)
            generated_sequence.append(predicted_word.item())

            # 如果生成了结束标记，停止生成
            if predicted_word.item() == end_token:
                break

            # 更新下一步的输入
            current_input = predicted_word

        return generated_sequence

# 创建解码器
zh_vocab_size = len(zh_vocab)
decoder = Decoder(zh_vocab_size, embedding_dim, hidden_dim)

print(f"🏗️ 解码器创建完成！")
print(f"📏 参数数量: {sum(p.numel() for p in decoder.parameters()):,}")

# 测试解码器
test_tgt_input = sample_batch['tgt_input'][:2]
print(f"\n🧪 测试目标输入形状: {test_tgt_input.shape}")

with torch.no_grad():
    # 使用编码器的隐状态作为解码器的初始状态
    decoder_outputs, _ = decoder(test_tgt_input, (hidden, cell))
    print(f"✅ 解码器输出形状: {decoder_outputs.shape}")
    print(f"✅ 输出词汇分布维度: {decoder_outputs.size(-1)} (应该等于中文词汇表大小 {zh_vocab_size})")


# 模型参数和计算复杂度分析
print("📊 模型参数分析")
print("=" * 50)

# 显示编码器参数详情
print(f"\n🔍 编码器参数详细分析:")
total_params = 0
for name, param in encoder.named_parameters():
    param_count = param.numel()
    total_params += param_count
    print(f"    {name:25s}: {param.shape} -> {param_count:,} 参数")

print(f"    {'总计':25s}: {total_params:,} 参数")

# 计算参数组成
vocab_size = len(en_vocab)
embedding_dim = 64
hidden_dim = 128

print(f"\n🧮 参数计算验证:")
embedding_params = vocab_size * embedding_dim
lstm_params = 4 * (embedding_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)           # LSTM公式
print(f"  词嵌入层：{vocab_size} × {embedding_dim} = {embedding_params:,}")
print(f"  LSTM层：4 × ({embedding_dim} × {hidden_dim} + {hidden_dim} × {hidden_dim} + {hidden_dim} = {lstm_params:,}")
print(f"  总计: {embedding_params + lstm_params:,}")

# 内存占用估计
print(f"\n💾 内存占用估计:")
bytes_per_param = 4         # float32
model_memory_mb = total_params * bytes_per_param / (1024**2)
print(f"  模型参数: {model_memory_mb:.2f} MB")

# 计算复杂度分析
print(f"\n⚡ 时间复杂度分析:")
print(f"  编码器前向传播: O(seq_len × embedding_dim × hidden_dim)")
print(f"  其中 seq_len ≈ {max([len(sent.split()) for sent, _ in raw_data])}")
print(f"      embedding_dim = {embedding_dim}")
print(f"      hidden_dim = {hidden_dim}")

# 实际测试编码器速度
import time
test_times = []
test_input = sample_batch['src'][:2]

print(f"\n🕒 实际性能测试:")
for i in range(5):
    start_time = time.time()
    with torch.no_grad():
        outputs, (hidden, cell) = encoder(test_input)
    end_time = time.time()
    test_times.append(end_time - start_time)

avg_time = sum(test_times) / len(test_times)
print(f"  编码器前向传播时间: {avg_time*1000:.2f} ms (平均)")
print(f"  处理速度: {test_input.shape[0]/avg_time:.1f} 句子/秒")


# 4.4 完整的Seq2Seq模型: 将编码器和解码器组合
# 完整的Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Seq2Seq模型
        Args:
            encoder: 编码器
            decoder: 解码器
            device: 计算设备 (cpu/gpu)
        """
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_seq, tgt_seq, teacher_forcing_ratio=1):
        """
        训练时的前向传播
        Args:
            src_seq: 源序列 [batch_size, src_len]
            tgt_seq: 目标序列 [batch_size, tgt_len]
            teacher_forcing_ratio: 教师强制比例
        Returns:
            outputs: 解码器输出 [batch_size, tgt_len, vocab_size]
        """
        batch_size = src_seq.size(0)
        tgt_len = tgt_seq.size(1)
        vocab_size = self.decoder.vocab_size

        # 存储解码器的所有输出
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # 1. 编码阶段: 获取源序列的表示
        _, hidden_state = self.encoder(src_seq)

        # 2. 解码阶段: 逐步生成目标序列
        # 解码器的第一个输入是SOS标记
        decoder_input = tgt_seq[:, :1]          # 第一个token (SOS)

        # 从第0个时间步开始训练，而不是从第1个时间步
        for t in range(tgt_len):
            # 解码器前向传播
            output, hidden_state = self.decoder(decoder_input, hidden_state)
            outputs[:, t:t+1, :] = output

            # 教师强制：决定下一个输入是真实标签还是模型预测
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and t < tgt_len - 1:
                # 使用真实的下一个词作为输入（但不要超出序列长度）
                decoder_input = tgt_seq[:, t+1:t+2]
            else:
                # 使用模型预测的词作为输入
                decoder_input = output.argmax(dim=-1)

        return outputs

    def translate(self, src_seq, max_length=20):
        """
        推理时的翻译功能
        Args:
            src_seq: 源序列 [1, src_len]
            max_length: 生成的最大长度
        Returns:
            generated_indices: 生成的词汇索引列表
        """
        self.eval()         # 设置为评估模式

        with torch.no_grad():
            # 编码源序列
            _, hidden_state = self.encoder(src_seq)

            # 生成目标序列
            generated_indices = []
            decoder_input = torch.tensor([[zh_vocab.word2idx[zh_vocab.SOS_TOKEN]]]).to(self.device)

            for _ in range(max_length):
                output, hidden_state = self.decoder(decoder_input, hidden_state)
                predicted_id = output.argmax(dim=-1).item()

                generated_indices.append(predicted_id)

                # 如果预测到结束标记，停止生成
                if predicted_id == zh_vocab.word2idx[zh_vocab.EOS_TOKEN]:
                    break

                # 下一步的输入是当前预测的词
                decoder_input = torch.tensor([[predicted_id]]).to(self.device)

        return generated_indices

# 创建设备对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建完整的Seq2Seq模型
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"🎯 Seq2Seq模型创建完成！")
print(f"📱 运行设备: {device}")
print(f"📏 总参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 显示模型结构
print(f"\n🏗️ 模型结构:")
print(f"  编码器参数: {sum(p.numel() for p in model.encoder.parameters()):,}")
print(f"  解码器参数: {sum(p.numel() for p in model.decoder.parameters()):,}")

# 测试模型
test_src = sample_batch['src'][:1].to(device)               # 取一个样本
test_tgt = sample_batch['tgt_input'][:1].to(device)

print(f"\n🧪 模型测试:")
print(f"  输入形状: {test_src.shape}")
print(f"  目标形状: {test_tgt.shape}")

with torch.no_grad():
    outputs = model(test_src, test_tgt, teacher_forcing_ratio=1.0)
    print(f"✅ 模型输出形状: {outputs.shape}")


# 训练我们的翻译模型
# 训练 vs 推理详细对比演示
print("🎭 训练模式 vs 推理模式详细对比")
print("=" * 70)

# 使用一个简单的例子来演示
demo_en = "hello"
demo_zh = "你好"

print(f"📝 演示句子: '{demo_en}' → '{demo_zh}'")
print("-" * 50)

# 准备输入数据
src_tensor = torch.tensor([en_vocab.encode_sentence(demo_en)]).to(device)
tgt_input = [zh_vocab.word2idx[zh_vocab.SOS_TOKEN]] + zh_vocab.encode_sentence(demo_zh, add_eos=False)
tgt_input_tensor = torch.tensor([tgt_input]).to(device)
tgt_output = zh_vocab.encode_sentence(demo_zh, add_eos=True)

print(f"🎓 训练模式演示:")
print(f"  输入编码: {src_tensor.tolist()[0]} -> {[en_vocab.idx2word[i] for i in src_tensor.tolist()[0]]}")
print(f"  目标输入: {tgt_input_tensor.tolist()[0]} -> {[zh_vocab.idx2word[i] for i in tgt_input_tensor.tolist()[0]]}")
print(f"  目标输出: {tgt_output} -> {[zh_vocab.idx2word[i] for i in tgt_output]}")

# 训练模式的详细步骤
model.train()
print(f"\n  🔄 训练步骤详解:")

# 编码阶段
with torch.no_grad():
    _, (encoder_hidden, encoder_cell) = model.encoder(src_tensor)
    print(f"  1. 编码器处理: '{demo_en}' -> 隐状态形状 {encoder_hidden.shape}")

    # 解码阶段（模拟）
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    print(f"  2. 解码器步骤:")
    for t in range(len(tgt_input)):
        current_input = tgt_input_tensor[:, t:t+1]          # 当前时间步输入

        # 解码器前向传播
        decoder_output, (decoder_hidden, decoder_cell) = model.decoder(current_input, (decoder_hidden, decoder_cell))
        predicted_id = decoder_output.argmax(dim=-1).item()
        predicted_word = zh_vocab.idx2word[predicted_id]

        if t < len(tgt_input):
            true_word = zh_vocab.idx2word[tgt_output[t]]
            print(f"    步骤{t+1}: 输入'{zh_vocab.idx2word[current_input.item()]}' -> 预测'{predicted_word}' (真实: '{true_word}')")
        else:
            print(f"    步骤{t+1}: 输入'{zh_vocab.idx2word[current_input.item()]}' -> 预测'{predicted_word}'")

print(f"\n🔮 推理模式演示:")
model.eval()

# 推理模式的详细步骤
print(f"    输入编码: {src_tensor.tolist()[0]} -> {[en_vocab.idx2word[i] for i in src_tensor.tolist()[0]]}")
print(f"    目标输出: 未知！需要逐步生成")

print(f"\n   🔄 推理步骤详解:")
with torch.no_grad():
    # 编码阶段
    _, (encoder_hidden, encoder_cell) = model.encoder(src_tensor)
    print(f"    1. 编码器处理: '{demo_en}' -> 隐状态形状 {encoder_hidden.shape}")

    # 解码阶段
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    current_input = torch.tensor([[zh_vocab.word2idx[zh_vocab.SOS_TOKEN]]]).to(device)
    generated_sequence = []

    print(f"    2. 解码器步骤:")
    for t in range(5):              # 最多生成5个词
        # 解码器前向传播
        decoder_output, (decoder_hidden, decoder_cell) = model.decoder(current_input, (decoder_hidden, decoder_cell))
        predicted_id = decoder_output.argmax(dim=-1).item()
        predicted_word = zh_vocab.idx2word[predicted_id]

        print(f"    步骤{t+1}: 输入'{zh_vocab.idx2word[current_input.item()]}' -> 预测'{predicted_word}'")

        generated_sequence.append(predicted_id)

        # 停止条件
        if predicted_id == zh_vocab.word2idx[zh_vocab.EOS_TOKEN]:
            print(f"        遇到结束符，停止生成")
            break

        # 下一步的输入是当前的预测（关键区别！）
        current_input = torch.tensor([[predicted_id]]).to(device)

    generated_text = zh_vocab.decode_sentence(generated_sequence)
    print(f"    3. 最终生成: '{generated_text}'")

print(f"\n💡 关键区别总结:")
print(f"   🎓 训练模式:")
print(f"      - 解码器输入: 使用真实的目标序列 (Teacher Forcing)")
print(f"      - 优点: 训练稳定、快速")
print(f"      - 缺点: 与推理不一致")
print(f"   🔮 推理模式:")
print(f"      - 解码器输入: 使用自己的预测结果")
print(f"      - 优点: 真实的使用场景")
print(f"      - 缺点: 错误会累积传播")

print(f"\n⚠️  曝光偏差 (Exposure Bias):")
print(f"   问题: 训练时模型从未见过自己的错误预测")
print(f"   后果: 推理时一旦出错，可能一错到底")
print(f"   解决方案: 调度采样、强化学习等高级技术")


# 训练和评估函数
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备上
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(src, tgt_input, teacher_forcing_ratio=0.5)

        # 计算损失（忽略填充符号）
        # 不再去掉第一个时间步，因为现在模型会学习预测第一个字符
        output = output.reshape(-1, output.size(-1))                # 展平所有时间步
        tgt_output = tgt_output.reshape(-1)                         # 展平所有时间步

        loss = criterion(output, tgt_output)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0:              # 每5个batch打印一次
            print(f'  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)

            # 前向传播（不使用教师强制）
            output = model(src, tgt_input, teacher_forcing_ratio=0)

            # 计算损失，不再去掉第一个时间步
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_length=20):
    """翻译单个句子"""
    model.eval()

    # 预处理句子
    tokens = sentence.lower().split()
    indices = [src_vocab.word2idx.get(token, src_vocab.word2idx[src_vocab.UNK_TOKEN]) for token in tokens]
    indices.append(src_vocab.word2idx[src_vocab.EOS_TOKEN])

    # 转换为tensor
    src_tensor = torch.tensor([indices]).to(device)

    # 翻译
    with torch.no_grad():
        generated_indices = model.translate(src_tensor, max_length)

    # 解码为文本
    translation = tgt_vocab.decode_sentence(generated_indices)

    return translation

# 设置训练参数
criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab.word2idx[zh_vocab.PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🏋️ 训练设置完成！")
print(f"📉 损失函数: CrossEntropyLoss (忽略填充符号)")
print(f"🔧 优化器: Adam (学习率: 0.001)")
print(f"🎯 设备: {device}")


# 验证修复效果
print("🔧 验证修复效果")
print("=" * 50)

# 重新创建模型以应用修复
model = Seq2Seq(encoder, decoder, device).to(device)

# 简单测试训练是否正常
print("\n🧪 测试修复后的训练流程:")
test_batch = next(iter(dataloader))
src = test_batch['src'][:2].to(device)
tgt_input = test_batch['tgt_input'][:2].to(device)
tgt_output = test_batch['tgt_output'][:2].to(device)

print(f"   源序列形状: {src.shape}")
print(f"   目标输入形状: {tgt_input.shape}")
print(f"   目标输出形状: {tgt_output.shape}")

# 测试前向传播
model.train()
with torch.no_grad():
    output = model(src, tgt_input, teacher_forcing_ratio=1.0)
    print(f"    模型输出形状: {output.shape}")

    # 测试损失计算
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab.word2idx[zh_vocab.PAD_TOKEN])
    output_flat = output.reshape(-1, output.size(-1))
    tgt_output_flat = tgt_output.reshape(-1)
    loss = criterion(output_flat, tgt_output_flat)
    print(f"    损失计算成功，损失值: {loss.item():.4f}")

# 测试推理
print(f"\n🔮 测试修复后的推理:")
test_sentences = ["hi.", "hello!", "good morning"]
for sentence in test_sentences:
    translation = translate_sentence(model, sentence, en_vocab, zh_vocab, device)
    print(f"    '{sentence}' -> '{translation}'")

print(f"\n✅ 修复验证完成！")
print(f"注意：由于模型尚未重新训练，预测结果可能仍然不准确。")
print(f"但现在模型的架构已经修复，重新训练后应该能正确预测第一个字符。")

# 开始训练
print("🚀 开始训练Seq2Seq模型...")
print("=" * 50)

num_epochs = 50         # 由于数据集很小，我们多训练几轮
train_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")

    # 训练
    train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
    train_losses.append(train_loss)

    print(f"✅ 平均训练损失: {train_loss:.4f}")

    # 每10个epoch测试翻译效果
    if (epoch + 1) % 10 == 0:
        print("\n🔍 翻译测试:")
        test_sentences = ["hello", "thank you", "i love you", "good morning"]

        for sentence in test_sentences:
            translation = translate_sentence(model, sentence, en_vocab, zh_vocab, device)
            print(f"    '{sentence}' -> '{translation}'")

    # 保存最佳模型
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), "best_seq2seq_model.pt")

print(f"\n🎉 训练完成！最佳损失: {best_loss:.4f}")
print("模型已保存为 'best_seq2seq_model.pth'")
