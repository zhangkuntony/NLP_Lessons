# 导入必要的库
import os
import glob
import chardet  # 用于检测文件编码
import jieba    # 中文分词
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

print("🚀 所有必要的库都已成功导入！")
print("接下来我们开始处理数据...")

# 定义数据路径
data_path = "CN_Corpus/SogouC.reduced/Reduced/"
print(f"📁 数据路径: {data_path}")

# 检查数据目录
if os.path.exists(data_path):
    categories = os.listdir(data_path)
    print(f"📊 找到 {len(categories)} 个数据类别: {categories}")
else:
    print("❌ 数据目录不存在，请检查路径！")


# 解决中文编码问题
def detect_and_read_file(file_path, max_detect_bytes=10000):
    """
    智能检测文件编码并正确读取文件

    Args:
        file_path: 文件路径
        max_detect_bytes: 用于检测编码的最大字节数

    Returns:
        文件内容字符串，如果读取失败返回None
    """
    try:
        # 1. 读取部分文件内容来检测编码
        with open(file_path, "rb") as f:
            raw_data = f.read(max_detect_bytes)

        # 2. 使用chardet检测编码
        encoding_result = chardet.detect(raw_data)
        detected_encoding = encoding_result["encoding"]
        confidence = encoding_result["confidence"]

        print(f"🔍 检测到编码: {detected_encoding} (置信度: {confidence:.2f})")

        # 3. 尝试用检测到的编码读取文件
        try:
            with open(file_path, "r", encoding=detected_encoding) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            print(f"⚠️  用检测到的编码 {detected_encoding} 读取失败，尝试其他编码...")

        # 4. 如果检测的编码失败，尝试常见的中文编码
        common_encodings = ['gbk', 'gb2312', 'utf-8', 'utf-16', 'big5']
        for encoding in common_encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                print(f"✅ 成功用编码 {encoding} 读取文件")
                return content

            except (UnicodeDecodeError, UnicodeError):
                continue

        print(f"❌ 无法读取文件 {file_path}")
        return None

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return None

# 测试编码检测功能
print("🧪 测试编码检测功能...")
test_file = os.path.join(data_path, "C000008", "10.txt")
if os.path.exists(test_file):
    content = detect_and_read_file(test_file)
    if content:
        print(f"📝 文件内容预览 (前200字符):")
        print(content[:200])
        print(f"📊 文件总长度: {len(content)} 字符")
    else:
        print("❌ 无法读取测试文件")
else:
    print(f"❌ 测试文件不存在: {test_file}")


# 文本预处理和分词
import re
import string

class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        # 将英文标点和中文标点进行字符串拼接（string.punctuation为英文标点）
        self.punctuation = set(string.punctuation + '，。？！；：""''（）【】《》、—…')

    def clean_text(self, text):
        """清理文本：去除特殊字符、多余空格等"""
        # 1. 去除网址、邮箱等
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # 2. 去除数字和英文（可选，根据需求调整）
        text = re.sub(r'[a-zA-Z0-9]', '', text)

        # 3. 去除多余的空白字符
        text = re.sub(r'\s+', '', text)

        return text.strip()

    def segment_text(self, text):
        """中文分词"""
        # 使用jieba进行分词
        words = jieba.cut(text)

        # 过滤掉单字符和标点符号
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) > 1 and word not in self.punctuation:
                filtered_words.append(word)

        return filtered_words

    def process_file(self, file_path):
        """处理单个文件"""
        content = detect_and_read_file(file_path)
        if content is None:
            return []

        # 清理文本
        clean_content = self.clean_text(content)

        # 分词
        words = self.segment_text(clean_content)

        return words

# 创建预处理器
preprocessor = TextPreprocessor()

# 测试预处理功能
print("🧪 测试文本预处理功能...")
test_text = "今天天气真好！我们去公园玩吧。网址：http://example.com 邮箱：test@email.com"
print(f"原文: {test_text}")

clean_text = preprocessor.clean_text(test_text)
print(f"清理后：{clean_text}")

words = preprocessor.segment_text(clean_text)
print(f"分词结果：{words}")

# 处理一个实际文件
if os.path.exists(test_file):
    print(f"\n📝 处理文件: {test_file}")
    processed_words = preprocessor.process_file(test_file)
    print(f"分词数量：{len(processed_words)}")
    print(f"前20个词：{processed_words[:20]}")
else:
    print("⚠️ 测试文件不存在，创建模拟数据...")
    # 创建一些模拟的中文文本用于测试
    sample_text = "今天天气很好。明天可能会下雨。我喜欢在晴天的时候去公园散步。"
    words = preprocessor.segment_text(sample_text)
    print(f"模拟数据分词结果: {words}")


# 构建词汇表和数据加载器
class Vocabulary:
    """词汇表类"""

    def __init__(self, min_freq=2):
        """
        Args:
            min_freq: 词汇最少出现次数，低于此频率的词会被标记为未知词
        """
        self.min_freq = min_freq
        self.word2idx = {}              # 词 -> 索引
        self.idx2word = {}              # 索引 -> 词
        self.word_counts = Counter()    # 词频统计

        # 特殊标记
        self.UNK_TOKEN = '<UNK>'        # 未知词
        self.PAD_TOKEN = '<PAD>'        # 填充词（用于batch padding）

        # 添加特殊标记
        self._add_word(self.PAD_TOKEN)  # 索引 0
        self._add_word(self.UNK_TOKEN)  # 索引 1

    def _add_word(self, word):
        """添加词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_vocab(self, word_lists):
        """
        从词列表构建词汇表

        Args:
            word_lists: 词的列表的列表，如 [['今天', '天气'], ['明天', '下雨']]
        """
        print("📊 统计词频...")

        # 统计所有词的频率
        for words in word_lists:
            self.word_counts.update(words)

        print(f"📈 总共发现 {len(self.word_counts)} 个唯一词汇")
        print(f"📋 最常见的10个词: {self.word_counts.most_common(10)}")

        # 添加频率高于阈值的词
        added_count = 0
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self._add_word(word)
                added_count += 1

        print(f"✅ 词汇表构建完成！")
        print(f"📊 词汇表大小: {len(self.word2idx)} (其中 {added_count} 个常用词)")
        print(f"🗑️  过滤掉 {len(self.word_counts) - added_count} 个低频词")

    def word_to_idx(self, word):
        """将词转换为索引"""
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])

    def idx_to_word(self, idx):
        """将索引转换为词"""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def words_to_indices(self, words):
        """将此列表转换为索引列表"""
        return [self.word_to_idx(word) for word in words]

    def indices_to_words(self, indices):
        """将索引列表转换为词列表"""
        return [self.idx_to_word(idx) for idx in indices]

    def __len__(self):
        return len(self.word2idx)

# 创建一些示例数据来测试词汇表
print("🧪 创建测试数据...")
sample_word_lists = [
    ['今天', '天气', '很', '好'],
    ['明天', '可能', '会', '下雨'],
    ['我', '喜欢', '晴天', '的', '时候'],
    ['今天', '天气', '真', '好'],  # 重复的词用来测试词频
    ['天气', '预报', '说', '明天', '晴天']
]

# 构建词汇表
vocab = Vocabulary(min_freq=1)          # 设置最小频率为1， 这样所有词都会被保留
vocab.build_vocab(sample_word_lists)

# 测试词汇表功能
print(f"\n🧪 测试词汇表功能:")
test_words = ['今天', '天气', '未知词汇']
for word in test_words:
    idx = vocab.word_to_idx(word)
    back_word = vocab.idx_to_word(idx)
    print(f"  '{word}' → {idx} → '{back_word}'")

# 测试句子转换
test_sentence = ['今天', '天气', '很', '好']
indices = vocab.words_to_indices(test_sentence)
back_words = vocab.indices_to_words(indices)
print(f"\n📝 句子转换测试:")
print(f"  原句子: {test_sentence}")
print(f"  索引序列: {indices}")
print(f"  还原句子: {back_words}")


# NNLM数据集类
class NNLMDataset:
    """NNLM数据集类"""

    def __init__(self, word_lists, vocab, context_size=3):
        """
        Args:
            word_lists: 词的列表的列表
            vocab: 词汇表对象
            context_size: 上下文窗口大小（用前几个词预测下一个词）
        """
        self.vocab = vocab
        self.context_size = context_size
        self.data = []

        print(f"🔨 构建训练数据，上下文窗口大小: {context_size}")
        self._build_data(word_lists)

    def _build_data(self, word_lists):
        """构建训练数据对"""
        for words in word_lists:
            # 将词转换为索引
            indices = self.vocab.words_to_indices(words)

            # 构建上下文-目标对
            for i in range(len(indices) - self.context_size):
                context = indices[i:i + self.context_size]          # 前n个词
                target = indices[i + self.context_size]             # 下一个词
                self.data.append((context, target))

        print(f"✅ 数据构建完成！总共 {len(self.data)} 个训练样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, batch_size=32, shuffle=True):
        """获取批次数据"""
        if shuffle:
            indices = np.random.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)
        else:
            indices = list(range(min(batch_size, len(self.data))))

        contexts = []
        targets = []

        for idx in indices:
            context, target = self.data[idx]
            contexts.append(context)
            targets.append(target)

        return torch.tensor(contexts), torch.tensor(targets)

# 创建数据集
print("📦 创建NNLM数据集...")
dataset = NNLMDataset(sample_word_lists, vocab, context_size=3)

# 查看一些训练样本
print(f"\n📋 训练样本示例:")
# for i in range(min(5, len(dataset))):
for i in range(len(dataset)):
    context, target = dataset[i]
    context_words = vocab.indices_to_words(context)
    target_word = vocab.idx_to_word(target)
    print(f"  样本 {i+1}: {context_words} → {target_word}")

# 测试批次数据获取
print(f"\n🎲 测试批次数据获取:")
batch_contexts, batch_targets = dataset.get_batch(batch_size=3)
print(f"  批次上下文形状: {batch_contexts.shape}")
print(f"  批次目标形状: {batch_targets.shape}")
print(f"  批次上下文内容: {batch_contexts}")
print(f"  批次目标内容: {batch_targets}")

# 转换回词汇查看
print(f"\n📝 批次内容（词汇形式）:")
for i in range(len(batch_contexts)):
    context_words = vocab.indices_to_words(batch_contexts[i].tolist())
    target_word = vocab.idx_to_word(batch_targets[i].item())
    print(f"  批次样本 {i + 1}: {context_words} → {target_word}")


# NNLM 模型实现
class NNLM(nn.Module):
    """神经网络语言模型"""

    def __init__(self, vocab_size, context_size, embedding_dim=50, hidden_dim=128):
        """
        Args:
            vocab_size: 词汇表大小
            context_size: 上下文窗口大小
            embedding_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(NNLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 🧱 组件1：词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 🧱 组件2：隐藏层
        # 输入维度 = context_size * embedding_dim (拼接后的向量长度)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)

        # 🧱 组件3：输出层
        self.output = nn.Linear(hidden_dim, vocab_size)

        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)            # 使用LogSoftmax与NLLLoss配合

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        # 嵌入层初始化
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # 线性层初始化
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context):
        """
        前向传播

        Args:
            context: 上下文词索引，形状 [batch_size, context_size]

        Returns:
            输出概率分布，形状 [batch_size, vocab_size]
        """
        batch_size = context.size(0)

        # 步骤1：词嵌入
        # context: [batch_size, context_size] -> [batch_size, context_size, embedding_dim]
        embedded = self.embedding(context)

        # 步骤2：拼接
        # [batch_size, context_size, embedding_dim] -> [batch_size, context_size * embedding_dim]
        concatenated = embedded.view(batch_size, -1)

        # 步骤3：隐藏层
        # [batch_size, context_size * embedding_dim] -> [batch_size, hidden_dim]
        hidden_out = self.relu(self.hidden(concatenated))

        # 步骤4：输出层
        # [batch_size, hidden_dim] -> [batch_size, vocab_size]
        output = self.output(hidden_out)

        # 步骤5：概率分布
        log_probs = self.softmax(output)

        return log_probs

    def predict_next_word(self, context_words, vocab, top_k=5):
        """
        预测下一个词

        Args:
            context_words: 上下文词列表
            vocab: 词汇表对象
            top_k: 返回概率最高的前k个词

        Returns:
            [(word, probability), ...] 按概率降序排列
        """
        self.eval()             # 设置为评估模式

        with torch.no_grad():
            # 转换为索引
            context_indices = vocab.words_to_indices(context_words)

            # 确保上下文长度正确
            if len(context_indices) != self.context_size:
                raise ValueError(f"上下文长度应为 {self.context_size}, 但得到{len(context_indices)}")

            # 转换为tensor并添加batch维度
            context_tensor = torch.tensor([context_indices])

            # 前向传播
            log_probs = self.forward(context_tensor)
            probs = torch.exp(log_probs)            # 从log概率转换为概率

            # 获取top_k
            top_probs, top_indices = torch.topk(probs[0], k=top_k)

            # 转换回词汇
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                word = vocab.idx_to_word(idx.item())
                predictions.append((word, prob.item()))

            return predictions

# 创建模型
print("🏗️  创建NNLM模型...")
model = NNLM(
    vocab_size=len(vocab),
    context_size=3,
    embedding_dim=20,       # 较小的维度用于演示
    hidden_dim=50
)

print(f"📊 模型参数:")
print(f"  词汇表大小: {len(vocab)}")
print(f"  上下文窗口: 3")
print(f"  嵌入维度: 20")
print(f"  隐藏层维度: 50")

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数数: {total_params:,}")
print(f"  可训练参数数: {trainable_params:,}")

# 测试模型前向传播
print(f"\n🧪 测试模型前向传播...")
test_context = torch.tensor([[2, 5, 8]])        # 批次大小为1的测试输入 (注意添加了方括号)
test_output = model(test_context)
print(f"  输入形状: {test_context.shape}")
print(f"  输出形状: {test_output.shape}")
print(f"  输出概率和: {torch.exp(test_output).sum().item():.6f} (应该接近1.0)")