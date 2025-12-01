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
