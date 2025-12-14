# 解决OpenMP冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import jieba
from collections import Counter

# 设置随机种子
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 环境准备完成！")

#
# """
# 🎨 可视化演示：基础Seq2Seq模型的问题
#
# 这个函数用生动的图表来展示传统Seq2Seq模型的两大核心问题：
# 1. 信息瓶颈问题：所有信息被压缩到一个小向量中
# 2. 对齐问题：模型不知道输出词对应哪个输入词
# """
#
# def visualize_basic_seq2seq_problem():
#     """用图表展示基础的seq2seq的问题，让抽象概念变得直观易懂"""
#
#     # 创建两个子图：上方展示信息瓶颈，下方展示对齐问题
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
#
#     # ===================第一个图：信息瓶颈问题===================
#     ax1.set_title('问题1：信息瓶颈 - 就像把整本书的内容写在便条纸上',
#                   fontsize=16, fontweight='bold', pad=20)
#
#     # 用一个长句子来演示信息瓶颈问题
#     input_words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
#     print(f"📝 示例句子：{' '.join(input_words)}")
#     print("   这个句子有9个词，包含丰富的信息（颜色、动作、对象等）")
#
#     # 绘制输入词汇（用蓝色方块表示）
#     for i, word in enumerate(input_words):
#         # 每个词用一个小方块表示
#         rect = plt.Rectangle((i-0.3, 3), 0.6, 0.6, facecolor='lightblue', edgecolor='blue')
#         ax1.add_patch(rect)
#         ax1.text(i, 3.3, word, ha='center', va='center', fontsize=10)
#
#     # 绘制信息流动箭头（展示序列处理过程）
#     for i in range(len(input_words) - 1):
#         ax1.arrow(i + 0.3, 3.3, 0.4, 0, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
#
#     # 绘制瓶颈向量（用红色方块表示压缩后的信息）
#     ax1.add_patch(plt.Rectangle((4, 1.5), 1, 0.8, facecolor='red', edgecolor='darkred'))
#     ax1.text(4.5, 1.9, '语义向量\n(信息瓶颈)', ha='center', va='center',
#              fontsize=12, fontweight='bold', color='white')
#
#     # 绘制压缩箭头（所有信息汇聚到一个向量）
#     ax1.arrow(8.3, 3, -3.5, -0.7, head_width=0.08, head_length=0.1, fc='red', ec='red', linewidth=2)
#     ax1.text(6, 2.5, '所有信息\n被压缩！', ha='center', va='center',
#              fontsize=11, color='red', fontweight='bold')
#
#     # 添加说明文字
#     ax1.text(1, 0.8, '问题：9个词的丰富信息 → 1个固定大小的向量',
#              fontsize=12, fontweight='bold', color='red')
#     ax1.text(1, 0.5, '   就像把一整本小说压缩成一句话！',
#              fontsize=10, color='darkred')
#
#     ax1.set_xlim(-1, 10)
#     ax1.set_ylim(0.3, 4)
#     ax1.axis('off')
#
#     # ===================第二个图：对齐问题===================
#     ax2.set_title('问题2：无法对齐 - 不知道哪个中文词对应哪个英文词',
#                   fontsize=16, fontweight='bold', pad=20)
#
#     # 用简单的翻译例子演示对齐问题
#     en_words = ['I', 'love', 'machine', 'learning']
#     zh_words = ['我', '喜欢', '机器', '学习']
#
#     print(f"\n翻译示例：")
#     print(f"   英文：{' '.join(en_words)}")
#     print(f"   中文：{''.join(zh_words)}")
#     print(f"   问题：模型不知道'machine'对应'机器'")
#
#     # 绘制英文词汇（绿色方块）
#     for i, word in enumerate(en_words):
#         rect = plt.Rectangle((i * 2 - 0.4, 2.5), 0.8, 0.6, facecolor='lightgreen', edgecolor='green')
#         ax2.add_patch(rect)
#         ax2.text(i * 2, 2.8, word, ha='center', va='center', fontsize=12)
#
#     # 绘制中文词汇（黄色方块）
#     for i, word in enumerate(zh_words):
#         rect = plt.Rectangle((i * 2 - 0.4, 0.5), 0.8, 0.6, facecolor='lightyellow', edgecolor='orange')
#         ax2.add_patch(rect)
#         ax2.text(i * 2, 0.8, word, ha='center', va='center', fontsize=12)
#
#     # 绘制正确的对应关系（绿色虚线表示理想的对应）
#     correct_alignments = [(0, 0), (1, 1), (2, 2), (3, 3)]
#     for en_idx, zh_idx in correct_alignments:
#         ax2.plot([en_idx * 2, zh_idx * 2], [2.5, 1.1], 'g--', linewidth=2, alpha=0.7)
#
#     # 添加问号和说明（表示模型的困惑）
#     ax2.text(3, 1.7, '？', fontsize=30, ha='center', va='center')
#     ax2.text(5, 1.7, '模型不知道\n对应关系！', ha='center', va='center',
#              fontsize=12, color='red', fontweight='bold')
#
#     # 添加说明文字
#     ax2.text(0.5, 3.3, 'US 英文输入', fontsize=12, fontweight='bold', color='green')
#     ax2.text(0.5, 0.1, 'CN 中文输出', fontsize=12, fontweight='bold', color='orange')
#     ax2.text(4.5, 1.3, '虚线 = 理想的对应关系', fontsize=10, color='green')
#
#     ax2.set_xlim(-1, 8)
#     ax2.set_ylim(0, 3.5)
#     ax2.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# # 运行可视化演示
# print("🎨 开始可视化演示基础Seq2Seq的问题...")
# print("=" * 60)
# visualize_basic_seq2seq_problem()
# print("=" * 60)
# print("🎯 看到这些问题了吗？这就是为什么我们需要注意力机制！")
# print("💡 注意力机制就是为了解决这两个核心问题而诞生的！")
#
#
# # 可视化注意力机制的工作原理
# def visualize_attention_mechanism():
#     fig, ax = plt.subplots(1, 1, figsize=(15, 8))
#
#     # 输入序列
#     input_words = ['The', 'red', 'car', 'is', 'very', 'fast']
#     output_words = ['这', '红色', '汽车', '非常', '快']
#
#     # 创建注意力权重矩阵（示例）
#     attention_weights = np.array([
#         [0.8, 0.1, 0.05, 0.03, 0.01, 0.01],  # 翻译"这"时主要关注"The"
#         [0.1, 0.8, 0.05, 0.02, 0.02, 0.01],  # 翻译"红色"时主要关注"red"
#         [0.05, 0.1, 0.8, 0.03, 0.01, 0.01],  # 翻译"汽车"时主要关注"car"
#         [0.02, 0.02, 0.03, 0.1, 0.8, 0.03],  # 翻译"非常"时主要关注"very"
#         [0.01, 0.01, 0.02, 0.06, 0.1, 0.8],  # 翻译"快"时主要关注"fast"
#     ])
#
#     # 绘制热力图
#     im = ax.imshow(attention_weights, cmap='Reds', aspect='auto')
#
#     # 设置标签
#     ax.set_xticks(range(len(input_words)))
#     ax.set_yticks(range(len(output_words)))
#     ax.set_xticklabels(input_words, fontsize=14)
#     ax.set_yticklabels(output_words, fontsize=14)
#
#     # 添加数值标注
#     for i in range(len(output_words)):
#         for j in range(len(input_words)):
#             text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
#                            ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white",
#                            fontsize=10, fontweight='bold')
#
#     # 设置标题和标签
#     ax.set_title('注意力权重矩阵：模型关注的焦点', fontsize=16, fontweight='bold', pad=20)
#     ax.set_xlabel('输入词（英文）', fontsize=14, fontweight='bold')
#     ax.set_ylabel('输出词（中文）', fontsize=14, fontweight='bold')
#
#     # 添加颜色条
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label('注意力权重', fontsize=12, fontweight='bold')
#
#     # 添加解释文字
#     ax.text(len(input_words) + 0.5, len(output_words) // 2,
#             '颜色越深\n= 关注度越高\n\n每行权重\n之和为1',
#             fontsize=12, ha='center', va='center',
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
#
#     plt.tight_layout()
#     plt.show()
#
# visualize_attention_mechanism()
# print("🎯 注意力机制让模型知道该关注什么！")
#
#
# """
# 🔧 动手实现：注意力机制的核心算法
#
# 让我们把刚才学到的"选朋友帮忙"理论转换成实际的代码！
# 这个简单的注意力模块完美演示了三个核心步骤。
# """
#
# class SimpleAttention(nn.Module):
#     """
#     简单注意力模块 - 把理论变成代码！
#
#     这就是我们刚才讨论的"选朋友帮忙"算法的代码实现：
#     1. 评估每个朋友的有用程度
#     2. 决定向每个朋友求助的比例
#     3. 综合所有朋友的建议
#     """
#
#     def __init__(self, hidden_dim):
#         super(SimpleAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         print(f"🏗️ 创建注意力模块，隐藏维度: {hidden_dim}")
#
#     def forward(self, decoder_hidden, encoder_outputs):
#         """
#         注意力机制的核心计算过程
#
#         参数解释（用我们的比喻）:
#             decoder_hidden: 你当前的"困惑状态" [batch_size, hidden_dim]
#                           （你要翻译什么词？你需要什么帮助？）
#             encoder_outputs: 所有朋友的"专业知识" [batch_size, seq_len, hidden_dim]
#                            （每个输入词能提供什么信息？）
#
#         返回结果:
#             context_vector: "综合所有建议的最终答案" [batch_size, hidden_dim]
#             attention_weights: "向每个朋友求助的比例" [batch_size, seq_len]
#         """
#
#         print(f"\n🧠 开始注意力计算...")
#         print(f"   当前状态形状: {decoder_hidden.shape}")
#         print(f"   输入信息形状: {encoder_outputs.shape}")
#
#         # =================== 步骤1: 评估朋友的有用程度 ===================
#         print("\n🎯 步骤1: 计算相关性分数（评估朋友有用程度）")
#
#         # 使用点积计算相似度 - 就像问“你的专长和我的需求有多匹配！”
#         # 数学原理：点积越大 = 向量越相似 = 朋友越有用
#         scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))
#         scores = scores.squeeze(2)          # [batch_size, seq_len]
#
#         print(f"   相关性分数: {scores.squeeze().detach().numpy()}")
#         print("   💡 分数越高 = 朋友越有用！")
#
#         # =================== 步骤2: 分配注意力比例 ===================
#         print("\n📊 步骤2: 计算注意力权重（分配求助比例）")
#
#         # 使用softmax确保所有权重加起来= 100%
#         # 就像把评分转换成百分比分配
#
#         attention_weights = F.softmax(scores, dim=1)            # [batch_size, seq_len]
#
#         print(f"   注意力权重: {attention_weights.squeeze().detach().numpy()}")
#         print(f"   权重总和: {attention_weights.sum().item():.4f} (应该=1.0)")
#         print("   💡 这就是给每个朋友分配的注意力比例！")
#
#         # =================== 步骤3: 综合所有建议 ===================
#         print("\n🤝 步骤3: 计算上下文向量（综合朋友建议）")
#
#         # 加权平均 - 按比例组合所有朋友的建议
#         # 权重高的朋友，他的建议影响更大
#         context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
#         context_vector = context_vector.squeeze(1)              # [batch_size, hidden_dim]
#
#         print(f"   最终上下文向量形状: {context_vector.shape}")
#         print("   💡 这就是综合所有朋友建议后的最终答案！")
#
#         return context_vector, attention_weights
#
# def demonstrate_attention():
#     """
#     🎪 注意力机制现场演示
#
#     用具体的数字来展示注意力机制是如何工作的，
#     让抽象的概念变得具体可见！
#     """
#
#     print("🎬 注意力机制现场演示开始！")
#     print("=" * 60)
#
#     # 设置演示参数
#     batch_size = 1          # 一次处理一个句子
#     seq_len = 4             # 输入句子有4个词（比如 "I love machine learning"）
#     hidden_dim = 8          # 每个词用8维向量表示
#
#     print(f"📝 演示设置：")
#     print(f"   句子长度: {seq_len}个词")
#     print(f"   向量维度: {hidden_dim}维")
#     print(f"   想象句子: 'I love machine learning'")
#     print(f"   要翻译的词: 'machine' → '机器'")
#
#     # 创建模拟数据（这些通常是神经网络训练出来的）
#     print("\n🎲 创建模拟数据...")
#     torch.manual_seed(42)           # 设置随机种子，确保结果可重现
#
#     # 解码器当前状态：表示“我现在要翻译machine这个词”
#     decoder_hidden = torch.randn(batch_size, hidden_dim)
#     print(f"    解码器状态（要翻译'machine'）：已创建")
#
#     # 编码器输出：表示每个输入词的信息
#     encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
#     print(f"    编码器输出（4个词的信息）：已创建")
#
#     # 创建注意力模块
#     print("\n🏗️ 创建注意力模块...")
#     attention = SimpleAttention(hidden_dim)
#
#     # 🎬 开始注意力计算！
#     print("\n🚀 开始注意力计算...")
#     context_vector, attention_weights = attention(decoder_hidden, encoder_outputs)
#
#     # 📊 结果分析
#     print("\n" + "="*60)
#     print("📊 计算结果分析：")
#     print("="*60)
#
#     weights_array = attention_weights.squeeze().detach().numpy()
#
#     # 创建词汇标签（模拟）
#     word_labels = ['I', 'love', 'machine', 'learning']
#
#     print("\n🎯 注意力权重分析：")
#     for i, (word, weight) in enumerate(zip(word_labels, weights_array)):
#         percentage = weight * 100
#         bar = "█" * int(percentage // 5)                # 简单的条形图
#         print(f"   {word:>8}: {weight:.3f} ({percentage:5.1f}%) {bar}")
#
#     # 找出最关注的词
#     max_idx = weights_array.argmax()
#     max_word = word_labels[max_idx]
#     max_weight = weights_array[max_idx]
#
#     print(f"\n💡 模型最关注: '{max_word}' (权重: {max_weight:.3f})")
#     print(f"   这说明翻译'machine'时，模型主要参考'{max_word}'这个词！")
#
#     # 可视化注意力权重
#     print("\n🎨 绘制可视化图表...")
#     plt.figure(figsize=(14, 5))
#
#     # 左图：条形图
#     plt.subplot(1, 3, 1)
#     colors = ['lightblue' if i != max_idx else 'red' for i in range(len(weights_array))]
#     bars = plt.bar(range(len(weights_array)), weights_array, color=colors, edgecolor='navy')
#     plt.title('注意力权重分布', fontsize=14, fontweight='bold')
#     plt.xlabel('输入词')
#     plt.ylabel('注意力权重')
#     plt.xticks(range(len(word_labels)), word_labels, rotation=45)
#     plt.ylim(0, 1)
#     plt.grid(True, alpha=0.3)
#
#     # 添加数值标签
#     for bar, weight in zip(bars, weights_array):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                  f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
#
#     # 中图：饼图
#     plt.subplot(1, 3, 2)
#     colors_pie = ['lightblue', 'lightgreen', 'red', 'lightyellow']
#     plt.pie(weights_array, labels=word_labels, autopct='%1.1f%%',
#             colors=colors_pie, startangle=90)
#     plt.title('注意力权重比例', fontsize=14, fontweight='bold')
#
#     # 右图：热力图
#     plt.subplot(1, 3, 3)
#     weights_matrix = weights_array.reshape(1, -1)
#     plt.imshow(weights_matrix, cmap='Reds', aspect='auto')
#     plt.title('注意力热力图', fontsize=14, fontweight='bold')
#     plt.xticks(range(len(word_labels)), word_labels)
#     plt.yticks([0], ['attention'])
#
#     # 添加数值
#     for i, weight in enumerate(weights_array):
#         plt.text(i, 0, f'{weight:.2f}', ha='center', va='center',
#                  color='white' if weight > 0.5 else 'black', fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#     print("\n" + "=" * 60)
#     print("🎉 演示完成！")
#     print("✅ 这就是注意力机制的核心计算过程！")
#     print("💡 模型学会了动态地选择最相关的输入信息！")
#     print("=" * 60)
#
# # 🎬 开始演示！
# print("🎪 欢迎来到注意力机制现场演示！")
# demonstrate_attention()


# 动手实践：构建完整的注意力翻译系统
"""
📊 第一步：数据准备 - 为我们的翻译机器人准备"学习材料"

就像教孩子学英语需要准备课本一样，我们的翻译机器人也需要大量的中英对照句子来学习。
这一步我们将从cmn.txt文件中加载真实的翻译数据集，并把文字转换成机器能理解的数字。
"""

import re
import string
class EnhancedTranslationDatast:
    """
    增强版翻译数据集 - 我们的"电子课本"

    这个类的作用就像一个智能的语言课本，它能：
    1. 从cmn.txt文件中加载大量的中英对照句子
    2. 把文字转换成机器能理解的数字
    3. 为训练过程提供规整的数据
    """

    def __init__(self, data_file="cmn.txt", max_pairs=5000):
        print("📚 正在准备翻译数据集...")
        print(f"📁 从文件 {data_file} 加载数据...")

        # 从cmn.txt文件中加载数据
        self.pairs = self.load_data_from_file(data_file, max_pairs)

        print(f"📝 成功加载了 {len(self.pairs)} 对中英句子")
        print("💡 这些句子来自真实的翻译数据集")

        # 开始构建词汇表
        self.prepare_vocabularies()

    def load_data_from_file(self, filename, max_pairs=5000):
        """
        从cmn.txt文件中加载翻译数据

        cmn.txt文件格式通常是：
        英文句子 \t 中文句子 \t 其他信息
        """
        pairs = []

        try:
            with open(filename, 'r', encoding='utf-8') as file:
                print("📖 正在读取数据文件...")

                for line_num, line in enumerate(file):
                    if line_num >= max_pairs:
                        break

                    # 去除换行符并分割
                    line = line.strip()
                    if not line:
                        continue

                    # 通常cmn.txt格式是用制表符分隔的
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        en_sentence = parts[0].strip()
                        zh_sentence = parts[1].strip()

                        # 数据清洗：去除标点和特殊字符
                        en_sentence = self.clean_english_sentence(en_sentence)
                        zh_sentence = self.clean_chinese_sentence(zh_sentence)

                        # 过滤掉过长或过短的句子
                        if 3 <= len(en_sentence.split()) <= 12 and 2 <= len(zh_sentence) <= 15:
                            pairs.append((en_sentence, zh_sentence))

                    # 显示进度
                    if (line_num + 1) % 1000 == 0:
                        print(f"   已处理 {line_num + 1} 行...")

            print(f"✅ 文件读取完成！共处理了 {line_num + 1} 行")
            print(f"✅ 筛选出 {len(pairs)} 对符合条件的句子")

        except FileNotFoundError:
            print(f"❌ 找不到文件 {filename}")
            print("💡 使用默认的示例数据...")
            # 如果文件不存在，使用默认数据
            pairs = [
                ("I love you", "我爱你"),
                ("Hello world", "你好世界"),
                ("Good morning", "早上好"),
                ("How are you", "你好吗"),
                ("Thank you very much", "非常感谢你"),
                ("See you later", "再见"),
                ("I am very happy", "我非常开心"),
                ("This is really good", "这真的很好"),
                ("I like eating apples", "我喜欢吃苹果"),
                ("Today is very sunny", "今天非常晴朗"),
                ("I want to drink water", "我想喝水"),
                ("You are very nice", "你很好"),
                ("I need your help", "我需要你的帮助"),
                ("This book is easy", "这本书很容易"),
                ("I want to go home", "我想回家"),
                ("The red car is fast", "红色汽车很快"),
                ("She likes beautiful flowers", "她喜欢美丽的花"),
                ("We study machine learning", "我们学习机器学习"),
                ("The weather is nice today", "今天天气很好"),
                ("I enjoy reading books", "我喜欢读书"),
            ]

        # 显示数据样本
        print(f"\n📋 数据样本预览:")
        for i, (en, zh) in enumerate(pairs[:5]):
            print(f"   样本{i + 1}: '{en}' → '{zh}'")

        return pairs

    def clean_english_sentence(self, sentence):
        """清洗英文句子：去除特殊字符，统一格式"""
        # 转换为小写
        sentence = sentence.lower()
        # 去除标点符号（保留基本标点）
        sentence = re.sub(r'[^\w\s\']', '', sentence)
        # 去除多余空格
        sentence = ' '.join(sentence.split())
        return sentence

    def clean_chinese_sentence(self, sentence):
        """清洗中文句子：去除特殊字符，统一格式"""
        # 去除英文字符和标点
        sentence = re.sub(r'[a-zA-Z\d\s.,!?;:\"\'()[\]{}]', '', sentence)
        # 去除特殊标点符号
        sentence = re.sub(r'[。，！？；：""''（）【】{}]', '', sentence)
        return sentence.strip()

    def prepare_vocabularies(self):
        """
        构建词汇表 - 制作"字典"

        就像学外语要先制作字典一样，我们需要：
        1. 找出所有出现的英文单词和中文词汇
        2. 给每个词分配一个唯一的数字ID
        3. 添加特殊标记（开始、结束、未知词等）
        """
        print("\n🔨 开始构建词汇表...")

        # 特殊标记 - 就像标点符号一样重要
        # <PAD>: 填充符，用于让所有句子长度一致
        # <START>: 句子开始标记
        # <END>: 句子结束标记
        # <UNK>: 未知词标记，用于处理没见过的词
        self.en_vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.zh_vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}

        print("   添加特殊标记: <PAD>, <START>, <END>, <UNK>")

        # 收集所有词汇
        en_words = set()  # 英文单词集合
        zh_words = set()  # 中文词汇集合

        print("\n🔍 扫描所有句子，收集词汇...")
        for i, (en_sentence, zh_sentence) in enumerate(self.pairs):
            # 英文按空格分词
            en_words.update(en_sentence.lower().split())
            # 中文使用jieba分词
            zh_words.update(jieba.cut(zh_sentence))

            if i < 3:               # 显示前3个例子
                print(f"   例子{i + 1}: '{en_sentence}' → '{zh_sentence}'")
                print(f"           英文词: {en_sentence.lower().split()}")
                print(f"           中文词: {list(jieba.cut(zh_sentence))}")

        print(f"\n📊 词汇统计:")
        print(f"   发现英文单词: {len(en_words)} 个")
        print(f"   发现中文词汇: {len(zh_words)} 个")

        # 构建词汇表（给每个词分配ID）
        # 从ID=4开始，因为0-3被特殊标记占用
        for i, word in enumerate(sorted(en_words), 4):
            self.en_vocab[word] = i

        for i, word in enumerate(sorted(zh_words), 4):
            self.zh_vocab[word] = i

        # 创建反向词汇表（从ID查找词汇）
        self.en_idx2word = {idx: word for word, idx in self.en_vocab.items()}
        self.zh_idx2word = {idx: word for word, idx in self.zh_vocab.items()}

        print(f"\n✅ 词汇表构建完成！")
        print(f"   英文词汇表大小: {len(self.en_vocab)}")
        print(f"   中文词汇表大小: {len(self.zh_vocab)}")

        # 显示一些词汇表内容作为例子
        print(f"\n📖 英文词汇表示例:")
        for word, idx in list(self.en_vocab.items())[:8]:
            print(f"   '{word}' → {idx}")

        print(f"\n📖 中文词汇表示例:")
        for word, idx in list(self.zh_vocab.items())[:8]:
            print(f"   '{word}' → {idx}")

    def sentence_to_indices(self, sentence, vocab, is_chinese=False):
        """
        将句子转换为数字序列

        机器只能理解数字，所以我们要把文字句子转换成数字序列。
        就像把 "I love you" 转换成 [5, 12, 8] 这样的数字列表。
        """
        if is_chinese:
            words = list(jieba.cut(sentence))           # 中文分词
        else:
            words = sentence.lower().split()            # 英文按空格分词

        # 查找每个词的ID，如果词汇表中没有就用<UNK>
        indices = [vocab.get(word, vocab["<UNK>"]) for word in words]
        return indices

    def get_training_data(self):
        """
        获取训练数据 - 把所有句子转换成数字序列

        这一步将所有的中英句子对转换成机器学习需要的数字格式。
        """
        print("\n🔄 将所有句子转换为数字序列...")

        en_sequences = []
        zh_sequences = []

        for i, (en_sentence, zh_sentence) in enumerate(self.pairs):
            # 转换英文句子
            en_indices = self.sentence_to_indices(en_sentence, self.en_vocab, False)

            # 转换中文句子（注意：中文句子前后要加<START>和<END>）
            zh_indices = ([self.zh_vocab["<START>"]] + self.sentence_to_indices(zh_sentence, self.zh_vocab, True) + [self.en_vocab["<END>"]])

            en_sequences.append(en_indices)
            zh_sequences.append(zh_indices)

            # 显示前3个转换例子
            if i < 3:
                print(f"\n   例子{i + 1}:")
                print(f"   英文: '{en_sentence}' → {en_indices}")
                print(f"   中文: '{zh_sentence}' → {zh_indices}")

        print(f"\n✅ 转换完成！共处理 {len(en_sequences)} 个句子对")
        return en_sequences, zh_sequences

# def pad_sequences(sequences, max_length=None, pad_value=0):
#     """
#     序列填充函数 - 让所有句子长度一致
#
#     就像排队时要站整齐一样，我们需要让所有句子长度一致，
#     这样机器才能批量处理。短句子用<PAD>填充到统一长度。
#     """
#     if max_length is None:
#         max_length = max(len(seq) for seq in sequences)
#
#     print(f"\n📏 填充序列到统一长度 {max_length}...")
#
#     padded_sequences = []
#     for seq in sequences:
#         if len(seq) < max_length:
#             # 短句子用PAD填充
#             padded_seq = seq + [pad_value] * (max_length - len(seq))
#         else:
#             # 长句子截断（虽然我们的数据集中不会出现这种情况
#             padded_seq = seq[:max_length]
#         padded_sequences.append(padded_seq)
#
#     return padded_sequences
#
# # 🚀 开始数据准备过程！
# print("🎬 开始数据准备过程...")
# print("="*60)
#
# # 创建数据集
# dataset = EnhancedTranslationDatast()
#
# # 获取训练数据
# en_sequences, zh_sequences = dataset.get_training_data()
#
# # 计算序列长度统计
# max_en_length = max(len(seq) for seq in en_sequences)
# max_zh_length = max(len(seq) for seq in zh_sequences)
#
# print(f"\n📊 序列长度统计:")
# print(f"   最长英文句子: {max_en_length} 个词")
# print(f"   最长中文句子: {max_zh_length} 个词")
#
# # 展示长度分布
# en_lengths = [len(seq) for seq in en_sequences]
# zh_lengths = [len(seq) for seq in zh_sequences]
# print(f"   英文长度分布: 最短{min(en_lengths)}, 最长{max(en_lengths)}, 平均{sum(en_lengths)/len(en_lengths):.1f}")
# print(f"   中文长度分布: 最短{min(zh_lengths)}, 最长{max(zh_lengths)}, 平均{sum(zh_lengths)/len(zh_lengths):.1f}")
#
# # 填充序列
# en_padded = pad_sequences(en_sequences, max_en_length, dataset.en_vocab['<PAD>'])
# zh_padded = pad_sequences(zh_sequences, max_zh_length, dataset.zh_vocab['<PAD>'])
#
# # 转换为PyTorch张量
# en_tensor = torch.tensor(en_padded, dtype=torch.long)
# zh_tensor = torch.tensor(zh_padded, dtype=torch.long)
#
# print(f"\n🎯 最终数据格式:")
# print(f"   英文张量形状: {en_tensor.shape} (句子数 × 最大长度)")
# print(f"   中文张量形状: {zh_tensor.shape} (句子数 × 最大长度)")
#
# print("\n" + "="*60)
# print("✅ 数据准备完成！我们的翻译机器人现在有了学习材料！")
# print("💡 接下来我们将构建模型的三个核心部件...")
# print("="*60)


# 📚 第二步：词汇表构建与数据转换
print("🔧 开始词汇表构建阶段...")
print("="*60)

# 创建数据集实例（这会自动加载数据并构建词汇表）
dataset = EnhancedTranslationDatast(data_file='cmn.txt', max_pairs=3000)

print("\n" + "="*60)
print("✅ 数据加载和词汇表构建完成！")
print("="*60)


# 📝 第三步：将文本转换为数字序列
print("🔄 开始文本到数字的转换...")
print("="*50)

# 获取训练数据（转换为数字序列）
en_sequences, zh_sequences = dataset.get_training_data()

# 📊 分析序列长度分布
max_en_length = max(len(seq) for seq in en_sequences)
max_zh_length = max(len(seq) for seq in zh_sequences)

print(f"\n📊 序列长度统计:")
print(f"  最长英文句子：{max_en_length}个词")
print(f"  最长中文句子：{max_zh_length}个词")

# 展示长度分布
en_lengths = [len(seq) for seq in en_sequences]
zh_lengths = [len(seq) for seq in zh_sequences]
print(f"  英文长度分布：最短{min(en_lengths)}，最长{max(en_lengths)}，平均{sum(en_lengths)/len(en_lengths):.1f}")
print(f"  中文长度分布：最短{min(zh_lengths)}，最长{max(zh_lengths)}，平均{sum(zh_lengths)/len(zh_lengths):.1f}")

print("\n" + "="*50)
print("✅ 文本转换完成！")
print("="*50)


# 📏 第四步：序列填充与张量转换
print("📐 开始序列填充处理...")
print("="*50)

def pad_sequences(sequences, max_length=None, pad_value=0):
    """
    序列填充函数 - 让所有句子长度一致

    就像排队时要站整齐一样，我们需要让所有句子长度一致，
    这样机器才能批量处理。短句子用<PAD>填充到统一长度。
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    print(f"   📏 填充序列到统一长度 {max_length}...")

    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # 短句子用PAD填充
            padded_seq = seq + [pad_value] * (max_length - len(seq))
        else:
            # 长句子截断（虽然我们的数据集中不会出现这种情况）
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)

    return padded_sequences

#执行填充
en_padded = pad_sequences(en_sequences, max_en_length, dataset.en_vocab["<PAD>"])
zh_padded = pad_sequences(zh_sequences, max_zh_length, dataset.zh_vocab["<PAD>"])

# 转换为PyTorch张量
en_tensor = torch.tensor(en_padded, dtype=torch.long)
zh_tensor = torch.tensor(zh_padded, dtype=torch.long)

print(f"\n🎯 最终数据格式:")
print(f"  英文张量形状：{en_tensor.shape}（句子数×最大长度）")
print(f"  中文张量形状：{zh_tensor.shape}（句子数×最大长度）")

# 显示填充前后的对比示例
print(f"\n📋 填充效果示例（前3个句子）:")
for i in range(3):
    print(f"  句子{i+1}:")
    print(f"    原始英文长度：{len(en_sequences[i])}")
    print(f"    填充后长度：{len(en_padded[i])}")
    print(f"    原始中文长度：{len(zh_sequences[i])}")
    print(f"    填充后长度：{len(zh_padded[i])}")

print("\n" + "="*50)
print("✅ 数据预处理全部完成！我们的翻译机器人现在有了学习材料！")
print("💡 接下来我们将构建模型的三个核心部件...")
print("="*50)


# 🧠 组件1：编码器 - "理解专家"
print("🏗️ 开始构建编码器...")

class AttentionEncoder(nn.Module):
    """
    🧠 编码器 - "理解专家"

    这是我们翻译机器人的第一个部件，它的工作是：
    1. 接收英文句子
    2. 理解每个单词的含义
    3. 保存所有单词的信息（不压缩！）

    与传统Seq2Seq不同，这里我们保留所有时间步的输出，
    而不是只保留最后一个状态。这就是注意力机制的关键！
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(AttentionEncoder, self).__init__()

        print(f"🧠 构建编码器（理解专家）:")
        print(f"  词汇量：{vocab_size}")
        print(f"  词嵌入维度：{embedding_dim}")
        print(f"  隐藏层维度：{hidden_dim}")

        # 词嵌入层：把词汇ID转换成密集向量
        # 就像给每个英文单词制作一张“身份证”
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM层：理解句子的上下文和语法
        # 就像一个英文专家，能理解句子的深层含义
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input_sequence):
        """
        编码器的工作流程

        输入: 英文句子的数字序列，如 [5, 12, 8, 3] 表示 "I love machine learning"
        输出: 每个英文单词的深度理解表示
        """
        # 步骤1：词嵌入 - 把数字ID转换成向量
        embedded = self.embedding(input_sequence)

        # 步骤2：LSTM处理 - 理解上下文
        # ⚠️ 关键：我们返回所有时间步的输出，不只是最后一个！
        encoder_outputs, (hidden, cell) = self.lstm(embedded)

        return encoder_outputs, (hidden, cell)

print("✅ 编码器构建完成！")

# 👁️ 组件2：Bahdanau注意力机制 - "关注专家"
print("🏗️ 开始构建注意力模块...")

class BahdanauAttention(nn.Module):
    """
    👁️ Bahdanau注意力机制 - "关注专家"

    这是我们翻译机器人的第二个部件，它的工作是：
    1. 分析当前要翻译的词需要什么帮助
    2. 扫描所有英文单词，评估每个词的有用程度
    3. 计算注意力权重分布

    这就是我们之前学习的"选朋友帮忙"算法的专业实现！
    """

    def __init__(self, hidden_dim, attention_type='Bahdanau'):
        super(BahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type

        print(f"👁️ 构建注意力模块（关注专家）:")
        print(f"   使用 {attention_type} 注意力机制")
        print(f"   隐藏维度: {hidden_dim}")

        if attention_type == 'Bahdanau':
            # 三个线性变换层 - Bahdanau注意力的核心组件
            # 这三个层实现了我们学过的“相关性计算”公式

            # W_h: 处理编码器输出（朋友们的专长）
            self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
            print("   📊 W_h: 处理编码器输出的变换层")

            # W_s: 处理解码器状态（当前的困惑）
            self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
            print("   📊 W_s: 处理解码器状态的变换层")

            # v: 计算最终分数（综合评估）
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            print("   📊 v: 计算最终相关性分数的层")

    def forward(self, decoder_hidden, encoder_outputs):
        """
        注意力计算的核心过程

        这就是我们学过的"选朋友帮忙三步法"的代码实现！

        参数:
            decoder_hidden: 当前的"困惑状态" [batch_size, hidden_dim]
            encoder_outputs: 所有"朋友的专长" [batch_size, seq_len, hidden_dim]
        """
        seq_len = encoder_outputs.size(1)

        # 🎯 步骤1: 准备数据 - 让decoder_hidden和encoder_outputs尺寸匹配
        # 将decoder_hidden从[batch_size, hidden_dim]扩展为[batch_size, seq_len, hidden_dim]
        # 1. unsqueeze(1)在第1维增加一个维度,变成[batch_size, 1, hidden_dim]
        # 2. repeat(1, seq_len, 1)将第1维重复seq_len次,保持其他维度不变
        # 这样做是为了让decoder_hidden的维度与encoder_outputs匹配,便于后续计算
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # 🎯 步骤2: 计算相关性分数
        if self.attention_type == 'dot':
            # 点积注意力计算
            # 公式: e_ij = h_j · s_i
            # 使用批量矩阵乘法(bmm)计算点积注意力分数
            scores = torch.bmm(encoder_outputs, decoder_hidden_expanded)

        else:
            # Bahdanau注意力计算
            # 公式：e_ij = v^T * tanh(W_h * h_j + W_s * s_i)
            # 1. 使用W_h变换层处理编码器输出(朋友专长)
            # 输入: encoder_outputs [batch_size, seq_len, hidden_dim]
            # 输出: W_h_out [batch_size, seq_len, hidden_dim]
            W_h_out = self.W_h(encoder_outputs)             # 处理“朋友专长”

            # 2. 使用W_s变换层处理解码器状态(当前困惑)
            # 输入: decoder_hidden_expanded [batch_size, seq_len, hidden_dim]
            # 输出: W_s_out [batch_size, seq_len, hidden_dim]
            W_s_out = self.W_s(decoder_hidden_expanded)     # 处理“当前困惑”

            # 3. 将两个变化结果相加并通过tanh激活函数
            # 输入: W_h_out + W_s_out [batch_size, seq_len, hidden_dim]
            # 输出: energy [batch_size, seq_len, hidden_dim]
            energy = torch.tanh(W_h_out + W_s_out)          # 非线性组合

            # 4. 使用v层计算最终注意力分数并压缩最后一维
            # 输入: energy [batch_size, seq_len, hidden_dim]
            # 输出: scores [batch_size, seq_len]
            scores = self.v(energy).squeeze(2)              # 计算最终分数

        # 🎯 步骤3: 计算注意力权重 - softmax归一化
        attention_weights = F.softmax(scores, dim=1)

        # 🎯 步骤4: 计算上下文向量 - 加权平均
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights

print("✅ 注意力模块构建完成！")


# ✍️ 组件3：解码器 - "翻译专家"
print("🏗️ 开始构建解码器...")

class AttentionDecoder(nn.Module):
    """
    ✍️ 解码器 - "翻译专家"

    这是我们翻译机器人的第三个部件，它的工作是：
    1. 接收注意力模块提供的上下文信息
    2. 结合当前输入词的信息
    3. 生成下一个中文词

    与传统解码器不同，这里我们使用注意力提供的上下文信息，
    而不是固定的编码器最终状态。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention, n_layers=1):
        super(AttentionDecoder, self).__init__()

        print(f"✍️ 构建解码器（翻译专家）:")
        print(f"   中文词汇量: {vocab_size}")
        print(f"   词嵌入维度: {embedding_dim}")
        print(f"   隐藏层维度: {hidden_dim}")

        # 中文词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 注意力模块（我们刚才创建的）
        self.attention = attention

        # ⚠️ 关键设计：LSTM的输入是词嵌入 + 上下文向量的拼接
        # 这样解码器就能同时使用"当前词信息"和"注意力上下文"
        lstm_input_size = embedding_dim + hidden_dim
        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        print(f"   🔗 LSTM输入维度: {lstm_input_size} (词嵌入 + 上下文)")

        # 输出投影层：把LSTM输出转换成词汇概率分布
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        print(f"   📤 输出层: {hidden_dim} → {vocab_size}")

    def forward(self, input_token, hidden_state, encoder_outputs):
        """
        解码器的工作流程 - 生成一个中文词

        这是"边看边译"的核心实现：
        1. 根据当前状态计算注意力
        2. 获得当前最相关的英文信息
        3. 结合当前输入生成下一个中文词
        """
        # 步骤1：词嵌入
        embedded = self.embedding(input_token)

        # 步骤2：计算注意力 - 这是关键！
        # 使用当前解码器状态来决定关注哪些英文词
        current_hidden = hidden_state[0][-1]            # 取最后一层的隐藏状态
        context_vector, attention_weights = self.attention(current_hidden, encoder_outputs)

        # 步骤3：拼接输入
        # 把“当前词信息”和“注意力上下文”组合起来
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)

        # 步骤4：LSTM处理
        lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)

        # 步骤5：生成输出概率分布
        output = self.output_projection(lstm_output)

        return output, new_hidden_state, attention_weights

print("✅ 解码器构建完成！")


# 🤖 组装完整的翻译系统
print("🔧 开始组装完整的翻译系统...")

class AttentionSeq2Seq(nn.Module):
    """
    🤖 完整的带注意力机制的翻译系统

    这是我们的总指挥官，它协调三个部件：
    🧠 编码器 + 👁️ 注意力模块 + ✍️ 解码器

    实现了完整的"边看边译"翻译流程！
    """

    def __init__(self, en_vocab_size, zh_vocab_size, embedding_dim, hidden_dim):
        super(AttentionSeq2Seq, self).__init__()

        print(f"🤖 组装完整翻译系统...")
        print(f"   英文词汇量: {en_vocab_size}")
        print(f"   中文词汇量: {zh_vocab_size}")
        print(f"   嵌入维度: {embedding_dim}")
        print(f"   隐藏维度: {hidden_dim}")

        # 创建三个核心组件
        print("\n🔧 创建系统组件:")

        # 组件1：编码器
        self.encoder = AttentionEncoder(en_vocab_size, embedding_dim, hidden_dim)

        # 组件2：注意力模块
        self.attention = BahdanauAttention(hidden_dim)

        # 组件3：解码器（使用上面创建的注意力模块）
        self.decoder = AttentionDecoder(zh_vocab_size, embedding_dim, hidden_dim, self.attention)

        self.zh_vocab_size = zh_vocab_size

        print("\n✅ 翻译系统组装完成！")

    def forward(self, en_sequence, zh_sequence=None, teacher_forcing_ratio=0.5):
        """
        翻译系统的主工作流程

        支持两种模式：
        1. 训练模式：有正确答案，用于学习
        2. 推理模式：没有答案，真正翻译
        """
        batch_size = en_sequence.size(0)

        # 第一步：编码 - 理解英文
        encoder_outputs, encoder_hidden = self.encoder(en_sequence)

        if zh_sequence is not None:
            # 训练模式
            return self._train_forward(encoder_outputs, encoder_hidden, zh_sequence, teacher_forcing_ratio)
        else:
            # 推理模式
            return self._inference_forward(encoder_outputs, encoder_hidden, max_length=15)

    def _train_forward(self, encoder_outputs, encoder_hidden, target_sequence, teacher_forcing_ratio):
        """训练模式：有标准答案，边学边练"""
        batch_size, target_length = target_sequence.shape
        outputs = torch.zeros(batch_size, target_length - 1, self.zh_vocab_size).to(device)
        all_attention_weights = []

        decoder_hidden = encoder_hidden
        decoder_input = target_sequence[:, 0].unsqueeze(1)          # <START>标记

        for t in range(1, target_length):
            # 生成当前词
            output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            outputs[:, t-1, :] = output.squeeze(1)
            all_attention_weights.append(attention_weights)

            # Teacher Forcing策略
            if random.random() < teacher_forcing_ratio:
                # 使用正确答案（老师指导）
                decoder_input = target_sequence[:, t].unsqueeze(1)
            else:
                # 使用模型预测（独立思考）
                decoder_input = output.argmax(2)

        return outputs, torch.stack(all_attention_weights, dim=1)

    def _inference_forward(self, encoder_outputs, encoder_hidden, max_length):
        """推理模式：真正的翻译，逐词生成直到句子结束"""
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_hidden
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).to(device)          # <START>

        outputs = []
        attention_weights_list = []

        for step in range(max_length):
            # 生成下一个词
            output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            predicted = output.argmax(2)
            outputs.append(predicted.squeeze(1))
            attention_weights_list.append(attention_weights)

            # 下一步的输入就是当前的预测
            decoder_input = predicted

            # 如果生成了<END>标记就停止
            if predicted.item() == 2:
                break

        if outputs:
            return torch.stack(outputs, dim=1), torch.stack(attention_weights_list, dim=1)
        else:
            return torch.empty(batch_size, 0, dtype=torch.long).to(device), torch.empty(batch_size, 0, encoder_outputs.size(1)).to(device)

# 🚀 创建我们的翻译系统！
print("\n" + "="*60)
print("🎬 开始创建完整的注意力翻译系统...")
print("="*60)

attention_model = AttentionSeq2Seq(
    en_vocab_size=len(dataset.en_vocab),
    zh_vocab_size=len(dataset.zh_vocab),
    embedding_dim=128,                          # 词嵌入维度
    hidden_dim=256                              # 隐藏层维度
).to(device)

# 统计模型参数
total_params = sum(p.numel() for p in attention_model.parameters())
trainable_params = sum(p.numel() for p in attention_model.parameters() if p.requires_grad)

print("\n" + "="*60)
print("✅ 带注意力机制的翻译系统创建成功！")
print(f"📊 模型统计:")
print(f"   总参数量: {total_params:,}")
print(f"   可训练参数: {trainable_params:,}")
print(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
print("💡 我们的翻译机器人已经就绪，准备开始学习！")
print("="*60)


# 🎯 训练函数：教机器人学翻译
def train_attention_model(model, en_tensor, zh_tensor, epochs=300):
    """
    训练带注意力机制的翻译模型

    参数:
        model: 我们构建的注意力翻译模型
        en_tensor: 英文数据张量
        zh_tensor: 中文数据张量
        epochs: 训练轮数
    """
    print("🏋️ 开始训练带注意力机制的翻译模型...")
    print("=" * 50)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)             # 忽略<PAD>标记
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    print("🔧 训练配置:")
    print(f"   损失函数: CrossEntropyLoss (忽略PAD)")
    print(f"   优化器: Adam (学习率=0.001)")
    print(f"   训练轮数: {epochs}")
    print(f"   数据量: {len(en_tensor)} 个句子对")

    # 移动数据到设备
    en_tensor = en_tensor.to(device)
    zh_tensor = zh_tensor.to(device)

    model.train()                           # 设置为训练模式
    losses = []                             # 记录损失值

    print("\n🚀 开始训练循环...")
    for epoch in range(epochs):
        optimizer.zero_grad()               # 清零维度

        # 前向传播：让模型尝试翻译
        outputs, attention_weights = model(en_tensor, zh_tensor, teacher_forcing_ratio=0.8)

        # 计算损失：对比模型输出和正确答案
        loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                         zh_tensor[:, 1:].reshape(-1))

        # 反向传播：计算梯度
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新：根据梯度调整参数
        optimizer.step()

        # 记录损失
        losses.append(loss.item())

        # 定期输出训练进度
        if (epoch + 1) % 50 == 0:
            print(f"📊 Epoch {epoch + 1:3d}/{epochs}, Loss: {loss.item():.4f}")

            # 显示训练进度条
            progress = (epoch + 1) / epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"    进度: |{bar}| {progress:.1%}")

    print("\n✅ 训练完成！")
    return losses

# 🎬 开始训练！
print("🎬 开始训练我们的注意力翻译模型...")
print("💡 这个过程可能需要几分钟，请耐心等待...")
print("="*60)

attention_losses = train_attention_model(attention_model, en_tensor, zh_tensor, epochs=300)

print("\n" + "="*60)
print("🎉 模型训练完成！我们的翻译机器人学会翻译了！")
print("="*60)


# 📈 训练结果可视化
print("📊 分析训练结果...")

# 绘制训练损失曲线
plt.figure(figsize=(15, 5))

# 左图：完整训练过程
plt.subplot(1, 3, 1)
plt.plot(attention_losses, label='训练损失', color='red', linewidth=2)
plt.title('完整训练损失曲线', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

# 中图：训练后期损失（更清楚地看到收敛情况）
plt.subplot(1, 3, 2)
plt.plot(attention_losses[-100:], label='最后100轮', color='red', linewidth=2)
plt.title('训练后期损失曲线', fontsize=14, fontweight='bold')
plt.xlabel('训练轮次（最后100轮）')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：损失下降统计
plt.subplot(1, 3, 3)
initial_loss = attention_losses[0]
final_loss = attention_losses[-1]
min_loss = min(attention_losses)

stats = ['初始损失', '最终损失', '最低损失']
values = [initial_loss, final_loss, min_loss]
colors = ['lightcoral', 'lightgreen', 'lightblue']

bars = plt.bar(stats, values, color=colors, edgecolor='black', linewidth=1)
plt.title('训练效果统计', fontsize=14, fontweight='bold')
plt.ylabel('损失值')

# 添加数值标签
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出训练效果分析
print("\n📈 训练效果分析:")
print(f"   初始损失: {initial_loss:.4f}")
print(f"   最终损失: {final_loss:.4f}")
print(f"   最低损失: {min_loss:.4f}")
print(f"   损失降幅: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

# 判断训练效果
if final_loss < initial_loss * 0.3:
    print("   🎉 训练效果: 非常好！模型学习充分")
elif final_loss < initial_loss * 0.5:
    print("   😊 训练效果: 良好，模型基本学会")
else:
    print("   😐 训练效果: 一般，可能需要更多训练")

print("\n✅ 训练结果分析完成！接下来我们测试翻译效果...")


# 🔮 翻译测试函数：验证模型效果
def translate_with_attention(model, sentence, dataset, visualize=True):
    """
    使用带注意力机制的模型进行翻译，并可视化注意力权重

    参数:
        model: 训练好的翻译模型
        sentence: 要翻译的英文句子
        dataset: 数据集（包含词汇表）
        visualize: 是否可视化注意力权重
    """
    model.eval()  # 设置为评估模式

    print(f"🔤 正在翻译: '{sentence}'")

    with torch.no_grad():
        # 步骤1: 预处理输入句子
        print("   🔧 预处理输入...")
        en_indices = dataset.sentence_to_indices(sentence, dataset.en_vocab, False)

        # 填充到与训练数据相同的长度
        en_padded = en_indices + [dataset.en_vocab["<PAD>"]] * (max_en_length - len(en_indices))
        en_tensor = torch.tensor([en_padded], dtype=torch.long).to(device)

        print(f"      原句长度: {len(en_indices)} 词")
        print(f"      填充后长度: {len(en_padded)} 词")

        # 步骤2: 生成翻译和注意力权重
        print("   🤖 模型推理中...")
        output_indices, attention_weights = model(en_tensor)

        # 步骤3: 转换回中文
        print("   📝 生成译文...")
        translated_words = []
        for idx in output_indices[0]:
            word = dataset.zh_idx2word.get(idx.item(), "<UNK>")
            if word == "<END>":
                break
            if word not in ["<START>", "<PAD>"]:
                translated_words.append(word)

        translation = "".join(translated_words)

        print(f"   ✅ 翻译完成: '{translation}'")

        # 步骤4: 可视化注意力权重
        if visualize and len(translated_words) > 0:
            print("   🎨 生成注意力可视化...")
            visualize_attention_weights(
                sentence.split(),
                translated_words,
                attention_weights[0][:len(translated_words), :len(sentence.split())].cpu().numpy(),
                sentence
            )

        return translation


def visualize_attention_weights(input_words, output_words, attention_weights, original_sentence):
    """可视化注意力权重热力图"""
    plt.figure(figsize=(12, 8))

    # 创建热力图
    im = plt.imshow(attention_weights, cmap='Blues', aspect='auto')

    # 设置坐标轴标签
    plt.xticks(range(len(input_words)), input_words, rotation=45)
    plt.yticks(range(len(output_words)), output_words)

    # 添加数值标注
    for i in range(len(output_words)):
        for j in range(len(input_words)):
            plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                     ha="center", va="center",
                     color="white" if attention_weights[i, j] > 0.5 else "black",
                     fontsize=10, fontweight='bold')

    # 设置标题和标签
    plt.title(f'注意力权重可视化\n原句: "{original_sentence}"',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('US 输入词（英文）', fontsize=12, fontweight='bold')
    plt.ylabel('CN 输出词（中文）', fontsize=12, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('注意力权重', fontsize=12, fontweight='bold')

    # 添加解释文字
    plt.text(len(input_words) + 0.5, len(output_words) // 2,
             '颜色越深\n关注度越高\n\n每行权重\n之和为1',
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.show()


print("🔮 翻译测试函数准备完成！")

# 🧪 翻译效果测试：见证智能翻译的魅力
print("🧪 开始测试翻译效果...")
print("=" * 60)

# 准备测试句子（包含训练过的和没训练过的）
test_sentences = [
    "I love you",  # 简单情感表达
    "How are you",  # 日常问候
    "I want to go home",  # 意愿表达
    "The red car is fast",  # 描述性句子
    "Thank you very much",  # 感谢表达
]

print(f"📝 测试句子数量: {len(test_sentences)}")
print("💡 每个句子都会显示:")
print("   - 翻译过程")
print("   - 最终结果")
print("   - 注意力权重可视化")
print("   - 与训练数据的对比（如果存在）")

# 逐个测试翻译
for i, sentence in enumerate(test_sentences):
    print(f"\n{'=' * 50}")
    print(f"🎯 测试 {i + 1}/{len(test_sentences)}: {sentence}")
    print('=' * 50)

    # 执行翻译
    translation = translate_with_attention(attention_model, sentence, dataset, visualize=True)

    print(f"\n📊 翻译结果:")
    print(f"   输入: {sentence}")
    print(f"   输出: {translation}")

    # 查找训练数据中是否有对应的正确答案
    ground_truth = None
    for en_orig, zh_orig in dataset.pairs:
        if en_orig.lower().strip() == sentence.lower().strip():
            ground_truth = zh_orig
            break

    if ground_truth:
        print(f"   参考: {ground_truth}")
        # 简单的相似度评估
        if translation == ground_truth:
            print("   ✅ 完全匹配！")
        elif len(set(translation) & set(ground_truth)) > 0:
            print("   😊 部分匹配，有相同字符")
        else:
            print("   😐 不太匹配，但可能仍然合理")
    else:
        print("   📝 新句子，无参考答案")

    print("\n" + "─" * 50)

print("\n" + "=" * 60)
print("🎉 翻译测试完成！")
print("💡 观察发现:")
print("   1. 注意力权重清楚显示了词汇对应关系")
print("   2. 模型学会了基本的英中翻译模式")
print("   3. 对于训练过的句子，翻译质量较高")
print("   4. 对于新句子，模型也能给出合理翻译")
print("=" * 60)
