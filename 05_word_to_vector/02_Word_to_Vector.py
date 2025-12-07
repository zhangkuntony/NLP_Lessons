import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import defaultdict, Counter
import re
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sympy.printing.pretty.pretty_symbology import line_width
from torch.utils.hipify.hipify_python import preprocessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
random.seed(42)

print("✅ 所有库导入成功！")

def generate_training_samples(sentence, window_size=2, model_type='skip-gram'):
    """
    从句子生成训练样本
    :param sentence: 分词后的句子列表
    :param window_size: 上下文窗口大小
    :param model_type: 'skip-gram' 或 'cbow'
    :return: 训练样本列表
    """
    samples = []

    for i, center_word in enumerate(sentence):
        # 获取上下文窗口
        start_idx = max(0, i - window_size)
        end_idx = min(len(sentence), i + window_size + 1)

        context_words = []
        for j in range(start_idx, end_idx):
            if j != i:          # 排除中心词本身
                context_words.append(sentence[j])

        if model_type == 'skip-gram':
            # skip-gram: 中心词 -> 上下文词
            for context_word in context_words:
                samples.append((center_word, context_word))

        elif model_type == 'cbow':
            # CBOW: 上下文词 -> 中心词
            if context_words:       # 确保有上下文词
                samples.append((context_words, center_word))

    return samples

# 演示样本生成
sentence = ['我', '喜欢', '吃', '苹果', '和', '香蕉']
print("原句子:", sentence)
print('\n' + '=' * 50)

# skip-gram 样本
skip_gram_samples = generate_training_samples(sentence, window_size=2, model_type='skip-gram')
print("🎯 Skip-gram 训练样本 (中心词 → 上下文词):")
for i, (center, context) in enumerate(skip_gram_samples):          # 只显示前10个
    print(f"  {i + 1}. '{center}' -> '{context}'")

print('\n' + '=' * 50)

# CBOW样本
cbow_samples = generate_training_samples(sentence, window_size=2, model_type='cbow')
print("📊 CBOW 训练样本 (上下文词 → 中心词):")
for i, (context, center) in enumerate(cbow_samples):
    context_str = ", ".join(context)
    print(f"  {i + 1}. [{context_str}] -> '{center}'")



# 从头开始实现一个简化版的Word2Vec模型！我们将实现Skip-gram架构，并使用负采样优化。

# 准备训练语料
corpus = [
    "我 喜欢 吃 苹果",
    "我 喜欢 吃 香蕉",
    "苹果 和 香蕉 都 很 好吃",
    "水果 很 有营养",
    "我 每天 都 吃 水果",
    "苹果 是 红色 的",
    "香蕉 是 黄色 的",
    "我 喜欢 红色 的 苹果",
    "新鲜 的 水果 很 甜",
    "这个 苹果 很 甜 很 好吃",
    "那个 香蕉 很 甜",
    "我 买 了 很多 水果",
    "水果 市场 有 很多 苹果",
    "我 在 水果 店 买 香蕉",
    "这些 水果 都 很 新鲜"
]

class Word2VecDataPreprocessor:
    """Word2Vec数据预处理器
    注意:这个预处理器只生成了正样本对(中心词,上下文词),没有进行负采样
    """

    def __init__(self, min_count=1):
        self.min_count = min_count          # 词频阈值
        self.word2idx = {}                  # 词到索引的映射
        self.idx2word = {}                  # 索引到词的映射
        self.word_counts = Counter()        # 词频统计
        self.vocab_size = 0                 # 词汇表大小

    def build_vocab(self, corpus):
        """构建词汇表，但不涉及负采样"""
        # 统计词频
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                self.word_counts[word] += 1

        # 过滤低频词，但不是负采样
        filtered_words = [word for word, count in self.word_counts.items()
                          if count >= self.min_count]

        # 构建词汇映射
        for i, word in enumerate(filtered_words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        self.vocab_size = len(self.word2idx)
        print(f"词汇表大小: {self.vocab_size}")
        print(f"词汇表: {list(self.word2idx.keys())}")

    def sentence_to_indices(self, sentence):
        """将句子转换为索引"""
        words = sentence.split()
        return [self.word2idx[word] for word in words if word in self.word2idx]

    def generate_skip_gram_samples(self, corpus, window_size=2):
        """生成skip-gram训练样本
        只生成正样本对,负采样在模型训练时进行
        """
        samples = []

        for sentence in corpus:
            indices = self.sentence_to_indices(sentence)

            for i, center_idx in enumerate(indices):
                # 获取上下文窗口
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)

                # 只生成正样本对
                for j in range(start, end):
                    if j != i:              # 排除中心词
                        context_idx = indices[j]
                        samples.append((center_idx, context_idx))

        return samples

# 初始化预处理器
preprocessor = Word2VecDataPreprocessor(min_count=1)
preprocessor.build_vocab(corpus)

# 生成训练样本（只有正样本）
training_samples = preprocessor.generate_skip_gram_samples(corpus, window_size=2)
print(f"\n训练样本数量：{len(training_samples)}")
print("前10个训练样本")
for i, (center, context) in enumerate(training_samples[:10]):
    center_word = preprocessor.idx2word[center]
    context_word = preprocessor.idx2word[context]
    print(f"  {i + 1}. {center_word}({center}) -> {context_word}({context})")


# Word2Vec 模型实现，实现核心的Skip-gram模型
class Word2VecSkipGram:
    """Skip-gram Word2Vec模型
    使用负采样优化的Skip-gram模型实现。主要包含:
    1. 词向量矩阵初始化
    2. 负采样
    3. 前向传播计算
    4. 反向传播更新
    """

    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01, neg_samples=5):
        self.vocab_size = vocab_size                # 词汇表大小
        self.embedding_dim = embedding_dim          # 词向量维度
        self.learning_rate = learning_rate          # 学习率
        self.neg_samples = neg_samples              # 每个正样本对应的负样本数量

        # 初始化两个权重矩阵：
        # 1. W_in: 输入词向量矩阵, shape=(vocab_size, embedding_dim)
        # 2. W_out: 输出上下文矩阵, shape=(embedding_dim, vocab_size)
        # 使用均匀分布初始化, 范围为[-0.5/dim, 0.5/dim]
        self.W_in = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
        self.W_out = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (embedding_dim, vocab_size))

    def sigmoid(self, x):
        """Sigmoid激活函数
        为防止数值溢出,将输入限制在[-500, 500]范围内
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def negative_sampling(self, target_idx, num_samples):
        """
        负采样：随机选择指定数量的负样本词
        :param target_idx: 目标词的索引（正样本）
        :param num_samples: 需要采样的负样本的数量
        :return: 负样本索引列表
        注意：确保不会采样到目标词本身作为负样本
        """
        negative_samples = []
        while len(negative_samples) < num_samples:
            neg_idx = np.random.randint(0, self.vocab_size)
            if neg_idx != target_idx:           # 排除目标词
                negative_samples.append(neg_idx)
        return negative_samples

    def forward_pass(self, center_idx, context_idx, negative_indices):
        """
        前向传播计算
        :param center_idx: 中心词索引
        :param context_idx: 上下文词索引（正样本）
        :param negative_indices: 负样本词索引列表
        :return:
            center_embedding: 中心词向量
            pos_score: 正样本得分
            pos_prob: 正样本概率
            neg_scores: 负样本得分列表
            neg_probs: 负样本概率列表
        """

        # 1. 获取中心词向量
        center_embedding = self.W_in[center_idx]        # shape: (embedding_dim, )

        # 2. 计算正样本得分和概率
        pos_score = np.dot(center_embedding, self.W_out[:, context_idx])
        pos_prob = self.sigmoid(pos_score)

        # 3. 计算所有负样本的得分和概率
        neg_scores = []
        neg_probs = []
        for neg_idx in negative_indices:
            neg_score = np.dot(center_embedding, self.W_out[:, neg_idx])
            # 对负样本使用-score, 因为我们希望最大化正样本概率，最小化负样本概率
            neg_prob = self.sigmoid(-neg_score)
            neg_scores.append(neg_score)
            neg_probs.append(neg_prob)

        return center_embedding, pos_score, pos_prob, neg_scores, neg_probs

    def backward_pass(self, center_idx, context_idx, negative_indices,
                      center_embedding, pos_prob, neg_probs):
        """
        反向转播更新权重
        使用随机梯度下降（SGD）更新词向量
        :param center_idx: 中心词索引
        :param context_idx: 上下文词索引（正样本）
        :param negative_indices: 负样本词索引列表
        :param center_embedding: 中心词向量
        :param pos_prob: 正样本预测概率
        :param neg_probs: 负样本预测概率列表
        """

        # 1. 计算并更新正样本相关的权重
        pos_grad = (1 - pos_prob) * self.learning_rate
        self.W_out[:, context_idx] += pos_grad * center_embedding       # 更新输出矩阵
        center_grad = pos_grad * self.W_out[:, context_idx]             # 累积中心词梯度

        # 2. 计算并更新负样本相关的权重
        for i, neg_idx in enumerate(negative_indices):
            # 负样本梯度（注意符号相反）
            neg_grad = -(1 - neg_probs[i]) * self.learning_rate
            # 更新负样本在输出矩阵中的权重
            self.W_out[:, neg_idx] += neg_grad * center_embedding
            # 累积负样本对中心词的梯度贡献
            center_grad += neg_grad * self.W_out[:, neg_idx]

        # 3. 最后更新中心词的词向量
        self.W_in[center_idx] += center_grad

    def train_step(self, center_idx, context_idx):
        """单步训练
        执行一次前向传播和反向传播的完整训练步骤
        """

        # 1. 为当前样本进行负采样
        negative_indices = self.negative_sampling(context_idx, self.neg_samples)

        # 2. 前向传播计算
        center_embedding, pos_score, pos_prob, neg_scores, neg_probs = self.forward_pass(center_idx, context_idx, negative_indices)

        # 3. 计算损失（交叉熵）
        pos_loss = -np.log(pos_prob + 1e-10)            # 正样本损失
        neg_loss = -np.sum([np.log(neg_prob + 1e-10) for neg_prob in neg_probs])       # 负样本损失
        total_loss = pos_loss + neg_loss

        # 4. 反向传播更新参数
        self.backward_pass(center_idx, context_idx, negative_indices, center_embedding, pos_prob, neg_probs)

        return total_loss

    def train(self, training_samples, epochs=10):
        """训练模型
        Args:
            training_samples: 训练样本列表,每个样本是(中心词,上下文词)对
            epochs: 训练轮数
        Returns:
            losses: 每个epoch的平均损失列表
        """
        print("开始训练Word2Vec模型...")
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(training_samples)             # 打乱训练样本顺序

            # 遍历所有训练样本
            for center_idx, context_idx, in training_samples:
                loss = self.train_step(center_idx, context_idx)
                total_loss += loss

            # 计算并记录当前epoch的平均损失
            avg_loss = total_loss / len(training_samples)
            losses.append(avg_loss)

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")

        print("训练完成！")
        return losses

    def get_word_vector(self, word_idx):
        """获取指定词的词向量"""
        return self.W_in[word_idx]

    def similarity(self, word_idx1, word_idx2):
        """计算两个词的余弦相似度"""
        vec1 = self.get_word_vector(word_idx1)
        vec2 = self.get_word_vector(word_idx2)

        # 对词向量进行L2归一化
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        return np.dot(vec1_norm, vec2_norm)

# 初始化模型
model = Word2VecSkipGram(
    vocab_size=preprocessor.vocab_size,
    embedding_dim=50,
    learning_rate=0.1,
    neg_samples=5
)

print("✅ Word2Vec模型初始化完成!")
print(f"词汇表大小: {model.vocab_size}")
print(f"嵌入维度: {model.embedding_dim}")
print(f"学习率: {model.learning_rate}")
print(f"负采样数量: {model.neg_samples}")


# 训练与可视化
# 训练模型
losses = model.train(training_samples, epochs=20)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b-', linewidth=2)
plt.title('Word2Vec 训练损失曲线', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('平均损失', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"最终损失：{losses[-1]:.4f}")
print(f"损失下降：{losses[0]:.4f} -> {losses[-1]:.4f}")


# 分析训练结果
def find_most_similar_words(model, preprocessor, target_word, top_k=5):
    """找到与目标词最相似的词"""
    if target_word not in preprocessor.word2idx:
        return []

    target_idx = preprocessor.word2idx[target_word]
    similarities = []

    for word, idx in preprocessor.word2idx.items():
        if word != target_word:             # 排除自己
            sim = model.similarity(target_idx, idx)
            similarities.append((word, sim))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# 测试词汇相似性
test_words = ["苹果", "我", "吃", "很"]

print("🔍 词汇相似性分析:")
print("="*50)

for word in test_words:
    if word in preprocessor.word2idx:
        similar_words = find_most_similar_words(model, preprocessor, word, top_k=5)
        print(f"\n与 '{word}' 最相似的词：")
        for i, (sim_word, sim_score) in enumerate(similar_words):
            print(f"  {i+1}. {sim_word}: {sim_score:.4f}")
    else:
        print(f"\n词 '{word}' 不在词汇表中")

# 计算一些有趣的词对相似度
print("\n" + "="*50)
print("📊 特定词对相似度:")

word_pairs = [
    ("苹果", "香蕉"),   # 两种水果
    ("苹果", "水果"),   # 具体和抽象
    ("红色", "黄色"),   # 两种颜色
    ("喜欢", "吃"),     # 动作词
    ("我", "很"),       # 不相关的词
]

for word1, word2 in word_pairs:
    if word1 in preprocessor.word2idx and word2 in preprocessor.word2idx:
        idx1 = preprocessor.word2idx[word1]
        idx2 = preprocessor.word2idx[word2]
        sim = model.similarity(idx1, idx2)
        print(f"'{word1}' 和 '{word2}': {sim:.4f}")
    else:
        print(f"'{word1}' 或 '{word2}' 不在词汇表中")


# 词向量可视化
def visualize_word_vectors(model, preprocessor, method='pca', figsize=(12, 8)):
    """可视化词向量"""
    word_vectors = []
    words = []

    for word, idx in preprocessor.word2idx.items():
        vector = model.get_word_vector(idx)
        word_vectors.append(vector)
        words.append(word)

    word_vectors = np.array(word_vectors)

    # 降维
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        vectors_2d = reducer.fit_transform(word_vectors)
        title = f"Word2Vec 词向量可视化(PCA降维)"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words) - 1))
        vectors_2d = reducer.fit_transform(word_vectors)
        title = f'Word2Vec 词向量可视化(t-SNE降维)'

    # 绘图
    plt.figure(figsize=figsize)

    # 定义颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(words)))

    # 绘制散点图
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                          c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # 添加词汇标签
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('维度1', fontsize=12)
    plt.ylabel('维度2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return vectors_2d

# PCA 可视化
print("📊 使用PCA进行2D可视化:")
vectors_2d_pca = visualize_word_vectors(model, preprocessor, method='pca')


# 实际应用
# 词汇类比任务
def word_analogy(model, preprocessor, word_a, word_b, word_c, top_k=3):
    """
    词汇类比: word_a - word_b + word_c = ?
    例如: 苹果 - 红色 + 黄色 = 香蕉
    """
    # 检查词汇是否在词汇表中
    words = [word_a, word_b, word_c]
    for word in words:
        if word not in preprocessor.word2idx:
            print(f"词 '{word}' 不在词汇表中")
            return[]

    # 获取词向量
    vec_a = model.get_word_vector(preprocessor.word2idx[word_a])
    vec_b = model.get_word_vector(preprocessor.word2idx[word_b])
    vec_c = model.get_word_vector(preprocessor.word2idx[word_c])

    # 计算目标向量: a - b + c
    target_vector = vec_a - vec_b + vec_c

    # 归一化
    target_vector = target_vector / np.linalg.norm(target_vector)

    # 计算与所有词的相似度
    similarities = []
    exclude_words = {word_a, word_b, word_c}            # 排除输入的三个词

    for word, idx in preprocessor.word2idx.items():
        if word not in exclude_words:
            word_vector = model.get_word_vector(idx)
            word_vector = word_vector / np.linalg.norm(word_vector)

            similarity = np.dot(target_vector, word_vector)
            similarities.append((word, similarity))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# 测试词汇类比
print("🎯 词汇类比测试:")
print("="*50)

# 虽然我们的小语料库可能无法完美展示复杂的语义关系，
# 但我们可以尝试一些简单的类比

analogy_tests = [
    ("苹果", "红色", "黄色"),  # 苹果 - 红色 + 黄色 = 香蕉?
    ("我", "喜欢", "吃"),      # 我 - 喜欢 + 吃 = ?
    ("水果", "苹果", "香蕉"), # 水果 - 苹果 + 香蕉 = ?
]

for word_a, word_b, word_c in analogy_tests:
    print(f"\n🔍 {word_a} - {word_b} + {word_c} = ?")
    results = word_analogy(model, preprocessor, word_a, word_b, word_c, top_k=3)

    if results:
        print("可能的答案：")
        for i, (word, similarity) in enumerate(results):
            print(f"  {i+1}. {word} (相似度: {similarity:.4f})")
    else:
        print("无法计算类比")

# 词向量运算可视化
print("\n" + "="*50)
print("📊 词向量运算可视化")

def visualize_vector_arithmetic(model, preprocessor, word_a, word_b, word_c):
    """可视化词向量运算"""
    # 检查词汇
    words = [word_a, word_b, word_c]
    for word in words:
        if word not in preprocessor.word2idx:
            print(f"词 '{word}' 不在词汇表中")
            return

    # 获取词向量
    vec_a = model.get_word_vector(preprocessor.word2idx[word_a])
    vec_b = model.get_word_vector(preprocessor.word2idx[word_b])
    vec_c = model.get_word_vector(preprocessor.word2idx[word_c])
    target_vector = vec_a - vec_b + vec_c

    # 收集所有向量用于可视化
    vectors = np.array([vec_a, vec_b, vec_c, target_vector])
    labels = [word_a, word_b, word_c, f"{word_a} - {word_b} + {word_c}"]

    # PCA降维
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)

    # 绘图
    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green', 'purple']
    for i, (vec_2d, label, color) in enumerate(zip(vectors_2d, labels, colors)):
        plt.scatter(vec_2d[0], vec_2d[1], c=color, s=100, alpha=0.7, label=label)
        plt.annotate(label, (vec_2d[0], vec_2d[1]), xytext=(5, 5), textcoords='offset points',
                     fontsize=10, fontweight='bold')

    plt.title(f'词向量运算: {word_a} - {word_b} + {word_c}', fontsize=14)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 可视化一个词向量运算
if "苹果" in preprocessor.word2idx and "红色" in preprocessor.word2idx and "黄色" in preprocessor.word2idx:
    visualize_vector_arithmetic(model, preprocessor, "苹果", "红色", "黄色")


# 使用现成的Word2Vec库
# 使用Gensim训练Word2Vec (需要安装: pip install gensim)
from gensim.models import Word2Vec

# 准备语料(Gensim需要句子列表，每个句子是词的列表)
sentences = [sentence.split() for sentence in corpus]

# 训练Word2Vec模型
gensim_model = Word2Vec(
    sentences=sentences,
    vector_size=50,                     # 词向量维度
    window=2,                           # 上下文窗口大小
    min_count=1,                        # 最小词频
    workers=1,                          # 线程数
    sg=1,                               # 1表示skip-gram, 0表示CBOW
    epochs=20                           # 训练轮数
)

print("✅ Gensim Word2Vec模型训练完成!")
print(f"词汇表大小: {len(gensim_model.wv.key_to_index)}")

# 测试相似度
print("\n🔍 Gensim模型相似词:")
test_word = "苹果"
if test_word in gensim_model.wv:
    similar_words = gensim_model.wv.most_similar(test_word, topn=3)
    print(f"与'{test_word}'最相似的词：")
    for word, similarity in similar_words:
        print(f"  {word}: {similarity:.4f}")

# 词汇类比
print("\n🎯 Gensim词汇类比:")
try:
    # positive表示要加的词, negative表示要减的词
    result = gensim_model.wv.most_similar(
        positive=['黄色', '苹果'],
        negative=['红色'],
        topn=3
    )
    print("苹果 - 红色 + 黄色 = ")
    for word, similarity in result:
        print(f"  {word}: {similarity:.4f}")

except:
    print("类比计算失败（可能是词汇表太小）")


# 使用Gensim训练自定义Word2Vec模型
# 准备更丰富的训练语料
extended_corpus = [
    "我 喜欢 吃 苹果",
    "我 喜欢 吃 香蕉",
    "苹果 和 香蕉 都 很 好吃",
    "水果 很 有营养",
    "我 每天 都 吃 水果",
    "苹果 是 红色 的",
    "香蕉 是 黄色 的",
    "我 喜欢 红色 的 苹果",
    "新鲜 的 水果 很 甜",
    "这个 苹果 很 甜 很 好吃",
    "那个 香蕉 很 甜",
    "我 买 了 很多 水果",
    "水果 市场 有 很多 苹果",
    "我 在 水果 店 买 香蕉",
    "这些 水果 都 很 新鲜",
    "妈妈 给 我 买 了 苹果",
    "爸爸 喜欢 吃 香蕉",
    "小朋友 都 喜欢 水果",
    "健康 的 生活 需要 多 吃 水果",
    "苹果 含有 丰富 的 维生素",
    "香蕉 含有 钾 元素",
    "水果 沙拉 很 好吃",
    "我 最 喜欢 的 水果 是 苹果",
    "红色 苹果 比 绿色 苹果 甜",
    "成熟 的 香蕉 是 黄色 的",
    "新鲜 水果 营养 价值 高",
    "每天 吃 水果 有益 健康",
    "水果 店 里 有 各种 水果",
    "苹果 树 结出 红色 果实",
    "香蕉 树 生长 在 热带 地区"
]

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging

# 设置日志级别以查看训练进度
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("✅ Gensim库导入成功!")
print(f"扩展语料库大小: {len(extended_corpus)} 句")

# 准备训练数据（Gensim需要句子列表，每个句子是词的列表）
sentences = [sentence.split() for sentence in extended_corpus]
print(f"预处理后的句子数量: {len(sentences)}")
print(f"示例句子: {sentences[0]}")

if sentences is not None:
    # 自定义训练监督器
    class TrainingMonitor(CallbackAny2Vec):
        """训练过程监控器"""

        def __init__(self):
            self.epoch = 0
            self.losses = []

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            if self.epoch == 0:
                self.current_loss = loss
            else:
                self.current_loss = loss - self.previous_loss
            self.losses.append(self.current_loss)
            self.previous_loss = loss
            self.epoch += 1
            print(f'Epoch {self.epoch}: Loss = {self.current_loss:.4f}')

    # 创建监控器
    monitor = TrainingMonitor()

    print("🚀 开始使用Gensim训练Word2Vec模型...")
    print("=" * 50)

    # 配置1: Skip-gram + 负采样
    print("\n📝 配置1: Skip-gram + 负采样")
    gensim_skipgram = Word2Vec(
        sentences=sentences,
        vector_size=50,  # 词向量维度
        window=2,  # 上下文窗口大小
        min_count=1,  # 最小词频（保留所有词）
        workers=1,  # 线程数
        sg=1,  # Skip-gram
        hs=0,  # 不使用层次softmax
        negative=5,  # 负采样数量
        epochs=20,  # 训练轮数
        alpha=0.025,  # 初始学习率
        min_alpha=0.0001,  # 最小学习率
        seed=42,  # 随机种子
        compute_loss=True,  # 计算损失
        callbacks=[monitor]  # 训练监控
    )

    print(f"✅ Skip-gram模型训练完成!")
    print(f"   词汇表大小: {len(gensim_skipgram.wv.key_to_index)}")
    print(f"   向量维度: {gensim_skipgram.wv.vector_size}")

else:
    print("❌ 无法进行Gensim训练（缺少Gensim库）")


if sentences is not None:
    # 训练多种配置进行对比
    print("\n" + "=" * 50)
    print("🔄 训练不同配置的模型进行对比")

    # 配置2: CBOW + 负采样
    print("\n📝 配置2: CBOW + 负采样")
    monitor_cbow = TrainingMonitor()
    gensim_cbow = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        workers=1,
        sg=0,  # CBOW
        hs=0,
        negative=5,
        epochs=20,
        alpha=0.025,
        seed=42,
        compute_loss=True,
        callbacks=[monitor_cbow]
    )

    print(f"✅ CBOW模型训练完成!")

    # 配置3: Skip-gram + 层次Softmax
    print("\n📝 配置3: Skip-gram + 层次Softmax")
    monitor_hs = TrainingMonitor()
    gensim_hs = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        workers=1,
        sg=1,  # Skip-gram
        hs=1,  # 层次softmax
        negative=0,  # 不使用负采样
        epochs=20,
        alpha=0.025,
        seed=42,
        compute_loss=True,
        callbacks=[monitor_hs]
    )

    print(f"✅ 层次Softmax模型训练完成!")

    # 绘制损失对比
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(monitor.losses, 'b-', linewidth=2, label='Skip-gram + 负采样')
    plt.title('Skip-gram + 负采样 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(monitor_cbow.losses, 'r-', linewidth=2, label='CBOW + 负采样')
    plt.title('CBOW + 负采样 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(monitor_hs.losses, 'g-', linewidth=2, label='Skip-gram + 层次Softmax')
    plt.title('Skip-gram + 层次Softmax 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(monitor.losses, 'b-', linewidth=2, label='Skip-gram + 负采样')
    plt.plot(monitor_cbow.losses, 'r-', linewidth=2, label='CBOW + 负采样')
    plt.plot(monitor_hs.losses, 'g-', linewidth=2, label='Skip-gram + 层次Softmax')
    plt.title('所有配置损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\n📊 训练结果总结:")
    print(f"Skip-gram + 负采样 最终损失: {monitor.losses[-1]:.4f}")
    print(f"CBOW + 负采样 最终损失: {monitor_cbow.losses[-1]:.4f}")
    print(f"Skip-gram + 层次Softmax 最终损失: {monitor_hs.losses[-1]:.4f}")


if sentences is not None:
    # 模型效果对比评估
    print("\n" + "="*60)
    print("🧪 不同模型效果对比测试")

    models = {
        'Skip-gram + 负采样': gensim_skipgram,
        'CBOW + 负采样': gensim_cbow,
        'Skip-gram + 层次Softmax': gensim_hs
    }

    # 1. 相似词测试
    print("\n🔍 相似词测试对比:")
    test_words = ['苹果', '我', '水果']

    for word in test_words:
        if word in gensim_skipgram.wv:              # 检查词是否在词汇表中
            print(f"\n📝 与 '{word}' 最相似的词:")

            for model_name, model in models.items():
                try:
                    similar_words = model.wv.most_similar(word, topn=3)
                    similar_str = ", ".join([f"{w}({s:.3f})" for w, s in similar_words])
                    print(f"  {model_name:<25}: {similar_str}")
                except:
                    print(f"  {model_name:<25}: 计算失败")

    # 2. 词汇类比测试
    print("\n🎯 词汇类比任务对比:")
    analogy_tests = [
        ('苹果', '红色', '黄色'),  # 苹果 - 红色 + 黄色 = 香蕉?
        ('我', '喜欢', '吃'),  # 我 - 喜欢 + 吃 = ?
    ]

    for word_a, word_b, word_c in analogy_tests:
        print(f"\n🔍 {word_a} - {word_b} + {word_c} = ?")

        for model_name, model in models.items():
            try:
                if all(word in model.wv for word in [word_a, word_b, word_c]):
                    result = model.wv.most_similar(
                        positive=[word_a, word_c],
                        negative=[word_b],
                        topn=2
                    )
                    result_str = ", ".join([f"{w}({s:.3f})" for w, s in result])
                    print(f"  {model_name:<25}: {result_str}")
                else:
                    print(f"  {model_name:<25}: 缺少词汇")
            except Exception as e:
                print(f"  {model_name:<25}: 计算失败")

    # 3. 词对相似度对比
    print("\n📊 词对相似度对比:")
    word_pairs = [
        ('苹果', '香蕉'),
        ('苹果', '水果'),
        ('红色', '黄色'),
        ('我', '喜欢')
    ]

    print(f"{'词对':<15} | {'Skip-gram+负采样':<15} | {'CBOW+负采样':<15} | {'Skip-gram+层次':<15}")
    print("-" * 70)

    for word1, word2 in word_pairs:
        row = f"{word1}-{word2}"

        for model_name, model in models.items():
            try:
                if word1 in model.wv and word2 in model.wv:
                    sim = model.wv.similarity(word1, word2)
                    row += f" | {sim:<15.4f}"
                else:
                    row += f" | {'缺少词汇':<15}"
            except:
                row += f" | {'错误':<15}"

        print(row)


# 模型保存与加载
if sentences is not None:
    import os

    print("💾 模型保存与加载演示")
    print("="*50)

    # 创建保存目录
    save_dir = "word2vec_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 方法1. 保存完整模型(推荐用于继续训练)
    print("\n📁 方法1: 保存完整模型")
    model_path = os.path.join(save_dir, "gensim_skipgram.model")
    gensim_skipgram.save(model_path)
    print(f"✅ 完整模型已保存到: {model_path}")

    # 方法2. 仅保存词向量(推荐用于推理)
    print("\n📁 方法2: 仅保存词向量")
    vectors_path = os.path.join(save_dir, "word_vectors.kv")
    gensim_skipgram.wv.save(vectors_path)
    print(f"✅ 词向量已保存到: {vectors_path}")

    # 方法3. 保存为Word2Vec格式(兼容性好)
    print("\n📁 方法3: 保存为Word2Vec格式")
    w2v_path = os.path.join(save_dir, "vectors.txt")
    gensim_skipgram.wv.save_word2vec_format(w2v_path, binary=False)
    print(f"✅ Word2Vec格式已保存到: {w2v_path}")

    print("\n🔄 模型加载演示:")

    # 加载完整模型
    print("\n📂 加载完整模型:")
    try:
        loaded_model = Word2Vec.load(model_path)
        print(f"✅ 完整模型加载成功")
        print(f"   词汇表大小: {len(loaded_model.wv.key_to_index)}")

        # 可以继续训练
        print("   📚 可以继续训练...")
        loaded_model.train(sentences, total_examples=len(sentences), epochs=2)
        print("   ✅ 额外训练完成")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

    # 加载词向量
    print("\n📂 加载词向量:")
    try:
        from gensim.models import KeyedVectors
        loaded_vectors = KeyedVectors.load(vectors_path)
        print(f"✅ 词向量加载成功")
        print(f"   词汇表大小: {len(loaded_vectors.key_to_index)}")

        # 测试功能
        if '苹果' in loaded_vectors:
            similar = loaded_vectors.most_similar('苹果', topn=2)
            print(f"   与'苹果'最相似的词: {similar}")

    except Exception as e:
        print(f"❌ 词向量加载失败: {e}")

    # 加载Word2Vec格式
    print("\n📂 加载Word2Vec格式:")
    try:
        loaded_w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        print(f"✅ Word2Vec格式加载成功")
        print(f"   词汇表大小: {len(loaded_w2v.key_to_index)}")

    except Exception as e:
        print(f"❌ Word2Vec格式加载失败: {e}")

    print("\\n📊 文件大小对比:")
    try:
        for filename in os.listdir(save_dir):
            filepath = os.path.join(save_dir, filename)
            if os.path.isfile(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename:<20}: {size_mb:.2f}MB")
    except:
        pass

    print("\\n💡 保存格式选择建议:")
    print("""
        1. 🔄 继续训练场景：
           - 使用 model.save() 保存完整模型
           - 包含训练状态，可继续训练

        2. 🚀 生产部署场景：
           - 使用 model.wv.save() 仅保存词向量
           - 文件更小，加载更快

        3. 🔗 跨平台兼容：
           - 使用 save_word2vec_format() 
           - 标准格式，各种工具都能读取
        """)


# Gensim vs 从零实现 对比
if sentences is not None:
    print("⚔️ Gensim vs 从零实现 模型对比")
    print("=" * 60)
    model = Word2VecSkipGram(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=50,
        learning_rate=0.1,
        neg_samples=5
    )

    # 对比相似词结果
    print("\n🔍 相似词对比测试:")
    comparison_words = ['苹果', '我', '水果']

    for word in comparison_words:
        if word in gensim_skipgram.wv and word in preprocessor.word2idx:
            print(f"\n📝 与 '{word}' 最相似的词:")

            # Gensim模型结果
            try:
                gensim_similar = gensim_skipgram.wv.most_similar(word, topn=3)
                gensim_str = ", ".join([f"{w}({s:.3f})" for w, s in gensim_similar])
                print(f"  Gensim Skip-gram      : {gensim_str}")
            except:
                print(f"  Gensim Skip-gram      : 计算失败")

            # 我们的模型结果
            try:
                word_idx = preprocessor.word2idx[word]
                our_similar = find_most_similar_words(model, preprocessor, word, top_k=3)
                our_str = ", ".join([f"{w}({s:.3f})" for w, s in our_similar])
                print(f"  我们的Skip-gram实现   : {our_str}")
            except Exception as e:
                print(f"  我们的Skip-gram实现   : 计算失败")
                print(e)

    # 对比词对相似度
    print("\n📊 词对相似度对比:")
    test_pairs = [('苹果', '香蕉'), ('苹果', '水果'), ('我', '喜欢')]

    print(f"{'词对':<12} | {'Gensim':<10} | {'我们的实现':<10} | {'差异':<10}")
    print("-" * 50)

    for word1, word2 in test_pairs:
        try:
            # Gensim相似度
            if word1 in gensim_skipgram.wv and word2 in gensim_skipgram.wv:
                gensim_sim = gensim_skipgram.wv.similarity(word1, word2)
            else:
                gensim_sim = None

            # 我们的实现相似度
            if word1 in preprocessor.word2idx and word2 in preprocessor.word2idx:
                idx1 = preprocessor.word2idx[word1]
                idx2 = preprocessor.word2idx[word2]
                our_sim = model.similarity(idx1, idx2)
            else:
                our_sim = None

            # 计算差异
            if gensim_sim is not None and our_sim is not None:
                diff = abs(gensim_sim - our_sim)
                print(f"{word1} - {word2:<6} | {gensim_sim:<10.4f} | {our_sim:<10.4f} | {diff:<10.4f}")
            else:
                print(f"{word1} - {word2:<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

        except Exception as e:
            print(f"{word1}-{word2:<6} | {'错误':<10} | {'错误':<10} | {'错误':<10}")

    print("\\n📈 性能对比分析见下方总结")