import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和emoji兼容字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def simple_attention_demo():
    """
    简单的注意力机制演示 - 不使用可学习参数
    通过直接的向量相似度计算来理解注意力
    """
    print("=== 🎯 简单注意力机制演示 ===\n")

    # 使用简单的词向量表示（不涉及可学习参数）
    sentence = "小明喜欢苹果"
    words = ['小明', '喜欢', '苹果']

    # 手工设计的词向量（仅用于演示）
    # 维度含义：[人物，动作，物体]
    word_vectors = {
        '小明': [1, 0, 0],            # 纯人物
        '喜欢': [0, 1, 0],            # 纯动作
        '苹果': [0, 0, 1]             # 纯物体
    }

    print(f"句子: {sentence}")
    print(f"词语: {words}")
    print("\n词向量表示（人物, 动作, 物体）:")
    for word in words:
        print(f"  {word}: {word_vectors[word]}")

    # 计算注意力权重
    print("\n=== 计算注意力过程 ===")

    # 假设我们要分析"喜欢"这个词
    query_word = "喜欢"
    query_vector = np.array(word_vectors[query_word])

    print(f"\n🔍 分析词语: {query_word}")
    print(f"Query向量: {query_vector}")

    # 计算与所有词的相似度
    similarities = []
    for word in words:
        word_vec = np.array(word_vectors[word])
        # 使用点积计算相似度
        similarity = np.dot(query_vector, word_vec)
        similarities.append(similarity)
        print(f"  与'{word}'的相似度: {query_vector} · {word_vec} = {similarity}")

    # 转换为注意力权重
    similarities = np.array(similarities)
    attention_weights = np.exp(similarities) / np.sum(np.exp(similarities))

    print(f"\n📊 注意力权重:")
    for i, word in enumerate(words):
        print(f"  {word}: {attention_weights[i]:.3f}")

    print(f"\n✅ 权重总和: {np.sum(attention_weights):.6f}")

    # 分析结果
    print(f"\n🧠 结果分析:")
    max_idx = np.argmax(attention_weights)
    print(f"  '{query_word}'最关注的词是: {words[max_idx]} (权重: {attention_weights[max_idx]:.3f})")

    return attention_weights, words

# 执行演示
weights, words = simple_attention_demo()


# 可视化注意力权重
def visualize_attention_weights(weights, words, query_word="喜欢"):
    """
    使用热力图可视化注意力权重
    """
    plt.figure(figsize=(12, 8))

    # 创建一个2x2的子图布局

    # 1. 条形图 - 显示注意力权重分布
    plt.subplot(2, 2, 1)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(words, weights, color=colors, alpha=0.7, edgecolor='black')
    plt.title(f'"{query_word}"的注意力权重分布', fontsize=14, fontweight='bold')
    plt.ylabel('注意力权重')
    plt.xlabel('词语')

    # 添加数值标签
    for bar, weight in zip(bars, weights):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 热力图 - 主要的可视化方式
    plt.subplot(2, 2, 2)
    weights_matrix = weights.reshape(1, -1)
    sns.heatmap(weights_matrix, annot=True, fmt='.3f',
                xticklabels=words, yticklabels=[f'查询:{query_word}'],
                cmap='YlOrRd', cbar_kws={'label': '注意力权重'},
                linewidths=0.5, linecolor='black')
    plt.title('注意力权重热力图', fontsize=14, fontweight='bold')

    # 3. 饼图 - 显示注意力分布比例
    plt.subplot(2, 2, 3)
    plt.pie(weights, labels=words, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=[0.1 if w == max(weights) else 0 for w in weights])
    plt.title('注意力权重比例', fontsize=14, fontweight='bold')

    # 4. 解释说明
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # 创建解释文本
    explanation_text = f"""
    注意力权重解释：

    查询词: "{query_word}"

    权重分析:
    """

    for i, (word, weight) in enumerate(zip(words, weights)):
        explanation_text += f"\n  • {word}: {weight:.3f}"
        if weight == max(weights):
            explanation_text += " * (最高关注)"

    explanation_text += f"""

    理解：
    • 权重越高，关注度越大
    • 所有权重之和 = 1.0
    • 热力图颜色越深，权重越高
    """

    plt.text(0.05, 0.95, explanation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 输出重要统计信息
    print("=== 📊 注意力权重统计 ===")
    print(f"最高关注词: {words[np.argmax(weights)]} (权重: {max(weights):.3f})")
    print(f"最低关注词: {words[np.argmin(weights)]} (权重: {min(weights):.3f})")
    print(f"权重总和: {np.sum(weights):.6f}")


# 可视化结果
visualize_attention_weights(weights, words, query_word="喜欢")


### 🧮 简化版自注意力计算

def simple_self_attention_demo():
    """
    用最简单的方式演示自注意力计算
    不涉及复杂的可学习参数，专注于理解核心思想
    """
    print("=== 🎯 动手计算自注意力 ===\n")

    # 使用一个简单的句子
    sentence = "小明喜欢苹果"
    words = ['小明', '喜欢', '苹果']

    print(f"句子: {sentence}")
    print(f"词语: {words}")

    # 简化的词向量表示（手工设计，不是学习得到的）
    # 每个词用3维向量表示: [人物特征, 动作特征, 物体特征]
    word_vectors = np.array([
        [1.0, 0.0, 0.0],        # 小明: 纯人物
        [0.2, 1.0, 0.2],        # 喜欢: 主要是动作，但也涉及人物和物体
        [0.0, 0.0, 1.0]         # 苹果: 纯物体
    ])

    print("\n词向量表示 [人物, 动作, 物体]:")
    for i, word in enumerate(words):
        print(f"  {word}: {word_vectors[i]}")

    print("\n=== 自注意力计算过程 ===")

    # 在自注意力中，每个词都会关注所有词（包括自己）
    # 这里简化处理：直接使用词向量作为Q、K、V
    Q = word_vectors                # 查询矩阵
    K = word_vectors                # 键矩阵
    V = word_vectors                # 值矩阵

    print("\n1. 📊 计算注意力得分矩阵")
    # 计算注意力得分：Q @ K^T
    attention_scores = Q @ K.T
    print("注意力得分矩阵 (Q @ K^T):")
    print(f"{'':>6}", end="")
    for word in words:
        print(f"{word:>8}", end="")
    print()

    for i, query_word in enumerate(words):
        print(f"{query_word:>6}", end="")
        for j, key_word in enumerate(words):
            score = attention_scores[i, j]
            print(f"{score:>7.2f}", end="")
        print()

    print("\n2. 🎯 应用Softmax得到注意力权重")
    # 应用softmax
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

    print("注意力权重矩阵 (每行和为1):")
    print(f"{'':>6}", end="")
    for word in words:
        print(f"{word:>8}", end="")
    print()

    for i, query_word in enumerate(words):
        print(f"{query_word:>6}", end="")
        for j, key_word in enumerate(words):
            weight = attention_weights[i, j]
            print(f"{weight:>7.3f}", end="")
        print(f"  (和:{np.sum(attention_weights[i]):.3f})")

    print("\n3. 🧠 分析注意力模式")
    for i, query_word in enumerate(words):
        max_attention_idx = np.argmax(attention_weights[i])
        max_attention_word = words[max_attention_idx]
        max_attention_weight = attention_weights[i, max_attention_idx]

        print(f"  '{query_word}' 最关注: '{max_attention_word}' (权重: {max_attention_weight:.3f})")

    print("\n4. 🎯 计算最终输出")
    # 计算最终输出：attention_weights @ V
    output = attention_weights @ V

    print("最终输出 (每个词的新表示):")
    for i, word in enumerate(words):
        print(f"  {word}: {output[i]}")

    print("\n💡 关键洞察:")
    print("• 每个词都能'看到'整个句子的信息")
    print("• 注意力权重决定了词语之间的关联强度")
    print("• 输出是所有词信息的加权融合")
    print("• 这就是自注意力的魔法！")

    return attention_weights, words


# 执行演示
attention_matrix, words = simple_self_attention_demo()


# 可视化自注意力权重矩阵
def visualize_self_attention(attention_matrix, words):
    plt.figure(figsize=(12, 5))

    # 热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(attention_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=words,
                yticklabels=words,
                cmap='Blues',
                cbar_kws={'label': '注意力权重'})
    plt.title('自注意力权重矩阵')
    plt.xlabel('键位置 (被关注的位置)')
    plt.ylabel('查询位置 (关注的位置)')

    # 每个位置的注意力分布
    plt.subplot(1, 2, 2)
    x_pos = np.arange(len(words))
    width = 0.25

    for i, query_word in enumerate(words):
        plt.bar(x_pos + i * width, attention_matrix[i],
                width, label=f'查询: {query_word}', alpha=0.7)

    plt.xlabel('键位置')
    plt.ylabel('注意力权重')
    plt.title('各位置的注意力分布')
    plt.xticks(x_pos + width, words)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 可视化结果
visualize_self_attention(attention_matrix, words)


# 可视化自注意力在代词指代方面的强大能力
def visualize_pronoun_resolution():
    plt.figure(figsize=(15, 10))

    # 使用优化后的注意力矩阵（重新定义以确保可用性）
    sentence = "小明喜欢苹果，他每天都吃。"
    words = ['小明', '喜欢', '苹果', '他', '每天', '都', '吃']

    # 重新定义注意力权重矩阵
    attention_matrix = np.array([
        [0.7, 0.1, 0.1, 0.05, 0.02, 0.01, 0.02],  # 小明 -> 主要关注自己
        [0.3, 0.4, 0.2, 0.05, 0.02, 0.01, 0.02],  # 喜欢 -> 关注主语和宾语
        [0.1, 0.2, 0.6, 0.05, 0.02, 0.01, 0.02],  # 苹果 -> 主要关注自己
        [0.8, 0.05, 0.05, 0.05, 0.02, 0.01, 0.02],  # 他 -> 高度关注"小明"！
        [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1],  # 每天 -> 关注时间相关
        [0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1],  # 都 -> 关注动作相关
        [0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05]  # 吃 -> 关注主语、动作和宾语
    ])

    # 创建三个子图来展示不同的视角

    # 1. 注意力权重热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(attention_matrix,
                annot=True,
                fmt='.2f',
                xticklabels=words,
                yticklabels=words,
                cmap='Reds',
                cbar_kws={'label': '注意力权重'})
    plt.title('自注意力权重热力图', fontsize=14, fontweight='bold')
    plt.xlabel('被关注的词')
    plt.ylabel('查询词')

    # 2. 代词"他"的注意力分布
    plt.subplot(2, 2, 2)
    pronoun_attention = attention_matrix[3]  # "他"的注意力分布
    bars = plt.bar(words, pronoun_attention,
                   color=['red' if w == '小明' else 'lightblue' for w in words],
                   alpha=0.8)
    plt.title('代词"他"的注意力分布', fontsize=14, fontweight='bold')
    plt.ylabel('注意力权重')
    plt.xticks(rotation=45)

    # 高亮最高权重
    max_idx = np.argmax(pronoun_attention)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(1.0)

    # 添加数值标签
    for i, (bar, weight) in enumerate(zip(bars, pronoun_attention)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. 动词"喜欢"的注意力分布
    plt.subplot(2, 2, 3)
    verb_attention = attention_matrix[1]  # "喜欢"的注意力分布
    bars2 = plt.bar(words, verb_attention,
                    color=['orange' if w in ['小明', '苹果'] else 'lightgray' for w in words],
                    alpha=0.8)
    plt.title('动词"喜欢"的注意力分布', fontsize=14, fontweight='bold')
    plt.ylabel('注意力权重')
    plt.xticks(rotation=45)

    # 添加数值标签
    for i, (bar, weight) in enumerate(zip(bars2, verb_attention)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')

    # 4. 注意力流向图
    plt.subplot(2, 2, 4)
    # 创建一个简化的注意力流向可视化
    pos_y = np.arange(len(words))

    # 绘制词语
    for i, word in enumerate(words):
        if word == '他':
            plt.text(0, i, word, fontsize=16, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        elif word == '小明':
            plt.text(0, i, word, fontsize=16, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        else:
            plt.text(0, i, word, fontsize=14,
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    # 绘制"他"到"小明"的强连接
    he_idx = words.index('他')
    ming_idx = words.index('小明')

    plt.arrow(0.1, he_idx, 0, ming_idx - he_idx - 0.1,
              head_width=0.05, head_length=0.1,
              fc='red', ec='red', linewidth=3, alpha=0.8)

    plt.text(0.2, (he_idx + ming_idx) / 2, '0.80',
             fontsize=12, fontweight='bold', color='red')

    plt.xlim(-0.3, 0.4)
    plt.ylim(-0.5, len(words) - 0.5)
    plt.title('代词指代关系可视化', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 输出解释
    print("\n🎯 自注意力机制的强大能力展示:")
    print("=" * 50)
    print("1. 代词解析: '他' → '小明' (权重: 0.80)")
    print("   自注意力成功识别出代词与其指代对象的关系")
    print()
    print("2. 语法关系: '喜欢' 连接 '小明' 和 '苹果'")
    print("   动词同时关注主语和宾语，体现语法结构")
    print()
    print("3. 语义联系: '吃' 关注相关实体")
    print("   动作词关注执行者、对象和相关动作")
    print()
    print("这些例子展示了自注意力机制在理解语言结构、")
    print("建立词语关系方面的强大能力！")


# 执行可视化
visualize_pronoun_resolution()



