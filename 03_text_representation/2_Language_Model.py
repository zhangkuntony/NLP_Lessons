# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import re
import math
import random
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("📚 欢迎来到语言模型课程！")


# 🎲 概率论概念的可视化演示

# 1. 🪙 抛硬币实验
def coin_flip_simulation():
    """模拟抛硬币实验，展示概率的基本概念"""
    print("🪙 抛硬币实验：观察概率如何稳定")
    print("=" * 40)

    # 模拟不同次数的抛硬币
    import random

    flip_counts = [10, 50, 100, 500, 1000, 5000]
    head_ratios = []

    for n_flips in flip_counts:
        heads = sum(1 for _ in range(n_flips) if random.choice(['正面', '反面']) == '正面')
        ratio = heads / n_flips
        head_ratios.append(ratio)
        print(f"抛 {n_flips:4d} 次：正面 {heads:4d} 次，比例 {ratio:.3f} ({ratio * 100:.1f}%)")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：随着实验次数增加，概率如何稳定
    ax1.plot(flip_counts, head_ratios, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='理论概率 50%')
    ax1.set_xlabel('抛硬币次数')
    ax1.set_ylabel('正面朝上的比例')
    ax1.set_title('大数定律：概率如何稳定到理论值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale('log')

    # 右图：概率分布的直观展示
    outcomes = ['正面', '反面']
    probabilities = [0.5, 0.5]
    colors = ['gold', 'silver']

    bars = ax2.bar(outcomes, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('概率')
    ax2.set_title('抛硬币的概率分布')
    ax2.set_ylim(0, 0.6)

    # 在柱状图上显示概率值
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{prob * 100:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


# 运行所有演示
print("🎯 开始概率论基础演示！")
coin_flip_simulation()


# 2. 🌧️ 条件概率演示：天气预测
def weather_prediction_demo():
    """演示条件概率在天气预测中的应用"""
    print("\n🌧️ 条件概率演示：看云识天气")
    print("=" * 40)

    # 设定不同云类型下的下雨概率
    cloud_types = ['乌云', '白云', '晴空']
    rain_probs = [0.8, 0.2, 0.05]
    cloud_colors = ['darkgray', 'lightgray', 'skyblue']

    # 可视化条件概率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：不同云类型的下雨概率
    bars = ax1.bar(cloud_types, rain_probs, color=cloud_colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('下雨概率')
    ax1.set_title('条件概率：不同云类型下的下雨概率')
    ax1.set_ylim(0, 1)

    # 显示概率值
    for bar, prob in zip(bars, rain_probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{prob * 100:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 右图：贝叶斯推理示例
    ax2.text(0.1, 0.9, '🕵️ 贝叶斯推理示例', fontsize=16, fontweight='bold')
    ax2.text(0.1, 0.8, '问题：如果下雨了，最可能是什么云？', fontsize=12)

    # 简化的贝叶斯计算
    cloud_prior = [0.3, 0.5, 0.2]  # 各种云的先验概率

    ax2.text(0.1, 0.65, '先验概率（平时各种云出现的概率）：', fontsize=11, fontweight='bold')
    for i, (cloud, prior) in enumerate(zip(cloud_types, cloud_prior)):
        ax2.text(0.15, 0.6 - i * 0.05, f'{cloud}: {prior * 100:.0f}%', fontsize=10)

    # 计算后验概率（下雨时是各种云的概率）
    evidence = sum(p_rain * p_cloud for p_rain, p_cloud in zip(rain_probs, cloud_prior))
    # evidence = 0.8×0.3 + 0.2×0.5 + 0.05×0.2 = 0.35
    posteriors = [(p_rain * p_cloud) / evidence for p_rain, p_cloud in zip(rain_probs, cloud_prior)]
    # posteriors = [0.8×0.3/0.35, 0.2×0.5/0.35, 0.05×0.2/0.35] = [0.686， 0.286， 0.029]

    ax2.text(0.1, 0.35, '后验概率（下雨时是各种云的概率）：', fontsize=11, fontweight='bold')
    for i, (cloud, posterior) in enumerate(zip(cloud_types, posteriors)):
        ax2.text(0.15, 0.3 - i * 0.05, f'{cloud}: {posterior * 100:.1f}%', fontsize=10)

    best_cloud = cloud_types[np.argmax(posteriors)]
    ax2.text(0.1, 0.1, f'🎯 结论：下雨时最可能是{best_cloud}！',
             fontsize=12, fontweight='bold', color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

weather_prediction_demo()


# 3. 💬 语言中的概率演示
def language_probability_demo():
    """演示语言中的概率概念"""
    print("\n💬 语言中的概率：'接话'游戏")
    print("=" * 40)

    # 模拟简单的语言概率
    context_words = {
        "我今天心情": {"很好": 0.4, "不错": 0.3, "一般": 0.2, "不好": 0.1},
        "今天天气": {"很好": 0.5, "不错": 0.3, "一般": 0.15, "很差": 0.05},
        "这道菜": {"很香": 0.4, "不错": 0.35, "一般": 0.2, "难吃": 0.05}
    }

    # 可视化语言概率分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (context, words_probs) in enumerate(context_words.items()):
        words = list(words_probs.keys())
        probs = list(words_probs.values())

        bars = axes[i].bar(words, probs, alpha=0.7,
                           color=['green', 'lightgreen', 'orange', 'red'])
        axes[i].set_title(f'"{context}..."\n下一个词的概率分布')
        axes[i].set_ylabel('概率')
        axes[i].tick_params(axis='x', rotation=45)

        # 显示概率值
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{prob * 100:.0f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 互动式演示
    print("\n🎮 互动演示：")
    for context, words_probs in context_words.items():
        print(f"\n当我说：'{context}...'")
        print("你觉得下一个词最可能是什么？")

        # 按概率排序
        sorted_words = sorted(words_probs.items(), key=lambda x: x[1], reverse=True)
        for rank, (word, prob) in enumerate(sorted_words, 1):
            if rank == 1:
                print(f"  🥇 第{rank}名: '{word}' ({prob * 100:.0f}% 概率) ← 最可能！")
            else:
                print(f"  📍 第{rank}名: '{word}' ({prob * 100:.0f}% 概率)")

language_probability_demo()

print("\n✨ 概率论基础演示完成！")
print("现在你应该对概率有了直观的理解：")
print("• 📊 概率就是衡量'可能性'的数字")
print("• 🔗 条件概率告诉我们上下文如何影响结果")
print("• 🧠 贝叶斯思维帮我们根据新信息更新判断")
print("• 💬 语言模型就是在计算词语出现的概率！")


# 🎯 让我们用图片来看看"接话"是怎么工作的！

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：模拟大脑"接话"的过程
ax1.text(0.5, 0.95, '🧠 大脑如何"接话"', ha='center', fontsize=16, fontweight='bold', color='darkblue')

# 输入部分
ax1.add_patch(plt.Rectangle((0.1, 0.7), 0.3, 0.15, facecolor='lightblue', edgecolor='blue', linewidth=2))
ax1.text(0.25, 0.775, '输入:\n"我爱"', ha='center', va='center', fontsize=12, fontweight='bold')

# 箭头
ax1.annotate('', xy=(0.6, 0.775), xytext=(0.4, 0.775),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))
ax1.text(0.5, 0.82, '大脑思考', ha='center', fontsize=10, color='orange')

# 预测结果
predictions = [
    ('🇨🇳 中国', 60, 'red'),
    ('📚 学习', 30, 'blue'),
    ('💻 编程', 10, 'green')
]

y_positions = [0.6, 0.4, 0.2]
for i, (word, prob, color) in enumerate(predictions):
    width = prob / 100 * 0.25  # 按概率调整宽度
    ax1.add_patch(plt.Rectangle((0.6, y_positions[i]), width, 0.08,
                               facecolor=color, alpha=0.7, edgecolor=color))
    ax1.text(0.6 + width + 0.02, y_positions[i] + 0.04, f'{word} {prob}%',
             va='center', fontsize=11, fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# 右图：三种"记忆"方式的比较
ax2.text(0.5, 0.95, '🎯 三种"记忆"方式', ha='center', fontsize=16, fontweight='bold', color='darkblue')

memory_types = [
    ('🎲 随机猜测', '完全靠运气', 'lightcoral', 0.8),
    ('📊 统计规律', '看历史经验', 'lightgreen', 0.55),
    ('🧠 深度学习', '模拟大脑思考', 'lightblue', 0.3)
]

for name, desc, color, y in memory_types:
    # 绘制方框
    ax2.add_patch(plt.Rectangle((0.1, y-0.08), 0.8, 0.15,
                               facecolor=color, alpha=0.7, edgecolor='black'))
    ax2.text(0.15, y, name, fontsize=12, fontweight='bold', va='center')
    ax2.text(0.15, y-0.04, desc, fontsize=10, va='center', style='italic')

# 添加重点标记
ax2.text(0.75, 0.55, '← 今天学这个！', fontsize=12, fontweight='bold',
         color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
plt.show()

print("🎉 太好了！现在你知道什么是语言模型了！")
print("💡 简单说：语言模型就是教计算机学会'接话'的方法！")


# 🎯 让我们一起来做个"数数游戏"！
def demonstrate_simple_counting():
    """用最简单的方式演示计算机如何学习"""

    print("🎪 欢迎来到'数数训练营'！")
    print("我们要教计算机学会统计词语的规律")
    print()

    # 准备简单的训练数据
    training_sentences = [
        "我爱中国",
        "我爱学习",
        "我喜欢中国",
        "他爱学习",
        "她爱中国"
    ]

    print("📚 计算机的'教材'（训练数据）：")
    for i, sentence in enumerate(training_sentences, 1):
        print(f"   第{i}课: {sentence}")
    print()

    # 手工统计，让过程更直观
    print("🔍 现在让我们像计算机一样'数数'：")
    print()

    # 统计每个词
    all_words = []
    for sentence in training_sentences:
        words = list(sentence)  # 把句子拆成字
        all_words.extend(words)

    # 统计单个字出现的次数
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    print("📊 第一步：数数每个字出现了多少次")
    for word, count in sorted(word_counts.items()):
        print(f"   '{word}' 出现了 {count} 次")
    print()

    # 统计词对（2-gram）
    print("👥 第二步：数数哪两个字经常在一起")

    word_pairs = {}
    for sentence in training_sentences:
        words = list(sentence)
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            word_pairs[pair] = word_pairs.get(pair, 0) + 1

    for pair, count in sorted(word_pairs.items()):
        print(f"   '{pair[0]}' → '{pair[1]}' 出现了 {count} 次")
    print()

    # 计算概率
    print("🧮 第三步：计算概率（做除法）")
    print("如果看到'我'，下一个字是什么的概率最大？")
    print()

    # 找出"我"后面跟的所有字
    me_followers = {}
    me_total = 0

    for (w1, w2), count in word_pairs.items():
        if w1 == '我':
            me_followers[w2] = count
            me_total += count

    print(f"   '我' 后面总共有 {me_total} 个字")
    for follower, count in me_followers.items():
        probability = count / me_total * 100
        print(f"   '我' → '{follower}': {count}/{me_total} = {probability:.1f}%")

    print()
    print("🎯 结论：看到'我'字后，下一个字是'爱'的概率最高！")
    print()

    # 预测测试
    print("🎮 现在来测试一下：")
    test_input = "我"
    print(f"输入：'{test_input}'")
    print("计算机的预测：")

    # 找到概率最高的下一个字
    if test_input in [pair[0] for pair in word_pairs.keys()]:
        candidates = [(w2, count / me_total * 100) for (w1, w2), count in word_pairs.items() if w1 == test_input]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for i, (word, prob) in enumerate(candidates[:3], 1):
            if i == 1:
                print(f"   🥇 第{i}名: '{word}' (概率: {prob:.1f}%) ← 最有可能！")
            else:
                print(f"   📍 第{i}名: '{word}' (概率: {prob:.1f}%)")

    print()
    print("✨ 就是这样！计算机通过'数数'学会了预测下一个字！")

demonstrate_simple_counting()


# 🔧 平滑技术演示
def demonstrate_smoothing():
    """演示不同平滑技术的效果"""

    # 简单的训练数据
    sentences = ["我 爱 学习", "我 爱 编程", "他 喜欢 学习"]

    # 统计计数
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    vocabulary = set()

    for sentence in sentences:
        words = sentence.split()
        words = ['<s>'] + words + ['</s>']

        for word in words:
            vocabulary.add(word)
            unigram_counts[word] += 1

        for i in range(1, len(words)):
            bigram_counts[(words[i - 1], words[i])] += 1

    V = len(vocabulary)  # 词汇表大小

    print("📊 训练数据统计：")
    print(f"词汇表大小: {V}")
    print(f"词汇表: {sorted(vocabulary)}")

    # 测试未见过的词对
    test_bigram = ("我", "讨厌")

    print(f"\n🧪 测试词对: {test_bigram}")

    # 1. 原始最大似然估计
    if test_bigram in bigram_counts:
        mle_prob = bigram_counts[test_bigram] / unigram_counts[test_bigram[0]]
    else:
        mle_prob = 0
    print(f"最大似然估计: P(讨厌|我) = {mle_prob}")

    # 2. 拉普拉斯平滑
    laplace_prob = (bigram_counts[test_bigram] + 1) / (unigram_counts[test_bigram[0]] + V)
    print(
        f"拉普拉斯平滑: P(讨厌|我) = ({bigram_counts[test_bigram]} + 1) / ({unigram_counts[test_bigram[0]]} + {V}) = {laplace_prob:.4f}")

    # 3. Add-k平滑 (k=0.5)
    k = 0.5
    add_k_prob = (bigram_counts[test_bigram] + k) / (unigram_counts[test_bigram[0]] + k * V)
    print(
        f"Add-k平滑(k=0.5): P(讨厌|我) = ({bigram_counts[test_bigram]} + {k}) / ({unigram_counts[test_bigram[0]]} + {k * V}) = {add_k_prob:.4f}")

    # 可视化不同平滑方法的概率分布
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['原始MLE', '拉普拉斯', 'Add-k(0.5)']
    probabilities = [mle_prob, laplace_prob, add_k_prob]
    colors = ['red', 'blue', 'green']

    bars = ax.bar(methods, probabilities, color=colors, alpha=0.7)
    ax.set_ylabel('概率')
    ax.set_title('不同平滑方法的概率对比\n(测试词对: "我" → "讨厌")')

    # 在柱状图上显示数值
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.0001,
                f'{prob:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

demonstrate_smoothing()


# 🎯 困惑度公式可视化演示
def visualize_perplexity_formula():
    """可视化困惑度公式的计算过程"""

    print("🎯 困惑度公式可视化演示")
    print("=" * 50)

    # 模拟两种不同的预测场景
    scenarios = {
        "确定预测": {
            "words": ["我", "爱", "中国"],
            "probs": [0.9, 0.8, 0.85],
            "color": "lightgreen"
        },
        "困惑预测": {
            "words": ["我", "爱", "中国"],
            "probs": [0.1, 0.15, 0.2],
            "color": "lightcoral"
        }
    }

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 概率分布比较
    x_pos = np.arange(len(scenarios["确定预测"]["words"]))
    width = 0.35

    ax1.bar(x_pos - width / 2, scenarios["确定预测"]["probs"], width,
            label='确定预测', color=scenarios["确定预测"]["color"], alpha=0.7)
    ax1.bar(x_pos + width / 2, scenarios["困惑预测"]["probs"], width,
            label='困惑预测', color=scenarios["困惑预测"]["color"], alpha=0.7)

    ax1.set_xlabel('词语')
    ax1.set_ylabel('预测概率')
    ax1.set_title('不同预测场景的概率分布')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios["确定预测"]["words"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 对数概率计算
    log_probs_certain = [math.log2(p) for p in scenarios["确定预测"]["probs"]]
    log_probs_confused = [math.log2(p) for p in scenarios["困惑预测"]["probs"]]

    ax2.bar(x_pos - width / 2, log_probs_certain, width,
            label='确定预测', color=scenarios["确定预测"]["color"], alpha=0.7)
    ax2.bar(x_pos + width / 2, log_probs_confused, width,
            label='困惑预测', color=scenarios["困惑预测"]["color"], alpha=0.7)

    ax2.set_xlabel('词语')
    ax2.set_ylabel('log₂(概率)')
    ax2.set_title('对数概率比较')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios["确定预测"]["words"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 困惑度计算过程
    def calculate_perplexity(probs):
        """计算困惑度"""
        total_log_prob = sum(math.log2(p) for p in probs)
        avg_log_prob = total_log_prob / len(probs)
        return 2 ** (-avg_log_prob)

    pp_certain = calculate_perplexity(scenarios["确定预测"]["probs"])
    pp_confused = calculate_perplexity(scenarios["困惑预测"]["probs"])

    perplexities = [pp_certain, pp_confused]
    labels = ['确定预测', '困惑预测']
    colors = [scenarios["确定预测"]["color"], scenarios["困惑预测"]["color"]]

    bars = ax3.bar(labels, perplexities, color=colors, alpha=0.7)
    ax3.set_ylabel('困惑度')
    ax3.set_title('最终困惑度比较')
    ax3.grid(True, alpha=0.3)

    # 在柱状图上显示数值
    for bar, pp in zip(bars, perplexities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{pp:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. 计算步骤详解
    ax4.text(0.05, 0.95, '📊 困惑度计算步骤：', fontsize=14, fontweight='bold', transform=ax4.transAxes)

    # 确定预测的计算
    ax4.text(0.05, 0.85, '🟢 确定预测场景：', fontsize=12, fontweight='bold', color='green', transform=ax4.transAxes)
    total_log_certain = sum(log_probs_certain)
    avg_log_certain = total_log_certain / len(log_probs_certain)
    ax4.text(0.05, 0.80, f'• 总对数概率: {total_log_certain:.3f}', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.75, f'• 平均对数概率: {avg_log_certain:.3f}', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.70, f'• 困惑度: 2^(-{avg_log_certain:.3f}) = {pp_certain:.2f}', fontsize=10,
             transform=ax4.transAxes)

    # 困惑预测的计算
    ax4.text(0.05, 0.60, '🔴 困惑预测场景：', fontsize=12, fontweight='bold', color='red', transform=ax4.transAxes)
    total_log_confused = sum(log_probs_confused)
    avg_log_confused = total_log_confused / len(log_probs_confused)
    ax4.text(0.05, 0.55, f'• 总对数概率: {total_log_confused:.3f}', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.50, f'• 平均对数概率: {avg_log_confused:.3f}', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.45, f'• 困惑度: 2^(-{avg_log_confused:.3f}) = {pp_confused:.2f}', fontsize=10,
             transform=ax4.transAxes)

    # 结论
    ax4.text(0.05, 0.35, '🎯 结论：', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.05, 0.30, f'• 确定预测困惑度更低 ({pp_certain:.2f})', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.25, f'• 困惑预测困惑度更高 ({pp_confused:.2f})', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.20, '• 困惑度越低，模型越好！', fontsize=10, color='blue', fontweight='bold',
             transform=ax4.transAxes)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    # 打印详细计算过程
    print("\n🔍 详细计算过程：")
    print(f"确定预测: 概率 {scenarios['确定预测']['probs']} → 困惑度 {pp_certain:.2f}")
    print(f"困惑预测: 概率 {scenarios['困惑预测']['probs']} → 困惑度 {pp_confused:.2f}")
    print(f"\n💡 解释：困惑度 {pp_certain:.2f} 意味着模型平均在 {pp_certain:.0f} 个选择中纠结")
    print(f"      困惑度 {pp_confused:.2f} 意味着模型平均在 {pp_confused:.0f} 个选择中纠结")

visualize_perplexity_formula()


# 📈 困惑度计算演示
def calculate_perplexity_demo():
    """演示困惑度的计算过程"""

    # 训练数据
    train_sentences = [
        "我 爱 学习 编程",
        "我 喜欢 学习 数学",
        "他 爱 编程 语言",
        "她 喜欢 数学 公式"
    ]

    # 测试数据
    test_sentences = [
        "我 爱 数学",
        "她 喜欢 编程"
    ]

    print("🏋️ 训练数据：")
    for sentence in train_sentences:
        print(f"  {sentence}")

    print("\n🧪 测试数据：")
    for sentence in test_sentences:
        print(f"  {sentence}")

    # 构建Bigram模型
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    # 训练
    for sentence in train_sentences:
        words = ['<s>'] + sentence.split() + ['</s>']
        for i in range(len(words)):
            unigram_counts[words[i]] += 1
            if i > 0:
                bigram_counts[(words[i - 1], words[i])] += 1

    print("\n📊 模型统计：")
    print(f"总词数: {sum(unigram_counts.values())}")
    print(f"不同词数: {len(unigram_counts)}")
    print(f"不同bigram数: {len(bigram_counts)}")

    # 计算困惑度
    def calculate_sentence_probability(sentence, smoothing=True):
        """计算句子概率"""
        words = ['<s>'] + sentence.split() + ['</s>']
        log_prob = 0.0

        for i in range(1, len(words)):
            w1, w2 = words[i - 1], words[i]

            if smoothing:  # 使用拉普拉斯平滑
                V = len(unigram_counts)
                prob = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)
            else:  # 原始MLE
                if unigram_counts[w1] > 0:
                    prob = bigram_counts[(w1, w2)] / unigram_counts[w1]
                else:
                    prob = 1e-10  # 避免log(0)

            log_prob += math.log2(prob)

        return log_prob, len(words) - 1  # 减1因为不算<s>

    # 计算每个测试句子的困惑度
    total_log_prob = 0
    total_words = 0

    print("\n🔍 详细计算过程：")
    for sentence in test_sentences:
        log_prob, num_words = calculate_sentence_probability(sentence)
        sentence_perplexity = 2 ** (-log_prob / num_words)

        print(f"\n句子: \"{sentence}\"")
        print(f"  对数概率: {log_prob:.4f}")
        print(f"  词数: {num_words}")
        print(f"  困惑度: 2^(-{log_prob:.4f}/{num_words}) = {sentence_perplexity:.2f}")

        total_log_prob += log_prob
        total_words += num_words

    # 总体困惑度
    overall_perplexity = 2 ** (-total_log_prob / total_words)
    cross_entropy = -total_log_prob / total_words

    print(f"\n📊 总体评估结果：")
    print(f"  总对数概率: {total_log_prob:.4f}")
    print(f"  总词数: {total_words}")
    print(f"  交叉熵: {cross_entropy:.4f}")
    print(f"  困惑度: {overall_perplexity:.2f}")

    # 可视化困惑度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：每个句子的困惑度
    sentence_perplexities = []
    for sentence in test_sentences:
        log_prob, num_words = calculate_sentence_probability(sentence)
        pp = 2 ** (-log_prob / num_words)
        sentence_perplexities.append(pp)

    ax1.bar(range(len(test_sentences)), sentence_perplexities,
            color=['skyblue', 'lightcoral'])
    ax1.set_xlabel('测试句子')
    ax1.set_ylabel('困惑度')
    ax1.set_title('各句子困惑度')
    ax1.set_xticks(range(len(test_sentences)))
    ax1.set_xticklabels([f'句子{i + 1}' for i in range(len(test_sentences))])

    # 在柱状图上显示数值
    for i, pp in enumerate(sentence_perplexities):
        ax1.text(i, pp + 0.5, f'{pp:.1f}', ha='center', va='bottom')

    # 右图：困惑度解释
    ax2.text(0.1, 0.8, '困惑度解释:', fontsize=14, fontweight='bold')
    ax2.text(0.1, 0.6, f'• 总体困惑度: {overall_perplexity:.1f}', fontsize=12)
    ax2.text(0.1, 0.5, f'• 模型平均在 {overall_perplexity:.0f} 个词中选择', fontsize=11)
    ax2.text(0.1, 0.3, '• 困惑度越低，模型预测越准确', fontsize=11)
    ax2.text(0.1, 0.2, '• 理想情况：困惑度 = 1', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    return overall_perplexity

perplexity = calculate_perplexity_demo()
print(f"\n✅ 困惑度计算完成！最终困惑度: {perplexity:.2f}")


class NgramLanguageModel:
    """N-gram语言模型实现"""

    def __init__(self, n=2, smoothing='laplace', k=1.0):
        """
        初始化N-gram语言模型

        参数：
        n: N-gram的阶数 (1=unigram, 2=bigram, 3=trigram, ...)
        smoothing: 平滑方法 ('laplace', 'add_k', 'interpolation')
        k: 平滑参数
        """
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.total_words = 0

    def preprocess_text(self, text):
        """文本预处理"""
        # 简单的分词和清理
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        return words

    def get_ngrams(self, words):
        """获取N-gram序列"""
        # 添加开始和结束标记
        padded_words = ['<s>'] * (self.n - 1) + words + ['</s>']

        ngrams = []
        for i in range(len(padded_words) - self.n + 1):
            ngram = tuple(padded_words[i:i + self.n])
            ngrams.append(ngram)

        return ngrams

    def train(self, texts):
        """训练模型"""
        print(f"🚀 开始训练 {self.n}-gram 语言模型...")

        for text in texts:
            words = self.preprocess_text(text)
            self.vocabulary.update(words)
            self.total_words += len(words)

            # 获取N-gram
            ngrams = self.get_ngrams(words)

            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                # 计算上下文(前n-1个词)的计数
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

        print(f"✅ 训练完成！")
        print(f"   词汇表大小: {len(self.vocabulary)}")
        print(f"   总词数: {self.total_words}")
        print(f"   {self.n}-gram总数: {len(self.ngram_counts)}")

    def get_probability(self, ngram):
        """计算N-gram概率"""
        if self.n == 1:
            # Unigram概率
            if self.smoothing == 'laplace':
                return (self.ngram_counts[ngram] + self.k) / (self.total_words + self.k * len(self.vocabulary))
            else:
                return self.ngram_counts[ngram] / self.total_words if self.total_words > 0 else 0
        else:
            # N-gram条件概率
            context = ngram[:-1]

            if self.smoothing == 'laplace':
                numerator = self.ngram_counts[ngram] + self.k
                denominator = self.context_counts[context] + self.k * len(self.vocabulary)
                return numerator / denominator if denominator > 0 else 0
            else:
                if self.context_counts[context] > 0:
                    return self.ngram_counts[ngram] / self.context_counts[context]
                else:
                    return 0

    def sentence_probability(self, sentence):
        """计算句子概率"""
        words = self.preprocess_text(sentence)
        ngrams = self.get_ngrams(words)

        log_prob = 0.0
        for ngram in ngrams:
            prob = self.get_probability(ngram)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # 避免log(0)

        return math.exp(log_prob)

    def perplexity(self, test_sentences):
        """计算困惑度"""
        total_log_prob = 0.0
        total_words = 0

        for sentence in test_sentences:
            words = self.preprocess_text(sentence)
            ngrams = self.get_ngrams(words)

            for ngram in ngrams:
                prob = self.get_probability(ngram)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(1e-10)
                total_words += 1

        if total_words > 0:
            return 2 ** (-total_log_prob / total_words)
        else:
            return float('inf')

    def generate_text(self, start_words=None, max_length=20):
        """生成文本"""
        if start_words is None:
            start_words = ['<s>'] * (self.n - 1)

        words = start_words.copy()

        for _ in range(max_length):
            # 获取当前上下文
            if self.n == 1:
                context = tuple()
            else:
                context = tuple(words[-(self.n - 1):])

            # 找到所有可能的下一个词
            candidates = []
            for ngram, count in self.ngram_counts.items():
                if self.n == 1 or (len(ngram) == self.n and ngram[:-1] == context):
                    next_word = ngram[-1]
                    prob = self.get_probability(ngram)
                    candidates.append((next_word, prob))

            if not candidates:
                break

            # 根据概率选择下一个词
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 简单的贪心选择最高概率的词
            next_word = candidates[0][0]

            if next_word == '</s>':
                break

            words.append(next_word)

        # 移除特殊标记
        generated = [w for w in words if w not in ['<s>', '</s>']]
        return ' '.join(generated)

print("🎯 N-gram语言模型类定义完成！")


# 🎮 实战演示：完整的语言模型训练和评估
def run_language_model_demo():
    """运行完整的语言模型演示"""
    # 准备训练数据 (模拟一些简单的中文句子)
    train_texts = [
        "我爱学习自然语言处理",
        "自然语言处理是人工智能的重要分支",
        "机器学习算法可以用于文本分析",
        "深度学习在语言模型中有重要应用",
        "我喜欢研究机器学习算法",
        "人工智能技术发展很快",
        "文本分析需要用到统计方法",
        "语言模型可以用于文本生成",
        "我对深度学习很感兴趣",
        "自然语言理解是一个挑战性问题",
        "机器学习模型需要大量数据训练",
        "人工智能在各个领域都有应用"
    ]

    # 测试数据
    test_texts = [
        "我爱人工智能",
        "深度学习很有趣",
        "机器学习算法很重要"
    ]

    print("📚 训练数据示例：")
    for i, text in enumerate(train_texts[:3], 1):
        print(f"  {i}. {text}")
    print(f"  ... (共{len(train_texts)}个句子)")

    print("\n🧪 测试数据：")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")

    # 比较不同的N-gram模型
    models = {}
    perplexities = {}

    for n in [1, 2, 3]:
        print(f"\n{'=' * 50}")
        print(f"🔄 训练 {n}-gram 模型")
        print('=' * 50)

        model = NgramLanguageModel(n=n, smoothing='laplace', k=1.0)
        model.train(train_texts)
        models[n] = model

        # 计算困惑度
        perplexity = model.perplexity(test_texts)
        perplexities[n] = perplexity
        print(f"📊 困惑度: {perplexity:.2f}")

        # 计算每个测试句子的概率
        print("📝 句子概率：")
        for sentence in test_texts:
            prob = model.sentence_probability(sentence)
            print(f"  '{sentence}': {prob:.2e}")

    # 可视化比较结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 困惑度比较
    ns = list(perplexities.keys())
    pps = list(perplexities.values())
    bars1 = ax1.bar(ns, pps, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_xlabel('N-gram阶数')
    ax1.set_ylabel('困惑度')
    ax1.set_title('不同N-gram模型的困惑度比较')
    ax1.set_xticks(ns)

    # 在柱状图上显示数值
    for bar, pp in zip(bars1, pps):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{pp:.1f}', ha='center', va='bottom')

    # 2. 词汇表大小比较
    vocab_sizes = [len(models[n].vocabulary) for n in ns]
    bars2 = ax2.bar(ns, vocab_sizes, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_xlabel('N-gram阶数')
    ax2.set_ylabel('词汇表大小')
    ax2.set_title('词汇表大小')
    ax2.set_xticks(ns)

    for bar, size in zip(bars2, vocab_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{size}', ha='center', va='bottom')

    # 3. N-gram数量比较
    ngram_counts = [len(models[n].ngram_counts) for n in ns]
    bars3 = ax3.bar(ns, ngram_counts, color=['lightcyan', 'lightgreen', 'mistyrose'])
    ax3.set_xlabel('N-gram阶数')
    ax3.set_ylabel('N-gram总数')
    ax3.set_title('N-gram总数比较')
    ax3.set_xticks(ns)

    for bar, count in zip(bars3, ngram_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{count}', ha='center', va='bottom')

    # 4. 模型性能总结
    ax4.text(0.1, 0.8, '模型性能总结:', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.6, f'• Unigram困惑度: {perplexities[1]:.1f}', fontsize=12)
    ax4.text(0.1, 0.5, f'• Bigram困惑度: {perplexities[2]:.1f}', fontsize=12)
    ax4.text(0.1, 0.4, f'• Trigram困惑度: {perplexities[3]:.1f}', fontsize=12)

    best_model = min(perplexities.items(), key=lambda x: x[1])
    ax4.text(0.1, 0.2, f'🏆 最佳模型: {best_model[0]}-gram (困惑度: {best_model[1]:.1f})',
             fontsize=12, fontweight='bold', color='red')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

    return models


# 运行演示
print("🚀 开始语言模型完整演示...")
trained_models = run_language_model_demo()


# 🎨 文本生成演示（修复版）
def demonstrate_text_generation():
    """演示不同N-gram模型的文本生成能力"""

    print("🎨 文本生成演示")
    print("=" * 50)

    # 检查trained_models是否存在
    if 'trained_models' not in globals():
        print("⚠️ 模型尚未训练，正在训练模型...")
        global trained_models
        trained_models = run_language_model_demo()

    # 使用之前训练好的模型进行文本生成
    for n in [1, 2, 3]:
        if n in trained_models:
            model = trained_models[n]

            print(f"\n📝 {n}-gram 模型生成的文本：")

            # 生成几个不同的文本
            for i in range(3):
                try:
                    if n == 1:
                        # Unigram 模型，从词汇表中随机开始
                        vocab_list = list(model.vocabulary)
                        if vocab_list:
                            import random
                            start_word = random.choice(vocab_list)
                            generated = model.generate_text_improved([start_word], max_length=8)
                        else:
                            generated = "无法生成文本（词汇表为空）"
                    else:
                        # 使用不同的起始词
                        start_options = [['我'], ['深度'], ['机器']]
                        if i < len(start_options):
                            start_words = start_options[i]
                        else:
                            start_words = ['我']

                        generated = model.generate_text_improved(start_words, max_length=10)

                    print(f"  生成 {i + 1}: {generated}")
                except Exception as e:
                    print(f"  生成 {i + 1}: 生成失败 - {str(e)}")
        else:
            print(f"\n❌ {n}-gram 模型不存在")

    # 比较不同起始词的生成效果
    print(f"\n🔍 固定起始词 '我' 的生成效果比较：")
    print("-" * 40)

    for n in [2, 3]:  # Unigram不需要起始词上下文
        if n in trained_models:
            try:
                model = trained_models[n]
                generated = model.generate_text_improved(['我'], max_length=8)
                print(f"{n}-gram: {generated}")
            except Exception as e:
                print(f"{n}-gram: 生成失败 - {str(e)}")


# 为NgramLanguageModel类添加改进的文本生成方法
def add_improved_generation_method():
    """为模型类添加改进的文本生成方法"""

    def generate_text_improved(self, start_words=None, max_length=20):
        """改进的文本生成方法"""
        import random

        if start_words is None or len(start_words) == 0:
            # 随机选择一个起始词
            if self.vocabulary:
                start_words = [random.choice(list(self.vocabulary))]
            else:
                return "无法生成（词汇表为空）"

        words = start_words.copy()

        for step in range(max_length):
            # 获取当前上下文
            if self.n == 1:
                context = tuple()
            else:
                context = tuple(words[-(self.n - 1):])

            # 找到所有可能的下一个词（排除结束符）
            candidates = []
            for ngram, count in self.ngram_counts.items():
                if self.n == 1:
                    next_word = ngram[0]
                    if next_word != '</s>' and next_word != '<s>':
                        prob = self.get_probability(ngram)
                        candidates.append((next_word, prob))
                elif len(ngram) == self.n and ngram[:-1] == context:
                    next_word = ngram[-1]
                    if next_word != '</s>' and next_word != '<s>':
                        prob = self.get_probability(ngram)
                        candidates.append((next_word, prob))

            if not candidates:
                # 如果没有候选词，尝试回退策略
                if self.n > 1 and len(context) > 0:
                    # 回退到更短的上下文
                    shorter_context = context[1:] if len(context) > 1 else tuple()
                    for ngram, count in self.ngram_counts.items():
                        if len(ngram) == self.n and ngram[:-2] == shorter_context:
                            next_word = ngram[-1]
                            if next_word != '</s>' and next_word != '<s>':
                                prob = self.get_probability(ngram)
                                candidates.append((next_word, prob))

                # 如果还是没有候选词，随机选择一个词汇表中的词
                if not candidates and self.vocabulary:
                    vocab_words = [w for w in self.vocabulary if w not in ['<s>', '</s>']]
                    if vocab_words:
                        next_word = random.choice(vocab_words)
                        candidates.append((next_word, 0.01))

            if not candidates:
                break

            # 使用概率加权的随机选择，而不是贪心选择
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 选择前几个高概率的候选词进行随机选择
            top_candidates = candidates[:min(3, len(candidates))]
            total_prob = sum(prob for _, prob in top_candidates)

            if total_prob > 0:
                # 按概率随机选择
                rand_val = random.random() * total_prob
                cumulative_prob = 0
                selected_word = top_candidates[0][0]  # 默认选择

                for word, prob in top_candidates:
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        selected_word = word
                        break
            else:
                selected_word = top_candidates[0][0]

            words.append(selected_word)

        # 移除特殊标记并返回结果
        generated = [w for w in words if w not in ['<s>', '</s>']]
        return ' '.join(generated) if generated else "无法生成文本"

    # 将方法添加到类中
    NgramLanguageModel.generate_text_improved = generate_text_improved


# 添加改进的生成方法
add_improved_generation_method()
print("✅ 已为模型添加改进的文本生成方法")

# 运行改进后的演示
demonstrate_text_generation()


# 🔬 模型分析与优化实验
def analyze_and_optimize_models():
    """分析模型性能并进行优化实验"""

    print("🔬 语言模型分析与优化")
    print("=" * 50)

    # 1. 平滑参数对性能的影响
    print("\n📊 实验1: 平滑参数对Bigram模型的影响")
    print("-" * 40)

    train_texts = [
        "我爱学习自然语言处理",
        "自然语言处理是人工智能的重要分支",
        "机器学习算法可以用于文本分析",
        "深度学习在语言模型中有重要应用"
    ]

    test_texts = ["我爱人工智能", "深度学习很有趣"]

    k_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    perplexities_by_k = []

    for k in k_values:
        model = NgramLanguageModel(n=2, smoothing='laplace', k=k)
        model.train(train_texts)
        pp = model.perplexity(test_texts)
        perplexities_by_k.append(pp)
        print(f"k={k}: 困惑度={pp:.2f}")

    # 2. 训练数据大小对性能的影响
    print("\n📊 实验2: 训练数据大小对性能的影响")
    print("-" * 40)

    full_train_texts = [
        "我爱学习自然语言处理",
        "自然语言处理是人工智能的重要分支",
        "机器学习算法可以用于文本分析",
        "深度学习在语言模型中有重要应用",
        "我喜欢研究机器学习算法",
        "人工智能技术发展很快",
        "文本分析需要用到统计方法",
        "语言模型可以用于文本生成"
    ]

    data_sizes = [2, 4, 6, 8]
    perplexities_by_size = []

    for size in data_sizes:
        model = NgramLanguageModel(n=2, smoothing='laplace', k=1.0)
        model.train(full_train_texts[:size])
        pp = model.perplexity(test_texts)
        perplexities_by_size.append(pp)
        print(f"训练句子数={size}: 困惑度={pp:.2f}")

    # 可视化实验结果
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 实验1：平滑参数影响
    ax1.plot(k_values, perplexities_by_k, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('平滑参数 k')
    ax1.set_ylabel('困惑度')
    ax1.set_title('平滑参数对困惑度的影响')
    ax1.grid(True, alpha=0.3)

    # 在图上标注最佳参数
    best_k_idx = np.argmin(perplexities_by_k)
    best_k = k_values[best_k_idx]
    best_pp = perplexities_by_k[best_k_idx]
    ax1.annotate(f'最佳: k={best_k}\\n困惑度={best_pp:.2f}',
                 xy=(best_k, best_pp),
                 xytext=(best_k + 0.3, best_pp + 5),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # 实验2：训练数据大小影响
    ax2.plot(data_sizes, perplexities_by_size, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('训练句子数量')
    ax2.set_ylabel('困惑度')
    ax2.set_title('训练数据大小对困惑度的影响')
    ax2.grid(True, alpha=0.3)

    # 实验3：模型复杂度 vs 性能
    if 'trained_models' in globals():
        ns = list(trained_models.keys())
        model_perplexities = []
        for n in ns:
            pp = trained_models[n].perplexity(["我爱人工智能", "深度学习很有趣"])
            model_perplexities.append(pp)

        ax3.bar(ns, model_perplexities, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        ax3.set_xlabel('N-gram 阶数')
        ax3.set_ylabel('困惑度')
        ax3.set_title('模型复杂度 vs 性能')
        ax3.set_xticks(ns)

        # 在柱状图上显示数值
        for i, pp in enumerate(model_perplexities):
            ax3.text(ns[i], pp + 0.5, f'{pp:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 3. 性能分析总结
    print("\\n📈 模型分析总结：")
    print(f"• 最佳平滑参数: k={best_k} (困惑度: {best_pp:.2f})")
    print(f"• 数据量效果: 更多数据通常带来更好性能")
    print(f"• 模型复杂度: 需要在复杂度和泛化能力间平衡")

    return {
        'best_k': best_k,
        'k_perplexities': dict(zip(k_values, perplexities_by_k)),
        'size_perplexities': dict(zip(data_sizes, perplexities_by_size))
    }


# 运行分析
analysis_results = analyze_and_optimize_models()

