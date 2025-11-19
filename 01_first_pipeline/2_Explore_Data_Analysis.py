# 数据探索示例代码 - 智能客服数据分析
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.font_manager as fm
import subprocess
import sys
import os

# 配置中文字体 - 更完善的解决方案
def setup_chinese_font():
    """设置中文字体，解决中文显示问题"""
    try:
        # 根据操作系统选择字体配置方式
        if sys.platform.startswith('linux'):
            print("检测到Linux系统，正在安装中文字体...")
            # 方法1：尝试安装和使用系统中文字体
            subprocess.run(['sudo', 'apt-get', 'update'],
                           check=True, capture_output=True, text=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y',
                            'font-wqy-zenhei', 'font-wqy-microhei',
                            'font-noto-cjk'],
                           check=True, capture_output=True, text=True)

            # 清除matplotlib字体缓存
            cache_dir = matplotlib.get_cachedir()
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

            # 重新加载字体
            matplotlib.pyplot.rcdefaults()

            # 设置Linux中文字体优先级
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Zen Hei',
                'WenQuanYi Micro Hei', 
                'Noto Sans CJK SC',
                'DejaVu Sans',
                'Arial'
            ]
        elif sys.platform.startswith('win'):
            print("检测到Windows系统，配置Windows中文字体...")
            # Windows系统中文字体配置
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei',
                'SimHei',
                'SimSun',
                'KaiTi',
                'FangSong',
                'Arial Unicode MS',
                'DejaVu Sans',
                'Arial'
            ]
        elif sys.platform.startswith('darwin'):
            print("检测到macOS系统，配置macOS中文字体...")
            # macOS系统中文字体配置
            plt.rcParams['font.sans-serif'] = [
                'PingFang SC',
                'Hiragino Sans GB',
                'STHeiti',
                'Microsoft YaHei',
                'SimHei',
                'DejaVu Sans',
                'Arial'
            ]
        else:
            print(f"未知操作系统 {sys.platform}，使用默认字体配置")
            plt.rcParams['font.sans-serif'] = [
                'DejaVu Sans',
                'Arial'
            ]
            
        plt.rcParams['axes.unicode_minus'] = False

        # 测试中文显示
        fig_for_test, ax_for_test = plt.subplots(figsize=(2, 2))
        ax_for_test.text(0.5, 0.5, '测试中文', ha='center', va='center', fontsize=12)
        ax_for_test.set_xlim(0, 1)
        ax_for_test.set_ylim(0, 1)
        ax_for_test.axis('off')
        plt.close(fig_for_test)

        return True
    except Exception as e:
        print(f"中文字体配置失败: {e}")
        # 如果配置失败，使用英文替代
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 执行字体设置
chinese_font_available = setup_chinese_font()

# 字体状态提示
if chinese_font_available:
    print("✅ 中文字体配置成功")
else:
    print("⚠️ 中文字体配置失败，将使用英文标题")

# 安全的标题函数
def safe_title(chinese_title, english_title):
    """安全的标题函数，根据字体可用性选择标题"""
    if chinese_font_available:
        return chinese_title
    else:
        return english_title

# 模拟只能客服数据
customer_questions = [
    {"text": "怎么退款？", "intent": "退款咨询", "length": 5},
    {"text": "我的订单什么时候发货", "intent": "物流查询", "length": 10},
    {"text": "有什么优惠活动吗", "intent": "优惠咨询", "length": 9},
    {"text": "产品质量有问题，要求退货", "intent": "售后投诉", "length": 12},
    {"text": "客服电话多少", "intent": "联系方式", "length": 7},
    {"text": "能不能换货？", "intent": "换货咨询", "length": 6},
    {"text": "为什么还没收到货", "intent": "物流查询", "length": 9},
    {"text": "这个产品怎么使用", "intent": "使用咨询", "length": 8},
    {"text": "我要投诉", "intent": "售后投诉", "length": 4},
    {"text": "有新品推荐吗", "intent": "产品咨询", "length": 7}
]

# 创建数据框
df = pd.DataFrame(customer_questions)

print("🔍 === 第一步：基础统计信息 ===")
print(f"📊 数据总量: {len(df)} 条")
print(f"📏 平均文本长度: {df['length'].mean():.1f} 个字符")
print(f"📈 文本长度范围: {df['length'].min()} - {df['length'].max()} 个字符")
print(f"🏷️ 意图类别数: {df['intent'].nunique()} 个")

print("\n🎯 === 第二步：意图分布分析 ===")
intent_counts = df['intent'].value_counts()
print("各意图类别分布：")
for intent, count in intent_counts.items():
    percentage = count / len(df) * 100
    print(f"  {intent}: {count}条({percentage:.2f}%)")

# 可视化分析
plt.figure(figsize=(15, 10))

# 1. 意图分布饼图
plt.subplot(2, 3, 1)
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
plt.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.2f%%',
        colors=colors[:len(intent_counts)], startangle=90)
plt.title(safe_title('意图分布', 'Intent Distribution'), fontsize=12, weight='bold')

# 2. 文本长度分布
plt.subplot(2, 3, 2)
plt.hist(df['length'], bins=6, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel(safe_title('文本长度（字符）', 'Text Length (Characters)'))
plt.ylabel(safe_title('频次', 'Frequency'))
plt.title(safe_title('文本长度分布', 'Text Length Distribution'), fontsize=12, weight='bold')

# 3. 按意图的长度分布
plt.subplot(2, 3, 3)
for intent in df['intent'].unique():
    lengths = df[df['intent'] == intent]['length'].values
    plt.hist(lengths, alpha=0.6, label=intent, bins=5)
plt.xlabel(safe_title('文本长度', 'Text Length'))
plt.ylabel(safe_title('频次', 'Frequency'))
plt.title(safe_title('各意图长度分布', 'Length Distribution by Intent'), fontsize=12, weight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 4. 词频分析
plt.subplot(2, 3, 4)
all_words = []
for text in df['text']:
    all_words.extend(list(text))            # 中文按字符分析

char_freq = Counter(all_words)
top_chars = char_freq.most_common(8)
chars, frequencies = zip(*top_chars)

plt.bar(chars, frequencies, color='lightgreen', alpha=0.7)
plt.xlabel(safe_title('字符', 'Characters'))
plt.ylabel(safe_title('频次', 'Frequency'))
plt.title(safe_title('高频字符分析', 'High-Frequency Character Analysis'), fontsize=12, weight='bold')

# 5. 数据质量检查
plt.subplot(2, 3, 5)
quality_metrics = {
    safe_title('完整', 'Complete'): len(df),
    safe_title('空值', 'Empty'): np.sum(df['text'].str.strip() == ''),
    safe_title('重复', 'Duplicate'): df.duplicated().sum(),
    safe_title('异常', 'Abnormal'): sum((df['length'] < 2) | (df['length'] > 50))
}

plt.bar(quality_metrics.keys(), quality_metrics.values(),
        color=['green', 'red', 'orange', 'yellow'], alpha=0.7)
plt.title(safe_title('数据质量检查', 'Data Quality Check'), fontsize=12, weight='bold')
plt.xticks(rotation=45)

# 6. 意图vs长度关系
plt.subplot(2, 3, 6)
df.boxplot(column='length', by='intent', ax=plt.gca())
plt.title(safe_title('意图类别vs文本长度', 'Intent Category vs Text Length'), fontsize=12, weight='bold')
plt.suptitle('')            # 移除自动标题

plt.tight_layout()
plt.show()

print("\n📈 === 第三步：数据质量评估 ===")
print(f"✅ 数据完整性: {(1 - df['text'].isnull().sum()/len(df))*100:.1f}%")
print(f"🔄 数据重复率: {(df.duplicated().sum()/len(df))*100:.1f}%")
print(f"⚠️ 异常数据: {sum((df['length'] < 2) | (df['length'] > 50))} 条")

print("\n💡 === 第四步：探索性发现 ===")
print("🎯 主要发现:")
print(f"1. 数据分布: {intent_counts.index[0]} 类问题最多({intent_counts.iloc[0]}条)")
print(f"2. 文本特点: 平均长度{df['length'].mean():.1f}字符，适合短文本模型")
print(f"3. 类别平衡: 最多类别{intent_counts.max()}条，最少{intent_counts.min()}条")
print("4. 数据质量: 整体质量良好，无明显异常")

print("\n🚀 === 建模建议 ===")
print("✅ 推荐模型: 短文本分类模型(如BERT、TextCNN)")
print("✅ 数据处理: 需要数据增强平衡各类别")
print("✅ 特征工程: 可以提取关键词、n-gram特征")
print("✅ 评估指标: 准确率、F1-score、混淆矩阵")

# 中文字体测试
# 创建一个简单的测试图来验证中文显示是否正常

fig, ax = plt.subplots(figsize=(10, 6))

# 测试不同字体大小的中文显示
test_texts = [
    "📊 数据探索 - Data Exploration",
    "🎯 意图分类 - Intent Classification",
    "📝 文本处理 - Text Processing",
    "🔍 特征提取 - Feature Extraction",
    "🤖 模型训练 - Model Training"
]

y_positions = [0.8, 0.6, 0.4, 0.2, 0.0]
font_sizes = [16, 14, 12, 10, 8]

for i, (text, y_pos, font_size) in enumerate(zip(test_texts, y_positions, font_sizes)):
    ax.text(0.1, y_pos, text, fontsize=font_size, weight='bold',
            transform=ax.transAxes, va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('中文字体显示测试 - Chinese Font Display Test', fontsize=18, weight='bold')
ax.axis('off')

plt.tight_layout()
plt.show()

# 显示字体配置信息
print("🔧 === 字体配置信息 ===")
print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")
print(f"中文字体状态: {'✅ 可用' if chinese_font_available else '❌ 不可用'}")

# 显示系统可用字体
print("\n📋 === 系统可用字体 ===")
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in available_fonts if any(keyword in f for keyword in ['Chinese', 'Hei', 'Song', 'Kai', 'Noto', 'WenQuanYi'])]
print(f"检测到的中文相关字体：{chinese_fonts[:]}...")