# 特征工程实战代码
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import re

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 使用智能客服数据进行特征工程演示
sample_texts = [
    "怎么退款？急急急",
    "我的订单什么时候发货呢",
    "有什么优惠活动吗？想了解一下",
    "产品质量有问题，要求退货！",
    "客服电话多少？联系不上",
    "能不能换货？不满意这个颜色",
    "为什么还没收到货？已经一周了",
    "这个产品怎么使用？说明书看不懂",
    "我要投诉！服务态度太差了",
    "有新品推荐吗？想买点东西"
]

intents = [
    "退款咨询", "物流查询", "优惠咨询", "售后投诉", "联系方式",
    "换货咨询", "物流查询", "使用咨询", "售后投诉", "产品咨询"
]

df = pd.DataFrame({
    'text': sample_texts,
    'intent': intents
})

print("📊 === 特征工程数据准备 ===")
print(f"数据量: {len(df)} 条")
print("样本数据:")
for idx, (i, row) in enumerate(df.iterrows()):
    print(f"    {idx+1}. '{row['text']}' -> {row['intent']}")

print("\n⚙️ === 第1层：基础统计特征 ===")

def extract_basic_features(input_text):
    """提取基础统计特征"""
    features = {'text_length': len(input_text), 'word_count': len(jieba.lcut(input_text)),  # 文本长度特征
                'question_marks': input_text.count('？') + input_text.count('?'),  # 标点符号特征
                'exclamation_marks': input_text.count('！') + input_text.count('!'),
                'punctuation_count': len(re.findall(r'[，。！？、；：]', input_text))}

    # 特殊词汇特征
    urgent_words = ['急', '快', '马上', '立即', '赶紧']
    features['urgent_words'] = sum(1 for word in urgent_words if word in input_text)

    negative_words = ['不', '没', '差', '坏', '烂', '糟']
    features['negative_words'] = sum(1 for word in negative_words if word in input_text)

    return features

# 提取基础特征
basic_features_list = []
for text in df['text']:
    basic_features_list.append(extract_basic_features(text))

basic_features_df = pd.DataFrame(basic_features_list)
print("基础特征示例：")
print(basic_features_df.head())

print(f"\n基础特征统计:")
print(basic_features_df.describe())

print("\n⚙️ === 第2层：词袋模型特征 ===")

# 中文分词预处理
def preprocess_chinese(chinese_text):
    """中文文本预处理"""
    # 分词
    words = jieba.lcut(chinese_text)
    # 去除停用词和标点
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '上', '也', '很', '到', '说',
                  '要', '去', '你', '会', '着', '没有', '看', '好', '这'}
    words = [w for w in words if w not in stop_words and len(w.strip()) > 1 and not re.match(r'\W', w)]
    return ' '.join(words)

# 对文本进行预处理
processed_texts = [preprocess_chinese(text) for text in df['text']]
print("预处理后的文本：")
for i, (orig, proc) in enumerate(zip(df['text'], processed_texts)):
    print(f"{i + 1}. 原文：{orig}")
    print(f"    处理：{proc}")
    print()

# 词袋模型特征提取
print("构建词袋模型")
count_vectorizer = CountVectorizer(
    max_features=100,               # 最多保留100个特征
    ngram_range=(1, 2)              # 1-gram和2-gram
)

bow_features = count_vectorizer.fit_transform(processed_texts)
feature_names = count_vectorizer.get_feature_names_out()

print(f"词袋模型特征维度: {bow_features.shape}")
print(f"特征词汇示例: {list(feature_names[:])}")

# 展示部分特征矩阵
bow_df = pd.DataFrame(bow_features.toarray()[:5, :10], columns=feature_names[:10])
print("词袋特征矩阵示例:")
print(bow_df)

print("\n⚙️ === 第3层：TF-IDF特征 ===")

# TF-IDF特征提取
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    sublinear_tf=True       # 使用次线性缩放
)

tfidf_features = tfidf_vectorizer.fit_transform(processed_texts)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"TF-IDF特征维度：{tfidf_features.shape}")

# 展示TF-IDF特征重要性
feature_importance = np.array(tfidf_features.sum(axis=0)).flatten()
importance_df = pd.DataFrame({
    'feature': tfidf_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("TF-IDF特征重要性Top10：")
print(importance_df.head(10))

print("\n⚙️ === 第4层：组合特征 ===")

# 将不同类型的特征组合
# 将基础特征转换为稀疏矩阵格式
basic_features_sparse = csr_matrix(basic_features_df.values)

# 组合所有特征
combined_features = hstack([
    basic_features_sparse,          # 基础统计特征
    tfidf_features                  # TF-IDF特征
])

print(f"组合特征维度: {combined_features.shape}")
print(f"  - 基础特征: {basic_features_sparse.shape[1]} 维")
print(f"  - TF-IDF特征: {tfidf_features.shape[1]} 维")

print("\n📊 === 特征工程效果可视化 ===")

# 可视化不同特征的分布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 文本长度分布
axes[0, 0].hist(basic_features_df['text_length'], bins=5, alpha=0.7, color='skyblue')
axes[0, 0].set_title('文本长度分布')
axes[0, 0].set_xlabel('文本长度')
axes[0, 0].set_ylabel('频次')

# 2. 标点符号使用
punct_data = basic_features_df[['question_marks', 'exclamation_marks']].sum()
axes[0, 1].bar(punct_data.index, punct_data.values, color=['orange', 'red'], alpha=0.7)
axes[0, 1].set_title('标点符号使用情况')
axes[0, 1].set_ylabel('总数')

# 3. 特征重要性
top_features = importance_df.head(8)
axes[1, 0].barh(top_features['feature'], top_features['importance'], color='green', alpha=0.7)
axes[1, 0].set_title('TF-IDF特征重要性')
axes[1, 0].set_xlabel('重要性得分')

# 4. 特征类型分布
feature_types = ['基础特征', 'TF-IDF特征']
feature_counts = [basic_features_sparse.shape[1], tfidf_features.shape[1]]
axes[1, 1].pie(feature_counts, labels=feature_types, autopct='%1.1f%%',
              colors=['lightblue', 'lightgreen'], startangle=90)
axes[1, 1].set_title('特征类型分布')

plt.tight_layout()
plt.show()

print("\n✅ === 特征工程总结 ===")
print("🎯 完成的特征类型:")
print("  ✅ 基础统计特征: 文本长度、标点符号、特殊词汇")
print("  ✅ 词袋模型特征: 词频统计")
print("  ✅ TF-IDF特征: 词频-逆文档频率")
print("  ✅ N-gram特征: 1-gram + 2-gram")
print("  ✅ 组合特征: 多种特征融合")

print(f"\n📊 最终特征矩阵:")
print(f"  - 样本数: {combined_features.shape[0]} 条")
print(f"  - 特征维度: {combined_features.shape[1]} 维")
print(f"  - 特征密度: {combined_features.nnz / (combined_features.shape[0] * combined_features.shape[1]):.4f}")

print("\n💡 特征工程建议:")
print("✅ 基础特征能捕捉文本的基本统计信息")
print("✅ TF-IDF特征能识别重要的词汇信息")
print("✅ N-gram特征能捕捉词汇搭配信息")
print("✅ 组合特征提供了更全面的文本表示")
print("✅ 特征已准备好进入模型训练阶段！")