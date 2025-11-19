# 建模实战代码
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import time

# 准备更大的模拟数据集用于建模
np.random.seed(42)

# 生成模拟的智能客服数据
def generate_sample_data(n_samples=500):
    """生成模拟的智能客服数据"""
    intents = ['退款咨询', '物流查询', '优惠咨询', '售后投诉', '联系方式']

    # 不同意图的模板
    templates = {
        '退款咨询': ['怎么退款', '退款流程', '申请退款', '钱什么时候退回来', '退款要多久'],
        '物流查询': ['什么时候发货', '查询物流', '快递到哪了', '多久能收到货', '物流信息'],
        '优惠咨询': ['有什么优惠', '打折活动', '优惠券怎么用', '促销信息', '会员价格'],
        '售后投诉': ['产品有问题', '质量不好', '要投诉', '服务态度差', '不满意'],
        '联系方式': ['客服电话', '联系方式', '人工客服', '在线客服', '客服QQ']
    }

    sample_texts = []
    sample_labels = []

    for _ in range(n_samples):
        intent = np.random.choice(intents)
        template = np.random.choice(templates[intent])

        # 添加一些随机变化
        variations = ['？', '！', '呢', '吗', '啊', '，急急急', '，谢谢']
        text = template + np.random.choice(variations)

        sample_texts.append(text)
        sample_labels.append(intent)

    return sample_texts, sample_labels

# 生成训练数据
texts, labels = generate_sample_data(500)
df = pd.DataFrame({'text': texts, 'intent': labels})

print("🤖 === 建模数据准备 ===")
print(f"数据量: {len(df)} 条")
print("意图分布:")
print(df['intent'].value_counts())

# 文本预处理
def preprocess_text(text):
    """简单的文本预处理"""
    words = jieba.lcut(text)
    # 去除停用词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '上', '也', '很', '到', '说',
                  '要', '去', '你', '会', '着', '没有', '看', '好', '这'}
    words = [w for w in words if w not in stop_words and len(w.strip()) > 1]
    return ' '.join(words)

# 预处理文本
processed_texts = [preprocess_text(text) for text in df['text']]

# 特征提取
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
x = tfidf.fit_transform(processed_texts)
y = df['intent']

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集: {x_train.shape[0]} 条")
print(f"测试集: {x_test.shape[0]} 条")
print(f"特征维度: {x_train.shape[1]} 维")

print("\n🧠 === 模型训练对比 ===")

# 定义多个模型进行对比
models = {
    '朴素贝叶斯': MultinomialNB(),
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    '支持向量机': SVC(kernel='linear', random_state=42),
    'K近邻': KNeighborsClassifier(n_neighbors=5)
}

# 存储结果
results = {}
traning_times = {}

print("开始训练各种模型...")
for name, model in models.items():
    print(f"\n训练 {name}...")

    # 训练时间测量
    start_time = time.time()

    # 交叉验证
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

    # 训练完整模型
    model.fit(x_train, y_train)

    training_time = time.time() - start_time

    # 预测
    y_pred = model.predict(x_test)
    test_accuracy = (y_pred == y_test).mean()

    # 存储结果
    results[name] = {
        'cv_neam': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'model': model
    }
    traning_times[name] = training_time

    print(f"  交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  测试集准确率: {test_accuracy:.4f}")
    print(f"  训练时间: {training_time:.2f}秒")

print("\n📊 === 模型性能对比 ===")