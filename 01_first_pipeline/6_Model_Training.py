# 建模实战代码
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import time

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

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
training_times = {}

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
    test_accuracy = np.mean(y_pred == y_test)

    # 存储结果
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'model': model
    }
    training_times[name] = training_time

    print(f"  交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  测试集准确率: {test_accuracy:.4f}")
    print(f"  训练时间: {training_time:.2f}秒")

print("\n📊 === 模型性能对比 ===")

# 创建结果DataFrame
results_df = pd.DataFrame({
    name: {
        '交叉验证准确率': data['cv_mean'],
        '测试集准确率': data['test_accuracy'],
        '训练时间（秒）': training_times[name]
    }
    for name, data in results.items()
}).T

print("各模型性能汇总:")
print(results_df.round(4))

# 找出最佳模型
best_model_name = max(results.keys(), key=lambda key: results[key]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"测试集准确率: {results[best_model_name]['test_accuracy']:.4f}")

print("\n📈 === 性能可视化 ===")

# 可视化模型性能对比
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 准确率对比
model_names = list(results.keys())
test_accs = [results[name]['test_accuracy'] for name in model_names]
cv_accs = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0, 0].bar(x - width / 2, cv_accs, width, label='交叉验证', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width / 2, test_accs, width, label='测试集', alpha=0.8, color='lightcoral')
axes[0, 0].set_xlabel('模型')
axes[0, 0].set_ylabel('准确率')
axes[0, 0].set_title('模型准确率对比')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 训练时间对比
train_times = [training_times[name] for name in model_names]
axes[0, 1].bar(model_names, train_times, color='lightgreen', alpha=0.8)
axes[0, 1].set_xlabel('模型')
axes[0, 1].set_ylabel('训练时间（秒）')
axes[0, 1].set_title('训练时间对比')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. 最佳模型的混淆矩阵
y_pred_best = best_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_,
            ax=axes[1, 0])
axes[1, 0].set_title(f'{best_model_name} 混淆矩阵')
axes[1, 0].set_xlabel('预测标签')
axes[1, 0].set_ylabel('真实标签')

# 4. 准确率 vs 速度 散点图
axes[1, 1].scatter(train_times, test_accs, s=100, alpha=0.7, c='purple')
for i, name in enumerate(model_names):
    axes[1, 1].annotate(name, (train_times[i], test_accs[i]),
                        xytext=(5, 5), textcoords='offset points',)
axes[1, 1].set_xlabel('训练时间（秒）')
axes[1, 1].set_ylabel('测试准确率')
axes[1, 1].set_title('准确率 vs 训练时间')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🔧 === 超参数调优示例 ===")

# 对最佳模型进行超参数调优
if best_model_name == '随机森林':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
    }
elif best_model_name == '逻辑回归':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
else:
    param_grid = {}

if param_grid:
    print(f"对 {best_model_name} 进行超参数调优...")

    grid_search = GridSearchCV(
        models[best_model_name],
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1
    )

    grid_search.fit(x_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 在测试集上评估调优后的模型
    optimized_score = grid_search.score(x_test, y_test)
    print(f"调优后测试准确率: {optimized_score:.4f}")

    # 性能提升
    improvement = optimized_score - results[best_model_name]['test_accuracy']
    print(f"性能提升: {improvement:.4f}")

print("\n✅ === 建模总结 ===")
print("🎯 模型训练完成情况:")
print(f"  ✅ 测试了 {len(models)} 种不同模型")
print(f"  ✅ 最佳模型: {best_model_name}")
print(f"  ✅ 最高准确率: {max(test_accs):.4f}")
print(f"  ✅ 平均训练时间: {np.mean(train_times):.2f}秒")

print("\n💡 建模建议:")
print("✅ 简单模型(朴素贝叶斯)训练快速，适合快速原型")
print("✅ 复杂模型(随机森林)效果更好，但需要更多计算资源")
print("✅ 线性模型(逻辑回归)平衡了效果和速度")
print("✅ 可以根据业务需求选择合适的模型")
print("✅ 模型已准备好进入评估阶段！")
