# 数据分割实战代码
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 使用前面清理好的数据，这里创建一个更大的模拟数据集
np.random.seed(42)

# 模拟智能客服数据集
intents = ['退款咨询', '物流查询', '优惠咨询', '售后投诉', '联系方式',
           '换货咨询', '使用咨询', '产品咨询', '订单查询', '技术支持']

# 模拟不平衡的数据分布（符合实际情况）
intent_weights = [0.25, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]

# 生成模拟数据
data_size = 1000
texts = []
labels = []

for i in range(data_size):
    intent = np.random.choice(intents, p=intent_weights)
    # 简单模拟文本
    text = f"这是一条关于{intent}的用户问询{i}"
    texts.append(text)
    labels.append(intent)

df = pd.DataFrame({
    'text': texts,
    'intent': labels}
)

print("📊 === 原始数据概览 ===")
print(f"总数据量: {len(df)} 条")
print("\n各意图分布:")
intent_counts = df['intent'].value_counts()
for intent, count in intent_counts.items():
    percentage = count / len(df) * 100
    print(f"  {intent}: {count:3d}条 ({percentage:.2f}%)")

# 可视化原始数据分布
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
intent_counts.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('原始数据分布', fontsize=12, weight='bold')
plt.xlabel('意图类别')
plt.ylabel('数量')
plt.xticks(rotation=45)

print("\n✂️ === 方法1: 简单随机分割 ===")

# 方法1：简单随机分割
x = df['text']
y = df['intent']

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {len(x_train)} 条 ({len(x_train)/len(df)*100:.1f}%)")
print(f"验证集: {len(x_val)} 条 ({len(x_val)/len(df)*100:.1f}%)")
print(f"测试集: {len(x_test)} 条 ({len(x_test)/len(df)*100:.1f}%)")

# 检查随机分割后的类别分布
print("\n各集合中的类别分布：")
for dataset_name, y_data in [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]:
    print(f"\n{dataset_name}:")
    dist = y_data.value_counts(normalize=True).sort_index()
    for intent in intents[:]:
        if intent in dist:
            print(f"  {intent}: {dist[intent] * 100:.2f}%")

# 方法2：分层分割，保证各类别比例一致
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(sss.split(x, y))

x_train_strat = x.iloc[train_idx]
y_train_strat = y.iloc[train_idx]
x_temp_strat = x.iloc[temp_idx]
y_temp_strat = y.iloc[temp_idx]

# 再次分割验证集和测试集
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(x_temp_strat, y_temp_strat))

x_val_strat = x_temp_strat.iloc[val_idx]
y_val_strat = y_temp_strat.iloc[val_idx]
x_test_strat = x_temp_strat.iloc[test_idx]
y_test_strat = y_temp_strat.iloc[test_idx]

print(f"训练集: {len(x_train_strat)} 条 ({len(x_train_strat)/len(df)*100:.1f}%)")
print(f"验证集: {len(x_val_strat)} 条 ({len(x_val_strat)/len(df)*100:.1f}%)")
print(f"测试集: {len(x_test_strat)} 条 ({len(x_test_strat)/len(df)*100:.1f}%)")

print("\n分层分割后的类别分布：")
for dataset_name, y_data in [('训练集', y_train_strat), ('验证集', y_val_strat), ('测试集', y_test_strat)]:
    print(f"\n{dataset_name}:")
    dist = y_data.value_counts(normalize=True).sort_index()
    for intent in intents[:]:
        if intent in dist:
            print(f"  {intent}: {dist[intent] * 100:.2f}%")

# 可视化分割结果比较
plt.subplot(1, 2, 2)
comparison_data = {
    '原始': intent_counts,
    '训练集': y_train_strat.value_counts(),
    '测试集': y_test_strat.value_counts(),
}

x = np.arange(len(intent_counts))  # 使用所有类别的数量
width = 0.25

for i, (name, data) in enumerate(comparison_data.items()):
    # 确保所有数据集都包含相同的类别顺序
    data_aligned = data.reindex(intent_counts.index, fill_value=0)
    plt.bar(x + i * width, data_aligned.values, width, label=name, alpha=0.8)

plt.title('分层分割效果对比', fontsize=12, weight='bold')
plt.xlabel('意图类别')
plt.ylabel('数量')
plt.xticks(x + width, intent_counts.index, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

print("\n🎯 === 分割质量验证 ===")

# 计算分布差异
def calculate_distribution_difference(original, subset):
    """计算分布差异"""
    orig_dist = original.value_counts(normalize=True).sort_index()
    subset_dist = subset.value_counts(normalize=True).sort_index()

    # 确保所有类别都存在
    all_classes = orig_dist.index.union(subset_dist.index)
    orig_dist = orig_dist.reindex(all_classes, fill_value=0)
    subset_dist = subset_dist.reindex(all_classes, fill_value=0)

    # 计算平均绝对差异
    diff = np.mean(np.abs(orig_dist - subset_dist))
    return diff

print("与原始分布的差异度（越小越好）：")
print(f"随机分割 - 训练集：{calculate_distribution_difference(y, y_train):.4f}")
print(f"随机分割 - 测试集：{calculate_distribution_difference(y, y_test):.4f}")
print(f"分层分割 - 训练集：{calculate_distribution_difference(y, y_train_strat):.4f}")
print(f"分层分割 - 测试集：{calculate_distribution_difference(y, y_test_strat):.4f}")

print("\n✅ === 数据分割总结 ===")
print("🎯 推荐方案: 分层分割")
print("📊 分割比例: 70% 训练 + 15% 验证 + 15% 测试")
print("✅ 优势: 保证了各类别在不同集合中的分布一致性")
print("✅ 结果: 模型训练更稳定，评估更可靠")

print("\n💾 === 保存分割后的数据 ===")
print("✅ 训练集已准备好用于模型训练")
print("✅ 验证集已准备好用于超参数调优")
print("✅ 测试集已准备好用于最终评估")
print("✅ 可以开始下一步：特征工程！")
