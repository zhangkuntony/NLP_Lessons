# 模型评估实战代码
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 假设我们已经有了训练好的最佳模型（从前面的建模步骤）
# 这里我们重新创建一个简化的示例

# 模拟测试数据和预测结果
np.random.seed(42)

# 创建测试数据
test_texts = [
    "怎么申请退款", "退款流程是什么", "我要退钱",
    "订单什么时候发货", "查询物流状态", "快递到哪了",
    "有什么优惠活动", "优惠券怎么使用", "打折信息",
    "产品质量有问题", "要投诉客服", "服务态度差",
    "客服电话多少", "联系人工客服", "怎么找客服"
]

true_labels = [
    "退款咨询", "退款咨询", "退款咨询",
    "物流查询", "物流查询", "物流查询",
    "优惠咨询", "优惠咨询", "优惠咨询",
    "售后投诉", "售后投诉", "售后投诉",
    "联系方式", "联系方式", "联系方式"
]

# 模拟模型预测结果（添加一些错误）
predicted_labels = [
    "退款咨询", "退款咨询", "优惠咨询",  # 第3个预测错误
    "物流查询", "物流查询", "物流查询",
    "优惠咨询", "优惠咨询", "优惠咨询",
    "售后投诉", "联系方式", "售后投诉",  # 第11个预测错误
    "联系方式", "联系方式", "联系方式"
]

# 模拟预测置信度
confidence_scores = np.random.uniform(0.7, 0.99, len(test_texts))
# 错误预测的置信度稍低
confidence_scores[2] = 0.65   # 退款咨询 -> 优惠咨询
confidence_scores[10] = 0.72  # 售后投诉 -> 联系方式

print("📈 === 模型评估开始 ===")
print(f"测试样本数: {len(test_texts)}")
print("测试样本示例:")
for i in range(5):
    print(f"  {i+1}. '{test_texts[i]}' 真实:{true_labels[i]} 预测:{predicted_labels[i]} 置信度:{confidence_scores[i]:.3f}")

print("\n📊 === 第1层：基础技术指标 ===")

# 计算基础指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision_macro = precision_score(true_labels, predicted_labels, average='macro')
recall_macro = recall_score(true_labels, predicted_labels, average='macro')
f1_macro = f1_score(true_labels, predicted_labels, average='macro')

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision_macro:.4f}")
print(f"召回率 (Recall): {recall_macro:.4f}")
print(f"F1分数: {f1_macro:.4f}")

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(true_labels, predicted_labels, digits=4))

print("\n📊 === 第2层：分类别详细分析 ===")

# 获取所有类别
unique_labels = sorted(set(true_labels + predicted_labels))
print("各类别详细指标：")

# 计算每个类别的指标
class_metrics = []
for label in unique_labels:
    # 将多分类转换为二分类
    y_true_binary = [1 if x == label else 0 for x in true_labels]
    y_pred_binary = [1 if x == label else 0 for x in predicted_labels]

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    class_metrics.append({
        '类别': label,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        '支持样本数': true_labels.count(label)
    })

class_df = pd.DataFrame(class_metrics)
print(class_df.round(4))

print("\n📊 === 第3层：混淆矩阵分析 ===")

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

plt.figure(figsize=(12, 8))

# 混淆矩阵热力图
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 标准化混淆矩阵（按行标准化）
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.subplot(2, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('标准化混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

# 各类别F1分数对比
plt.subplot(2, 2, 3)
f1_scores = [metric['F1分数'] for metric in class_metrics]
colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(f1_scores)))
bars = plt.bar(unique_labels, f1_scores, color=colors, alpha=0.8)
plt.title('各类别F1分数')
plt.ylabel('F1分数')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# 添加数值标签
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# 置信度分布分析
plt.subplot(2, 2, 4)
correct_predictions = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true == pred]
incorrect_predictions = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]

correct_confidence = [confidence_scores[i] for i in correct_predictions]
incorrect_confidence = [confidence_scores[i] for i in incorrect_predictions]

plt.hist(correct_confidence, bins=10, alpha=0.7, label='正确预测', color='green')
plt.hist(incorrect_confidence, bins=10, alpha=0.7, label='错误预测', color='red')
plt.title('预测置信度分布')
plt.xlabel('置信度')
plt.ylabel('频次')
plt.legend()

plt.tight_layout()
plt.show()

print("\n🔍 === 第4层：错误分析 ===")

# 错误样本分析
errors = []
error_patterns = []
for i, (text, true, pred, conf) in enumerate(zip(test_texts, true_labels, predicted_labels, confidence_scores)):
    if true != pred:
        errors.append({
            '样本': text,
            '真实标签': true,
            '预测标签': pred,
            '置信度': conf
        })

if errors:
    error_df = pd.DataFrame(errors)
    print("错误预测样本分析：")
    print(error_df)

    print("\n错误类型统计：")
    error_patterns = Counter([f"{err['真实标签']} -> {err['预测标签']}" for err in errors])
    for pattern, count in error_patterns.items():
        print(f"    {pattern}: {count} 次")
else:
    print("🎉 所有预测都正确！")

print("\n⏱️ === 第5层：性能指标 ===")

# 模拟推理时间测试
def simulate_inference_time(texts, model_complexity='medium'):
    """模拟不同复杂度模型的推理时间"""
    base_time = {
        'simple': 0.001,            # 朴素贝叶斯等简单模型
        'medium': 0.005,            # 逻辑回归等中等复杂度
        'complex': 0.050            # 深度学习模型
    }

    times = []
    for text_to_simulate_time in texts:
        # 基础时间 + 文本长度相关时间 + 随机波动
        time_per_char = base_time[model_complexity]
        text_time = len(text_to_simulate_time) * time_per_char * 0.1
        noise = np.random.normal(0, time_per_char * 0.1)
        total_time = base_time[model_complexity] + text_time + abs(noise)
        times.append(total_time)

    return times

# 测试不同模型的推理性能
model_types = ['simple', 'medium', 'complex']
model_names = ['朴素贝叶斯', '逻辑回归', '深度学习']

performance_results = {}
for model_type, model_name in zip(model_types, model_names):
    inference_times = simulate_inference_time(test_texts, model_type)
    
    # 计算性能指标 - 明确转换为float类型
    avg_time = float(np.mean(inference_times))
    max_time = float(np.max(inference_times))

    performance_results[model_name] = {
        '平均推理时间(ms)': avg_time * 1000.0,
        '最大推理时间(ms)': max_time * 1000.0,
        '吞吐量（条/秒）': 1.0 / avg_time,
    }

performance_df = pd.DataFrame(performance_results).T
print("模型性能对比:")
print(performance_df.round(2))

print("\n💼 === 第6层：业务指标模拟 ===")

# 模拟业务指标
def calculate_business_metrics(business_accuracy, avg_business_confidence):
    """根据技术指标估算业务指标"""
    # 用户满意度与准确率和置信度相关
    user_satisfaction_rate = min(0.95, business_accuracy * 0.8 + avg_business_confidence * 0.2)

    # 人工干预率与准确率负相关
    human_intervention_rate = max(0.05, 1 - business_accuracy)

    # 问题解决率略高于准确率
    problem_solving_rate = min(0.98, business_accuracy + 0.1)

    return user_satisfaction_rate, human_intervention_rate, problem_solving_rate

avg_confidence = np.mean(confidence_scores)
user_satisfaction, human_intervention, problem_solving = calculate_business_metrics(accuracy, avg_confidence)

business_metrics = {
    '技术指标': {
        '准确率': f"{accuracy:.1%}",
        '平均置信度': f"{avg_confidence:.1%}"
    },
    '业务指标': {
        '用户满意度': f"{user_satisfaction:.1%}",
        '人工干预率': f"{human_intervention:.1%}",
        '问题解决率': f"{problem_solving:.1%}"
    }
}

print("业务指标评估:")
for category, metrics in business_metrics.items():
    print(f"\n{category}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

print("\n✅ === 评估总结报告 ===")

# 生成评估总结
print("🎯 模型评估总结:")
print(f"  ✅ 总体准确率: {accuracy:.1%} ({'达标' if accuracy >= 0.85 else '未达标'})")
print(f"  ✅ 宏平均F1: {f1_macro:.3f} ({'达标' if f1_macro >= 0.8 else '未达标'})")
print(f"  ✅ 平均推理时间: {performance_results['逻辑回归']['平均推理时间(ms)']:.1f}ms ({'达标' if performance_results['逻辑回归']['平均推理时间(ms)'] < 100 else '未达标'})")

print("\n🔍 主要发现:")
if errors:
    print(f"  ⚠️ 发现 {len(errors)} 个错误预测")
    print(f"  ⚠️ 最容易混淆的类别: {list(error_patterns.keys())[0] if error_patterns else '无'}")
else:
    print("  🎉 所有测试样本预测正确")

print(f"  📊 最优类别: {class_df.loc[class_df['F1分数'].idxmax(), '类别']}")
print(f"  📊 待改进类别: {class_df.loc[class_df['F1分数'].idxmin(), '类别']}")

print("\n💡 改进建议:")
if accuracy < 0.9:
    print("  🔧 考虑增加训练数据或调整模型参数")
if len(errors) > 0:
    print("  🔧 重点关注错误样本，进行针对性优化")
print("  🔧 定期监控线上效果，持续优化模型")
print("  ✅ 模型评估完成，可以进入部署阶段！")
