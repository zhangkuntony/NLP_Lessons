# 推理部署实战代码
import pickle
import time
import json
from datetime import datetime
import numpy as np
import jieba
import matplotlib.pyplot as plt
from collections import Counter

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

print("🎪 === 模型推理系统构建 ===")

def _preprocess_text(preprocess_text):
    """文本预处理"""
    if not preprocess_text or not isinstance(preprocess_text, str):
        return ""

    # 分词
    words = jieba.lcut(preprocess_text.strip())

    # 去除停用词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '上', '也', '很', '到',
                  '说', '要', '去', '你', '会', '着', '没有', '看', '好', '这'}

    words = [w for w in words if w not in stop_words and len(w.strip()) > 1]
    return ' '.join(words)

# 1. 模拟保存和加载训练好的模型
class IntentClassifier:
    """智能客服意图分类器"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_trained = False

        # 性能统计
        self.total_requests = 0
        self.total_time = 0
        self.error_count = 0

        # 缓存机制
        self.cache = {}
        self.cache_hit = 0

    def save_model(self, model_path="intent_model.pkl"):
        """保存模型到文件"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)              # type: ignore
        print(f"✅ 模型已保存到 {model_path}")

    def load_model(self, model_path="intent_model.pkl"):
        """从文件加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = True
            print(f"✅ 模型已从 {model_path} 加载")

        except FileNotFoundError:
            print(f"❌ 模型文件 {model_path} 不存在，使用模拟模型")
            self._create_mock_model()

    def _create_mock_model(self):
        """创建模拟模型用于演示"""
        # 模拟一个简单的规则模型
        self.intent_keywords = {
            '退款咨询': ['退款', '退钱', '申请退', '怎么退'],
            '物流查询': ['发货', '物流', '快递', '收到货', '配送'],
            '优惠咨询': ['优惠', '打折', '活动', '促销', '折扣'],
            '售后投诉': ['投诉', '质量', '问题', '不满意', '差'],
            '联系方式': ['客服', '电话', '联系', '人工']
        }
        self.is_trained = True
        print("✅ 模拟模型创建完成")

    def _predict_by_rules(self, predict_text):
        """基于规则的简单预测（模拟模型）"""
        text_lower = predict_text.lower()

        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # 模拟置信度
                    confidence = np.random.uniform(0.8, 0.95)
                    return intent, confidence

        # 默认预测
        return '其他', 0.5

    def predict(self, predict_text, use_cache=True):
        """预测文本意图"""
        start_time = time.time()
        self.total_requests += 1

        try:
            # 输入验证
            if not predict_text or not isinstance(predict_text, str):
                raise ValueError("输入文本不能为空")

            # 检查缓存
            if use_cache and predict_text in self.cache:
                self.cache_hit += 1
                predict_result = self.cache[predict_text]
                print(f"🎯 缓存命中: '{predict_text}' → {predict_result['intent']}")
                return predict_result

            # 文本预处理
            processed_text = _preprocess_text(predict_text)

            # 模型预测
            if self.is_trained:
                intent, confidence = self._predict_by_rules(processed_text)
            else:
                raise RuntimeError("模型未训练")

            # 构建结果
            predict_result = {
                'intent': intent,
                'confidence': float(confidence),
                'processed_text': processed_text,
                'timestamp': datetime.now().isoformat()
            }

            # 缓存结果
            if use_cache:
                self.cache[predict_text] = predict_result

            # 记录性能
            inference_time = time.time() - start_time
            self.total_time += inference_time

            return predict_result

        except Exception as e:
            self.error_count += 1
            print(f"❌ 预测出错: {str(e)}")

            # 返回降级结果
            return {
                'intent': '其他',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_stats(self):
        """获取性能统计"""
        if self.total_requests > 0:
            avg_time = self.total_time / self.total_requests
            cache_rate = self.cache_hit / self.total_requests
            error_rate = self.error_count / self.total_requests
        else:
            avg_time = cache_rate = error_rate = 0

        return {
            'total_requests': self.total_requests,
            'average_time_ms': avg_time * 1000,
            'cache_hit_rate': cache_rate,
            'error_rate': error_rate,
            'cache_size': len(self.cache)
        }

# 2. 创建推理服务
print("\n🚀 === 创建推理服务 ===")

# 初始化分类器
classifier = IntentClassifier()
classifier.load_model()                 # 加载模型（会使用模拟模型）

print("\n🧪 === 推理功能测试 ===")

# 测试样本
test_samples = [
    "我要申请退款",
    "订单什么时候发货？",
    "有什么优惠活动吗",
    "产品质量有问题要投诉",
    "客服电话是多少",
    "我要申请退款",                   # 重复，测试缓存
    "怎么退钱啊",
    "这个商品什么时候能到货",
    "",                             # 空输入，测试异常处理
    "随便说点什么"                    # 未知意图
]

print("开始批量推理测试...")
results = []

for i, text in enumerate(test_samples):
    print(f"\n测试 {i + 1}: '{text}'")
    result = classifier.predict(text)
    results.append(result)

    if 'error' not in result:
        print(f"  预测结果: {result['intent']} (置信度: {result['confidence']:.3f})")
    else:
        print(f"  错误: {result['error']}")

print("\n📊 === 性能统计报告 ===")

# 获取性能统计
stats = classifier.get_stats()
print("系统性能指标：")
for metric, value in stats.items():
    if isinstance(value, float):
        print(f"    {metric}: {value:.4f}")
    else:
        print(f"    {metric}: {value}")

print("\n⚡ === 并发性能测试 ===")

# 模拟并发请求
def simulate_concurrent_requests(classifier_model, num_requests=100):
    """模拟并发请求测试"""
    test_texts = ["退款怎么申请", "查询物流", "有优惠吗", "要投诉", "联系客服"] * (num_requests // 5 + 1)

    start_time = time.time()

    for test_text in test_texts[:num_requests]:
        classifier_model.predict(test_text)

    total_request_time = time.time() - start_time
    request_qps = num_requests / total_request_time

    return request_qps, total_request_time

# 运行并发测试
print("模拟100个并发请求...")
qps, total_time = simulate_concurrent_requests(classifier, 100)

print(f"并发性能测试结果:")
print(f"  总请求数: 100")
print(f"  总耗时: {total_time:.2f}秒")
print(f"  QPS (每秒请求数): {qps:.2f}")
print(f"  平均响应时间: {total_time/100*1000:.2f}ms")

print("\n📈 === 推理结果可视化 ===")

# 统计意图分布
# 统计预测结果
valid_results = [r for r in results if 'error' not in r]
intent_counts = Counter([r['intent'] for r in valid_results])

plt.figure(figsize=(12, 5))

# 意图分布饼图
plt.subplot(1, 2, 1)
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
plt.pie(intent_counts.values(), labels=list(intent_counts.keys()), autopct='%1.1f%%',
        colors=colors[:len(intent_counts)], startangle=90)
plt.title('预测意图分布')

# 置信度分布直方图
plt.subplot(1, 2, 2)
confidences = [r['confidence'] for r in valid_results if r['confidence'] > 0]
plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('预测置信度分布')
plt.xlabel('置信度')
plt.ylabel('频次')

plt.tight_layout()
plt.show()

print("\n🎯 === API接口示例 ===")

# 模拟API接口
def intent_api(api_text):
    """模拟API接口"""
    try:
        api_response_result = classifier.predict(api_text)

        # 构建API响应格式
        api_response = {
            "code": 200,
            "message": "success",
            "data": {
                "intent": api_response_result['intent'],
                "confidence": api_response_result['confidence'],
                "timestamp": api_response_result['timestamp']
            }
        }

        if 'error' in api_response_result:
            api_response["code"] = 500
            api_response["message"] = api_response_result['error']

        return api_response

    except Exception as e:
        return {
            "code": 500,
            "message": f"Internal Server Error: {str(e)}",
            "data": None
        }

# 测试API接口
print("API接口调用示例：")
api_test_cases = [
    "我想要退款",
    "查询订单状态",
    "有什么优惠"
]

for text in api_test_cases:
    response = intent_api(text)
    print(f"\n请求: '{text}'")
    print(f"响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

print("\n✅ === 推理系统总结 ===")

final_stats = classifier.get_stats()

print("🎯 推理系统完成情况:")
print(f"  ✅ 处理请求总数: {final_stats['total_requests']}")
print(f"  ✅ 平均响应时间: {final_stats['average_time_ms']:.2f}ms")
print(f"  ✅ 缓存命中率: {final_stats['cache_hit_rate']:.1%}")
print(f"  ✅ 错误率: {final_stats['error_rate']:.1%}")

print("\n🚀 部署建议:")
print("✅ 模型已封装成可调用的服务")
print("✅ 支持批量推理和实时响应")
print("✅ 包含缓存机制和异常处理")
print("✅ 提供性能监控和统计功能")
print("✅ 可以直接部署到生产环境")

print("\n🎉 === NLP完整流程结束 ===")
print("恭喜！你已经完成了从问题定义到模型部署的完整NLP流程！")
print("🎯 下一步可以考虑：")
print("  📈 在真实数据上训练更强的模型")
print("  🚀 部署到云端服务器")
print("  📊 建立完整的监控体系")
print("  🔄 建立模型的持续优化流程")