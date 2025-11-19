# 数据清理和预处理实战代码
import re
import pandas as pd
import jieba

# 模拟智能客服的原始数据（包含各种问题）
raw_data = [
    {"text": "怎么退款？", "intent": "退款咨询"},
    {"text": "", "intent": ""},  # 空文本
    {"text": "我的订单什么时候发货？？？", "intent": "物流查询"},
    {"text": "    有什么优惠活动吗   ", "intent": "优惠咨询"},  # 多余空格
    {"text": "产品质量有问题，要求退货！！！", "intent": "售后投诉"},
    {"text": "客服电话多少 13812345678", "intent": "联系方式"},  # 包含手机号
    {"text": "怎么退款？", "intent": "退款咨询"},  # 重复数据
    {"text": "访问 https://www.example.com 了解更多", "intent": "其他"},  # 包含网址
    {"text": "###@@!!!", "intent": "--++"},  # 纯符号
    {"text": "能不能换货呢？🤔", "intent": "换货咨询"},  # 包含emoji
]

df_raw = pd.DataFrame(raw_data)

print("🔍 === 原始数据概览 ===")
print(f"原始数据量: {len(df_raw)} 条")
print("样本数据:")
for idx, (i, row) in enumerate(df_raw.iterrows()):
    print(f"    {idx+1}. '{row['text']}' -> {row['intent']}")

print("\n🧹 === 开始数据清理 ===")

# 步骤1：删除空文本和空标签
print("步骤1：删除空数据")
df_clean = df_raw[
    (df_raw['text'].str.strip() != '') &
    (df_raw['intent'].str.strip() != '')
].copy()
print(f"删除空数据后：{len(df_clean)}条")

# 步骤2：删除重复数据
print("步骤2：删除重复数据")
before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['text', 'intent'])
print(f"删除重复后：{len(df_clean)} 条 （删除了{before_dedup - len(df_clean)}条重复数据）")

# 步骤3：删除纯符号文本
print("步骤3：删除纯符号文本")
def is_valid_text(text):
    # 去除标点符号和空格后，检查是否还有有效字符
    replaced_text = re.sub(r'[^\w\s]', '', text.strip())
    return len(replaced_text) >= 2

df_clean = df_clean[df_clean['text'].apply(is_valid_text)]
print(f"删除纯符号后： {len(df_clean)} 条")

# 步骤4：文本清理函数
def clean_text(text):
    """文本清理函数"""
    if pd.isna(text):
        return ""

    text = str(text)

    # 去除网址
    text = re.sub(r'https?://\S+', '', text)

    # 去除邮箱
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 去除手机号码（简单规则）
    text = re.sub(r'1[3-9]\d{9}', '[手机号]', text)

    # 去除多余的标点符号
    text = re.sub(r'([！？。，])\1+', r'\1', text)  # 多个标点变一个

    # 去除emoji（简单版本）
    text = re.sub(r'[🀀-🿿]', '', text)

    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("步骤4：应用文本清理")
df_clean['text'] = df_clean['text'].apply(clean_text)

# 再次检查长度
df_clean = df_clean[df_clean['text'].str.len() >= 2]
print(f"最终清理后：{len(df_clean)} 条")

print("\n📊 === 清理结果对比 ===")
print("清理前 vs 清理后:")
for i, (orig, clean) in enumerate(zip(df_raw['text'][:5], df_clean['text'][:5])):
    print(f"{i+1}. 原始: '{orig}'")
    print(f"   清理: '{clean}'")
    print()

# 步骤5：中文分词处理
print("📝 === 分词处理 ===")

# 加载停用词（这里用简单的停用词列表）
stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
              '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

def tokenize_text(text):
    """中文分词"""
    words = jieba.lcut(text)
    # 去除停用词和标点
    words = [w for w in words if w not in stop_words and len(w.strip()) > 1 and not re.match(r'\W', w)]
    return words

# 为文本添加分词结果
df_clean['tokens'] = df_clean['text'].apply(tokenize_text)

print("分词结果示例：")
for i, row in df_clean.iterrows():
    print(f"原文：{row['text']}")
    print(f"分词：{row['tokens']}")
    print()

print("✅ === 数据清理完成 ===")
print(f"📊 原始数据: {len(df_raw)} 条")
print(f"📊 清理后: {len(df_clean)} 条")
print(f"📊 清理率: {(len(df_raw) - len(df_clean))/len(df_raw)*100:.1f}%")

# 保存清理后的数据
print("\n💾 数据清理总结:")
print("✅ 删除了空文本和无效数据")
print("✅ 去除了重复数据")
print("✅ 清理了网址、手机号等敏感信息")
print("✅ 统一了标点符号格式")
print("✅ 完成了中文分词")
print("✅ 数据已准备好进入下一步！")




