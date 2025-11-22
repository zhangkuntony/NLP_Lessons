# 🛠️ 步骤1：环境搭建与工具准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from collections import Counter

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# 忽略警告信息
warnings.filterwarnings('ignore')

# 安装jieba分词库（如果需要）
try:
    import jieba
    print("✅ jieba库已安装")
except ImportError:
    print("📦 正在安装jieba库...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'jieba'])
    import jieba
    print("✅ jieba库安装完成")

print("🎉 环境配置完成！所有必要的库已导入。")


# 📂 步骤2：数据加载与编码处理
def load_data_with_encoding():
    """尝试不同编码方式读取数据"""
    encodings = ['utf-8', 'gbk', 'gb18030']

    for encoding in encodings:
        try:
            load_data_with_encoding_comments_df = pd.read_csv('douban-dataset/comments.csv', encoding=encoding)
            print(f"✅ 使用 {encoding} 编码读取成功！")
            print(f"数据形状：{load_data_with_encoding_comments_df.shape}")
            return load_data_with_encoding_comments_df
        except FileNotFoundError as e:
            print(f"❌ {encoding} 编码失败: {e}")
            continue

    print("❌ 所有编码方式都失败了，请检查数据文件")
    return None


# 加载数据
comments_df = load_data_with_encoding()

if comments_df is not None:
    print("\n📊 数据基本信息：")
    print(comments_df.info())
    print(f"\n🔍 前3行数据：")
    print(comments_df.head(3))
    print(f"\n📋 列名：{list(comments_df.columns)}")
else:
    print("数据加载失败，请检查文件路径和编码")


# 🔍 步骤3：数据质量诊断
if comments_df is not None:
    print("=== 📊 数据质量诊断报告 ===")

    # 1. 基本信息
    print(f"📋 数据基本信息：")
    print(f"• 数据行数：{len(comments_df):,}")
    print(f"• 数据列数：{len(comments_df.columns)}")
    print(f"• 数据大小：{comments_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # 2. 缺失值检查
    print(f"\n🕳️ 缺失值统计：")
    missing_stats = comments_df.isnull().sum()
    missing_percent = (missing_stats / len(comments_df) * 100).round(2)

    missing_summary = pd.DataFrame({
        '缺失数量': missing_stats,
        '缺失比例(%)': missing_percent
    })
    print(missing_summary[missing_summary['缺失数量'] > 0])

    # 3. 重复值检查
    duplicates = comments_df.duplicated().sum()
    print(f"\n🔄 重复数据：{duplicates:,} 行 ({duplicates / len(comments_df) * 100:.2f}%)")

    # 4. 数据类型检查
    print(f"\n🏷️ 数据类型：")
    for col in comments_df.columns:
        print(f"• {col}: {comments_df[col].dtype}")

    # 5. 文本列基本统计（如果有CONTENT列）
    if 'CONTENT' in comments_df.columns:
        content_stats = comments_df['CONTENT'].astype(str).str.len().describe()
        print(f"\n📝 评论文本长度统计：")
        print(f"• 平均长度：{content_stats['mean']:.1f} 字符")
        print(f"• 最短评论：{content_stats['min']:.0f} 字符")
        print(f"• 最长评论：{content_stats['max']:.0f} 字符")
        print(f"• 中位数长度：{content_stats['50%']:.0f} 字符")

    # 6. 数据完整性检查
    rows_with_missing = comments_df.isnull().any(axis=1).sum()
    total_rows = len(comments_df)
    print(f"\n📈 数据完整性：")
    print(f"• 有缺失值的行数：{rows_with_missing:,}")
    print(f"• 完全无缺失的行数：{total_rows - rows_with_missing:,}")
    print(f"• 数据完整度：{((total_rows - rows_with_missing) / total_rows * 100):.1f}%")
else:
    print("❌ 数据未加载，跳过质量诊断")


# 🧹 步骤4：缺失值处理
# 制作数据备份，避免修改原数据
comments_cleaned = comments_df.copy()

if comments_df is not None:
    print("=== 🧹 缺失值处理 ===")

    print("📋 处理前的缺失值状况：")
    missing_before = comments_cleaned.isnull().sum()
    print(missing_before[missing_before > 0])

    print("\n🔧 开始缺失值处理...")

    # 处理策略：
    # 1. 对于评论内容CONTENT，如果缺失则删除该行（核心数据不能为空）
    if 'CONTENT' in comments_cleaned.columns:
        before_count = len(comments_cleaned)
        comments_cleaned = comments_cleaned.dropna(subset=['CONTENT'])
        after_count = len(comments_cleaned)
        if before_count != after_count:
            print(f"✅ 删除了 {before_count - after_count} 条无评论内容的记录")

    # 2. 对于其他文本字段，用"未知"填补
    text_columns = comments_cleaned.select_dtypes(include=['object']).columns
    for col in text_columns:
        if comments_cleaned[col].isnull().sum() > 0:
            comments_cleaned[col] = comments_cleaned[col].fillna('未知')
            print(f"✅ {col}列的缺失值已用'未知'填补")

    # 3. 对于数值字段，用适当的统计值填补
    numeric_columns = comments_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if comments_cleaned[col].isnull().sum() > 0:
            if col == 'RATING':  # 评分用中位数填补
                fill_value = comments_cleaned[col].median()
                comments_cleaned[col] = comments_cleaned[col].fillna(fill_value)
                print(f"✅ {col}列的缺失值已用中位数 {fill_value} 填补")
            else:  # 其他数值用均值填补
                fill_value = comments_cleaned[col].mean()
                comments_cleaned[col] = comments_cleaned[col].fillna(fill_value)
                print(f"✅ {col}列的缺失值已用均值 {fill_value:.2f} 填补")

    print("\n📊 处理后的缺失值状况：")
    missing_after = comments_cleaned.isnull().sum()

    if missing_after.sum() == 0:
        print("🎉 所有缺失值已处理完毕！")
    else:
        print("剩余缺失值：")
        print(missing_after[missing_after > 0])

    print(f"\n📈 缺失值处理效果：")
    print(f"• 处理前数据量：{len(comments_df):,} 行")
    print(f"• 处理后数据量：{len(comments_cleaned):,} 行")
    print(f"• 数据保留率：{len(comments_cleaned) / len(comments_df) * 100:.1f}%")

else:
    print("❌ 数据未加载，跳过缺失值处理")


# 🔄 步骤5：重复数据处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 🔄 重复数据检测与处理 ===")

    print(f"📊 处理前数据量：{len(comments_cleaned):,} 条")

    # 1. 完全重复记录检测
    duplicated_all = comments_cleaned.duplicated()
    duplicate_count_all = duplicated_all.sum()
    print(f"\n🔍 完全重复的记录数：{duplicate_count_all:,} ({duplicate_count_all / len(comments_cleaned) * 100:.2f}%)")

    # 2. 内容重复检测（同一用户对同一电影的重复评论）
    if 'CREATOR' in comments_cleaned.columns and 'MOVIEID' in comments_cleaned.columns and 'CONTENT' in comments_cleaned.columns:
        content_duplicated = comments_cleaned.duplicated(subset=['CREATOR', 'MOVIEID', 'CONTENT'])
        content_duplicate_count = content_duplicated.sum()
        print(
            f"🔍 内容重复的记录数：{content_duplicate_count:,} ({content_duplicate_count / len(comments_cleaned) * 100:.2f}%)")

    # 3. 去重处理
    print(f"\n🔧 开始去重处理...")

    # 删除完全重复的记录
    comments_deduped = comments_cleaned.drop_duplicates()
    step1_removed = len(comments_cleaned) - len(comments_deduped)
    print(f"✅ 删除完全重复记录：{step1_removed} 条")

    # 删除内容重复的记录（保留最新的）
    if 'ADD_TIME' in comments_deduped.columns:
        # 按时间排序，保留最新的记录
        comments_deduped = comments_deduped.sort_values('ADD_TIME').drop_duplicates(
            subset=['CREATOR', 'MOVIEID', 'CONTENT'], keep='last'
        )
    else:
        # 如果没有时间字段，就保留第一条
        comments_deduped = comments_deduped.drop_duplicates(
            subset=['CREATOR', 'MOVIEID', 'CONTENT'], keep='first'
        )

    step2_removed = len(comments_cleaned) - step1_removed - len(comments_deduped)
    print(f"✅ 删除内容重复记录：{step2_removed} 条")

    # 4. 去重效果统计
    total_removed = len(comments_cleaned) - len(comments_deduped)
    print(f"\n📈 去重处理总结：")
    print(f"• 原始数据：{len(comments_cleaned):,} 条")
    print(f"• 去重后数据：{len(comments_deduped):,} 条")
    print(f"• 总共去除：{total_removed:,} 条重复记录")
    print(f"• 数据保留率：{len(comments_deduped) / len(comments_cleaned) * 100:.1f}%")

    # 更新清洗后的数据
    comments_cleaned = comments_deduped

else:
    print("❌ 清洗后的数据不存在，跳过重复数据处理")


# 🚨 步骤6：异常值检测与处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 🚨 异常值检测与处理 ===")

    # 1. 文本长度异常检测
    if 'CONTENT' in comments_cleaned.columns:
        print("📏 文本长度异常检测：")
        comments_cleaned['text_length'] = comments_cleaned['CONTENT'].astype(str).str.len()

        length_stats = comments_cleaned['text_length'].describe()
        print(f"• 平均评论长度：{length_stats['mean']:.1f} 字符")
        print(f"• 最短评论：{length_stats['min']:.0f} 字符")
        print(f"• 最长评论：{length_stats['max']:.0f} 字符")
        print(f"• 中位数长度：{length_stats['50%']:.0f} 字符")

        # 检测过短评论（可能是无效评论）
        very_short = comments_cleaned[comments_cleaned['text_length'] <= 3]
        print(f"• 过短评论（≤3字符）：{len(very_short)} 条 ({len(very_short) / len(comments_cleaned) * 100:.2f}%)")
        if len(very_short) > 0:
            print(f"  示例：{list(very_short['CONTENT'].head(3))}")

        # 检测过长评论（可能是异常）
        q99 = comments_cleaned['text_length'].quantile(0.99)
        very_long = comments_cleaned[comments_cleaned['text_length'] > q99]
        print(
            f"• 过长评论（>99%分位数 {q99:.0f}字符）：{len(very_long)} 条 ({len(very_long) / len(comments_cleaned) * 100:.2f}%)")

    # 2. 重复字符异常检测
    print(f"\n🔍 文本内容异常检测：")

    def has_excessive_repetition(text_to_check_repetition):
        """检测是否有过多重复字符"""
        if pd.isna(text_to_check_repetition):
            return False
        text_to_check_repetition = str(text_to_check_repetition)
        # 检查是否有连续4个以上相同字符
        pattern = r'(.)\\1{3,}'
        return bool(re.search(pattern, text_to_check_repetition))

    def calculate_special_char_ratio(text_to_calc_special_char_ratio):
        """计算特殊字符占比"""
        if pd.isna(text_to_calc_special_char_ratio):
            return 0
        text_to_calc_special_char_ratio = str(text_to_calc_special_char_ratio)
        if len(text_to_calc_special_char_ratio) == 0:
            return 0

        # 计算非中文、非英文、非数字字符的比例
        special_chars = 0
        for char in text_to_calc_special_char_ratio:
            if not (char.isalnum() or '\\u4e00' <= char <= '\\u9fff'):
                special_chars += 1

        return special_chars / len(text_to_calc_special_char_ratio)

    # 检测重复字符异常
    repetitive_mask = comments_cleaned['CONTENT'].apply(has_excessive_repetition)
    repetitive_count = repetitive_mask.sum()
    print(f"• 包含过多重复字符的评论：{repetitive_count} 条 ({repetitive_count / len(comments_cleaned) * 100:.2f}%)")

    # 检测特殊字符异常
    comments_cleaned['special_char_ratio'] = comments_cleaned['CONTENT'].apply(calculate_special_char_ratio)
    high_special_char = comments_cleaned[comments_cleaned['special_char_ratio'] > 0.3]
    print(
        f"• 特殊字符占比>30%的评论：{len(high_special_char)} 条 ({len(high_special_char) / len(comments_cleaned) * 100:.2f}%)")

    # 3. 异常值处理决策
    print(f"\n🔧 异常值处理：")

    # 删除过短的无效评论（≤2个字符）
    before_count = len(comments_cleaned)
    comments_cleaned = comments_cleaned[comments_cleaned['text_length'] > 2]
    removed_short = before_count - len(comments_cleaned)
    if removed_short > 0:
        print(f"✅ 删除过短评论：{removed_short} 条")

    # 可选：删除特殊字符占比过高的评论（可能是乱码）
    before_count = len(comments_cleaned)
    comments_cleaned = comments_cleaned[comments_cleaned['special_char_ratio'] <= 0.5]
    removed_special = before_count - len(comments_cleaned)
    if removed_special > 0:
        print(f"✅ 删除特殊字符过多的评论：{removed_special} 条")

    print(f"\n📈 异常值处理总结：")
    print(f"• 处理后数据量：{len(comments_cleaned):,} 条")
    print(f"• 总计删除异常记录：{removed_short + removed_special} 条")

else:
    print("❌ 清洗后的数据不存在，跳过异常值检测")


# 📝 步骤7：文本深度清洗
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 📝 NLP文本深度清洗 ===")

    def clean_text_comprehensive(text_to_clean):
        """
        综合文本清洗函数 - NLP预处理核心步骤
        """
        if pd.isna(text_to_clean) or text_to_clean is None:
            return ""

        text_to_clean = str(text_to_clean)

        # 1. 去除HTML标签和实体
        text_to_clean = re.sub(r'<[^>]+>', '', text_to_clean)
        text_to_clean = re.sub(r'&[a-zA-Z]+;', ' ', text_to_clean)
        text_to_clean = re.sub(r'&lt;|&gt;', '', text_to_clean)

        # 2. 去除特殊符号和噪声字符
        text_to_clean = re.sub(r'[★☆※@#$%^&*]', '', text_to_clean)
        text_to_clean = re.sub(r'[�□\ue000-\uf8ff]', '', text_to_clean)

        # 3. 处理表情符号（保留部分情感信息）
        text_to_clean = re.sub(r'[\U0001F600-\U0001F64F]', '[表情]', text_to_clean)
        text_to_clean = re.sub(r'[\U0001F300-\U0001F5FF]', '', text_to_clean)

        # 4. 标准化网络用语
        text_to_clean = re.sub(r'h{3,}', '哈哈', text_to_clean, flags=re.IGNORECASE)
        text_to_clean = re.sub(r'2333+', '哈哈', text_to_clean)
        text_to_clean = re.sub(r'6{4,}', '厉害', text_to_clean)

        # 5. 去除过度重复的字符
        text_to_clean = re.sub(r'(.)\\1{3,}', r'\\1\\1', text_to_clean)
        text_to_clean = re.sub(r'[！!]{3,}', '！！', text_to_clean)
        text_to_clean = re.sub(r'[？?]{3,}', '？？', text_to_clean)
        text_to_clean = re.sub(r'[。.]{3,}', '...', text_to_clean)

        # 6. 清理空白字符
        text_to_clean = re.sub(r'\\s+', ' ', text_to_clean)
        text_to_clean = text_to_clean.strip()

        return text_to_clean


    # 应用文本清洗
    print("🔧 开始文本清洗处理...")

    # 展示清洗效果示例
    if 'CONTENT' in comments_cleaned.columns:
        # 随机选择几个评论展示清洗效果
        sample_comments = comments_cleaned['CONTENT'].dropna().sample(n=min(3, len(comments_cleaned)), random_state=42)

        print("\\n🔍 文本清洗示例：")
        for i, (idx, original) in enumerate(sample_comments.items(), 1):
            cleaned = clean_text_comprehensive(original)
            print(f"{i}. 原文：{str(original)[:80]}...")
            print(f"   清洗后：{cleaned[:80]}...")
            print()

        # 批量清洗所有评论
        print("💾 批量清洗所有评论文本...")
        comments_cleaned['CONTENT_CLEANED'] = comments_cleaned['CONTENT'].apply(clean_text_comprehensive)

        # 统计清洗效果
        original_avg_len = comments_cleaned['CONTENT'].astype(str).str.len().mean()
        cleaned_avg_len = comments_cleaned['CONTENT_CLEANED'].str.len().mean()
        length_reduction = (1 - cleaned_avg_len / original_avg_len) * 100

        print(f"\\n📊 文本清洗效果统计：")
        print(f"• 清洗前平均长度：{original_avg_len:.1f} 字符")
        print(f"• 清洗后平均长度：{cleaned_avg_len:.1f} 字符")
        print(f"• 平均长度减少：{length_reduction:.1f}%")

        # 检查清洗后的空文本
        empty_after_clean = np.sum(comments_cleaned['CONTENT_CLEANED'].str.len() == 0)
        print(f"• 清洗后变为空的文本：{empty_after_clean} 条")

        # 过滤掉清洗后为空的文本
        if empty_after_clean > 0:
            before_filter = len(comments_cleaned)
            comments_cleaned = comments_cleaned[comments_cleaned['CONTENT_CLEANED'].str.len() > 0]
            print(f"✅ 删除清洗后为空的记录：{before_filter - len(comments_cleaned)} 条")

        print(f"✅ 文本清洗完成！当前数据量：{len(comments_cleaned):,} 条")

else:
    print("❌ 清洗后的数据不存在，跳过文本清洗")


# ✂️ 步骤8：中文分词处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== ✂️ 中文分词处理 ===")

    # 添加电影领域自定义词汇
    movie_terms = [
        "复仇者联盟", "钢铁侠", "美国队长", "黑寡妇", "雷神", "绿巨人",
        "漫威", "DC", "诺兰", "漫威电影宇宙", "特效", "剧情", "演技",
        "导演", "编剧", "配乐", "票房", "口碑", "豆瓣", "评分"
    ]

    for term in movie_terms:
        jieba.add_word(term)

    print(f"✅ 已添加 {len(movie_terms)} 个电影领域专业词汇")


    def tokenize_text(text_to_tokenize):
        """
        中文分词处理函数
        """
        if pd.isna(text_to_tokenize) or not text_to_tokenize.strip():
            return []

        # 使用jieba精确模式分词
        tokenized_words = jieba.lcut(str(text_to_tokenize).strip())

        # 过滤长度小于2的词和纯标点符号
        filtered_tokenized_words = []
        for tokenized_word in tokenized_words:
            tokenized_word = tokenized_word.strip()
            if (len(tokenized_word) >= 2 and
                    not re.match(r'^[^\w\u4e00-\u9fff]+$', tokenized_word)):  # 不是纯标点符号
                filtered_tokenized_words.append(tokenized_word)

        return filtered_tokenized_words


    # 展示分词效果
    if 'CONTENT_CLEANED' in comments_cleaned.columns:
        print("\\n🔍 分词效果演示：")

        # 选择几个样本展示分词效果
        sample_texts = comments_cleaned['CONTENT_CLEANED'].dropna().sample(n=min(3, len(comments_cleaned)),
                                                                           random_state=42)

        for i, (idx, text) in enumerate(sample_texts.items(), 1):
            if len(text) > 10:  # 只处理有内容的文本
                words = tokenize_text(text)
                print(f"{i}. 原文：{text[:60]}...")
                print(f"   分词：{' / '.join(words[:15])}...")
                print(f"   词数：{len(words)}")
                print()

        # 批量分词处理
        print("💾 批量分词处理中...")
        comments_cleaned['WORDS'] = comments_cleaned['CONTENT_CLEANED'].apply(tokenize_text)

        # 统计分词效果
        word_counts = comments_cleaned['WORDS'].apply(len)
        avg_words = word_counts.mean()

        print(f"\\n📊 分词处理效果统计：")
        print(f"• 平均每条评论词数：{avg_words:.1f}")
        print(f"• 最多词数：{word_counts.max()}")
        print(f"• 最少词数：{word_counts.min()}")
        print(f"• 中位数词数：{word_counts.median():.1f}")

        # 统计高频词汇（前处理）
        all_words = []
        for words_list in comments_cleaned['WORDS'].head(1000):  # 取前1000条进行词频统计
            all_words.extend(words_list)

        if all_words:
            word_freq = Counter(all_words)
            print(f"\\n🔝 高频词汇（前10）：")
            for word, freq in word_freq.most_common(10):
                print(f"• {word}: {freq} 次")

        print(f"✅ 分词处理完成！")

else:
    print("❌ 清洗后的数据不存在，跳过分词处理")


# 🚫 步骤9：停用词过滤
if 'comments_cleaned' in locals() and comments_cleaned is not None and 'WORDS' in comments_cleaned.columns:
    print("=== 🚫 停用词过滤处理 ===")

    # 构建停用词表
    def create_stopwords():
        """创建适合电影评论的停用词表"""
        basic_stopwords = {
            # 基础停用词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这样', '现在', '比如', '什么', '如果', '还是', '只是', '这个', '那个',
            '可以', '但是', '因为', '所以', '虽然', '然后', '而且', '或者',
            # 语气词
            '吧', '呢', '啊', '嗯', '哦', '呀', '嘛', '呐',
            # 代词
            '这', '那', '这些', '那些', '我们', '你们', '他们', '她们',
            # 介词
            '从', '向', '对', '对于', '关于', '由于', '通过', '根据',
            # 电影评论常见停用词
            '电影', '影片', '片子', '这部', '观看', '看了', '看过', '感觉', '觉得',
            '认为', '个人', '我觉得', '我认为', '我感觉', '一部', '一个', '真的'
        }

        # 为了情感分析，保留一些程度副词
        sentiment_words = {'很', '非常', '特别', '十分', '超级', '太', '最', '极其', '相当'}

        return basic_stopwords - sentiment_words


    stopwords = create_stopwords()
    print(f"📋 停用词表大小：{len(stopwords)} 个词")


    def remove_stopwords(words_list_to_remove):
        """去除停用词"""
        if not words_list_to_remove:
            return []

        return [word_in_list for word_in_list in words_list_to_remove if word_in_list not in stopwords]


    # 展示停用词过滤效果
    print("\\n🔍 停用词过滤演示：")

    sample_indices = comments_cleaned[comments_cleaned['WORDS'].apply(len) > 5].sample(n=min(3, len(comments_cleaned)),
                                                                                       random_state=42).index

    for i, idx in enumerate(sample_indices, 1):
        original_words = comments_cleaned.loc[idx, 'WORDS']
        filtered_words = remove_stopwords(original_words)

        print(f"{i}. 原始分词：{' / '.join(original_words)}")
        print(f"   过滤停用词：{' / '.join(filtered_words)}")
        print(
            f"   词数变化：{len(original_words)} → {len(filtered_words)} (减少{len(original_words) - len(filtered_words)}词)")
        print()

    # 批量处理停用词过滤
    print("💾 批量停用词过滤中...")
    comments_cleaned['WORDS_FILTERED'] = comments_cleaned['WORDS'].apply(remove_stopwords)

    # 统计停用词过滤效果
    original_word_counts = comments_cleaned['WORDS'].apply(len)
    filtered_word_counts = comments_cleaned['WORDS_FILTERED'].apply(len)

    avg_reduction = (original_word_counts.mean() - filtered_word_counts.mean()) / original_word_counts.mean() * 100

    print(f"\\n📊 停用词过滤效果统计：")
    print(f"• 过滤前平均词数：{original_word_counts.mean():.1f}")
    print(f"• 过滤后平均词数：{filtered_word_counts.mean():.1f}")
    print(f"• 平均词数减少：{avg_reduction:.1f}%")

    # 统计过滤后的高频词汇
    all_filtered_words = []
    for words_list in comments_cleaned['WORDS_FILTERED'].head(1000):
        all_filtered_words.extend(words_list)

    if all_filtered_words:
        filtered_word_freq = Counter(all_filtered_words)
        print(f"\\n🔝 停用词过滤后高频词汇（前10）：")
        for word, freq in filtered_word_freq.most_common(10):
            print(f"• {word}: {freq} 次")

    # 过滤掉停用词处理后为空的记录
    empty_after_filter = np.sum(comments_cleaned['WORDS_FILTERED'].apply(len) == 0)
    if empty_after_filter > 0:
        before_filter = len(comments_cleaned)
        comments_cleaned = comments_cleaned[comments_cleaned['WORDS_FILTERED'].apply(len) > 0]
        print(f"\\n✅ 删除停用词过滤后为空的记录：{before_filter - len(comments_cleaned)} 条")

    print(f"✅ 停用词过滤完成！当前数据量：{len(comments_cleaned):,} 条")

else:
    print("❌ 分词数据不存在，跳过停用词处理")


# ✅ 步骤10：最终数据质量验证
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== ✅ 最终数据质量验证 ===")

    retention_rate = 0
    # 1. 数据量变化统计
    if 'comments_df' in locals() and comments_df is not None:
        original_count = len(comments_df)
        final_count = len(comments_cleaned)
        retention_rate = final_count / original_count * 100

        print(f"📊 数据处理总结：")
        print(f"• 原始数据量：{original_count:,} 条")
        print(f"• 最终数据量：{final_count:,} 条")
        print(f"• 数据保留率：{retention_rate:.1f}%")
        print(f"• 总计清理：{original_count - final_count:,} 条")

    # 2. 数据完整性验证
    print(f"\\n🔍 数据完整性检查：")
    missing_check = comments_cleaned.isnull().sum()
    if missing_check.sum() == 0:
        print("✅ 没有缺失值")
    else:
        print("⚠️ 仍有缺失值：")
        print(missing_check[missing_check > 0])

    print(comments_cleaned.head(5))

    # 3. 重复数据验证
    # 排除WORDS和WORDS_FILTERED列来检测重复数据
    cols_to_exclude = ['WORDS', 'WORDS_FILTERED']
    cols_to_check = [col for col in comments_cleaned.columns if col not in cols_to_exclude]
    remaining_duplicates = comments_cleaned[cols_to_check].duplicated().sum()
    print(f"🔄 重复数据检查：{remaining_duplicates} 条 (应为0)")

    # 4. 文本质量验证
    if 'CONTENT_CLEANED' in comments_cleaned.columns:
        print(f"\\n📝 文本质量检查：")

        # 检查文本长度分布
        text_lengths = comments_cleaned['CONTENT_CLEANED'].str.len()
        print(f"• 平均文本长度：{text_lengths.mean():.1f} 字符")
        print(f"• 最短文本：{text_lengths.min()} 字符")
        print(f"• 最长文本：{text_lengths.max()} 字符")

        # 检查是否有空文本
        empty_texts = np.sum(text_lengths == 0)
        print(f"• 空文本数量：{empty_texts} 条 (应为0)")

    # 5. 分词质量验证
    if 'WORDS_FILTERED' in comments_cleaned.columns:
        print(f"\\n✂️ 分词质量检查：")

        word_counts = comments_cleaned['WORDS_FILTERED'].apply(len)
        print(f"• 平均词数：{word_counts.mean():.1f}")
        print(f"• 最少词数：{word_counts.min()}")
        print(f"• 最多词数：{word_counts.max()}")

        # 检查是否有空词列表
        empty_words = np.sum(word_counts == 0)
        print(f"• 空词列表数量：{empty_words} 条 (应为0)")

    # 6. 数据采样检查
    print(f"\\n🎯 最终数据采样检查：")

    if len(comments_cleaned) > 0:
        sample_data = comments_cleaned.sample(n=min(3, len(comments_cleaned)), random_state=42)

        for i, (idx, row) in enumerate(sample_data.iterrows(), 1):
            print(f"\\n样本 {i}:")
            print(f"  原始评论：{str(row.get('CONTENT', 'N/A'))[:50]}...")
            if 'CONTENT_CLEANED' in row:
                print(f"  清洗后：{str(row['CONTENT_CLEANED'])[:50]}...")
            if 'WORDS_FILTERED' in row and row['WORDS_FILTERED']:
                print(f"  关键词：{' / '.join(row['WORDS_FILTERED'][:8])}...")

    # 7. 数据质量评分
    print(f"\\n🏆 数据质量综合评分：")

    score = 100
    issues = []

    # 检查各项指标
    if 'comments_df' in locals():
        if retention_rate < 80:
            score -= 20
            issues.append("数据保留率偏低")
        elif retention_rate < 90:
            score -= 10
            issues.append("数据保留率一般")

    if missing_check.sum() > 0:
        score -= 15
        issues.append("仍有缺失值")

    if remaining_duplicates > 0:
        score -= 10
        issues.append("仍有重复数据")

    if 'CONTENT_CLEANED' in comments_cleaned.columns:
        if np.sum(comments_cleaned['CONTENT_CLEANED'].str.len() == 0) > 0:
            score -= 15
            issues.append("存在空文本")

    if score >= 90:
        grade = "A+ 优秀"
        emoji = "🥇"
    elif score >= 80:
        grade = "A 良好"
        emoji = "🥈"
    elif score >= 70:
        grade = "B 合格"
        emoji = "🥉"
    else:
        grade = "C 需改进"
        emoji = "⚠️"

    print(f"{emoji} 综合质量评分：{score}/100 ({grade})")

    if issues:
        print(f"🔧 需要关注的问题：")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("🎉 数据质量优秀，无明显问题！")

    # 8. 保存清洗后的数据
    print(f"\\n💾 数据处理完成总结：")
    print(f"• 最终清洗后数据已保存在 comments_cleaned 变量中")
    print(f"• 主要字段：")
    for col in comments_cleaned.columns:
        print(f"  - {col}: {comments_cleaned[col].dtype}")

    print(f"\\n🎊 恭喜！数据预处理流程全部完成！")
    print(f"📈 处理后的数据可以用于后续的机器学习、情感分析、主题建模等任务")

else:
    print("❌ 没有可验证的数据")
