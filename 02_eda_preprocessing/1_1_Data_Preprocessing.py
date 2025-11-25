# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

print("✅ 环境配置完成！")

# 定义只能读取函数
def smart_read_csv(file_path, data_size=1000):
    """
    智能读取CSV文件，自动尝试不同编码

    参数:
        file_path: 文件路径
        sample_size: 测试样本大小

    返回:
        df_full: 读取的完整数据框
        encoding: 成功的编码格式
    """
    # 常见的中文编码列表
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']

    for encoding in encodings:
        try:
            print(f"🔍 尝试使用 {encoding} 编码读取文件...")

            # 先读取样本测试编码是否正确
            pd.read_csv(file_path, encoding=encoding, nrows=data_size)
            print(f"✅ 成功！使用 {encoding} 编码读取文件")

            # 测试成功后读取完整文件
            df_full = pd.read_csv(file_path, encoding=encoding)
            return df_full, encoding

        except Exception as ex:
            print(f"❌ {encoding} 编码失败: {str(ex)[:50]}...")
            continue

    print("😱 所有编码都失败了！")
    return None, None

print("🛠️ 智能读取函数定义完成！")

# 📖 读取电影数据
print("📖 正在读取电影信息数据...")

# 使用只能读取函数加载电影数据
movies_df, movies_encoding = smart_read_csv('douban-dataset/movies.csv')

# 检查读取结果
if movies_df is not None:
    print(f"🎬 电影数据读取成功！共有 {len(movies_df)} 部电影")
    print(f"📊 数据形状: {movies_df.shape}")
    print(f"🔤 使用编码: {movies_encoding}")
else:
    print("💔 电影数据读取失败！")

# 💬 读取评论数据（分批处理大文件）
print("💬 正在读取评论数据...")
print("⚠️  由于评论文件较大(68MB)，我们先读取前10000条进行探索")

# 使用只能读取函数处理评论数据
try:
    # 方法1：使用只能读取函数（推荐）
    comments_df, comment_encoding = smart_read_csv('douban-dataset/comments.csv', data_size=1000)

    # 如果文件太大，只取前10000条
    if comments_df is not None and len(comments_df) > 10000:
        comments_df = comments_df.head(10000)
        print(f"📝 为了演示方便，只保留前10000条评论")

    if comments_df is not None:
        print(f"✅ 评论数据读取成功！读取了 {len(comments_df)} 条评论")
        print(f"📊 数据形状: {comments_df.shape}")
        print(f"🔤 使用编码: {comment_encoding}")

except Exception as e:
    print(f"💔 评论数据读取失败: {e}")
    comments_df = None


# 数据质量全面诊断
if comments_df is not None:
    print("=== 📊 数据质量诊断报告 ===")

    # 1. 基本信息
    print(f"📋 数据基本信息：")
    print(f"• 数据行数：{len(comments_df):,}")
    print(f"• 数据列数：{len(comments_df.columns)}")
    print(f"• 数据大小：{comments_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. 缺失值检查
    print(f"\n🕳️ 缺失值统计：")
    missing_stats = comments_df.isnull().sum()
    missing_percent = (missing_stats / len(comments_df) * 100).round(2)

    for col in comments_df.columns:
        if missing_stats[col] > 0:
            print(f"• {col}: {missing_stats[col]:,} 个缺失值 ({missing_percent[col]}%)")
        else:
            print(f"• {col}: 无缺失值 ✅")

    # 3. 重复值检查
    duplicates = comments_df.duplicated().sum()
    print(f"\n🔄 重复数据：{duplicates:,} 行 ({duplicates / len(comments_df) * 100:.2f}%)")

    # 4. 数据类型检查
    print(f"\n🏷️ 数据类型：")
    for col in comments_df.columns:
        print(f"• {col}: {comments_df[col].dtype}")

    # 5. 文本列质量检查（CONTENT列）
    if 'CONTENT' in comments_df.columns:
        print(f"\n📝 评论文本质量分析：")
        content_lengths = comments_df['CONTENT'].astype(str).str.len()

        print(f"• 平均长度：{content_lengths.mean():.1f} 字符")
        print(f"• 最短评论：{content_lengths.min()} 字符")
        print(f"• 最长评论：{content_lengths.max()} 字符")
        print(f"• 中位数长度：{content_lengths.median():.0f} 字符")

        # 检查异常短的评论
        very_short = (content_lengths <= 5).sum()
        print(f"• 过短评论(≤5字符)：{very_short} 条 ({very_short / len(comments_df) * 100:.2f}%)")

        # 检查异常长的评论
        very_long = (content_lengths >= 500).sum()
        print(f"• 过长评论(≥500字符)：{very_long} 条 ({very_long/len(comments_df)*100:.2f}%)")

    print(f"\n✅ 数据质量诊断完成！发现 {missing_stats.sum()} 个缺失值，{duplicates} 个重复值")
else:
    print("❌ 数据未成功加载，跳过质量诊断")


# 系统化缺失值处理
comments_cleaned = comments_df.copy()
if comments_df is not None:
    print("=== 🎯 缺失值处理方案 ===")

    # 创建数据副本
    original_shape = comments_cleaned.shape
    print(f"原始数据形状：{original_shape}")

    # 计算缺失值比例
    missing_ratios = comments_cleaned.isnull().sum() / len(comments_cleaned)

    print("\n📊 各列缺失值比例：")
    for col in comments_cleaned.columns:
        ratio = missing_ratios[col] * 100
        status = "🔴高风险" if ratio > 20 else "🟡中等" if ratio > 5 else "🟢低风险"
        print(f"• {col}: {ratio:.2f}% {status}")

    # 处理策略执行
    print("\n🔧 执行处理策略：")

    # 1. 核心字段：评论内容不能为空，直接删除
    if 'CONTENT' in comments_cleaned.columns:
        before_count = len(comments_cleaned)
        comments_cleaned = comments_cleaned.dropna(subset=['CONTENT'])
        removed_count = before_count - len(comments_cleaned)
        if removed_count > 0:
            print(f"✅ 删除无评论内容的记录：{removed_count} 条")

    # 2. 用户ID字段：用“未知用户”填补
    if 'CREATOR' in comments_cleaned.columns:
        creator_missing = comments_cleaned['CREATOR'].isnull().sum()
        if creator_missing > 0:
            comments_cleaned['CREATOR'] = comments_cleaned['CREATOR'].fillna('未知用户')
            print(f"✅ 用户名缺失值填补：{creator_missing} 条 → '未知用户'")

    # 3. 评分字段：用中位数填补（避免异常值影响）
    if 'RATING' in comments_cleaned.columns:
        rating_missing = comments_cleaned['RATING'].isnull().sum()
        if rating_missing > 0:
            # 将评分转换为数值类型
            comments_cleaned['RATING'] = pd.to_numeric(comments_cleaned['RATING'], errors='coerce')
            median_rating = comments_cleaned['RATING'].median()
            comments_cleaned['RATING'] = comments_cleaned['RATING'].fillna(median_rating)
            print(f"✅ 评分缺失值填补：{rating_missing} 条 → {median_rating}")

    # 4. 其他字段：用“未知”填补
    text_columns = ['ID', 'TIME', 'MOVIEID', 'ADD_TIME']
    for col in text_columns:
        if col in comments_cleaned.columns:
            missing_count = comments_cleaned[col].isnull().sum()
            if missing_count > 0:
                comments_cleaned[col] = comments_cleaned[col].fillna('未知')
                print(f"✅ {col}缺失值填补：{missing_count} 条 → '未知'")

    # 处理结果统计
    final_shape = comments_cleaned.shape
    final_missing = comments_cleaned.isnull().sum().sum()

    print(f"\n📈 处理结果：")
    print(f"• 处理前：{original_shape[0]:,} 行")
    print(f"• 处理后：{final_shape[0]:,} 行")
    print(f"• 数据保留率：{final_shape[0] / original_shape[0] * 100:.1f}%")
    print(f"• 剩余缺失值：{final_missing} 个")

    if final_missing == 0:
        print("🎉 所有缺失值处理完成！")
    else:
        print(f"⚠️ 仍有 {final_missing} 个缺失值需要处理")

else:
    print("❌ 数据未加载，跳过缺失值处理")


# 重复数据全面处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 🔄 重复数据检测与处理 ===")

    original_count = len(comments_cleaned)
    print(f"处理前数据量：{original_count:,} 条")

    # 1. 完全重复检测
    print(f"\n🔍 检测完全重复...")
    duplicate_all = comments_cleaned.duplicated()
    duplicate_count_all = duplicate_all.sum()

    print(f"• 完全重复记录：{duplicate_count_all:,} 条 ({duplicate_count_all / original_count * 100:.2f}%)")

    if duplicate_count_all > 0:
        # 删除完全重复
        comments_cleaned = comments_cleaned.drop_duplicates()
        print(f"✅ 已删除完全重复记录")

    # 2. 内容重复检测（同一用户对同一电影的重复评论）
    print(f"\n🔍 检测内容重复...")
    if 'CREATOR' in comments_cleaned.columns and 'MOVIEID' in comments_cleaned.columns and 'CONTENT' in comments_cleaned.columns:

        # 检测同用户同电影的重复评论
        content_duplicates = comments_cleaned.duplicated(subset=['CREATOR', 'MOVIEID', 'CONTENT'])
        content_duplicate_count = content_duplicates.sum()

        print(f"• 内容重复记录：{content_duplicate_count:,} 条 ({content_duplicate_count / len(comments_cleaned) * 100:.2f}%)")

        if content_duplicate_count > 0:
            # 删除内容重复，保留最后一条
            if 'ADD_TIME' in comments_cleaned.columns:
                # 如果有时间字段，保留最新的
                comments_cleaned = comments_cleaned.sort_values('ADD_TIME').drop_duplicates(
                    subset=['CREATOR', 'MOVIEID', 'CONTENT'], keep='last'
                )
                print(f"✅ 已删除内容重复记录，保留最新的")
            else:
                # 没有时间字段，保留第一条
                comments_cleaned = comments_cleaned.drop_duplicates(
                    subset=['CREATOR', 'MOVIEID', 'CONTENT'], keep='first'
                )
                print(f"✅ 已删除内容重复记录，保留第一条")

    # 3. 用户重复评论检测（可选处理）
    print(f"\n🔍 检测用户重复评论...")
    if 'CREATOR' in comments_cleaned.columns and 'MOVIEID' in comments_cleaned.columns:

        # 统计每个用户对每部电影的评论数
        user_movie_counts = comments_cleaned.groupby(['CREATOR', 'MOVIEID']).size()
        multiple_reviews = user_movie_counts[user_movie_counts > 1]

        total_multiple = multiple_reviews.sum()
        unique_user_movie = len(multiple_reviews)

        print(f"• 重复评论的用户-电影组合：{unique_user_movie:,} 组")
        print(f"• 涉及重复评论总数：{total_multiple:,} 条")

        if unique_user_movie > 0:
            print(f"💡 建议：根据业务需求决定是否保留用户的多条评论")
            print(f"   - 情感分析：可保留多条，体现情感变化")
            print(f"   - 统计分析：建议每用户每电影只保留一条")

    # 处理结果统计
    final_count = len(comments_cleaned)
    removed_total = original_count - final_count

    print(f"\n📈 去重处理结果：")
    print(f"• 原始记录：{original_count:,} 条")
    print(f"• 处理后记录：{final_count:,} 条")
    print(f"• 删除记录：{removed_total:,} 条")
    print(f"• 数据保留率：{final_count / original_count * 100:.1f}%")

    # 验证去重效果
    remaining_duplicates = comments_cleaned.duplicated().sum()
    if remaining_duplicates == 0:
        print("🎉 所有重复记录已成功处理！")
    else:
        print(f"⚠️ 仍有 {remaining_duplicates} 条重复记录")

    print(f"\n💾 去重后的数据已保存为 comments_cleaned")

else:
    print("❌ 数据未准备好，跳过重复数据处理")


# 异常值检测与处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 🚨 异常值检测分析 ===")

    # 1. 文本长度异常检测
    print("🔍 检测文本长度异常...")
    content_lengths = comments_cleaned['CONTENT'].astype(str).str.len()

    # 过短文本
    too_short = content_lengths <= 3
    short_count = too_short.sum()
    print(f"• 过短评论(≤3字符)：{short_count:,} 条 ({short_count / len(comments_cleaned) * 100:.2f}%)")

    # 过长文本
    too_long = content_lengths >= 1000
    long_count = too_long.sum()
    print(f"• 过长评论(≥1000字符)：{long_count:,} 条 ({long_count / len(comments_cleaned) * 100:.2f}%)")

    # 展示异常例子
    if short_count > 0:
        print("📝 过短评论示例：")
        short_samples = comments_cleaned[too_short]['CONTENT'].head(3)
        for i, content in enumerate(short_samples, 1):
            print(f"   {i}. '{content}' (长度:{len(str(content))})")

    # 2. 重复字符异常检测
    print(f"\n🔍 检测重复字符异常...")
    def has_excessive_repetition(text):
        """检测是否有过多重复字符"""
        if pd.isna(text):
            return False
        text = str(text)
        # 检查连续4个以上相同字符
        pattern = r'(.)\\1{3,}'
        return bool(re.search(pattern, text))

    repetitive_mask = comments_cleaned['CONTENT'].apply(has_excessive_repetition)
    repetitive_count = repetitive_mask.sum()
    print(f"• 过度重复字符：{repetitive_count:,} 条 ({repetitive_count / len(comments_cleaned) * 100:.2f}%)")

    if repetitive_count > 0:
        print("📝 重复字符异常示例：")
        rep_samples = comments_cleaned[repetitive_mask]['CONTENT'].head(3)
        for i, content in enumerate(rep_samples, 1):
            preview = str(content)[:50] + "..." if len(str(content)) > 50 else str(content)
            print(f"   {i}. '{preview}'")

    # 3. 特殊字符占比异常
    print(f"\n🔍 检测特殊字符占比异常...")
    def calc_special_char_ratio(text):
        """计算特殊字符占比"""
        if pd.isna(text):
            return 0
        text = str(text)
        if len(text) == 0:
            return 0

        # 计算非中文、非英文、非数字字符的比例
        special_count = 0
        for char in text:
            if not (char.isalnum() or '\\u4e00' <= char <= '\\u9fff'):
                special_count += 1

        return special_count / len(text)

    special_ratios = comments_cleaned['CONTENT'].apply(calc_special_char_ratio)
    high_special = special_ratios > 0.5             # 特殊字符超过50%
    high_special_count = high_special.sum()

    print(f"• 特殊字符占比>50%：{high_special_count:,} 条 ({high_special_count / len(comments_cleaned) * 100:.2f}%)")

    # 4. 评分异常检测
    if 'RATING' in comments_cleaned.columns:
        print(f"\n🔍 检测评分异常...")

        # 转换为数值类型进行检查
        numeric_ratings = pd.to_numeric(comments_cleaned['RATING'], errors='coerce')

        # 检测评分范围异常（假设正常范围是1-5）
        invalid_ratings = (numeric_ratings < 1) | (numeric_ratings > 5)
        invalid_rating_count = invalid_ratings.sum()

        print(f"• 异常评分(不在1-5范围)：{invalid_rating_count:,} 条")

        if invalid_rating_count > 0:
            print("📊 异常评分分布：")
            abnormal_ratings = numeric_ratings[invalid_ratings].value_counts().head(5)
            for rating, count in abnormal_ratings.items():
                print(f"   评分{rating}: {count}条")

    # 5. 异常处理决策
    print(f"\n🔧 异常处理建议：")

    # 计算总异常数量
    total_anomalies = short_count + repetitive_count + high_special_count
    anomaly_rate = total_anomalies / len(comments_cleaned) * 100

    print(f"• 总异常记录：{total_anomalies:,} 条 ({anomaly_rate:.2f}%)")

    if anomaly_rate < 2:
        print("✅ 异常率较低，数据质量良好")
    elif anomaly_rate < 5:
        print("⚠️ 异常率中等，建议关注但可接受")
    else:
        print("🔴 异常率较高，建议深入分析原因")

    # 可选：删除明显的异常数据
    severe_anomalies = too_short | (special_ratios > 0.8)       # 过短或特殊字符占比过高
    severe_count = severe_anomalies.sum()

    if severe_count > 0:
        print(f"\n💡 发现严重异常 {severe_count} 条，是否删除？")
        print("   - 删除后数据更纯净，但可能丢失信息")
        print("   - 保留后便于后续深入分析")
        print("   📝 当前选择：保留所有数据，添加异常标记")

        # 添加异常标记列
        comments_cleaned['is_anomaly'] = severe_anomalies
        print(f"✅ 已添加异常标记列 'is_anomaly'")

    print(f"\n📊 异常检测完成！数据质量评估结果已生成")

else:
    print("❌ 数据未准备好，跳过异常值检测")

# NLP专业文本清洗函数
def clean_text_for_nlp(text):
    """NLP专用文本清洗函数"""

    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # 1. 去除HTML标签和实体
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&lt;|&gt;|&nbsp;', ' ', text)

    # 2. 处理网络用语标准化
    text = re.sub(r'h{3,}', '哈哈', text)  # hhh -> 哈哈
    text = re.sub(r'2333+', '哈哈', text)  # 2333 -> 哈哈
    text = re.sub(r'6{4,}', '厉害', text)  # 6666 -> 厉害
    text = re.sub(r'\\d{4,}', '', text)  # 去除长数字串

    # 3. 处理重复字符（保留一定的重复表达情感）
    text = re.sub(r'(.)\\1{4,}', r'\\1\\1\\1', text)  # 超过4个重复减少到3个
    text = re.sub(r'[！!]{4,}', '！！！', text)  # 多个感叹号
    text = re.sub(r'[？?]{4,}', '？？？', text)  # 多个问号
    text = re.sub(r'[。.]{3,}', '...', text)  # 多个句号

    # 4. 去除特殊符号（保留基本标点）
    text = re.sub(r'[★☆※@#$%^&*\\[]{}|\\\\]', '', text)
    text = re.sub(r'[~～]', '', text)

    # 5. 清理空白字符
    text = re.sub(r'\\s+', ' ', text)
    text = text.strip()

    return text

# 应用文本清洗
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("=== 🧹 NLP文本深度清洗 ===")

    print("🔧 正在进行文本清洗...")

    # 保存原始内容用于对比
    original_content_sample = comments_cleaned['CONTENT'].head(3).tolist()

    # 应用清洗函数
    comments_cleaned['CONTENT_CLEANED'] = comments_cleaned['CONTENT'].apply(clean_text_for_nlp)

    # 统计清洗效果
    original_total_length = comments_cleaned['CONTENT'].astype(str).str.len().sum()
    cleaned_total_length = comments_cleaned['CONTENT_CLEANED'].astype(str).str.len().sum()
    reduction_rate = (1 - cleaned_total_length / original_total_length) * 100

    print(f"📊 清洗效果统计：")
    print(f"• 原始总字符数：{original_total_length:,}")
    print(f"• 清洗后字符数：{cleaned_total_length:,}")
    print(f"• 压缩率：{reduction_rate:.1f}%")

    # 展示清洗前后对比
    print(f"\n📝 清洗前后对比示例：")
    cleaned_content_sample = comments_cleaned['CONTENT_CLEANED'].head(3).tolist()

    for i, (original, cleaned) in enumerate(zip(original_content_sample, cleaned_content_sample), 1):
        print(f"\n{i}. 原文：{original}")
        print(f"   清洗后：{cleaned}")

        # 计算个例压缩率
        if len(original) > 0:
            individual_reduction = (1 - len(cleaned) / len(original)) * 100
            print(f"   压缩率：{individual_reduction:.1f}%")

    # 检查清洗质量
    empty_after_cleaning = np.sum(comments_cleaned['CONTENT_CLEANED'].str.len() == 0)
    if empty_after_cleaning > 0:
        print(f"\n⚠️ 警告：有 {empty_after_cleaning} 条评论清洗后变为空")
        print("建议检查清洗规则是否过于严格")
    else:
        print(f"\n✅ 清洗质量检查通过，无评论变为空")

    print(f"\n💾 清洗后的文本已保存到 'CONTENT_CLEANED' 列")

else:
    print("❌ 数据未准备好，跳过文本清洗")

# 中文分词处理
# 安装和导入jieba
import jieba
# 自然语言处理
# 自然 语言 处理

# 电影领域自定义词典
movie_words = [
    # 电影名称
    "复仇者联盟", "钢铁侠", "美国队长", "黑寡妇", "雷神", "绿巨人",
    "蜘蛛侠", "奇异博士", "黑豹", "惊奇队长", "流浪地球", "哪吒",

    # 导演和演员
    "诺兰", "斯皮尔伯格", "张艺谋", "冯小刚", "徐峥", "王宝强",
    "漫威", "DC", "迪士尼", "环球影业", "索尼影业",

    # 电影术语
    "特效", "剧情", "演技", "配乐", "摄影", "剪辑", "编剧",
    "票房", "口碑", "评分", "首映", "上映", "下映", "点映",
    "IMAX", "3D", "4D", "杜比", "巨幕"
]

# 添加自定义词汇到jieba词典
print("🔧 加载电影领域词典...")
for word in movie_words:
    jieba.add_word(word)
print(f"✅ 已添加 {len(movie_words)} 个电影领域专有词汇")

# 分词处理函数
def segment_text(text, mode='accurate'):
    """
    中文分词函数
    mode: 'accurate'(精确), 'full'(全模式), 'search'(搜索模式)
    """
    if pd.isna(text) or not text.strip():
        return []

    text = str(text).strip()

    if mode == 'accurate':
        words = jieba.lcut(text)
    elif mode == 'full':
        words = jieba.lcut(text, cut_all=True)
    elif mode == 'search':
        words = jieba.lcut_for_search(text)
    else:
        words = jieba.lcut(text)

    # 过滤长度小于2的词和纯标点符号
    filtered_words = []
    for filter_word in words:
        filter_word = filter_word.strip()
        if (len(filter_word) >= 2 and
                not all(char in '，。！？、；：""''（）【】' for char in filter_word)):
            filtered_words.append(filter_word)

    return filtered_words

text = "我爱自然语言处理"

print(segment_text(text, mode="accurate"))
print(segment_text(text, mode="full"))
print(segment_text(text, mode="search"))

jieba.add_word("我爱自然语言处理")
print(segment_text(text, mode="accurate"))
print(segment_text(text, mode="full"))

# 应用分词处理
if 'comments_cleaned' in locals() and comments_cleaned is not None:
    print("\n=== ✂️ 中文分词处理 ===")

    # 选择用于分词的文本列
    text_column = 'CONTENT_CLEANED' if 'CONTENT_CLEANED' in comments_cleaned.columns else 'CONTENT'
    print(f"🔧 使用 '{text_column}' 列进行分词...")

    # 对前1000条进行分词演示（避免处理时间过长）
    sample_size = min(1000, len(comments_cleaned))
    sample_data = comments_cleaned.head(sample_size).copy()

    print(f"📊 处理样本：{sample_size} 条评论")

    # 执行分词
    print("🔄 正在进行分词处理...")
    sample_data['WORDS'] = sample_data[text_column].apply(
        lambda x: segment_text(x, mode='accurate')
    )

    # 统计分词效果
    total_words = sample_data['WORDS'].apply(len).sum()
    avg_words_per_review = total_words / len(sample_data)

    print(f"📈 分词统计结果：")
    print(f"• 总词汇数：{total_words:,}")
    print(f"• 平均每条评论词数：{avg_words_per_review:.1f}")

    # 展示分词示例
    print(f"\n📝 分词效果示例：")
    for i in range(3):
        if i < len(sample_data):
            original = sample_data.iloc[i][text_column]
            words = sample_data.iloc[i]['WORDS']

            print(f"\n{i + 1}. 原文：{original[:80]}...")
            print(f"   分词：{' / '.join(words[:15])}...")
            print(f"   词数：{len(words)}")

    # 词频统计
    print(f"\n📊 高频词汇分析：")
    all_words = []
    for words_list in sample_data['WORDS']:
        all_words.extend(words_list)

    from collections import Counter

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(10)

    print("🔝 TOP10高频词汇：")
    for word, freq in top_words:
        print(f"• {word}: {freq} 次")

    # 检查自定义词典效果
    custom_words_found = []
    for word in movie_words:
        if word in all_words:
            custom_words_found.append((word, word_freq[word]))

    if custom_words_found:
        print(f"\n🎬 发现的电影相关词汇：")
        for word, freq in sorted(custom_words_found, key=lambda x: x[1], reverse=True)[:5]:
            print(f"• {word}: {freq} 次")

    print(f"\n💾 分词结果已保存到 'WORDS' 列")

    # 保存到主数据集
    comments_cleaned = comments_cleaned.head(sample_size).copy()
    comments_cleaned['WORDS'] = sample_data['WORDS']

else:
    print("❌ 数据未准备好，跳过分词处理")

# 停用词处理
def create_stopwords_for_movie_reviews():
    """构建适合电影评论的停用词表"""

    # 基础停用词
    basic_stopwords = {
        # 助词和虚词
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '这个', '那个',

        # 语气词
        '吧', '呢', '啊', '嗯', '哦', '呀', '嘛', '呐',

        # 连接词
        '但是', '然后', '因为', '所以', '而且', '不过', '虽然', '如果', '那么', '或者',

        # 代词
        '这', '那', '这些', '那些', '我们', '你们', '他们', '她们', '它们',

        # 介词
        '从', '向', '对', '对于', '关于', '由于', '通过', '根据', '按照',

        # 时间词
        '现在', '当时', '之前', '以前', '之后', '以后', '今天', '明天', '昨天'
    }

    # 电影评论特定停用词
    movie_stopwords = {
        '电影', '影片', '片子', '这部', '这个', '一部', '整部',
        '观看', '看了', '看过', '观影', '看到', '看见',
        '感觉', '觉得', '认为', '个人', '我觉得', '我认为', '我感觉',
        '还是', '就是', '只是', '真的', '确实', '的确'
    }

    # 标点符号
    punctuation = {
        '，', '。', '！', '？', '；', '：', '"', '"', ''', ''',
        '（', '）', '【', '】', '、', '…', '—', '·'
    }

    return basic_stopwords | movie_stopwords | punctuation

def filter_stopwords(filter_words, scenario='general'):
    """
    根据不同场景过滤停用词
    scenario: 'general', 'sentiment', 'topic'
    """
    stopwords = create_stopwords_for_movie_reviews()

    if scenario == 'sentiment':
        # 情感分析：保留程度词
        degree_words = {'很', '非常', '特别', '太', '超级', '最', '极其', '相当', '比较', '有点', '稍微'}
        stopwords = stopwords - degree_words
    elif scenario == 'topic':
        # 主题分析：更严格的过滤
        extra_stopwords = {
            '比较', '可能', '应该', '什么', '怎么', '为什么',
            '时候', '地方', '方面', '问题', '东西', '事情', '方法'
        }
        stopwords = stopwords | extra_stopwords

    # 过滤停用词
    filtered_words = [filter_word for filter_word in filter_words if filter_word not in stopwords]
    return filtered_words


# 应用停用词过滤
if 'comments_cleaned' in locals() and 'WORDS' in comments_cleaned.columns:
    print("=== 🚫 停用词过滤处理 ===")

    # 构建停用词表
    stopwords = create_stopwords_for_movie_reviews()
    print(f"📋 停用词表大小：{len(stopwords)} 个词")
    print(f"🔍 停用词示例：{list(stopwords)[:15]}...")

    # 应用不同场景的停用词过滤
    scenarios = {
        'general': '通用场景',
        'sentiment': '情感分析',
        'topic': '主题分析'
    }

    for scenario, desc in scenarios.items():
        col_name = f'WORDS_{scenario.upper()}'
        comments_cleaned[col_name] = comments_cleaned['WORDS'].apply(
            lambda x: filter_stopwords(x, scenario)
        )

        # 统计过滤效果
        original_word_count = comments_cleaned['WORDS'].apply(len).sum()
        filtered_word_count = comments_cleaned[col_name].apply(len).sum()
        reduction_rate = (1 - filtered_word_count / original_word_count) * 100

        print(f"\\n🎯 {desc}过滤结果：")
        print(f"• 原始词数：{original_word_count:,}")
        print(f"• 过滤后词数：{filtered_word_count:,}")
        print(f"• 过滤率：{reduction_rate:.1f}%")

    # 展示过滤效果示例
    print(f"\\n📝 停用词过滤效果对比：")
    for i in range(3):
        if i < len(comments_cleaned):
            original_words = comments_cleaned.iloc[i]['WORDS']
            general_words = comments_cleaned.iloc[i]['WORDS_GENERAL']
            sentiment_words = comments_cleaned.iloc[i]['WORDS_SENTIMENT']
            topic_words = comments_cleaned.iloc[i]['WORDS_TOPIC']

            print(f"\\n{i + 1}. 原始分词：{' / '.join(original_words[:10])}...")
            print(f"   通用过滤：{' / '.join(general_words[:10])}...")
            print(f"   情感保留：{' / '.join(sentiment_words[:10])}...")
            print(f"   主题过滤：{' / '.join(topic_words[:10])}...")

    # 分析高频词变化
    print(f"\\n📊 停用词过滤后的高频词分析：")

    # 统计通用过滤后的词频
    all_filtered_words = []
    for words_list in comments_cleaned['WORDS_GENERAL']:
        all_filtered_words.extend(words_list)

    if all_filtered_words:
        from collections import Counter

        filtered_word_freq = Counter(all_filtered_words)
        top_filtered_words = filtered_word_freq.most_common(10)

        print("🔝 停用词过滤后TOP10词汇：")
        for word, freq in top_filtered_words:
            print(f"• {word}: {freq} 次")

    print(f"\\n💾 停用词过滤结果已保存到对应列")
    print("✅ 可根据具体应用场景选择合适的过滤结果")

else:
    print("❌ 分词结果未准备好，跳过停用词处理")

# 数据质量全面验证
def comprehensive_data_quality_check(df, original_df=None):
    """
    全面的数据质量检查函数
    """
    print("=== ✅ 数据质量综合验证报告 ===")

    quality_score = 0
    max_score = 100

    # 1. 完整性验证 (30分)
    print("\\n1️⃣ 完整性验证:")
    missing_count = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    completeness_rate = (1 - missing_count / total_cells) * 100

    print(f"• 总缺失值：{missing_count}")
    print(f"• 完整性：{completeness_rate:.1f}%")

    completeness_score = min(30, (completeness_rate / 100) * 30)
    quality_score += completeness_score
    print(f"• 完整性得分：{completeness_score:.1f}/30")

    # 2. 一致性验证 (25分)
    print("\\n2️⃣ 一致性验证:")
    consistency_issues = 0

    # 检查数据类型一致性
    for col in df.columns:
        if df[col].dtype == 'object':
            # 检查是否有异常的数据类型混合
            sample_values = df[col].dropna().head(100)
            types = set(type(val).__name__ for val in sample_values)
            if len(types) > 1:
                consistency_issues += 1
                print(f"• {col}列存在混合数据类型：{types}")

    # 检查文本格式一致性
    if 'CONTENT_CLEANED' in df.columns:
        cleaned_texts = df['CONTENT_CLEANED'].dropna()
        html_tags = cleaned_texts.str.contains('<[^>]+>', na=False).sum()
        if html_tags > 0:
            consistency_issues += 1
            print(f"• 仍有{html_tags}条记录包含HTML标签")

    consistency_score = max(0, 25 - consistency_issues * 5)
    quality_score += consistency_score
    print(f"• 一致性得分：{consistency_score:.1f}/25")

    # 3. 准确性验证 (25分)
    print("\\n3️⃣ 准确性验证:")
    accuracy_issues = 0

    # 检查分词结果准确性
    if 'WORDS' in df.columns:
        # 检查分词结果是否合理
        word_lengths = []
        for words_list in df['WORDS'].dropna():
            if isinstance(words_list, list):
                word_lengths.extend([len(word) for word in words_list])

        if word_lengths:
            avg_word_length = sum(word_lengths) / len(word_lengths)
            print(f"• 平均词长：{avg_word_length:.1f}字符")

            # 中文词汇平均长度应该在1.5-3之间
            if avg_word_length < 1.5 or avg_word_length > 4:
                accuracy_issues += 1
                print(f"• 警告：平均词长异常，可能存在分词问题")

    # 检查停用词过滤效果
    if 'WORDS_GENERAL' in df.columns:
        # 检查是否还有常见停用词
        common_stopwords = {'的', '了', '是', '在', '我', '有'}
        all_filtered_words = []
        for words_list in df['WORDS_GENERAL'].dropna():
            if isinstance(words_list, list):
                all_filtered_words.extend(words_list)

        remaining_stopwords = [word for word in all_filtered_words if word in common_stopwords]
        if len(remaining_stopwords) > len(all_filtered_words) * 0.05:  # 超过5%
            accuracy_issues += 1
            print(f"• 警告：仍有较多停用词未过滤")

    accuracy_score = max(0, 25 - accuracy_issues * 8)
    quality_score += accuracy_score
    print(f"• 准确性得分：{accuracy_score:.1f}/25")

    # 4. 有效性验证 (20分)
    print("\\n4️⃣ 有效性验证:")
    validity_issues = 0

    # 检查评分范围有效性
    if 'RATING' in df.columns:
        numeric_ratings = pd.to_numeric(df['RATING'], errors='coerce')
        valid_ratings = numeric_ratings.dropna()
        if len(valid_ratings) > 0:
            invalid_count = ((valid_ratings < 1) | (valid_ratings > 5)).sum()
            if invalid_count > 0:
                validity_issues += 1
                print(f"• 发现{invalid_count}个无效评分")

    # 检查文本长度有效性
    if 'CONTENT_CLEANED' in df.columns:
        text_lengths = df['CONTENT_CLEANED'].astype(str).str.len()
        too_short = (text_lengths <= 2).sum()
        if too_short > len(df) * 0.02:  # 超过2%
            validity_issues += 1
            print(f"• 过短文本过多：{too_short}条")

    validity_score = max(0, 20 - validity_issues * 7)
    quality_score += validity_score
    print(f"• 有效性得分：{validity_score:.1f}/20")

    # 总体评分
    print(f"\\n🏆 数据质量总分：{quality_score:.1f}/{max_score}")

    if quality_score >= 90:
        grade = "A+ 优秀"
        emoji = "🥇"
    elif quality_score >= 80:
        grade = "A 良好"
        emoji = "🥈"
    elif quality_score >= 70:
        grade = "B 合格"
        emoji = "🥉"
    else:
        grade = "C 需改进"
        emoji = "⚠️"

    print(f"{emoji} 质量等级：{grade}")

    return quality_score


# 进行数据质量验证
if 'comments_cleaned' in locals() and comments_cleaned is not None:

    # 对比处理前后的数据量
    if 'comments_df' in locals() and comments_df is not None:
        print("📊 数据处理前后对比：")
        print(f"• 原始数据：{len(comments_df):,} 条")
        print(f"• 处理后数据：{len(comments_cleaned):,} 条")
        print(f"• 数据保留率：{len(comments_cleaned) / len(comments_df) * 100:.1f}%")

    # 执行质量检查
    quality_score = comprehensive_data_quality_check(comments_cleaned,
                                                     comments_df if 'comments_df' in locals() else None)

    # 输出最终数据概览
    print(f"\\n📋 最终数据集概览：")
    print(f"• 数据行数：{len(comments_cleaned):,}")
    print(f"• 数据列数：{len(comments_cleaned.columns)}")
    print(f"• 主要列：{list(comments_cleaned.columns)}")

    # 展示最终处理后的数据样本
    print(f"\\n📝 处理后数据样本：")
    if len(comments_cleaned) > 0:
        for i in range(min(2, len(comments_cleaned))):
            row = comments_cleaned.iloc[i]
            print(f"\\n样本 {i + 1}:")
            print(f"• 原评论：{str(row.get('CONTENT', 'N/A'))[:60]}...")
            print(f"• 清洗后：{str(row.get('CONTENT_CLEANED', 'N/A'))[:60]}...")
            if 'WORDS_GENERAL' in row:
                words = row['WORDS_GENERAL']
                if isinstance(words, list) and len(words) > 0:
                    print(f"• 关键词：{' / '.join(words[:8])}...")

    print(f"\\n🎉 数据预处理流程全部完成！")
    print(f"📊 数据质量评分：{quality_score:.1f}/100")
    print(f"💾 最终数据集已保存在 comments_cleaned 变量中")

else:
    print("❌ 没有可验证的数据集")
