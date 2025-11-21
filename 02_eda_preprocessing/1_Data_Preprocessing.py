# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
def smart_read_csv(file_path, sample_size=1000):
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
            pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
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
    comments_df, comment_encoding = smart_read_csv('douban-dataset/comments.csv', sample_size=1000)

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
