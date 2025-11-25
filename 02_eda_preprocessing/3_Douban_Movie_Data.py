# 导入我们的数据探索工具箱
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 设置图表样式，让图表更好看
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# 设置中文字体，这样图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🎉 工具箱准备完毕！让我们开始探索数据吧！")
#
# # 🎨 示例1：创建一些模拟数据来演示seaborn
# print("🎬 创建一些模拟的电影评分数据来演示...")
#
# # 创建模拟数据：不同类型电影的评分
# np.random.seed(42)  # 保证结果可重复
# movie_data = {
#     '电影类型': ['动作'] * 100 + ['喜剧'] * 100 + ['爱情'] * 100 + ['科幻'] * 100,
#     '评分': (
#         np.random.normal(7.5, 1.2, 100).tolist() +  # 动作片评分
#         np.random.normal(8.0, 1.0, 100).tolist() +  # 喜剧片评分
#         np.random.normal(7.8, 1.1, 100).tolist() +  # 爱情片评分
#         np.random.normal(7.2, 1.3, 100).tolist()    # 科幻片评分
#     ),
#     '票房': np.random.lognormal(2, 1, 400),  # 票房数据（对数正态分布）
# }
#
# demo_df = pd.DataFrame(movie_data)
# print(f"✅ 创建了 {len(demo_df)} 条模拟电影数据")
# print("前5行数据预览：")
# print(demo_df.head())
#
# # 🆚 示例2：对比matplotlib vs seaborn
# print("🆚 对比 matplotlib 和 seaborn 的区别")
#
# # 创建对比图
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('📊 Matplotlib vs Seaborn 对比展示', fontsize=16, fontweight='bold')
#
# # matplotlib版本 - 直方图
# axes[0,0].hist(demo_df['评分'], bins=20, alpha=0.7, color='blue')
# axes[0,0].set_title('Matplotlib 直方图')
# axes[0,0].set_xlabel('评分')
# axes[0,0].set_ylabel('频次')
#
# # seaborn版本 - 直方图
# sns.histplot(data=demo_df, x='评分', bins=20, ax=axes[0,1])
# axes[0,1].set_title('Seaborn 直方图（更美观）')
#
# # matplotlib版本 - 箱线图
# box_data = [demo_df[demo_df['电影类型']==genre]['评分'].values
#            for genre in demo_df['电影类型'].unique()]
# axes[1,0].boxplot(box_data, labels=demo_df['电影类型'].unique())
# axes[1,0].set_title('Matplotlib 箱线图')
# axes[1,0].set_ylabel('评分')
#
# # seaborn版本 - 箱线图
# sns.boxplot(data=demo_df, x='电影类型', y='评分', ax=axes[1,1])
# axes[1,1].set_title('Seaborn 箱线图（一行代码！）')
#
# plt.tight_layout()
# plt.show()
#
# print("🎯 对比总结：")
# print("• Matplotlib：功能强大但需要更多代码")
# print("• Seaborn：简洁美观，自动处理分类数据")
#
#
# # ✨ 示例3：Seaborn的"超能力"展示
# print("✨ Seaborn的特色功能展示")
#
# # 创建展示图
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('🌟 Seaborn 特色功能展示', fontsize=16, fontweight='bold')
#
# # 1. 散点图 + 回归线（一行代码完成）
# sns.scatterplot(data=demo_df, x='票房', y='评分', hue='电影类型', ax=axes[0,0])
# axes[0,0].set_title('🎯 散点图：票房 vs 评分')
# axes[0,0].set_xlabel('票房（万元）')
#
# # 2. 小提琴图（显示分布形状）
# sns.violinplot(data=demo_df, x='电影类型', y='评分', ax=axes[0,1])
# axes[0,1].set_title('🎻 小提琴图：评分分布')
# axes[0,1].tick_params(axis='x', rotation=45)
#
# # 3. 计数图（柱状图的升级版）
# sns.countplot(data=demo_df, x='电影类型', ax=axes[1,0])
# axes[1,0].set_title('📊 计数图：各类型电影数量')
#
# # 4. 热力图（显示相关性）
# # 创建相关性矩阵
# corr_data = demo_df[['评分', '票房']].corr()
# sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
#             square=True, ax=axes[1,1])
# axes[1,1].set_title('🔥 热力图：相关性分析')
#
# plt.tight_layout()
# plt.show()
#
# print("\n🎓 每种图表的作用：")
# print("• 📈 散点图：发现两个变量之间的关系")
# print("• 🎻 小提琴图：比箱线图更详细地显示数据分布")
# print("• 📊 计数图：统计各类别的数量，比普通柱状图更智能")
# print("• 🔥 热力图：可视化数字之间的相关性，颜色越深关系越强")
# print("箱线图作用")
#
# # 使用seaborn的regplot或lmplot
# sns.regplot(data=demo_df, x='评分', y='票房', scatter_kws={'alpha':0.6})
# plt.title('电影评分与票房关系')
# plt.xlabel('评分')
# plt.ylabel('票房')
# plt.show()
#
#
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 创建图
# G = nx.Graph()
# G.add_edge('电影A', '演员1')
# G.add_edge('电影A', '导演1')
# G.add_edge('电影B', '演员1')
#
# # 绘制
# nx.draw(G, with_labels=True)
# plt.show()
#
#
# from pyvis.network import Network
#
# # 创建网络图，设置参数以解决模板问题
# net = Network(height='600px', width='100%', notebook=True, cdn_resources='remote')
#
# # 添加节点
# net.add_node('电影A', label='电影A', color='#FF6B6B')
# net.add_node('演员1', label='演员1', color='#4ECDC4')
#
# # 添加边
# net.add_edge('电影A', '演员1')
#
# # 在 Jupyter Notebook 中显示
# net.show('graph.html')


# 📚 定义智能读取函数
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

        except Exception as e:
            print(f"❌ {encoding} 编码失败: {str(e)[:50]}...")
            continue

    print("😱 所有编码都失败了！")
    return None, None


print("🛠️ 智能读取函数定义完成！")

# 📖 读取电影数据
print("📖 正在读取电影信息数据...")

# 使用智能读取函数加载电影数据
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

# 使用智能读取函数处理评论数据
try:
    # 方法1：使用智能读取函数（推荐）
    comments_sample, comment_encoding = smart_read_csv('douban-dataset/comments.csv', sample_size=10000)

    # 如果文件太大，只取前10000条
    if comments_sample is not None and len(comments_sample) > 10000:
        comments_sample = comments_sample.head(10000)
        print(f"📝 为了演示方便，只保留前10000条评论")

    if comments_sample is not None:
        print(f"✅ 评论数据读取成功！读取了 {len(comments_sample)} 条评论")
        print(f"📊 数据形状: {comments_sample.shape}")
        print(f"🔤 使用编码: {comment_encoding}")

except Exception as e:
    print(f"💔 评论数据读取失败: {e}")
    comments_sample = None
#
# # 电影数据概览
# if movies_df is not None:
#     print("🎬 电影数据的基本信息：")
#     print("前5行数据预览：")
#     print(movies_df.head())
#
#     print(f"\n数据形状: {movies_df.shape}")
#     print(f"行数（电影数量）: {movies_df.shape[0]}")
#     print(f"列数（特征数量）: {movies_df.shape[1]}")
#
#     print("\n列名信息：")
#     for i, col in enumerate(movies_df.columns):
#         print(f"第{i + 1}列: {col}")
#
#     print("\n数据类型：")
#     print(movies_df.dtypes)
# else:
#     print("无法显示电影数据概览")
#
#
# # 评论数据概览
# if comments_sample is not None:
#     print("\n💬 评论数据的基本信息：")
#     print("前5行数据预览：")
#     print(comments_sample.head())
#
#     print(f"\n数据形状: {comments_sample.shape}")
#     print(f"行数（评论数量）: {comments_sample.shape[0]}")
#     print(f"列数（特征数量）: {comments_sample.shape[1]}")
#
#     print("\n列名信息：")
#     for i, col in enumerate(comments_sample.columns):
#         print(f"第{i + 1}列: {col}")
#
#     print("\n数据类型：")
#     print(comments_sample.dtypes)
#
#     print("\n缺失值检查：")
#     missing_data = comments_sample.isnull().sum()
#     print(missing_data)
# else:
#     print("无法显示评论数据概览")
#
#
# # 📊 文本长度分析
# if comments_sample is not None:
#     print("📝 评论文本长度分析")
#     print("=" * 40)
#
#     # 假设评论在某一列，我们先检查列名
#     print("可用列名：", list(comments_sample.columns))
#
#     # 尝试找到评论文本列（通常可能叫comment、content、text等）
#     text_columns = []
#     for col in comments_sample.columns:
#         if any(keyword in col.lower() for keyword in ['comment', 'content', 'text', 'review']):
#             text_columns.append(col)
#
#     if text_columns:
#         comment_col = text_columns[0]
#         print(f"找到评论列：{comment_col}")
#
#         # 计算文本长度
#         comments_sample['text_length'] = comments_sample[comment_col].astype(str).str.len()
#
#         print("\n📏 评论长度统计：")
#         length_stats = comments_sample['text_length'].describe()
#         print(length_stats)
#
#         print(f"\n🎯 关键指标解读：")
#         print(f"• 平均评论长度: {length_stats['mean']:.1f} 个字符")
#         print(f"• 最短评论: {length_stats['min']:.0f} 个字符")
#         print(f"• 最长评论: {length_stats['max']:.0f} 个字符")
#         print(f"• 中位数长度: {length_stats['50%']:.1f} 个字符")
#
#     else:
#         print("未找到明确的评论文本列，显示所有列的基本统计：")
#
#
# # 📊 可视化1：评论长度分布直方图
# if comments_sample is not None and 'text_length' in comments_sample.columns:
#
#     # 创建图表
#     plt.figure(figsize=(12, 6))
#
#     # 左图：直方图
#     plt.subplot(1, 2, 1)
#     plt.hist(comments_sample['text_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.title('评论长度分布直方图')
#     plt.xlabel('评论长度（字符数）')
#     plt.ylabel('评论数量')
#     plt.grid(True, alpha=0.3)
#
#     # 添加统计信息
#     mean_length = comments_sample['text_length'].mean()
#     plt.axvline(mean_length, color='red', linestyle='--', label=f'平均值: {mean_length:.1f}')
#     plt.legend()
#
#     # 右图：箱线图
#     plt.subplot(1, 2, 2)
#     plt.boxplot(comments_sample['text_length'], labels=['评论长度'])
#     plt.title('评论长度箱线图')
#     plt.ylabel('评论长度（字符数）')
#     plt.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # 数据解读
#     print("🔍 图表解读：")
#     print("• 直方图显示了评论长度的分布模式")
#     print("• 箱线图帮助我们发现异常值（超长或超短的评论）")
#
#     # 找出异常值
#     Q1 = comments_sample['text_length'].quantile(0.25)
#     Q3 = comments_sample['text_length'].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = comments_sample[
#         (comments_sample['text_length'] < Q1 - 1.5 * IQR) |
#         (comments_sample['text_length'] > Q3 + 1.5 * IQR)
#     ]
#
#     print(f"• 发现 {len(outliers)} 个异常值（特别长或特别短的评论）")
#     if len(outliers) > 0:
#         print(f"• 最长评论有 {outliers['text_length'].max()} 个字符")
#         print(f"• 最短评论有 {outliers['text_length'].min()} 个字符")
#
#
# # 🔥 热门电影分析
# if comments_sample is not None:
#     print("🔥 热门电影分析")
#     print("=" * 40)
#
#     # 检查是否有电影ID或相关列
#     movie_columns = []
#     for col in comments_sample.columns:
#         if any(keyword in col.lower() for keyword in ['movie', 'film', 'id']):
#             movie_columns.append(col)
#
#     if movie_columns:
#         movie_col = movie_columns[0]
#         print(f"使用电影标识列: {movie_col}")
#
#         # 统计每部电影的评论数量
#         movie_comment_counts = comments_sample[movie_col].value_counts()
#
#         print(f"\n📊 评论数统计:")
#         print(f"• 总共有 {len(movie_comment_counts)} 部不同的电影")
#         print(f"• 平均每部电影有 {movie_comment_counts.mean():.1f} 条评论")
#         print(f"• 评论最多的电影有 {movie_comment_counts.max()} 条评论")
#         print(f"• 评论最少的电影有 {movie_comment_counts.min()} 条评论")
#
#         # 显示TOP 10热门电影
#         print(f"\n🏆 TOP 10 热门电影（按评论数量）:")
#         top_movies = movie_comment_counts.head(10)
#         for i, (movie_id, count) in enumerate(top_movies.items(), 1):
#             print(f"{i:2d}. 电影ID {movie_id}: {count} 条评论")
#
#         # 可视化热门电影
#         plt.figure(figsize=(12, 6))
#
#         # 左图：TOP 10电影评论数
#         plt.subplot(1, 2, 1)
#         top_movies.plot(kind='bar', color='lightcoral')
#         plt.title('🏆 TOP 10 热门电影')
#         plt.xlabel('电影ID')
#         plt.ylabel('评论数量')
#         plt.xticks(rotation=45)
#         plt.grid(True, alpha=0.3)
#
#         # 右图：评论数分布
#         plt.subplot(1, 2, 2)
#         plt.hist(movie_comment_counts.values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
#         plt.title('📊 电影评论数分布')
#         plt.xlabel('评论数量')
#         plt.ylabel('电影数量')
#         plt.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.show()
#
#     else:
#         print("未找到电影相关列，显示数据集的整体统计")
#
#
# # 🎭 电影类型分析 - 第1步：数据准备
# print("🎭 电影类型偏好分析 - 数据准备阶段")
# print("=" * 50)
#
# if movies_df is not None:
#     # 1️⃣ 检查现有数据结构
#     print("📋 电影数据列名:")
#     for i, col in enumerate(movies_df.columns):
#         print(f"  {i + 1}. {col}")
#
#     # 2️⃣ 创建模拟类型数据（实际项目中替换为真实数据）
#     print("\n🎨 创建模拟电影类型数据来演示分析方法...")
#
#     # 设置随机种子确保结果可重复
#     np.random.seed(42)
#
#     # 定义电影类型和权重分布（模拟真实市场分布）
#     movie_genres = ['动作', '喜剧', '爱情', '科幻', '悬疑', '动画', '剧情', '恐怖']
#     genre_weights = [0.15, 0.18, 0.12, 0.10, 0.08, 0.07, 0.20, 0.10]
#
#     # 为每部电影分配类型
#     simulated_genres = np.random.choice(movie_genres, size=len(movies_df), p=genre_weights)
#     movies_df_demo = movies_df.copy()
#     movies_df_demo['类型'] = simulated_genres
#
#     print(f"✅ 模拟数据创建完成！包含{len(movies_df_demo)}部电影")
#     print("📌 数据准备完成，可以进行后续分析")
#
# else:
#     print("❌ 电影数据未加载，无法进行分析")
#
#
# # 🎭 电影类型分析 - 第2步：基础统计
# if 'movies_df_demo' in locals():
#     # 统计各类型电影数量
#     genre_counts = movies_df_demo['类型'].value_counts()
#
#     print("📊 电影类型分布统计:")
#     print(f"• 总共有 {len(genre_counts)} 种不同类型")
#     print(f"• 数据集中共有 {len(movies_df_demo)} 部电影")
#
#     print(f"\n🏆 各类型电影数量排行:")
#     for i, (genre, count) in enumerate(genre_counts.items(), 1):
#         percentage = (count / len(movies_df_demo)) * 100
#         print(f"  {i}. {genre}: {count} 部 ({percentage:.1f}%)")
#
# else:
#     print("❌ 请先运行上一个cell创建模拟数据")
#
#
# # 🎭 电影类型分析 - 第3步：基础可视化
# if 'genre_counts' in locals():
#     # 创建基础图表：柱状图和饼图
#     plt.figure(figsize=(12, 5))
#
#     # 左图：柱状图
#     plt.subplot(1, 2, 1)
#     colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
#     bars = plt.bar(genre_counts.index, genre_counts.values, color=colors)
#     plt.title('🎬 电影类型数量分布', fontsize=14, fontweight='bold')
#     plt.xlabel('电影类型')
#     plt.ylabel('电影数量')
#     plt.xticks(rotation=45)
#
#     # 在柱状图上添加数值标签
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height + 10,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=10)
#
#     # 右图：饼图
#     plt.subplot(1, 2, 2)
#     plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%',
#             colors=colors, startangle=90)
#     plt.title('🥧 电影类型占比饼图', fontsize=14, fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#     print("📈 图表说明：")
#     print("• 柱状图：直观显示各类型的绝对数量")
#     print("• 饼图：显示各类型在总体中的占比")
#
# else:
#     print("❌ 请先运行前面的cell进行数据统计")
#
#
# # 🎭 电影类型分析 - 第4步：进阶可视化
# if 'genre_counts' in locals():
#     # 创建进阶图表：水平柱状图和累积分布图
#     plt.figure(figsize=(12, 5))
#
#     # 左图：水平柱状图（便于阅读长标签）
#     plt.subplot(1, 2, 1)
#     colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
#     plt.barh(genre_counts.index, genre_counts.values, color=colors)
#     plt.title('📊 电影类型分布（水平视图）', fontsize=14, fontweight='bold')
#     plt.xlabel('电影数量')
#
#     # 添加数值标签
#     for i, (label, value) in enumerate(zip(genre_counts.index, genre_counts.values)):
#         plt.text(value + 20, i, f'{value}', va='center', fontsize=10)
#
#     # 右图：累积百分比图
#     plt.subplot(1, 2, 2)
#     cumulative_pct = (genre_counts.cumsum() / genre_counts.sum() * 100)
#     plt.plot(range(len(cumulative_pct)), cumulative_pct.values, 'o-',
#              linewidth=2, markersize=8, color='darkblue')
#     plt.title('📈 类型累积分布图', fontsize=14, fontweight='bold')
#     plt.xlabel('类型排名')
#     plt.ylabel('累积百分比 (%)')
#     plt.xticks(range(len(cumulative_pct)), genre_counts.index, rotation=45)
#     plt.grid(True, alpha=0.3)
#
#     # 添加80%线
#     plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%线')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     print("📊 进阶图表说明：")
#     print("• 水平柱状图：方便阅读类型名称，便于比较")
#     print("• 累积分布图：显示主要类型的集中度，用于分析长尾效应")
#
# else:
#     print("❌ 请先运行前面的cell进行数据统计")
#
#
# # 🎭 电影类型分析 - 第5步：业务洞察
# if 'genre_counts' in locals():
#     print("🎯 电影类型分析洞察:")
#     print("=" * 40)
#
#     # 基础排名信息
#     print("🏆 类型受欢迎程度排名:")
#     print(f"• 🥇 最受欢迎类型: {genre_counts.index[0]} ({genre_counts.iloc[0]} 部)")
#     print(f"• 🥈 第二受欢迎: {genre_counts.index[1]} ({genre_counts.iloc[1]} 部)")
#     print(f"• 🥉 第三受欢迎: {genre_counts.index[2]} ({genre_counts.iloc[2]} 部)")
#
#     # 市场集中度分析
#     top3_percentage = (genre_counts.iloc[:3].sum() / genre_counts.sum()) * 100
#     print(f"\n📊 市场集中度分析:")
#     print(f"• 🔝 前三类型占总数的 {top3_percentage:.1f}%")
#
#     # 长尾效应分析
#     bottom_half = len(genre_counts) // 2
#     tail_percentage = (genre_counts.iloc[bottom_half:].sum() / genre_counts.sum()) * 100
#     print(f"• 📉 后半部分类型占 {tail_percentage:.1f}%（长尾效应）")
#
#     # 业务建议
#     print(f"\n💡 业务建议:")
#     if top3_percentage > 60:
#         print("• 市场集中度较高，建议重点关注头部类型")
#
#     if genre_counts.iloc[0] / genre_counts.iloc[1] > 1.5:
#         print(f"• {genre_counts.index[0]}类型明显领先，可作为主打类型")
#
#     print(f"• 投资策略：优先考虑{genre_counts.index[0]}、{genre_counts.index[1]}、{genre_counts.index[2]}类型")
#     print(f"• 差异化机会：{genre_counts.index[-1]}、{genre_counts.index[-2]}类型竞争较少")
#
# else:
#     print("❌ 请先运行前面的cell进行数据统计")


# 导入词云相关库
from wordcloud import WordCloud
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt

print("✅ 词云库安装完成！")