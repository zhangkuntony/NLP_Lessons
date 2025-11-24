# 🛠️ 准备工作环境
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')  # 先用默认样式，方便对比
warnings.filterwarnings('ignore')

# 创建示例数据 - 豆瓣电影评分数据
np.random.seed(42)
movie_data = {
    'movie_id': range(1, 301),
    'rating': np.random.normal(7.5, 1.5, 300),
    'genre': np.random.choice(['动作', '喜剧', '爱情', '科幻', '悬疑'], 300),
    'year': np.random.choice(range(2000, 2024), 300),
    'comments_count': np.random.exponential(100, 300).astype(int),
    'box_office': np.random.lognormal(3, 1, 300)
}

df = pd.DataFrame(movie_data)
# 确保评分在合理范围内
df['rating'] = np.clip(df['rating'], 1, 10)

print("📊 豆瓣电影示例数据创建完成！")
print(f"数据形状: {df.shape}")
print("\n前5行数据:")
print(df.head())

# 🎨 实战演示1：Pyplot接口（传统方法）

print("🎯 演示1：Pyplot接口 - 简单直接的绘图方式")

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建画布和子图布局（1行3列）
plt.figure(figsize=(15, 5))

# 子图1：评分分布直方图
plt.subplot(1, 3, 1)  # 第1行第3列的第1个位置
plt.hist(df['rating'], bins=20, alpha=0.7, color='skyblue')
plt.title('评分分布')
plt.xlabel('评分')
plt.ylabel('频次')

# 子图2：票房vs评分散点图
plt.subplot(1, 3, 2)  # 第1行第3列的第2个位置
plt.scatter(df['box_office'], df['rating'], alpha=0.6)
plt.title('票房vs评分')
plt.xlabel('票房')
plt.ylabel('评分')

# 子图3：电影类型分布柱状图
plt.subplot(1, 3, 3)  # 第1行第3列的第3个位置
genre_counts = df['genre'].value_counts()  # 统计各类型的数量
plt.bar(genre_counts.index, genre_counts.values)
plt.title('类型分布')
plt.xlabel('电影类型')
plt.ylabel('数量')
plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

# 自动调整子图间距，避免重叠
plt.tight_layout()
plt.show()

print("\n✅ Pyplot接口特点：")
print("• 简单直接，类似MATLAB语法")
print("• 适合快速探索和原型制作")
print("• 但在复杂布局时控制力有限")


# 🎨 实战演示2：面向对象接口（推荐方法）

print("🎯 演示2：面向对象接口 - 专业级的绘图方式")

# 创建Figure和Axes对象（这是关键区别！）
fig_01, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig是整个画布，axes是包含3个子图的数组

# 子图1：评分分布直方图
axes[0].hist(df['rating'], bins=20, alpha=0.7, color='lightcoral')
axes[0].set_title('评分分布')           # 注意：用set_title而不是title
axes[0].set_xlabel('评分')            # 注意：用set_xlabel而不是xlabel
axes[0].set_ylabel('频次')            # 注意：用set_ylabel而不是ylabel
axes[0].grid(True, alpha=0.3)           # 添加网格，alpha控制透明度

# 子图2：票房vs评分散点图
axes[1].scatter(df['box_office'], df['rating'], alpha=0.6, color='lightgreen')
axes[1].set_title('票房vs评分')
axes[1].set_xlabel('票房')
axes[1].set_ylabel('评分')
axes[1].grid(True, alpha=0.3)

# 子图3：电影类型分布柱状图
axes[2].bar(genre_counts.index, genre_counts.values, color='gold')
axes[2].set_title('类型分布')
axes[2].set_xlabel('电影类型')
axes[2].set_ylabel('数量')
axes[2].tick_params(axis='x', rotation=45)  # 旋转x轴标签
axes[2].grid(True, alpha=0.3)

# 设置整个Figure的标题（这是Figure级别的操作）
fig_01.suptitle('豆瓣电影数据分析 - 面向对象接口演示',
                fontsize=16, fontweight='bold')

# 自动调整布局
plt.tight_layout()
plt.show()

print("\n✅ 面向对象接口特点：")
print("• 明确区分Figure（画布）和Axes（绘图区）")
print("• 每个Axes独立控制，便于复杂布局")
print("• 代码结构清晰，便于调试和维护")
print("• 这是Matplotlib的推荐使用方式！")


# 🎨 Artist实战1：基础图表和获取Artist对象

print("🎯 步骤1：创建基础图表并获取Artist对象")

# 创建图表
fig_01, ax = plt.subplots(figsize=(10, 6))

# 绘制散点图并获取返回的Artist对象
scatter_artist = ax.scatter(df['box_office'], df['rating'],
                            c=df['comments_count'],                 # 颜色映射到评论数
                            s=60,                                   # 点的大小
                            alpha=0.7,                              # 透明度
                            cmap='viridis',                          # 颜色图
                            edgecolors='black',                     # 边框颜色
                            linewidth=0.5)                          # 边框宽度

print(f"✅ 获得的Artist对象类型: {type(scatter_artist)}")
print(f"✅ 这个对象包含 {len(df)} 个数据点")

# 基础的标签设置
ax.set_xlabel('票房（万元）')
ax.set_ylabel('评分')
ax.set_title('豆瓣电影：票房 vs 评分')

plt.tight_layout()
plt.show()

print("\n💡 重要概念：")
print("• scatter() 返回的是一个 PathCollection Artist对象")
print("• 这个对象包含了所有散点的信息")
print("• 我们可以通过这个对象来修改所有点的属性")


# 🎨 Artist实战2：精确控制图表样式

print("🎯 步骤2：通过Artist对象精确控制图表外观")

# 创建专业级图表
fig_02, ax = plt.subplots(figsize=(12, 8))

# 绘制散点图
scatter = ax.scatter(df['box_office'], df['rating'],
                     c=df['comments_count'], s=60, alpha=0.7,
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# 🎨 样式1：美化坐标轴标签
ax.set_xlabel('票房（万元）', fontsize=14, fontweight='bold', color='darkblue')
ax.set_ylabel('评分', fontsize=14, fontweight='bold', color='darkblue')
ax.set_title('🎬 豆瓣电影：票房 vs 评分 vs 评论数',
             fontsize=16, fontweight='bold', pad=20)

# 🎨 样式2：自定义网格和背景
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_facecolor('#f8f9fa')            # 设置背景色为浅灰

# 🎨 样式3：添加颜色条（这也是一个Artist对象！）
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('评论数量', fontsize=12, fontweight='bold')

# 🎨 样式4：设置坐标轴范围
ax.set_xlim(0, df['box_office'].max() * 1.1)
ax.set_ylim(0, 10.5)

# 🎨 样式5：自定义边框（spines也是Artist对象）
for spine in ax.spines.values():
    spine.set_linewidth(2)                      # 边框粗细
    spine.set_edgecolor('darkgray')             # 边框颜色

plt.tight_layout()
plt.show()

print("\n✅ 我们控制了哪些Artist对象？")
print("• Axes对象：背景色、网格、坐标轴标签")
print("• PathCollection对象：散点的样式")
print("• Colorbar对象：颜色条的标签")
print("• Spine对象：图表边框的样式")
print("• Text对象：标题和轴标签的字体样式")


# 🎨 Artist实战3：添加复杂注释和装饰

print("🎯 步骤3：添加注释文字和装饰元素")

# 创建图表
fig_03, ax = plt.subplots(figsize=(12, 8))

# 绘制基础散点图
scatter = ax.scatter(df['box_office'], df['rating'],
                     c=df['comments_count'], s=60, alpha=0.7,
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# 基础样式设置
ax.set_xlabel('票房 (万元)', fontsize=14, fontweight='bold')
ax.set_ylabel('评分', fontsize=14, fontweight='bold')
ax.set_title('豆瓣电影分析：重点标注版', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# 🎯 重点功能：添加注释
# 找到评分最高的电影
best_movie_idx = df['rating'].idxmax()
best_movie = df.loc[best_movie_idx]

# 创建箭头注释（这会创建Annotation Artist对象）
annotation = ax.annotate(
    f'最高评分电影\n评分: {best_movie["rating"]:.1f}分',                     # 注释文字
    xy=(best_movie['box_office'], best_movie['rating']),                    # 箭头指向的点
    xytext=(best_movie['box_office'] + 20, best_movie['rating'] + 0.8),     # 文字位置
    arrowprops=dict(
        arrowstyle='->',                    # 箭头样式
        color='red',                        # 箭头颜色
        lw=2,                               # 箭头粗细
        connectionstyle="arc3,rad=0.2"      # 箭头弯曲
    ),
    fontsize=12,
    fontweight='bold',
    color='red',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8)      # 文字框
)

# 添加数据来源标注（这会创建Text Artist对象）
source_text = ax.text(0.02, 0.02, '数据来源：豆瓣电影（模拟数据）',
                      transform=ax.transAxes,               # 使用相对坐标
                      fontsize=10,
                      alpha=0.7,
                      style='italic'
                      )

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('评论数量', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ 我们添加了哪些新的Artist对象？")
print("• Annotation对象：带箭头的注释文字")
print("• Text对象：数据来源说明")
print("• FancyBboxPatch对象：文字背景框")
print("• Arrow对象：指向特定数据点的箭头")

print("\n💡 Artist对象的威力：")
print("• 每个元素都可以独立控制和修改")
print("• 可以创建任意复杂的图表装饰")
print("• 这就是matplotlib如此强大的原因！")


# 🎨 Grammar of Graphics实战1：基础映射

print("🎯 演示：数据到视觉的逐步映射过程")

# 设置seaborn样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 步骤1：最基础的映射 - 只有位置
print("\n📍 步骤1：基础位置映射")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='box_office', y='rating')
plt.title('基础映射：票房（x）->评分（y）')
plt.show()

print("✅ 这里我们把：")
print("• 票房数据 → 映射到 → x轴位置")
print("• 评分数据 → 映射到 → y轴位置")
print("• 每个电影 → 映射到 → 一个点")


# 🎨 Grammar of Graphics实战2：添加颜色映射

print("\n🎨 步骤2：添加颜色维度")
plt.figure(figsize=(10, 6))

# 添加颜色映射：电影类型 → 颜色
sns.scatterplot(data=df, x='box_office', y='rating', hue='genre')
plt.title('颜色映射：电影类型 → 点的颜色')
plt.show()

print("✅ 现在我们添加了第三个维度：")
print("• 票房数据 → x轴位置")
print("• 评分数据 → y轴位置")
print("• 电影类型 → 点的颜色")
print("\n💡 观察：现在可以同时看到票房、评分、类型三个维度的信息！")


# 🎨 Grammar of Graphics实战3：添加大小映射

print("\n📏 步骤3：添加大小维度")
plt.figure(figsize=(12, 6))

# 添加大小映射：评论数量 → 点的大小
sns.scatterplot(data=df, x='box_office', y='rating',
               hue='genre', size='comments_count')
plt.title('大小映射：评论数量 → 点的大小')
plt.show()

print("✅ 现在我们有了四个维度：")
print("• 票房数据 → x轴位置")
print("• 评分数据 → y轴位置")
print("• 电影类型 → 点的颜色")
print("• 评论数量 → 点的大小")
print("\n💡 观察：一个图表现在包含了四个维度的信息！")
print("• 大点 = 评论多（热门）")
print("• 小点 = 评论少（冷门）")


# 🎨 Grammar of Graphics实战4：添加统计变换

print("\n📈 步骤4：添加统计变换（回归线）")
plt.figure(figsize=(12, 6))

# 绘制散点图
sns.scatterplot(data=df, x='box_office', y='rating',
               hue='genre', size='comments_count', alpha=0.7)

# 添加整体回归线（统计变换）
sns.regplot(data=df, x='box_office', y='rating',
           scatter=False,  # 不画散点，只画回归线
           color='red',
           line_kws={'linewidth': 3, 'alpha': 0.8})

plt.title('统计变换：添加回归线显示整体趋势')
plt.show()

print("✅ 现在我们有了：")
print("• 四个数据维度的视觉映射")
print("• 一个统计变换（回归线）")
print("\n💡 Grammar of Graphics的威力：")
print("• 数据 + 映射 + 几何对象 + 统计变换 = 丰富洞察")
print("• 相同数据通过不同映射可以发现不同模式")
print("• Seaborn让复杂的统计可视化变得简单！")


# 🎨 实战演示：Seaborn三层架构对比

print("🎯 演示：Seaborn三层架构的不同用法")

# 1. Figure-level functions (最简单)
print("\n1️⃣ Figure-level functions: 自动分面")
g1 = sns.relplot(data=df, x='box_office', y='rating',
                 col='genre', col_wrap=3,
                 height=4, aspect=0.8)
g1.figure.suptitle('Figure-level: 按电影类型自动分面', y=1.02)
plt.show()

# 2. Axes-level functions （平衡）
print("\n2️⃣ Axes-level functions: 手动控制布局")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Axes-level: 手动控制每个子图', fontsize=14)

genres = df['genre'].unique()
for i, genre in enumerate(genres):
    if i < 6:           # 只显示前6个
        row, col = i // 3, i % 3
        genre_data = df[df['genre'] == genre]

        # 使用axes-level函数
        sns.scatterplot(data=genre_data, x='box_office', y='rating',
                        ax=axes[row, col], color=sns.color_palette()[i])
        axes[row, col].set_title(f'{genre}电影')
        axes[row, col].grid(True, alpha=0.3)

# 隐藏空的子图
if len(genres) < 6:
    for i in range(len(genres), 6):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)

plt.tight_layout()
plt.show()


# 3. Grid objects (最复杂但最灵活)
print("\n3️⃣ Grid objects: 高级自定义")
# 创建自定义网格
g3 = sns.FacetGrid(df, col='genre', col_wrap=3, height=4, aspect=0.8)

# 自定义绘图函数
def custom_plot(x, y, **kwargs):
    plt.scatter(x, y, alpha=0.6, **kwargs)
    # 添加趋势线
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)

# 应用自定义函数
g3.map(custom_plot, 'box_office', 'rating')
g3.add_legend()
g3.figure.suptitle('Grid objects: 自定义绘图函数', y=1.02)
plt.show()

print("\n🎓 三层架构选择指南：")
print("• Figure-level: 快速探索，一行代码搞定分面")
print("• Axes-level: 需要精确控制布局时使用")
print("• Grid objects: 复杂的自定义可视化需求")


# 🎯 实战对比：快速探索 vs 精确控制

print("📊 场景1：快速数据探索")
print("任务：了解所有变量之间的关系")

# Seaborn: 一行代码的威力
print("\n🌟 Seaborn解决方案：")
# 注意：pairplot可能比较耗时，我们用简化版本
sns.pairplot(df[['rating', 'box_office', 'comments_count', 'genre']],
             hue='genre', diag_kind='hist', height=2.5)
plt.show()

print("\n🎓 Seaborn优势：")
print("• 一行代码生成复杂的多面板图表")
print("• 自动处理分类变量的颜色映射")
print("• 默认样式美观专业")

print("\n" + "="*60)
print("📊 场景2：精确控制图表细节")
print("任务：创建符合论文发表标准的图表")

# Matplotlib: 精确控制的威力
print("\n🥊 Matplotlib解决方案：")
fig, ax = plt.subplots(figsize=(10, 6))

# 精确控制每个细节
scatter = ax.scatter(df['box_office'], df['rating'],
                     c=df['comments_count'], s=50, alpha=0.7,
                     cmap='viridis', edgecolors='black', linewidth=0.5)

# 专业的坐标轴设置
ax.set_xlabel('Box Office (Million RMB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Rating Score', fontsize=12, fontweight='bold')
ax.set_title('Relationship between Box Office and Rating\nof Douban Movies',
             fontsize=14, fontweight='bold', pad=20)

# 自定义刻度
ax.set_xlim(0, df['box_office'].max() * 1.05)
ax.set_ylim(0, 10.5)
ax.grid(True, alpha=0.3, linestyle='--')

# 专业的颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Comments', fontsize=11, fontweight='bold')

# 添加统计信息
correlation = df['box_office'].corr(df['rating'])
ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print("\n🎓 Matplotlib优势：")
print("• 完全控制图表的每一个像素")
print("• 符合学术出版的精确标准")
print("• 可以实现任何想象得到的视觉效果")


# 📝 实用模板1：数据探索万能函数

print("🎯 模板1：一键数据探索函数 - 拿来即用！")

def explore_data(data_frame):
    """
    数据探索万能函数 - 自动分析数据特征并生成可视化

    参数说明:
    df: pandas.DataFrame - 要分析的数据集
    target_col: str - 目标变量列名（可选）

    功能:
    - 自动识别数据类型
    - 生成相关性热力图
    - 针对目标变量进行深入分析
    """
    print("📊 数据探索报告")
    print("=" * 50)

    # 📋 第1步：数据概览
    print(f"📏 数据维度: {data_frame.shape[0]}行 × {data_frame.shape[1]}列")
    print(f"🔢 数值型变量: {data_frame.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"📝 分类型变量: {data_frame.select_dtypes(include=['object']).columns.tolist()}")
    print(f"❓ 缺失值总数: {data_frame.isnull().sum().sum()}")

    # 📈 第2步：数值变量相关性分析
    numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        print(f"\n🔥 发现{len(numeric_cols)}个数值变量，绘制相关性热力图...")

        plt.figure(figsize = (10, 8))
        correlation_matrix = data_frame[numeric_cols].corr()

        # 创建精美的热力图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))        # 上三角遮罩
        sns.heatmap(correlation_matrix,
                    mask=mask,                  # 只显示下三角
                    annot=True,                 # 显示相关系数
                    cmap="coolwarm",            # 冷暖色调
                    center=0,                   # 以0为中心
                    square=True,                # 正方形格子
                    fmt='.3f',                  # 保留2位小数
                    cbar_kws={"shrink": .8})    # 缩小颜色条

        plt.title('变量间相关性热力图', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️  数值变量不足2个，跳过相关性分析")

    return f"✅ 探索完成！共分析 {data_frame.shape[0]} 条记录的 {data_frame.shape[1]} 个变量"

# 🚀 实际使用演示
print("\n📊 使用万能探索函数分析豆瓣电影数据:")
result = explore_data(df)
print(f"\n{result}")

# 📝 实用模板2：专业报告图表函数

print("🎯 模板2：专业级图表生成器 - 一键创建发布级图表")


def create_professional_plot(data_frame, x_col, y_col, title="", save_path=None):
    """
    创建专业级别的图表，适合报告、演示和学术论文

    参数说明:
    df: pandas.DataFrame - 数据集
    x_col: str - x轴变量名
    y_col: str - y轴变量名
    title: str - 图表标题（可选）
    save_path: str - 保存路径（可选，高分辨率保存）

    特色功能:
    - 自动美化字体和样式
    - 智能图例位置
    - 数据源标注
    - 高分辨率保存选项
    """
    # 🎨 第1步：设置专业字体和样式
    plt.rcParams['font.family'] = ['Microsoft YaHei']  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示

    # 🎨 第2步：使用专业样式模板
    # with plt.style.context('seaborn-v0_8-whitegrid'):     # 注释掉这一行，解决图表中文显示的问题
    professional_figure, professional_ax = plt.subplots(figsize=(10, 6))

    # 🎨 第3步：绘制核心图表
    sns.scatterplot(data=data_frame, x=x_col, y=y_col,
                    hue='genre',  # 按类型着色
                    size='comments_count',  # 按评论数调整大小
                    alpha=0.7,  # 半透明效果
                    ax=professional_ax)

    # 🎨 第4步：专业化标签设置
    professional_ax.set_xlabel(x_col.replace('_', ' ').title(),
                               fontsize=12, fontweight='bold')
    professional_ax.set_ylabel(y_col.replace('_', ' ').title(),
                               fontsize=12, fontweight='bold')
    professional_ax.set_title(title or f'{y_col.title()} vs {x_col.title()}',
                              fontsize=14, fontweight='bold', pad=20)

    # 🎨 第5步：网格和图例优化
    professional_ax.grid(True, alpha=0.3)  # 淡网格
    professional_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放右侧

    # 🎨 第6步：添加数据源标注（专业习惯）
    professional_ax.text(0.02, 0.02, '数据源: 豆瓣电影 (模拟数据)',
                         transform=professional_ax.transAxes, fontsize=10, alpha=0.7, style='italic')

    plt.tight_layout()

    # 💾 第7步：可选的高质量保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"📁 高分辨率图表已保存: {save_path}")

    plt.show()

    return "✅ 专业图表创建完成！"


# 🚀 使用演示
print("\n🎨 创建专业级电影分析图表:")
result = create_professional_plot(df, 'box_office', 'rating',
                                  "豆瓣电影：票房与评分关系分析")
print(result)


# 📝 实用模板3：快速分组对比分析

print("🎯 模板3：多角度对比分析器 - 一键看透分组差异")

def quick_comparison(data_frame, group_col, value_col):
    """
    快速分组对比分析 - 从多个角度对比不同组别的数据分布

    参数说明:
    df: pandas.DataFrame - 数据集
    group_col: str - 分组变量（分类变量）
    value_col: str - 对比变量（数值变量）

    输出内容:
    - 箱线图：查看分布差异和异常值
    - 小提琴图：查看分布形状和密度
    - 柱状图：查看平均值排序
    - 统计摘要：详细数值对比
    """
    # 🎨 第1步：设置字体
    plt.rcParams['font.family'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 🎨 第2步：创建三合一布局
    quick_comparison_figure, quick_comparison_axes = plt.subplots(1, 3, figsize=(18, 5))
    quick_comparison_figure.suptitle(f'📊 {group_col} vs {value_col} 多角度对比分析',
                                     fontsize=16, fontweight='bold', y=1.05)

    # 📦 第3步：箱线图 - 看分布位置和离散程度
    sns.boxplot(data=data_frame, x=group_col, y=value_col, ax=quick_comparison_axes[0])
    quick_comparison_axes[0].set_title('📦 分布对比：中位数、四分位数、异常值', fontweight='bold')
    quick_comparison_axes[0].tick_params(axis='x', rotation=45)
    quick_comparison_axes[0].grid(True, alpha=0.3)

    # 🎻 第4步：小提琴图 - 看分布形状
    sns.violinplot(data=data_frame, x=group_col, y=value_col, ax=quick_comparison_axes[1])
    quick_comparison_axes[1].set_title('🎻 分布形状：密度和对称性', fontweight='bold')
    quick_comparison_axes[1].tick_params(axis='x', rotation=45)
    quick_comparison_axes[1].grid(True, alpha=0.3)

    # 📊 第5步：均值柱状图 - 看平均水平排序
    mean_data = data_frame.groupby(group_col)[value_col].mean().sort_values(ascending=False)
    sns.barplot(x=mean_data.index, y=mean_data.values, ax=quick_comparison_axes[2], palette='viridis')
    quick_comparison_axes[2].set_title('📊 平均值排序对比', fontweight='bold')
    quick_comparison_axes[2].tick_params(axis='x', rotation=45)
    quick_comparison_axes[2].set_ylabel(f'平均{value_col}')
    quick_comparison_axes[2].grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值标签
    for index, v in enumerate(mean_data.values):
        quick_comparison_axes[2].text(index, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 📈 第6步：输出详细统计摘要
    print("\n📈 详细统计摘要表：")
    print("="*80)
    summary = data_frame.groupby(group_col)[value_col].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    summary.columns = ['样本量', '均值', '中位数', '标准差', '最小值', '最大值']
    print(summary.round(2))

    # 💡 第7步：自动洞察提取
    print(f"\n💡 快速洞察：")
    best_group = mean_data.index[0]
    worst_group = mean_data.index[-1]
    print(f"• 📈 {value_col}最高的组别: {best_group} (平均{mean_data.iloc[0]:.2f})")
    print(f"• 📉 {value_col}最低的组别: {worst_group} (平均{mean_data.iloc[-1]:.2f})")
    print(f"• 📏 最大组间差异: {mean_data.iloc[0] - mean_data.iloc[-1]:.2f}")

    return summary

# 🚀 使用演示
print("\n🔍 分析不同电影类型的评分差异:")
summary_result = quick_comparison(df, 'genre', 'rating')
print(f"\n✅ 对比分析完成！共分析了 {len(summary_result)} 个组别")
