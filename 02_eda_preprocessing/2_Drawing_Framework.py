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
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
fig.suptitle('豆瓣电影数据分析 - 面向对象接口演示',
             fontsize=16, fontweight='bold')

# 自动调整布局
plt.tight_layout()
plt.show()

print("\n✅ 面向对象接口特点：")
print("• 明确区分Figure（画布）和Axes（绘图区）")
print("• 每个Axes独立控制，便于复杂布局")
print("• 代码结构清晰，便于调试和维护")
print("• 这是Matplotlib的推荐使用方式！")
