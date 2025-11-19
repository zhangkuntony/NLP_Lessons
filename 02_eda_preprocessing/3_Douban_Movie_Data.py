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

# 🎨 示例1：创建一些模拟数据来演示seaborn
print("🎬 创建一些模拟的电影评分数据来演示...")

# 创建模拟数据：不同类型电影的评分
np.random.seed(42)  # 保证结果可重复
movie_data = {
    '电影类型': ['动作'] * 100 + ['喜剧'] * 100 + ['爱情'] * 100 + ['科幻'] * 100,
    '评分': (
        np.random.normal(7.5, 1.2, 100).tolist() +  # 动作片评分
        np.random.normal(8.0, 1.0, 100).tolist() +  # 喜剧片评分
        np.random.normal(7.8, 1.1, 100).tolist() +  # 爱情片评分
        np.random.normal(7.2, 1.3, 100).tolist()    # 科幻片评分
    ),
    '票房': np.random.lognormal(2, 1, 400),  # 票房数据（对数正态分布）
}

demo_df = pd.DataFrame(movie_data)
print(f"✅ 创建了 {len(demo_df)} 条模拟电影数据")
print("前5行数据预览：")
print(demo_df.head())

# 🆚 示例2：对比matplotlib vs seaborn
print("🆚 对比 matplotlib 和 seaborn 的区别")

# 创建对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('📊 Matplotlib vs Seaborn 对比展示', fontsize=16, fontweight='bold')

# matplotlib版本 - 直方图
axes[0,0].hist(demo_df['评分'], bins=20, alpha=0.7, color='blue')
axes[0,0].set_title('Matplotlib 直方图')
axes[0,0].set_xlabel('评分')
axes[0,0].set_ylabel('频次')

# seaborn版本 - 直方图
sns.histplot(data=demo_df, x='评分', bins=20, ax=axes[0,1])
axes[0,1].set_title('Seaborn 直方图（更美观）')

# matplotlib版本 - 箱线图
box_data = [demo_df[demo_df['电影类型']==genre]['评分'].values
           for genre in demo_df['电影类型'].unique()]
axes[1,0].boxplot(box_data, labels=demo_df['电影类型'].unique())
axes[1,0].set_title('Matplotlib 箱线图')
axes[1,0].set_ylabel('评分')

# seaborn版本 - 箱线图
sns.boxplot(data=demo_df, x='电影类型', y='评分', ax=axes[1,1])
axes[1,1].set_title('Seaborn 箱线图（一行代码！）')

plt.tight_layout()
plt.show()

print("🎯 对比总结：")
print("• Matplotlib：功能强大但需要更多代码")
print("• Seaborn：简洁美观，自动处理分类数据")


# ✨ 示例3：Seaborn的"超能力"展示
print("✨ Seaborn的特色功能展示")

# 创建展示图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🌟 Seaborn 特色功能展示', fontsize=16, fontweight='bold')

# 1. 散点图 + 回归线（一行代码完成）
sns.scatterplot(data=demo_df, x='票房', y='评分', hue='电影类型', ax=axes[0,0])
axes[0,0].set_title('🎯 散点图：票房 vs 评分')
axes[0,0].set_xlabel('票房（万元）')

# 2. 小提琴图（显示分布形状）
sns.violinplot(data=demo_df, x='电影类型', y='评分', ax=axes[0,1])
axes[0,1].set_title('🎻 小提琴图：评分分布')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. 计数图（柱状图的升级版）
sns.countplot(data=demo_df, x='电影类型', ax=axes[1,0])
axes[1,0].set_title('📊 计数图：各类型电影数量')

# 4. 热力图（显示相关性）
# 创建相关性矩阵
corr_data = demo_df[['评分', '票房']].corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
            square=True, ax=axes[1,1])
axes[1,1].set_title('🔥 热力图：相关性分析')

plt.tight_layout()
plt.show()

print("\n🎓 每种图表的作用：")
print("• 📈 散点图：发现两个变量之间的关系")
print("• 🎻 小提琴图：比箱线图更详细地显示数据分布")
print("• 📊 计数图：统计各类别的数量，比普通柱状图更智能")
print("• 🔥 热力图：可视化数字之间的相关性，颜色越深关系越强")
print("箱线图作用")

# 使用seaborn的regplot或lmplot
sns.regplot(data=demo_df, x='评分', y='票房', scatter_kws={'alpha':0.6})
plt.title('电影评分与票房关系')
plt.xlabel('评分')
plt.ylabel('票房')
plt.show()