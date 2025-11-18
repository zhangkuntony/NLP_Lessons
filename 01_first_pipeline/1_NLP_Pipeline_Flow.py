# 创建简洁明了的NLP Pipeline流程图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 设置字体（使用英文避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(15, 8))

# 定义9个核心步骤（使用英文标题避免字体问题）
steps = [
    "1. Problem\nDefinition",
    "2. Data\nAcquisition",
    "3. Data\nExploration",
    "4. Data\nCleaning",
    "5. Data\nSplitting",
    "6. Feature\nEngineering",
    "7. Model\nTraining",
    "8. Model\nEvaluation",
    "9. Model\nInference"
]

# 定义位置（3行3列布局）
positions = [
    (2, 7), (5, 7), (8, 7),         # 第一行
    (2, 4), (5, 4), (8, 4),         # 第二行
    (2, 1), (5, 1), (8, 1)          # 第三行
]

# 定义颜色
colors = ['#FFE5B4', '#B4E5FF', '#C8E6C8', '#FFB4B4', '#E6C8FF',
          '#FFD700', '#98FB98', '#FFA07A', '#DDA0DD']

# 绘制流程框
for i, (step, pos, color) in enumerate(zip(steps, positions, colors)):
    box = FancyBboxPatch((pos[0] - 0.8, pos[1] - 0.4), 1.6, 0.8,
                         boxstyle='round,pad=0.05',
                         facecolor=color,edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(pos[0], pos[1],step, ha='center', va='center', fontsize=12, weight='bold')

# 绘制箭头连接
arrow_props = dict(arrowstyle='->', lw=3, color='#2E8B57')

# 水平箭头
connections = [
    (0, 1), (1, 2),         # 第一行
    (3, 4), (4, 5),         # 第二行
    (6, 7), (7, 8)          # 第三行
]

# 垂直箭头
vertical_connections = [
    (2, 5), (5, 8),         # 从第一行到第二行，第二行到第三行
    (0, 3), (3, 6)          # 左侧垂直连接
]

# 绘制水平箭头
for start, end in connections:
    start_pos = positions[start]
    end_pos = positions[end]
    ax.annotate('', xy=(end_pos[0] - 0.8, end_pos[1]),
                xytext=(start_pos[0] + 0.8, start_pos[1]),
                arrowprops=arrow_props)

# 绘制垂直箭头
for start, end in vertical_connections:
    start_pos = positions[start]
    end_pos = positions[end]
    ax.annotate('', xy=(end_pos[0], end_pos[1] + 0.4),
                xytext=(start_pos[0], start_pos[1] - 0.4),
                arrowprops=arrow_props)

# 添加反馈循环箭头（从评估回到特征工程）
ax.annotate('', xy=(positions[5][0] + 0.5, positions[5][1] + 0.2),
            xytext=(positions[7][0] - 0.5, positions[7][1] + 0.2),
            arrowprops=dict(arrowstyle='->', lw=2, color='red',
                            connectionstyle='arc3,rad=0.3'))

ax.text(6.5, 2.5, 'Feedback Loop', ha='center', va='center',
        fontsize=10, color='red', weight='bold')

# 设置图形属性
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('NLP Project Pipeline - 9 Key Steps', fontsize=18, weight='bold', pad=20)

# 添加说明文字
ax.text(5, 0.2, 'Note: These steps may require multiple iterations and optimization in real projects',
        ha='center', va='center', fontsize=11, style='italic', color='gray')

plt.tight_layout()
plt.show()

print("🎯 智能客服机器人项目示例：")
print("1. 问题定义：自动回答用户常见问题")
print("2. 数据获取：收集客服对话记录")
print("3. 数据探索：分析对话长度、问题类型等")
print("4. 数据清理：去除无关信息，统一格式")
print("5. 数据分割：70%训练，30%测试")
print("6. 特征工程：提取关键词、意图特征")
print("7. 建模：训练分类模型")
print("8. 评估：测试准确率、响应速度")
print("9. 推理：部署到线上，实时回答用户问题")
