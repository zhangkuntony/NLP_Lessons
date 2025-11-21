import requests
from bs4 import BeautifulSoup

# 示例1. 从网页中抽取文本（带错误处理）
def extract_text_from_url(url, timeout=10):
    """从URL提取文本内容"""
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # 检测并设置编码
        response.encoding = response.apparent_encoding
        
        soup_text = BeautifulSoup(response.text, 'html.parser')

        # 移除脚本和样式标签
        for script in soup_text(["script", "style"]):
            script.decompose()

        # 提取文本
        text_from_url = soup_text.get_text()

        # 清理空白字符
        lines = (line.strip() for line in text_from_url.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_from_url = ' '.join(chunk for chunk in chunks if chunk)

        return text_from_url

    except requests.RequestException as e:
        print(f"网络请求失败：{e}")
        return None

print("从网页提取文字：")
# web_page_text = extract_text_from_url("https://export.shobserver.com/baijiahao/html/1022224.html", 30)
web_page_text = extract_text_from_url("https://finance.sina.com.cn/money/nmetal/roll/2025-11-20/doc-infxzkva3855674.shtml", 30)
print(web_page_text)

# 示例2. 使用本地HTML文件（更稳定的示例）
sample_html = """
<html>
<head><title>示例网页</title></head>
<body>
    <h1>自然语言处理教程</h1>
    <p>这是一个关于NLP的教程，涵盖了从数据预处理到模型部署的完整流程。</p>
    <p>在实际项目中，我们经常需要从网页、文档等多种来源提取文本信息。</p>
    <div class="ad">这是广告内容，通常需要过滤掉</div>
    <p>文本清理是NLP pipeline中的重要环节。</p>
</body>
</html>
"""

soup = BeautifulSoup(sample_html, 'html.parser')

# 移除广告等无关内容
for ad in soup.find_all('div', class_='ad'):
    ad.decompose()

# 提取段落文本
paragraphs = soup.find_all('p')
print("提取到的段落文本")
for i, p in enumerate(paragraphs, 1):
    print(f"{i}. {p.get_text()}")

# 提取所有文本
all_text = soup.get_text()
# 手动清理多余的空白字符
all_text = ' '.join(all_text.split())
print(f"\n完整文本：\n{all_text}")

# Unicode 标准化
text = "I feel really 😡. GOGOGO!! 💪💪💪  🤣🤣 ȀÆĎǦƓ"
print(text)
text2 = text.encode("utf-8")  # encode the strings in bytes
print(text2)

# 分段和分词
# 需要先下载NLTK数据
import nltk
nltk.download('punkt_tab')  # 取消注释以下载分词模型

from nltk.tokenize import sent_tokenize, word_tokenize

# 英文文本示例
english_text = """
Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
"""

print("=== 英文文本处理 ===")
## 句子分割
sents = sent_tokenize(english_text)

## 词汇分割
for i, sent in enumerate(sents, 1):
    print(f"句子 {i}: {sent.strip()}")
    print(f"分词结果: {word_tokenize(sent)}")
    print()

# 中文文本处理示例
print("=== 中文文本处理 ===")
chinese_text = "自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。中文分词是中文NLP的基础任务。"

# 方法1：简单的中文句子分割
chinese_sentences = chinese_text.split('。')
chinese_sentences = [s.strip() for s in chinese_sentences if s.strip()]

print("中文句子分割:")
for i, sent in enumerate(chinese_sentences, 1):
    print(f"句子 {i}: {sent}")

print("\n中文字符级分割:")
print(list(chinese_text))

# 推荐使用专门的中文分词工具
print("\n注意：中文分词建议使用专门工具如jieba、pkuseg等")
print("示例：pip install jieba")

import jieba
try:
    print("jieba分词结果:")
    words = jieba.lcut(chinese_text)
    print(words)
except ImportError:
    print("jieba未安装，可以运行: pip install jieba")
