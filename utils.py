import re

# 清理文本：去除标点并小写（英文适用，中文已分词）
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点
    return text.lower()

def read_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def write_file(file_path, content: str, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)