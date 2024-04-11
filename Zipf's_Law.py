import os
import jieba
import collections
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# 加载停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords


# 从文件夹中加载合并的文本内容
def load_text_from_folder(folder_path):
    total_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='ansi') as file:
                total_text += file.read()
    return total_text


# 分词、删除停用词并计算词频
def calculate_word_frequencies(text, stopwords):
    # 使用jieba进行分词
    words = jieba.lcut(text)
    # 删除停用词
    words = [word for word in words if word not in stopwords]
    # 使用collections.Counter计算词频
    word_frequencies = collections.Counter(words)
    return word_frequencies


# 绘制词频排名图并进行幂律拟合
def plot_zipf_law(word_frequencies):
    # 按照频率从高到低排序
    sorted_frequencies = sorted(word_frequencies.values(), reverse=True)
    # 计算排名（从1开始）
    ranks = np.arange(1, len(sorted_frequencies) + 1)
    # 转换为对数坐标轴
    log_ranks = np.log10(ranks)
    log_frequencies = np.log10(sorted_frequencies)

    # 绘制词频排名图
    plt.figure()
    plt.plot(log_ranks, log_frequencies, 'o', label='Data')
    plt.xlabel('Log10 Rank')
    plt.ylabel('Log10 Frequency')
    plt.title('Zipf\'s Law')

    # 进行幂律拟合（线性回归）
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_frequencies)
    plt.plot(log_ranks, slope * log_ranks + intercept, 'r-', label=f'Fit: slope={slope:.2f}')

    plt.legend()
    plt.show()

    # 输出幂律指数和拟合的r平方值
    print(f'幂律指数（Slope）: {slope}')
    print(f'拟合的R²值: {r_value ** 2}')


if __name__ == '__main__':
    # 设置文件夹路径和停用词列表路径
    folder_path = 'jyxstxtqj_downcc.com'  # 请替换为文件夹路径
    stopwords_file_path = '停词表.txt'  # 请替换为停用词文件路径

    # 加载停用词列表
    stopwords = load_stopwords(stopwords_file_path)

    # 从文件夹中读取文本并计算词频
    text = load_text_from_folder(folder_path)
    word_frequencies = calculate_word_frequencies(text, stopwords)

    # 绘制Zipf定律图并进行幂律拟合
    plot_zipf_law(word_frequencies)
