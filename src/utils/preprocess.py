import numpy as np


def parse_fasta(dir, number=None):
    """解析 FASTA 文件（带标签）"""
    print("开始解析 FASTA 文件（带标签）：", dir)
    names, sequences, labels = [], [], []
    if number is None:
        number = -1
    with open(dir, 'r') as f:
        data = f.readlines()
        for i in range(0, len(data[:number]), 2):
            line = data[i]
            if line.startswith('>'):
                # 提取标签（'|' 后的部分）
                label = int(line.split('|')[-1])
                labels.append(label)
                # 提取名称，并去掉 '|label' 部分
                name = line.strip()[1:].split('|')[0]
                names.append(name)
                sequences.append(data[i + 1].strip())
    print(f"解析完成，读取 {len(names)} 条序列")
    return np.array(names), np.array(sequences), np.array(labels)


def parse_fasta_nolabel(dir, number=None):
    """解析 FASTA 文件（预测模式，无标签）"""
    print("开始解析 FASTA 文件（预测模式）：", dir)
    names, sequences = [], []
    if number is None:
        number = -1
    with open(dir, 'r') as f:
        data = f.readlines()
        for i in range(0, len(data[:number]), 2):
            line = data[i]
            if line.startswith('>'):
                names.append(line.strip()[1:])
                sequences.append(data[i + 1].strip())
    print(f"解析完成，读取 {len(names)} 条序列")
    return np.array(names), np.array(sequences)
