"""
人民日报数据预处理

数据已经被初步处理了。现在数据格式为:
    中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O 书/O 记/O 、/O
    国/O 家/O 主/O 席/O 江/B_nr 泽/M_nr 民/E_nr

里面的实体有：人名、地名、机构名称
    人名： B_nr M_nr E_nr
    地名： B_ns M_ns E_ns
    机构名称: B_nt M_nt E_nt

由于这个数据的处理需要处理成:
人名：
    B_nr -> B-PER
    M_nr -> I-PER
    E_nr -> I-PER

地名：
    B_ns -> B-LOC
    M_ns -> I-LOC
    E_ns -> I-LOC

机构名称:
    B_nt -> B-ORG
    M_nt -> I-ORG
    E_nt -> I-ORG


将renmin3.txt 文件分割成两份分别为训练集和测试集，分割比例按照8:2进行分割。
分割后存储名称为: train.txt   test.txt

存储格式:(每个字符和标志为一行，字符和标志空格符号隔开。每一句用回车隔开)
char tag
树 O
立 O
企 O
业 O
形 O
象 O
的 O
需 O
要 O

大 B-ORG
连 I-ORG
海 I-ORG
富 I-ORG
集 I-ORG
团 I-ORG

"""

from Public.path import path_renmin_dir
import os


# 读取数据
def read_file(file_path: str) -> [str]:
    with open(file_path, 'r') as f:
        texts = f.read().split('\n')
    return texts


# 文本映射
def text_map(texts: [str]) -> [str]:
    """
    文本映射处理
    处理好的数据格式:
       ['需 O'
        '要 O'
        '大 B-ORG'
        '连 I-ORG'
        '海 I-ORG'
        '富 I-ORG'
        '集 I-ORG'
        '团 I-ORG']

    :param texts:  例如 中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O  的文本
    :return: [str] 处理好的数据
    """
    mapping = {'O': 'O',
               'B_nr': 'B-PER',
               'M_nr': 'I-PER',
               'E_nr': 'I-PER',
               'B_ns': 'B-LOC',
               'M_ns': 'I-LOC',
               'E_ns': 'I-LOC',
               'B_nt': 'B-ORG',
               'M_nt': 'I-ORG',
               'E_nt': 'I-ORG'
              }
    deal_texts = []
    for line in texts:
        sub_line = str(line).split(' ')
        for item in sub_line:
            item_list = str(item).split('/')
            if len(item_list) == 2:
                a = item_list[0]
                b = item_list[1]
                flag = mapping.get(b, 'O')
                deal_texts.append(f"{a} {flag}\n")
        deal_texts.append('\n')
    return deal_texts


# 数据预处理
def renminribao_preprocessing(split_rate: float = 0.8,
                              ignore_exist: bool = False) -> None:
    """
    人名日报数据预处理
    :param split_rate:数据分割比例，默认为 0.8
    :param ignore_exist 忽略已经存储的数据，则不需要在判断以及存好的数据
    :return:None
    """
    path_train = os.path.join(path_renmin_dir, "train.txt")
    path_test = os.path.join(path_renmin_dir, "test.txt")
    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        print("人民日报数据预处理已经完成")
        return
    else:
        print("正在对人民日报数据进行预处理......")
    path_org = os.path.join(path_renmin_dir, "renmin3.txt")
    texts = read_file(path_org)

    # 对数据进行分割处理
    if split_rate >= 1.0:  # 分割比例不得大于1.0
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # 对数据进行映射处理
    test_texts = text_map(test_texts)
    train_texts = text_map(train_texts)

    # 数据写入到本地文件
    with open(path_train, 'w') as f:
        f.write("".join(train_texts))
    with open(path_test, 'w') as f:
        f.write("".join(test_texts))
    print("人民日报数据进行预处理完成 ---- OK!")


if __name__ == '__main__':
    renminribao_preprocessing()




