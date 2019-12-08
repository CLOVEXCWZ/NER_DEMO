"""
MSRA 数据集预处理

原数据格式:
    当/o 希望工程/o 救助/o 的/o 百万/o 儿童/o 成长/o 起来/o ，/o 科教/o 兴/o 国/o 蔚然成风/o 时/o ，/o 今天/o 有/o 收藏/o 价值/o 的/o 书/o 你/o 没/o 买/o ，/o 明日/o 就/o 叫/o 你/o 悔不当初/o ！/o

标记了3种实体:
    nr - 人名
    ns - 地名
    nt - 机构名

由于这个数据的处理需要处理成:
人名：
    nr -> B-PER、I-PER
地名：
    ns -> B-LOC、I-LOC
机构名称:
    nt -> B-ORG、I-ORG


将train1.txt 文件分割成两份分别为训练集和测试集，分割比例按照8:2(默认)进行分割。
分割后存储名称为: train.txt   test.txt

存储格式:(每个字符和标志为一行，字符和标志空格符号隔开。每一句用回车隔开)
char tag
另 O
一 O
方 O
面 O
还 O
要 O
看 O
到 O
， O
中 B-LOC
国 I-LOC

"""

from Public.path import path_msra_dir
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
    mapping = {
        'nr': 'PER',
        'ns': 'LOC',
        'nt': 'ORG'
    }
    deal_texts = []
    for line in texts:
        sub_line = str(line).split(' ')
        for item in sub_line:
            item_list = str(item).split('/')
            if len(item_list) == 2:
                a = item_list[0]
                b = item_list[1]
                if b in mapping:
                    flag = mapping[b]
                    for i, char in enumerate(a):
                        if i == 0:
                            deal_texts.append(f"{char} B-{flag}\n")
                        else:
                            deal_texts.append(f"{char} I-{flag}\n")
                else:
                    for char in a:
                        deal_texts.append(f"{char} O\n")
        deal_texts.append('\n')
    return deal_texts


def msra_preprocessing(split_rate: float = 0.8,
                       ignore_exist: bool = False) -> None:
    """
    MSRA数据预处理
    :param split_rate:数据分割比例，默认为 0.8
    :param ignore_exist 忽略已经存储的数据，则不需要在判断以及存好的数据
    :return:None
    """
    path_train = os.path.join(path_msra_dir, "train.txt")
    path_test = os.path.join(path_msra_dir, "test.txt")
    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        print("MSRA数据预处理已经完成")
        return
    else:
        print("正在对MSRA数据进行预处理......")
    path_train1 = os.path.join(path_msra_dir, "train1.txt")
    texts = read_file(path_train1)

    if split_rate >= 1.0:  # 分割比例不得大于1.0
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    test_ = text_map(test_texts)
    train_ = text_map(train_texts)

    with open(path_train, 'w') as f:
        f.write("".join(train_))
    with open(path_test, 'w') as f:
        f.write("".join(test_))
    print("MSRA数据进行预处理完成 ---- OK!")


if __name__ == '__main__':
    msra_preprocessing(ignore_exist=True)

    pass


