"""
data2 数据预处理

数据从all 文件夹中分割出两粉数据
"""

from Public.path import path_data2_dir
import os


# data2 数据预处理
def data2_preprocessing(split_rate: float = 0.8,
                        ignore_exist: bool = False) -> None:
    """
    data2数据预处理
    :param split_rate: 训练集和测试集切分比例
    :param ignore_exist: 是否忽略已经存在的文件(如果忽略，处理完一遍后不会再进行第二遍处理)
    :return: None
    """
    path = os.path.join(path_data2_dir, "all.txt")
    path_train = os.path.join(path_data2_dir, "train.txt")
    path_test = os.path.join(path_data2_dir, "test.txt")

    if not ignore_exist and os.path.exists(path_train) and os.path.exists(path_test):
        return

    texts = []
    with open(path, 'r') as f:
        line_t = []
        for l in f:
            if l != '\n':
                line_t.append(l)
            else:
                texts.append(line_t)
                line_t = []

    if split_rate >= 1.0:
        split_rate = 0.8
    split_index = int(len(texts) * split_rate)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # 分割和存数文本
    def split_save(texts: [str], save_path: str) -> None:
        data = []
        for line in texts:
            for item in line:
                data.append(item)
            data.append("\n")
        with open(save_path, 'w') as f:
            f.write("".join(data))

    split_save(texts=train_texts, save_path=path_train)
    split_save(texts=test_texts, save_path=path_test)


if __name__ == '__main__':
    data2_preprocessing()


