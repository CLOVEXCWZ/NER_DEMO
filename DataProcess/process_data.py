from DataProcess.vocab import *
from Public.path import path_data_dir, path_data2_dir, path_msra_dir, path_renmin_dir
import numpy as np
import os

from DataProcess.data2_preprocessing import data2_preprocessing
from DataProcess.msra_preprocessing import msra_preprocessing
from DataProcess.renminribao_preprocessing import renminribao_preprocessing


class DataProcess(object):
    def __init__(self,
                 max_len=100,
                 data_type='data',  # 'data', 'data2', 'msra', 'renmin'
                 model='other',  # 'other'、'bert' bert 数据处理需要单独进行处理
                 ):
        """
        数据处理
        :param max_len: 句子最长的长度，默认为保留100
        :param data_type: 数据类型，当前支持四种数据类型
        """
        self.w2i = get_w2i()  # word to index
        self.tag2index = get_tag2index()  # tag to index
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len
        self.model = model

        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)

        if data_type == 'data':
            self.base_dir = path_data_dir
        elif data_type == 'data2':
            self.base_dir = path_data2_dir
            data2_preprocessing()
        elif data_type == 'msra':
            self.base_dir = path_msra_dir
            msra_preprocessing()
        elif data_type == 'renmin':
            self.base_dir = path_renmin_dir
            renminribao_preprocessing()
        else:
            raise RuntimeError('type must be "data", "msra", "renmin" or "data2"')

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """
        # 拼接地址
        path_train = os.path.join(self.base_dir, "train.txt")
        path_test = os.path.join(self.base_dir, "test.txt")

        # 读取数据
        if self.model == 'bert':
            train_data, train_label = self.__bert_text_to_index(path_train)
            test_data, test_label = self.__bert_text_to_index(path_test)
        else:
            train_data, train_label = self.__text_to_indexs(path_train)
            test_data, test_label = self.__text_to_indexs(path_test)

        # 进行 one-hot处理
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0]*self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)
            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
        else:
            train_label = np.expand_dims(train_label, 2)
            test_label = np.expand_dims(test_label, 2)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    # texts 转化为 index序列
    def __text_to_indexs(self, file_path: str) -> ([], []):
        data, label = [], []
        with open(file_path, 'r') as f:
            line_data,  line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    char_index = self.w2i.get(w, self.w2i[self.unk_flag])
                    tag_index = self.tag2index.get(t, 0)
                    line_data.append(char_index)
                    line_label.append(tag_index)
                else:
                    if len(line_data) < self.max_len:
                        pad_num = self.max_len - len(line_data)
                        line_data = [self.pad_index]*pad_num + line_data
                        line_label = [0]*pad_num + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    label.append(line_label)
                    line_data, line_label = [], []
        return np.array(data), np.array(label)

    def __bert_text_to_index(self, file_path: str):
        """
        bert的数据处理
        处理流程 所有句子开始添加 [CLS] 结束添加 [SEP]
        bert需要输入 ids和types所以需要两个同时输出
        由于我们句子都是单句的，所以所有types都填充0

        :param file_path:  文件路径
        :return: [ids, types], label_ids
        """
        data_ids = []
        data_types = []
        label_ids = []
        with open(file_path, 'r') as f:
            line_data_ids = []
            line_data_types = []
            line_label = []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
                    w_index = self.w2i.get(w, self.unk_index)
                    t_index = self.tag2index.get(t, 0)
                    line_data_ids.append(w_index)  # index
                    line_data_types.append(0)  # types
                    line_label.append(t_index)  # label index
                else:
                    # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
                    max_len_buff = self.max_len-2
                    if len(line_data_ids) > max_len_buff: # 先进行截断
                        line_data_ids = line_data_ids[:max_len_buff]
                        line_data_types = line_data_types[:max_len_buff]
                        line_label = line_label[:max_len_buff]
                    line_data_ids = [self.cls_index] + line_data_ids + [self.sep_index]
                    line_data_types = [0] + line_data_types + [0]
                    line_label = [0] + line_label + [0]

                    # padding
                    if len(line_data_ids) < self.max_len: # 填充到最大长度
                        pad_num = self.max_len - len(line_data_ids)
                        line_data_ids = [self.pad_index]*pad_num + line_data_ids
                        line_data_types = [0] * pad_num + line_data_types
                        line_label = [0] * pad_num + line_label
                    data_ids.append(np.array(line_data_ids))
                    data_types.append(np.array(line_data_types))
                    label_ids.append(np.array(line_label))
                    line_data_ids = []
                    line_data_types = []
                    line_label = []
        return [np.array(data_ids), np.array(data_types)], np.array(label_ids)



if __name__ == '__main__':

    # dp = DataProcess(data_type='data')
    # x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    #
    # print(y_train[:1, :1, :100])

    dp = DataProcess(data_type='data', model='bert')
    x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(x_test[0].shape)
    print(x_test[1].shape)
    print(y_test.shape)

    print(y_train[:1, :1, :100])

    pass



