"""
IDCNN(空洞CNN) 当卷积Conv1D的参数dilation_rate>1的时候，便是空洞CNN的操作
"""

from keras.models import  Model
from keras.layers import Embedding, Dense, Dropout, Input
from keras.layers import Conv1D
from keras_contrib.layers import CRF

class IDCNNCRF(object):
    def __init__(self,
                 vocab_size: int,  # 词的数量(词表的大小)
                 n_class: int,  # 分类的类别(本demo中包括小类别定义了7个类别)
                 max_len: int = 100,  # 最长的句子最长长度
                 embedding_dim: int = 128,  # 词向量编码长度
                 drop_rate: float = 0.5,  # dropout比例
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        pass

    def creat_model(self):
        """
        本网络的机构采用的是，
           Embedding
           直接进行2个常规一维卷积操作
           接上一个空洞卷积操作
           连接全连接层
           最后连接CRF层

        kernel_size 采用2、3、4

        cnn  特征层数: 64、128、128
        """

        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Conv1D(filters=64,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=1)(x)
        x = Conv1D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=1)(x)
        x = Conv1D(filters=128,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=2)(x)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class)(x)
        self.crf = CRF(self.n_class, sparse_target=False)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()
        return self.model

    def compile(self):
        self.model.compile('adam',
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])


if __name__ == '__main__':

    from DataProcess.process_data import DataProcess
    from sklearn.metrics import f1_score
    import numpy as np
    from keras.utils.vis_utils import plot_model

    dp = DataProcess(max_len=100, data_type='msra')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    model_class = IDCNNCRF(vocab_size=dp.vocab_size, n_class=7, max_len=100)
    model_class.creat_model()
    model = model_class.model

    plot_model(model, to_file='picture/IDCNN_CRF.png', show_shapes=True)
    exit()

    model.fit(train_data, train_label, batch_size=64, epochs=5,
              validation_data=[test_data, test_label])

    # 对比测试数据的tag
    y = model.predict(test_data)

    label_indexs = []
    pridict_indexs = []

    num2tag = dp.num2tag()
    i2w = dp.i2w()
    texts = []
    texts.append(f"字符\t预测tag\t原tag\n")
    for i, x_line in enumerate(test_data):
        for j, index in enumerate(x_line):
            if index != 0:
                char = i2w.get(index, ' ')
                t_line = y[i]
                t_index = np.argmax(t_line[j])
                tag = num2tag.get(t_index, 'O')
                pridict_indexs.append(t_index)

                t_line = test_label[i]
                t_index = np.argmax(t_line[j])
                org_tag = num2tag.get(t_index, 'O')
                label_indexs.append(t_index)

                texts.append(f"{char}\t{tag}\t{org_tag}\n")
        texts.append('\n')

    f1score = f1_score(label_indexs, pridict_indexs, average='macro')
    print(f"f1score:{f1score}")

    """ epochs=1

    - val_loss: 0.0518 - val_crf_viterbi_accuracy: 0.9830
    
        epochs=2
    
    - val_loss: 0.0386 - val_crf_viterbi_accuracy: 0.9867
    
        epochs=3
        
    - val_loss: 0.0338 - val_crf_viterbi_accuracy: 0.9880
        
        epochs=4
        
    - val_loss: 0.0304 - val_crf_viterbi_accuracy: 0.9889
        
        epochs=5
    
    - val_loss: 0.0283 - val_crf_viterbi_accuracy: 0.9890
    
    f1score:0.8564147211248077
    

    """

    exit()

    with open('./pre.txt', 'w') as f:
        f.write("".join(texts))








