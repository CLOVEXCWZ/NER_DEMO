from keras.regularizers import L1L2
from keras.engine.topology import Layer
from keras import backend as K

from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from keras_contrib.layers import CRF


class AttentionSelf(Layer):
    """
        self attention,
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(name='WKV',
                                        shape=(3, input_shape[2], self.output_dim),
                                        initializer='uniform',
                                        regularizer=L1L2(0.0000032),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        print("WQ.shape",WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)
        QK = K.softmax(QK)
        print("QK.shape",QK.shape)
        V = K.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)


class BILSTMAttentionCRF(object):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate
        pass

    def creat_model(self):
        inputs = Input(shape=(None,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(x)
        x = AttentionSelf(300)(x)
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

    dp = DataProcess(data_type='msra')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    model_class = BILSTMAttentionCRF(vocab_size=dp.vocab_size, n_class=7)
    model_class.creat_model()
    model = model_class.model

    plot_model(model, to_file='picture/BILSTM_ATTENTION_CRF.png', show_shapes=True)
    exit()

    model.fit(train_data, train_label, batch_size=64, epochs=2,
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
    
    - val_loss: 0.1038 - val_crf_viterbi_accuracy: 0.9664
    
        epochs=2
        
    - val_loss: 0.0531 - val_crf_viterbi_accuracy: 0.9819
    
    f1score:0.7481255117004026
        
        epochs=3
        
        
        
        epochs=4
        
        
        epochs=5

    """

    exit()

    with open('./pre.txt', 'w') as f:
        f.write("".join(texts))

