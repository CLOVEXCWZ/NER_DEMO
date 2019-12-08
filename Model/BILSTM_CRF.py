from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from keras_contrib.layers import CRF


class BILSTMCRF():
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

    def creat_model(self):
        inputs = Input(shape=(None,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(x)
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

    lstm_crf = BILSTMCRF(vocab_size=dp.vocab_size, n_class=7)
    lstm_crf.creat_model()
    model = lstm_crf.model

    plot_model(model, to_file='picture/BILSTM_CRF.png', show_shapes=True)
    exit()

    model.fit(train_data, train_label, batch_size=64, epochs=2,
              validation_data=[test_data, test_label])

    # 对比测试数据的tag
    y = model.predict(test_data)

    label_indexs =[]
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

    """ epochs=2
    
     - val_loss: 0.0422 - val_crf_viterbi_accuracy: 0.9852
     
     f1score:0.8070982542924598
     
    """

    exit()

    with open('./pre.txt', 'w') as f:
        f.write("".join(texts))




