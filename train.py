from Model.BERT_BILSTM_CRF import BERTBILSTMCRF
from Model.BILSTM_Attetion_CRF import BILSTMAttentionCRF
from Model.BILSTM_CRF import BILSTMCRF
from Model.IDCNN_CRF import IDCNNCRF
from Model.IDCNN5_CRF import IDCNNCRF2

from sklearn.metrics import f1_score, recall_score
import numpy as np
import pandas as pd

from Public.utils import *
from keras.callbacks import EarlyStopping
from DataProcess.process_data import DataProcess

max_len = 100


def train_sample(train_model='BERTBILSTMCRF',
                 # ['BERTBILSTMCRF', 'BILSTMAttentionCRF', 'BILSTMCRF',
                 # 'IDCNNCRF', 'IDCNNCRF2']
                 epochs=15,
                 log = None,
                 ):

    # bert需要不同的数据参数 获取训练和测试数据
    if train_model == 'BERTBILSTMCRF':
        dp = DataProcess(data_type='msra', max_len=max_len, model='bert')
    else:
        dp = DataProcess(data_type='msra', max_len=max_len)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    log.info("----------------------------数据信息 START--------------------------")
    log.info(f"当前使用数据集 MSRA")
    # log.info(f"train_data:{train_data.shape}")
    log.info(f"train_label:{train_label.shape}")
    # log.info(f"test_data:{test_data.shape}")
    log.info(f"test_label:{test_label.shape}")
    log.info("----------------------------数据信息 END--------------------------")

    if train_model == 'BERTBILSTMCRF':
        model_class = BERTBILSTMCRF(dp.vocab_size, dp.tag_size, max_len=max_len)
    elif train_model == 'BILSTMAttentionCRF':
        model_class = BILSTMAttentionCRF(dp.vocab_size, dp.tag_size)
    elif train_model == 'BILSTMCRF':
        model_class = BILSTMCRF(dp.vocab_size, dp.tag_size)
    elif train_model == 'IDCNNCRF':
        model_class = IDCNNCRF(dp.vocab_size, dp.tag_size, max_len=max_len)
    else:
        model_class = IDCNNCRF2(dp.vocab_size, dp.tag_size, max_len=max_len)

    model = model_class.creat_model()

    callback = TrainHistory(log=log, model_name=train_model)  # 自定义回调 记录训练数据
    early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
    model.fit(train_data, train_label, batch_size=32, epochs=epochs,
              validation_data=[test_data, test_label],
              callbacks=[callback, early_stopping])

    # 计算 f1 和 recall值

    pre = model.predict(test_data)
    pre = np.array(pre)
    test_label = np.array(test_label)
    pre = np.argmax(pre, axis=2)
    test_label = np.argmax(test_label, axis=2)
    pre = pre.reshape(pre.shape[0] * pre.shape[1], )
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], )

    f1score = f1_score(pre, test_label, average='macro')
    recall = recall_score(pre, test_label, average='macro')

    log.info("================================================")
    log.info(f"--------------:f1: {f1score} --------------")
    log.info(f"--------------:recall: {recall} --------------")
    log.info("================================================")

    # 把 f1 和 recall 添加到最后一个记录数据里面
    info_list = callback.info
    if info_list and len(info_list)>0:
        last_info = info_list[-1]
        last_info['f1'] = f1score
        last_info['recall'] = recall

    return info_list


if __name__ == '__main__':

    # 需要测试的模型
    train_modes = ['IDCNNCRF', 'IDCNNCRF2', 'BILSTMAttentionCRF', 'BILSTMCRF', 'BERTBILSTMCRF']

    # 定义文件路径（以便记录数据）
    log_path = os.path.join(path_log_dir, 'train_log.log')
    df_path = os.path.join(path_log_dir, 'df.csv')
    log = create_log(log_path)

    # 训练同时记录数据写入的df文件中
    columns = ['model_name','epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
    df = pd.DataFrame(columns=columns)
    for model in train_modes:
        info_list = train_sample(train_model=model, epochs=15, log=log)
        for info in info_list:
            df = df.append([info])
        df.to_csv(df_path)

