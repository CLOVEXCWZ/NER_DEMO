import logging
import keras
import os
from Public.path import path_log_dir


def create_log(path, stream=False):
    """
    获取日志对象
    :param path: 日志文件路径
    :param stream: 是否输出控制台
                False: 不输出到控制台
                True: 输出控制台，默认为输出到控制台
    :return:日志对象
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    if stream:
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)

    # 设置文件日志s
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, log=None, model_name=None):
        super(TrainHistory, self).__init__()
        if not log:
            path = os.path.join(path_log_dir, 'callback.log')
            log = create_log(path=path, stream=False)
        self.log = log
        self.model_name = model_name
        self.epoch = 0
        self.info = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        self.log.info(message)

    def on_epoch_end(self, epoch, logs={}):
        message = f'end epoch: {epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]} acc:{logs["crf_viterbi_accuracy"]} val_acc:{logs["val_crf_viterbi_accuracy"]}'
        self.log.info(message)
        dict = {
            'model_name':self.model_name,
            'epoch': self.epoch+1,
            'loss': logs["loss"],
            'acc': logs['crf_viterbi_accuracy'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_crf_viterbi_accuracy']
        }
        self.info.append(dict)

    def on_batch_end(self, batch, logs={}):
        message = f'{self.model_name} epoch: {self.epoch} batch:{batch} loss:{logs["loss"]}  acc:{logs["crf_viterbi_accuracy"]}'
        self.log.info(message)

