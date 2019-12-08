<font size=10>**NER（中文实体命名识别）**</font>



**光健字: 中文命名实体识别 NER  BILSTM CRF IDCNN BERT**



**摘要：对中文命名实体识别一直处于知道却未曾真正实践过的状态，此次主要是想了解和实践一些主流的中文命名实体识别的神经网络算法。通过对网上博客的阅读了解，中文命名实体识别比较主流的方法是BILSTM+CRF、IDCNN+CRF、BERT+BILSTM+CRF这几种神经网络算法，这个demo也用Keras实现了这几个算法，并且采用几个比较通用的数据集进行训练测试。这个demo是以了解和学习为目的的，所以未得出任何结论**



**注意：由于算力和时间的问题，对神经网络的参数未进行太多调试，所以模型现在的参数并不是最佳参数**



# 主要库的版本

本项目是基于keras（Using TensorFlow backend）以下是主要库的版本

- python = 3.6.8



- keras == 2.2.4
- keras_contrib == 0.0.2
- keras_bert == 0.80.0
- tensorflow == 1.14.0



# 项目目录结构

- data 数据目录
  - 具体请查看数据目录文件夹下的README文件
- DataProcess 数据处理文件夹
  - 具体请查看数据处理文件夹下的README文件
- Pubilc 公共工具
  - path 定义文件（文件夹）的路径
  - utils 工具
    - 创建log
    - keras的callback调类
- Model 模型（总共定义了5个模型，具体结构请查看Model文件夹下的README文件）
  - BERT+BILST+CRF
  - BILSTM+Attention+CRF
  - BILSTM+CRF
  - IDCNN+CRF（1）
  - IDCNN+CRF（2）
- log 记录数据

# 运行项目

**注意：需要用到bert网络的需要提前下载BERT预训练模型解压到data文件夹下**



- 直接在IDE里运行项目
  - 直接运行 train.py文件

- 命令行
  - python train.py



# 运行结果



运行GPU： GeForceRTX2080Ti（GPU显存 10.0G, 算力7.5）



训练周期为15个周期，提前停止条件：2个周期验证集准确率没有提升。

BERT采用batch_size=32 因为值为64的时候所使用GPU内存不够

 **以下数据基于MSRA数据集，以8:2的拆分(训练集:测试集)。测试结果**

|         模型         | 准确率 |  F1   | 召回率 |
| :------------------: | :----: | :---: | :----: |
|      IDCNN_CRF       | 0.988  | 0.860 | 0.871  |
|     IDCNN_CRF_2      | 0.990  | 0.872 | 0.897  |
| BILSTM_Attention_CRF | 0.987  | 0.850 | 0.848  |
|      BILSTMCRF       | 0.989  | 0.870 | 0.863  |
|   BERT_BILSTM_CRF    | 0.996  | 0.954 | 0.950  |



很显然BERT+BILIST+CRF的组合效果会好很多



**提示:log文件夹里有每个训练周期记录的数据**



