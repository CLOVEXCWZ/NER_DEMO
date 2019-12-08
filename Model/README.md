<font size=10>**Model**</font>

​	在通过查阅一些博客后，在实体命名识别方面，几大主流的模型就是BILSTM+CRF、IDCNN+CRF、BERT+BILSTM+CRF。所以在此也是搭建了这几种模型分别进行测试。



**注意:由于算力和时间原因，并没有对模型进行很好的调参~现在所搭建的模型并没有进行调参对比。**



本demo总共采用了5个模型分别是:

- BERT+BILSTM+CRF
- BILSTM+Attention+CRF
- BILSTM+CRF
- IDCNN+CRF（1）
- IDCNN+CRF（2）  与（1）的CNN的filters数和全连接层会有些区别。



# BERT+LSTM+CRF

采用BERT输出直接接上LSTM再接上CRF ，并没有进行太多参数的调整。

但从网络结构上看，LSTM和全连接层和dropout部分是可以做一些调整和尝试的，由于算力和时间的问题实在没办法做太多尝试。等有机会了再做更多尝试吧。



<img src="picture/BERT_BILSTM_CRF.png">





# BILSTM+Attention+CRF

BILSTM接上self attention层在家CRF。

其中AttentionSelf层是根据大神实现的代码实现的



<img src="picture/BILSTM_ATTENTION_CRF.png">





# BILSTM+CRF

<img src="picture/BILSTM_CRF.png">

# IDCNN+CRF（1）

<img src="picture/IDCNN_CRF.png">



# IDCNN+CRF（2）

<img src="picture/IDCNN_CRF_2.png">