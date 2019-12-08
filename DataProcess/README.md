

<font size=8>**数据处理模块**</font>

​	此模块主要对数据data、data2、msra、人名日报 4份数据进行处理，可直接提供给模型使用。



# 文件构成

- data2_preprocessing.py  data2数据预处理文件
- msra_preprocessing.py MSRA数据预处理
- process_data.py  数据处理
- renminribao_preprocessing.py 人民日报数据处理
- vocab.py  词表 
  - 提供词表 和 tag 标签表格
  - 词表采用 BERT 预训练文件中vocab.txt 
  - tag 采用 O、B-PER、I-PER、B-LOC、I-LOC、B-ORG、I-ORG



# 文件简介

## data2_preprocessing.py 



data2 原文件all.txt 文件中的格式，这便是我们预处理的标准格式，所以这里只需要将文件切分成test.txt和train.txt两份数据即可。

```python
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```



## msra_preprocessing.py MSRA数据预处理

MAST是微软亚洲研究院开源数据

train1.txt 数据格式:

```python
当/o 希望工程/o 救助/o 的/o 百万/o 儿童/o 成长/o 起来/o ，/o 科教/o 兴/o 国/o 蔚然成风/o 时/o ，/o 今天/o 有/o 收藏/o 价值/o 的/o 书/o 你/o 没/o 买/o ，/o 明日/o 就/o 叫/o 你/o 悔不当初/o ！/o 
```

其中需要做几个映射:

```
人名：
    nr -> B-PER、I-PER
地名：
    ns -> B-LOC、I-LOC
机构名称:
    nt -> B-ORG、I-ORG
```

我们需要处理成以下格式:

```python
当 O
希 O
望 O
工 O
程 O
救 O
助 O
的 O
百 O
万 O
```

并且切分成训练和测试数据集



## process_data.py   数据处理

数据处理直接提供给模型训练使用。

数据处理有两种一种是一般的模型输入只需要一个2维数组

- 一般模型
  - inputs = ids(n_sample x max_len)   只需要一个二维输入
  - labels = label(n_sample x max_len x one_hot_num)  (转化为noe-hot为可选项目) 三维数据

- BERT
  - inputs =[ids(n_sample x max_len) , types(n_sample x max_len) ]   只需要2个二维输入
  - labels = label(n_sample x max_len x one_hot_num)  (转化为noe-hot为可选项目) 三维数据

 所以 一般模型的输入和bert数据输入需要进行区分



- 数据处理 (过程)
  - data（texts）（一般处理）
    - 获取char to index 字典
    - 把文本转化为index的形式
    - 截取过长的文本、填充长度不够文本
  - data（texts）(bert数据处理)
    - 获取char to index 字典
    - ids把文本转化为index的形式   type 全部填充0 因为只有一个句子
    - 截取过长的文本
    - 首尾进行填充（开始填充[CLS] 结尾填充 [SEP]）
  - labels（）
    - 获取tag to index 字典
    - 把label处理成Index的形式
    - 截取和填充(bert需要注意填充了[CLS] [SEP])
    - 转化成one-hot（如果需要）
- 最终输出(以label为one-hot为例子)
  - 一般数据
    - datas.shape=(n_sample, max_len)
    - labels.shape=(n_sample, max_len, one_hot_len)
  - bert数据
    - datas = [ ids.shape=(n_sample, max_len),  types.shape=(n_sample, max_len) ]
    - lables.shape=(n_sample, max_len, one_hot_len)

- bert数据处理







## renminribao_preprocessing.py 

人名日报标注数据预处理



原数据格式 (其实这个是别人稍稍处理过的--------------  )

```python
迈/O 向/O 充/O 满/O 希/O 望/O 的/O 新/O 世/O 纪/O —/O —/O 一/O 九/O 九/O 八/O 年/O 新/O 年/O 讲/O 话/O （/O 附/O 图/O 片/O １/O 张/O ）/O 
中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O 书/O 记/O 、/O 国/O 家/O 主/O 席/O 江/B_nr 泽/M_nr 民/E_nr 
（/O 一/O 九/O 九/O 七/O 年/O 十/O 二/O 月/O 三/O 十/O 一/O 日/O ）/O 
１/O ２/O 月/O ３/O １/O 日/O ，/O 中/B_nt 共/M_nt 中/M_nt 央/E_nt 总/O 书/O 记/O 、/O 国/O 家/O 主/O 席/O 江/B_nr 泽/M_nr 民/E_nr 发/O 表/O １/O ９/O ９/O ８/O 年/O 新/O 年/O 讲/O 话/O 《/O 迈/O 向/O 充/O 满/O 希/O 望/O 的/O 新/O 世/O 纪/O 》/O 。/O （/O 新/B_nt 华/M_nt 社/E_nt 记/O 者/O 兰/B_nr 红/M_nr 光/E_nr 摄/O ）/O 
```



需要做的标注映射:

```python
人名：
    B_nr -> B-PER
    M_nr -> I-PER
    E_nr -> I-PER

地名：
    B_ns -> B-LOC
    M_ns -> I-LOC
    E_ns -> I-LOC

机构名称:
    B_nt -> B-ORG
    M_nt -> I-ORG
    E_nt -> I-ORG
```



处理成通用的格式

```python
迈 O
向 O
充 O
满 O
希 O
望 O
的 O
新 O
世 O
纪 O
— O
— O
```





## vocab.py  词表

- 提供词表 和 tag 标签表格
- 词表采用 BERT 预训练文件中vocab.txt 
- tag 采用 O、B-PER、I-PER、B-LOC、I-LOC、B-ORG、I-ORG