简介:  基于keras框架，优化了之前的模型，实现GCN、BOW、ELMo跨媒体检索。

作者: 卢宇航 

------

#### gcn_train.py

- 数据读取、准备，模型训练

#### models/gcn_net.py

- GCN + VGG net

#### models/fc_net.py

- 训练Bi-lstm代码
- 训练时默认模型保存文件名"save"

#### models/gcn_utils.py

- 预处理代码

