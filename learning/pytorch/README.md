# PyTorch 学习

| 类别 | 练习 |
| --- | --- |
| 基础 | [张量与自动求导](basics/pytorch_basics/main.py)、[逻辑回归](basics/logistic_regression/main.py)、[前馈网络](basics/feedforward_neural_network/main.py) |
| 视觉 | [CNN](vision/convolutional_neural_network/main.py)、[ResNet](vision/deep_residual_network/main.py) |
| 序列 | [双向循环网络](sequence/bidirectional_recurrent_neural_network/bidirectional_recurrent_neural_network.py)、[语言模型数据工具](sequence/language_model/data_utils.py) |

这些是个人学习复现，各自保留原有完成度；语言模型训练入口原为空文件，本次归档到本地占位备份。基础教程中的自定义 Dataset 仍是待实现示意代码。

MNIST 统一读取 `data/mnist/`，CIFAR-10 统一读取 `data/cifar10/`；权重输出到 `artifacts/pytorch/<练习名>/`。保留原脚本的下载开关，缺少数据且 `download=False` 时需先下载。基础教程还会下载预训练 ResNet。

参考 [上游教程](../../resources/README.md) 与 [PyTorch 笔记](../../notes/pytorch/README.md)。
