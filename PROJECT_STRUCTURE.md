# 项目结构

```text
normalization_learning/
├── examples/                 # 可单独运行的小型示例
│   ├── normalization/demo.py
│   └── vision/               # CNN、数据增强、图像处理示例
├── projects/                 # 相对完整的训练项目
│   ├── cnn_classifier/       # 模型、训练脚本与冒烟测试
│   ├── dino/simple_dino.py
│   └── lenet/leNet.py
├── algorithms/               # 算法练习
├── tutorials/                # 本地复现的 PyTorch 教程
├── pytorch-tutorial/         # 外部教程子模块（保留原位置）
├── data/mnist/               # MNIST 数据
├── assets/figures/           # README 与示例生成的图片
└── artifacts/                # 训练产生的权重（Git 忽略）
```

常用入口：

```powershell
python examples/normalization/demo.py
python -m projects.cnn_classifier.scripts.smoke_test
python projects/dino/simple_dino.py
python projects/lenet/leNet.py
```

`projects/cnn_classifier/scripts/train.py` 的训练集和测试集路径仍是本地配置；运行前请修改为自己的 `ImageFolder` 数据集目录。
