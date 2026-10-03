# normalization_learning

用于学习数据归一化、计算机视觉、PyTorch 和 C++ 的个人仓库。

| 分区 | 内容 | 入口 |
| --- | --- | --- |
| 学习区 `learning/` | 按算法、C++、PyTorch、计算机视觉分类的代码练习 | [学习目录](learning/README.md) |
| 笔记区 `notes/` | 原理总结、学习档案、实验记录、维护说明 | [笔记目录](notes/README.md) |
| 实践区 `projects/` | CNN 分类、MNIST、DINO、火灾检测、视觉 SLAM | [项目目录](projects/README.md) |
| 参考区 `resources/` | 固定版本的外部 PyTorch 教程 | [参考资料](resources/README.md) |
| 素材区 `assets/` | 文档使用的静态插图 | [插图目录](assets/figures/) |

完整结构和新增内容的归档规则见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

## 环境与运行

```powershell
python -m pip install -r requirements.txt
python learning/vision/normalization/demo.py
python -m projects.vision.cnn_classifier.scripts.smoke_test
python learning/vision/data_pipeline/playground.py
```

从仓库根目录运行。训练练习还需要 PyTorch、torchvision、Pillow；图像处理需要 OpenCV，增强示例需要 albumentations，CNN 项目训练需要 tqdm。按所选练习安装；PyTorch 请按机器的 CPU/CUDA 环境选择版本。

下载数据统一使用 `data/mnist/`、`data/cifar10/` 等目录，运行输出放入 `artifacts/`。这两个目录只保存在本机，Git 不跟踪；数据管线自带的六张小样例图片继续随代码同步。

## 获取外部教程

```powershell
git clone --recurse-submodules https://github.com/garand-spec/neural-network.git
# 已克隆的仓库：
git submodule update --init --recursive
```

外部教程作为子模块保存在 `resources/pytorch-tutorial/`；自己的复现写在学习区。迁移前的教程修改保存在 [可恢复补丁](notes/pytorch/upstream-local-changes.patch)，用法见 [PyTorch 笔记](notes/pytorch/README.md)。
