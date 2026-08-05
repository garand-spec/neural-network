# 学习形象档案（用于 AI 对话自我介绍）

> 用法：把本文件内容复制到任意 AI 对话开头，作为背景介绍。用后如有变化请更新。

---

## 我是谁

- 身份：计算机视觉方向学生，基础薄弱，正在系统补课。
- 编程语言：主要使用 Python 和 PyTorch。
- 操作系统：Windows，习惯 PowerShell 命令行。

## 我掌握的内容（能独立写出并理解）

- **CNN**：能写 `Conv2d -> ReLU -> MaxPool2d -> Linear` 的分类网络，知道卷积核、padding、stride 对特征图尺寸的影响（`projects/cnn_classifier/model/cnn.py`）。
- **MNIST**：会用 `torchvision.datasets.MNIST` + `DataLoader` 做分类训练。
- **RNN / BiRNN / LSTM**：能写 `nn.LSTM`，理解双向网络要把 hidden_size 翻倍、取最后时刻输出（`tutorials/reimplementations/bidirectional_recurrent_neural_network/`）。
- **DINOv3 简化版**：写过 student-teacher 结构、EMA 参数更新、跨视图一致性 loss、Patch Embedding、CLS token、Vision Transformer 主干（`projects/dino/simple_dino.py`）。
- **基础训练循环**：`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` 的流程。

## 我的弱项

- 对敲过的代码记忆不深，需要回看才能复述。
- 数据管线只停留在「会用 `DataLoader`」的阶段，不清楚内部的取样、装填、变换、并行机制。
- 对数据预处理（归一化、标准化、增强）为什么要这么做、底层是什么，理解不深。
- 张量形状变换（reshape/transpose/view）偶尔会算错。

## 当前学习目标

- 系统掌握 CV 数据管线：数据集组织 → 变换（transforms）→ 采样 → 批处理（batching）→ 多进程加载。
- 理解归一化/标准化/增强的数学原理和代码实现。
- 边学边写代码验证，并保留到本仓库。

## 学习偏好

- 希望用「先直观例子、再讲原理、再写代码、再画结构图」的方式教学。
- 希望解释时多用比喻，少用术语堆砌。
- 中文交流。
- 写过的代码会放在本仓库（normalization_learning），AI 可随时翻看评估我的进度。

## 环境

- 仓库：D:\normalization_learning
- 常用入口：
  - `python examples/normalization/demo.py`
  - `python -m projects.cnn_classifier.scripts.smoke_test`
  - `python projects/dino/simple_dino.py`
