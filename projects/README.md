# 实践区

当前项目均归入计算机视觉：

| 项目 | 入口 | 当前状态 |
| --- | --- | --- |
| CNN 分类 | [模型](vision/cnn_classifier/model/cnn.py)、[训练](vision/cnn_classifier/scripts/train.py)、[冒烟验证](vision/cnn_classifier/scripts/smoke_test.py) | 模型和训练脚本已有实现；训练/测试数据路径需自行配置 |
| MNIST 分类 | [leNet.py](vision/lenet/leNet.py) | 简单 CNN 训练练习，保留原文件名；并非经典 LeNet 结构 |
| DINO | [simple_dino.py](vision/dino/simple_dino.py) | student/teacher 与 EMA 的简化自监督练习 |
| 火灾检测 | [数据路径入口](vision/fire_detection/src/main.py) | 只有初始路径练习，检测模型尚未实现 |
| 视觉 SLAM | [项目说明](vision/visual_slam/README.md) | 原先为空目录，保留待实现框架 |

从仓库根目录运行 CNN 冒烟验证：

```powershell
python -m projects.vision.cnn_classifier.scripts.smoke_test
```

生成权重归入 `artifacts/`，数据归入 `data/`。项目原有完成度保留。
