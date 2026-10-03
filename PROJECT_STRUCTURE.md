# 项目结构与归档规则

```text
normalization_learning/
├── learning/                         # 学习区：练习、示例、教程复现
│   ├── algorithms/                   # 二分查找、二项式展开
│   ├── cpp/basics/                   # C++ 入门
│   ├── pytorch/
│   │   ├── basics/                   # 张量、逻辑回归、前馈网络
│   │   ├── vision/                   # CNN、ResNet 复现
│   │   └── sequence/                 # BiRNN、语言模型数据工具
│   └── vision/
│       ├── normalization/            # 归一化与 Softmax
│       ├── data_pipeline/            # Dataset、DataLoader 与小样例数据
│       ├── augmentation/             # 图像与 YOLO 标注增强
│       ├── image_processing/         # 模糊、letterbox、切片坐标
│       ├── classification/           # CNN 分类练习
│       ├── fine_tuning/              # backbone 冻结
│       └── point_cloud/pointnetpp/    # PointNet++ 组件与分类器
├── notes/                            # 笔记区
│   ├── algorithms/                   # 算法笔记索引
│   ├── cpp/                          # C++ 笔记索引
│   ├── pytorch/                      # 框架索引、外部教程修改补丁
│   ├── vision/                       # CV 索引、数据管线原笔记
│   ├── learning/                     # 学习档案、目标与偏好
│   ├── experiments/                  # 服务器使用与训练记录
│   └── maintenance/                 # 本次迁移映射与去重记录
├── projects/vision/                  # 实践区
│   ├── cnn_classifier/
│   ├── lenet/
│   ├── dino/
│   ├── fire_detection/               # 初始数据路径练习
│   └── visual_slam/                  # 尚未实现的项目框架
├── resources/pytorch-tutorial/       # 上游教程 Git 子模块
├── assets/figures/                   # 已有静态插图
├── data/                             # 本地数据，Git 忽略
├── artifacts/                        # 本地权重、图表和编译产物，Git 忽略
├── requirements.txt
└── README.md
```

## 新内容放在哪里

- 单项练习放 `learning/<领域>/<主题>/`；自己的复现和参考源码分开存放。
- 原理、踩坑、推导放 `notes/<领域>/`，通过相对链接关联配套代码；正文只维护一份。
- 具有模型、训练、评估等多个模块的实践放 `projects/<领域>/<项目>/`。
- 外部仓库放 `resources/`，优先使用可初始化的子模块，不将个人修改留在子模块里。
- 小型必要测试素材可随代码跟踪；完整数据集放 `data/`，生成结果放 `artifacts/`。
- 同主题但内容不同的练习保留；仅对确认相同的内容、空占位文件、缓存和生成产物做去重或取消跟踪。

## 常用入口

```powershell
python learning/vision/normalization/demo.py
python learning/vision/data_pipeline/MyImageDataset.py
python -m projects.vision.cnn_classifier.scripts.smoke_test
python learning/vision/point_cloud/pointnetpp/PointNet2Classifier/PointNet2Classifier.py
python projects/vision/dino/simple_dino.py
python projects/vision/lenet/leNet.py
```

完整训练会消耗计算资源或下载数据。`cnn_classifier/scripts/train.py` 和 `fire_detection/src/main.py` 仍需要自己的数据路径；其余保留练习的完成度见各区索引。本次整理的旧路径对照见 [迁移记录](notes/maintenance/reorganization-20261003.md)。
