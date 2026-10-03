# 仓库整理记录（2026-10-03）

## 分类与迁移

学习代码集中于 `learning/`，知识正文集中于 `notes/`，实践项目位于 `projects/vision/`，外部教程位于 `resources/`。根目录只保留仓库入口和公共配置。

| 原路径 | 新路径 |
| --- | --- |
| `algorithms/binarySearch` | `learning/algorithms/binary_search` |
| `algorithms/binomial_expand` | `learning/algorithms/binomial_expand` |
| `algorithms/tiles` | `learning/vision/image_processing/tiles` |
| `algorithms/pointnetpp` | `learning/vision/point_cloud/pointnetpp` |
| `examples/data_pipline/README.md` | `notes/vision/data_pipeline.md` |
| `examples/data_pipline` | `learning/vision/data_pipeline` |
| `examples/normalization` | `learning/vision/normalization` |
| `examples/fine_tune` | `learning/vision/fine_tuning` |
| `examples/vision/augmentation` | `learning/vision/augmentation` |
| `examples/vision/image_processing/GaussBlur_picture.py` | `learning/vision/image_processing/GaussBlur_picture.py` |
| `examples/vision/image_processing/test.py` | `learning/vision/image_processing/letterbox.py` |
| `examples/vision/image_processing/test.jpg` | `learning/vision/image_processing/test.jpg` |
| `examples/vision/basic_cnn_train.py` | `learning/vision/classification/basic_cnn_train.py` |
| `Cpp/test/test.cpp` | `learning/cpp/basics/hello_world.cpp` |
| `tutorials/reimplementations/pytorch_basics` | `learning/pytorch/basics/pytorch_basics` |
| `tutorials/reimplementations/logitic_regression` | `learning/pytorch/basics/logistic_regression` |
| `tutorials/reimplementations/feedforward_neural_network` | `learning/pytorch/basics/feedforward_neural_network` |
| `tutorials/reimplementations/convolutional_neural_network` | `learning/pytorch/vision/convolutional_neural_network` |
| `tutorials/reimplementations/deep_residual_network` | `learning/pytorch/vision/deep_residual_network` |
| `tutorials/reimplementations/bidirectional_recurrent_neural_network` | `learning/pytorch/sequence/bidirectional_recurrent_neural_network` |
| `tutorials/reimplementations/language_model` | `learning/pytorch/sequence/language_model` |
| `projects/cnn_classifier` | `projects/vision/cnn_classifier` |
| `projects/lenet` | `projects/vision/lenet` |
| `projects/dino` | `projects/vision/dino` |
| `fire_detected` | `projects/vision/fire_detection` |
| `visual_slam` | `projects/vision/visual_slam` |
| `pytorch-tutorial` | `resources/pytorch-tutorial` |
| `LEARNING_PROFILE.md` | `notes/learning/LEARNING_PROFILE.md` |
| `服务器使用情况简单说明.md` | `notes/experiments/server_usage.md` |

## 去重与保留

- 按 SHA-256 检查源码和小型素材，未发现完全相同的非空源码。同主题但内容不同的教程复现、增强练习继续保留。
- 两个相同的空占位文件 `cnn_classifier/scripts/test.py`、`language_model/main.py` 从代码区移到本地 `artifacts/placeholders/`；语言模型数据工具仍在学习区。
- 外部教程不重复复制到主仓库；修复原来缺少 `.gitmodules` 的子模块登记，保留上游版本和许可证。上游个人修改以补丁保存。
- 旧权重统一移动到本地 `artifacts/legacy_checkpoints/`，C++ 编译产物移到 `artifacts/cpp/`，语言模型采样输出移到 `artifacts/language_model/`。
- 已被 Git 跟踪的 MNIST 数据、权重、Python 缓存和 `.codex/backups/` 取消跟踪；本机数据与历史备份继续保留，旧版本也可通过 Git 历史找回。
- MNIST 的完整压缩文件和解压文件都保留在同一个数据集目录，供 torchvision 使用；另一个不完整下载保留到 `data/downloads/mnist/*.partial`，不当作相同数据删除。
- 本机 `.vscode/` 编译调试配置保留并忽略；新 C++ 源码纳入学习区并同步。
- 整理前未提交的二项式展开变量修正继续保留；原知识笔记只更新代码路径。

## 路径与恢复

更新 CNN 包导入、教程数据目录、权重输出目录、归一化图表输出目录和 letterbox 默认素材路径。教程数据统一放仓库根目录下的 `data/`，输出统一放 `artifacts/`；使用文件自身位置定位路径。

整理前的源码、文档和编辑器配置快照保存在本机 `.codex/backups/reorganization-20261003/`，其中 `original-manifest.json` 记录原始 SHA-256。原 PointNet++ 备份仍保留在 `.codex/backups/pointnetpp-variable-names-20260819/`。这些目录不参与远端同步。

旧入口已迁移，请使用 [根目录 README](../../README.md) 和 [项目结构](../../PROJECT_STRUCTURE.md) 中的新路径。整理不改变原练习的算法逻辑或完成度。

## 验证范围

- 38 个 Python 文件通过语法检查，19 份导航与笔记文档中的 75 个相对链接均有效。
- 73 份整理前源码和资料快照通过 SHA-256 核对。
- 归一化示例、CNN 冒烟验证、自定义 Dataset、DataLoader 练习、PointNet++ 前向计算、backbone 冻结示例运行成功。
- 外部教程保持干净，原修改补丁通过 `git apply --check`；补丁文件关闭换行转换，保留其格式空白。
- 未启动完整训练或下载新的训练数据；YOLO 增强依赖 albumentations，本机尚未安装，因此未运行该示例。
