# PyTorch 笔记索引

代码按 [基础、视觉、序列](../../learning/pytorch/README.md) 分类。Dataset、Transforms、DataLoader 的已有总结见 [数据管线笔记](../vision/data_pipeline.md)。

## 外部教程中的个人修改

[upstream-local-changes.patch](upstream-local-changes.patch) 保存整理前上游工作区的全部修改：前馈网络的 MNIST 下载镜像配置，以及两个文件的末尾换行差异。镜像配置也已存在于个人前馈网络复现中。

补丁对应上游提交 `0500d3df5a2a8080ccfccbc00aca0eacc21818db`。如需恢复，先初始化子模块，再从仓库根目录执行：

```powershell
git -C resources/pytorch-tutorial apply --check ../../notes/pytorch/upstream-local-changes.patch
git -C resources/pytorch-tutorial apply ../../notes/pytorch/upstream-local-changes.patch
```

应用补丁后子模块会产生本地修改。日常继续复现时，优先将代码写在学习区。
