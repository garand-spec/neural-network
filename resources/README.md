# 外部参考资料

[pytorch-tutorial](pytorch-tutorial/README.md) 来自 [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)，作为 Git 子模块固定在提交 `0500d3df5a2a8080ccfccbc00aca0eacc21818db`。保留上游目录结构和许可证。

```powershell
git submodule update --init --recursive
```

自己的复现位于 [learning/pytorch](../learning/pytorch/README.md)。原有上游修改已提取为 [补丁](../notes/pytorch/upstream-local-changes.patch)，不会丢失，也避免额外复制整套教程。
