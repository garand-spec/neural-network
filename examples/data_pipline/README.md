# 数据管线总结（四课锚点）

> 配套代码：`examples/data_pipline/MyImageDataset.py`、`examples/data_pipline/playground.py`

## 一图流：一次迭代发生了什么

```
for inputs, labels in train_loader:
        │ 每次迭代取一批
        ▼
① Sampler     按顺序给出 batch_size 个索引（shuffle=True 先洗牌）
        ▼
② Dataset.__getitem__(idx)
     → 读 PIL 图 → transforms 加工 → 返回 (img, label[, path])
        ▼ ×batch_size 次
③ collate_fn   把一批样本"叠"成一个张量 [B, C, H, W]
        ▼
模型收到: inputs [B,C,H,W]   labels [B]
```

## 四个工位与职责

| 工位 | 职责 | 常见实现 | 关键认知 |
|------|------|---------|---------|
| Dataset | 登记路径清单，按编号出单张样本 | `ImageFolder`, 自定义类 | **惰性**：`__init__` 只记路径不读图 |
| Transforms | 加工单张图片 | `Compose([Resize, ToTensor, Normalize])` | 处理对象是**单张**，与批次无关 |
| Sampler | 决定取样本的顺序 | `RandomSampler`(shuffle), `SequentialSampler` | `shuffle=True` 是**排列**，不重不漏 |
| DataLoader | 拼批 + 多进程搬运 | `DataLoader(...)` | `len()` = 批次数，不是样本数 |

## Transform 三件套的数学

```
Resize      : 只改 H×W（双线性插值），不改数值含义
ToTensor    : [H,W,C]uint8 0~255 → [C,H,W]float 0.0~1.0（翻转通道轴 + 除以255）
Normalize   : (x - mean)/std，逐通道，把 [0,1] 映射到 [-1,1]
```

- 换数据集时用**数据集的真实 mean/std**，ImageNet 标准值是 `(0.485,0.456,0.406)/(0.229,0.224,0.225)`。
- `transforms.Normalize` 用**写死的常数**；BN/LN/GN 是**运行时按轴现算**，同族不同源。

## collate_fn：必须和 __getitem__ 配对

- `__getitem__` 返回几个值，collate 就拆几个值（`zip(*batch)`）。
- 返回了字符串（路径）后，默认 collate 拼不了 → 必须自定义 `collate_fn`。

## 关键坑位清单

1. `self.transform = transforms`（模块）vs `transform`（参数）——差一个字母。
2. DataLoader 吐出 `[B,1,H,W]`，RNN 要 `[B, 时间步, 特征]` → 中间 `reshape` 转接。
3. Windows 下多进程（num_workers>0）必须放在 `if __name__ == "__main__":` 保护内。
4. `drop_last=False`（默认）时，最后一批可能比 batch_size 小。

## 下一步（可继续的方向）

- 手写 Dataset 时用 `__getitem__` 返回归一化后的 mask/bbox，配多字段 collate。
- 自定义 `__getitem__` 内部做随机增强（相当于把 augmentation 搬进 Dataset）。
- 学 `WeightedRandomSampler` 解决类别不平衡。
