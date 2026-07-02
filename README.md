# 归一化学习包

## 结论

归一化不是“求平均”，而是把数据放到更适合比较、优化或输出概率的尺度上。

## 运行

```powershell
cd C:\Users\20147\Documents\Codex\2026-06-26\https-www-zhihu-com-question-326034346\outputs\normalization_learning
python -m pip install -r requirements.txt
python normalization_demo.py
```

运行后会生成：

- `figures/distribution_normalization.png`
- `figures/softmax_temperature.png`

## 学习顺序

1. `MinMax`：把数据压到 `[0, 1]`，保留相对大小，怕异常值。
2. `Z-score`：变成均值约 `0`、标准差约 `1`，机器学习最常用。
3. `Softmax`：把一组分数变成概率，温度越低越“赢家通吃”。
4. `BatchNorm`：按 batch 和空间位置统计每个通道，CNN 常见。
5. `LayerNorm`：按单个样本内部特征统计，Transformer 常见。
6. `GroupNorm`：把通道分组统计，小 batch 场景更稳定。

## 关键公式

```text
MinMax:   x' = (x - min) / (max - min)
Z-score:  x' = (x - mean) / std
Softmax:  p_i = exp(x_i / T) / sum(exp(x_j / T))
Norm:     y = gamma * ((x - mean) / sqrt(var + eps)) + beta
```

## 看代码重点

- `minmax()` 和 `z_score()`：普通表格数据归一化。
- `softmax()`：分类概率和温度参数。
- `batch_norm_nchw()`：看 `axis=(0, 2, 3)`，表示每个通道单独统计。
- `layer_norm_last_dim()`：看 `axis=-1`，表示每个样本自己的特征内部统计。
- `group_norm_nchw()`：先 reshape 分组，再在组内统计。

## 最小练习

1. 把 `raw` 里的异常值 `[120, 140, 160]` 删除，再看图变化。
2. 修改 `softmax` 的 `temperature`，观察概率是否更尖锐。
3. 把 toy tensor 的通道尺度 `[1, 5, 10, 20]` 改大，看 BN/LN/GN 的输出变化。
"# neural-network" 
