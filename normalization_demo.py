import os

import matplotlib.pyplot as plt
import numpy as np


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def z_score(x):
    return (x - x.mean()) / x.std()


def softmax(x, temperature=1.0):
    z = x / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def batch_norm_nchw(x, eps=1e-5):
    # CNN常用输入形状: N=batch, C=channel, H/W=空间位置
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def layer_norm_last_dim(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def group_norm_nchw(x, groups=2, eps=1e-5):
    n, c, h, w = x.shape
    assert c % groups == 0
    x_grouped = x.reshape(n, groups, c // groups, h, w)
    mean = x_grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = x_grouped.var(axis=(2, 3, 4), keepdims=True)
    y = (x_grouped - mean) / np.sqrt(var + eps)
    return y.reshape(n, c, h, w)


def save_distribution_plot(raw, out_path):
    variants = {
        "raw": raw,
        "minmax": minmax(raw),
        "z_score": z_score(raw),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (name, data) in zip(axes, variants.items()):
        ax.hist(data, bins=40, color="#3b82f6", alpha=0.75)
        ax.axvline(data.mean(), color="#ef4444", linewidth=2, label="mean")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_softmax_plot(out_path):
    logits = np.array([1.0, 2.0, 3.0, 6.0])
    temps = [0.2, 1.0, 3.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(logits))
    width = 0.24
    for i, temp in enumerate(temps):
        probs = softmax(logits, temperature=temp)
        ax.bar(x + (i - 1) * width, probs, width=width, label=f"T={temp}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"logit {v:g}" for v in logits])
    ax.set_ylim(0, 1)
    ax.set_title("Softmax temperature")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_stats(name, x):
    print(f"\n{name}")
    print(f"shape: {x.shape}")
    print(f"mean:  {np.round(x.mean(axis=tuple(range(x.ndim)) if x.ndim else None), 4)}")
    print(f"std:   {np.round(x.std(), 4)}")


def main():
    #固定随机数，每次运行结果都一样
    np.random.seed(7)
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    raw = np.concatenate([
        np.random.normal(loc=10, scale=2, size=900),
        np.random.normal(loc=40, scale=5, size=90),
        np.array([120, 140, 160]),
    ])

    print("=== scalar normalization ===")
    print(f"raw mean/std:     {raw.mean():.3f}, {raw.std():.3f}")
    print(f"minmax mean/std:  {minmax(raw).mean():.3f}, {minmax(raw).std():.3f}")
    print(f"zscore mean/std:  {z_score(raw).mean():.3f}, {z_score(raw).std():.3f}")

    print("\n=== softmax ===")
    logits = np.array([1.0, 2.0, 3.0, 6.0])
    for temp in [0.2, 1.0, 3.0]:
        print(f"T={temp}: {np.round(softmax(logits, temp), 4)}")

    print("\n=== BN / LN / GN toy tensor ===")
    x = np.random.normal(size=(2, 4, 3, 3)) * np.array([1, 5, 10, 20]).reshape(1, 4, 1, 1)
    bn = batch_norm_nchw(x)
    ln = layer_norm_last_dim(x)
    gn = group_norm_nchw(x, groups=2)
    print_stats("raw", x)
    print_stats("batch_norm", bn)
    print_stats("layer_norm", ln)
    print_stats("group_norm", gn)

    save_distribution_plot(raw, os.path.join(out_dir, "distribution_normalization.png"))
    save_softmax_plot(os.path.join(out_dir, "softmax_temperature.png"))
    print(f"\nfigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
