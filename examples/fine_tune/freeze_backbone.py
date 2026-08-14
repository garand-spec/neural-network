"""
冻结模型 backbone 的示例

常见场景：迁移学习 / 微调时，预训练特征提取部分（backbone）已经学得很好了，
我们希望只训练后面的分类头（head），从而加快训练、防止过拟合、节省显存。

三种冻结方式（从简单到灵活）：
  1. 按模块整体冻结： model.backbone.requires_grad_(False)
  2. 遍历命名参数，按名字前缀冻结
  3. 只把需要训练的参数传给优化器（这是真正起作用的步骤）

核心要点：
  - requires_grad = False 只是让该参数不计算梯度；
  - optimizer 只接收 requires_grad=True 的参数，才能保证冻结生效。
"""
import torch
import torch.nn as nn
import torch.optim as optim


# 一个简单的“backbone + head”结构，与项目里 simplecnn 的结构对应
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x


class ClassifierHead(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.fc = nn.Linear(32 * 56 * 56, num_class)

    def forward(self, x):
        return self.fc(x)


class MyModel(nn.Module):
    def __init__(self, num_class=4):
        super().__init__()
        self.backbone = Backbone()
        self.head = ClassifierHead(num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


# ---------------- 方式一：按模块整体冻结 ----------------
def freeze_module(model: nn.Module, module_name: str):
    """把指定子模块（如 backbone）的所有参数设为不更新。"""
    module = getattr(model, module_name)
    for param in module.parameters():
        param.requires_grad = False


# ---------------- 方式二：按参数名前缀冻结 ----------------
def freeze_by_name(model: nn.Module, prefix: str):
    """冻结所有名字以 prefix 开头的参数，比如 'backbone'。"""
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = False


# ---------------- 方式三：通用 freeze / unfreeze ----------------
def set_requires_grad(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


# ---------------- 只把可训练参数交给优化器（关键步骤） ----------------
def make_optimizer(model: nn.Module, lr=0.001):
    # 只有 requires_grad=True 的参数才会被优化器更新
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数量: {len(trainable)} / {sum(1 for _ in model.parameters())}")
    return optim.Adam(trainable, lr=lr)


# ---------------- 统计各参数是否可训练 ----------------
def print_trainable(model: nn.Module):
    for name, param in model.named_parameters():
        status = "可训练" if param.requires_grad else "已冻结"
        print(f"{name:30s} {status}")


if __name__ == "__main__":
    model = MyModel(num_class=4)
    print("========== 冻结前 ==========")
    print_trainable(model)

    # 选择一种冻结方式即可，这里演示“方式一”
    freeze_module(model, "backbone")
    print("\n========== 冻结 backbone 后 ==========")
    print_trainable(model)

    # 只把 head 的参数交给优化器（backbone 不参与更新）
    optimizer = make_optimizer(model, lr=0.001)

    # 简单模拟一次训练步骤，验证 backbone 权重确实没变
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(2, 3, 224, 224)      # 模拟 batch=2 的 224x224 图像
    y = torch.randint(0, 4, (2,))        # 模拟 4 分类标签

    before = {n: p.clone() for n, p in model.named_parameters()}
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    print("\n========== 验证 ==========")
    for name, param in model.named_parameters():
        changed = not torch.equal(before[name], param)
        if name.startswith("backbone"):
            assert not changed, f"backbone 参数 {name} 被更新了，冻结失败！"
            print(f"{name:30s} 权重未变 (冻结生效)")
        else:
            assert changed, f"head 参数 {name} 没有更新，训练失败！"
            print(f"{name:30s} 权重已更新")
