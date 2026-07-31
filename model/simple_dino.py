import copy
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_image(size=64):
    """
    创建一张简单的模拟图像
    黑色背景中间放一个亮色方块

    返回形状
    [3, 64, 64]
    """

    image = torch.zeros(3, size, size)

    #在图像中央画一个亮色方块
    image[0, 20:44, 20:44] = 1.0 #红色通道
    image[1, 20:44, 20:44] = 0.5 #绿色通道
    image[2, 20:44, 20:44] = 0.1 #蓝色通道

    #添加少量随机噪声
    image += torch.randn_like(image) * 0.03

    return image.clamp(0, 1)

#对同一张图生成不同视图
def random_view(image):
    """
    剪裁
    缩放
    水平翻转
    亮度
    颜色
    
    输入：
    [3, H, W]

    输出：
    [3, 64, 64]
    """
    _, height, width= image.shape

    #随机裁剪尺寸
    crop_size = random.randint(44, 64)

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)

    cropped = image[
        :,
        top:top + crop_size,
        left:left + crop_size
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cropped = cropped.to(device)

    #增加batch维度， 方便使用interpolate
    cropped = cropped.unsqueeze(0)

    #恢复到64*64
    cropped = F.interpolate(
        cropped,
        size=(64*64),
        mode="bilinear",
        align_corners=False
    )

    cropped = cropped.squeeze(0)

    #随机水平翻转
    if random.random() < 0.5:
        cropped = torch.flip(cropped, dims=[2])

    #随机亮度变化
    brightness = random.uniform(0.6, 1.4)
    cropped = cropped * brightness

    

    #每个颜色通道分别做轻微扰动
    color_scale = torch.tensor([
        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2)
    ],
    dtype=cropped.dtype,
    device=cropped.device
    ).view(3, 1, 1)

    cropped = cropped * color_scale

    return cropped.clamp(0, 1)

#Patch Embedding
class PatchEmbedding(nn.Module):
    """
    使用卷积实现Patch切分

    输入
    [B, 3, 64, 64]

    patch_size = 8时
    64 / 8 = 8

    最终得到
    8*8 = 64个Patch
    """

    def __init__(self, image_size=64, patch_size=8, embed_dim=64):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.grad_size = image_size // patch_size
        self.num_patches = self.grad_size ** 2

        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        #输入
        #[B, 3, 64, 64]

        x = self.projection(x)

        #经过卷积
        #[B, 64, 8, 8]

        x = x.flatten(2)

        #展平空间维度
        #[B, 64, 64]
        #
        #第一个64， 特征维度
        #第二个64， Patch数量
        x = x.transpose(1, 2)

        #转换为
        #[B, 64, 64]
        #
        #维度含义
        #[batch, patch数量, 特征维度]

        return x

#一个简单的 Vision Transformer
class TinyVisionTransformer(nn.Moudule):

    def __init__(self, image_size=64, patch_size=8, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embedding.num_patches

        #CLS token: 负责描述整张图
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        #位置编码 告诉模型每个patch在哪里
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model= embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        #投影头 用于计算跨视图的一致性
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

if __name__ == "__main__":
    image = create_image()
    image = random_view(image=image)
    image = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("imshow", image)
    cv2.waitKey(0)