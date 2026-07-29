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
    _, height, width = image.shape

    #随机裁剪尺寸
    crop_size = random.randint(44, 64)

    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)

    cropped = image[
        :,
        top:top + crop_size,
        left:left + crop_size
    ]

    #增加batch维度， 方便使用interpolate
    cropped = cropped.unsqueeze(0)

    #恢复到64*64
    cropped = F.interpolate(
        cropped,
        size=(64*64),
        mode="bilinear",
        align_corners=False
    )

    cropped = cropped.squeeze()

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
    ]).view()

    cropped = cropped * color_scale

    return cropped.clamp(0, 1)

if __name__ == "__main__":
    image = create_image().permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("imshow", image)
    cv2.waitKey(0)