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
class TinyVisionTransformer(nn.Module):

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
            dim_feedforward=embed_dim * 4,
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

    def forward(self, x):
        #得到patch token
        patch_tokens = self.patch_embedding(x)

        batch_size = patch_tokens.shape[0]

        #扩展CLS Token
        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        #cls token 放在所有patch token前面
        tokens = torch.cat(
            [cls_tokens, patch_tokens],
            dim=1
        )

        #tokens形状:
        #[B, 65, 64]
        #65 = 1cls + 64patch

        #加入位置编码
        tokens = tokens + self.position_embedding

        #transformer提取特征
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        #第0个token是CLS TOKEN
        cls_features = tokens[:, 0]

        #剩余token是patch特征
        patch_features = tokens[:, 1:]

        #投影后的cls特征， 用于一致性训练
        projected_feature = self.projection(self.cls_token)

        #归一化， 使其更适合计算余弦相似度
        projected_feature = F.normalize(
            projected_feature,
            dim=-1
        )

        return projected_feature, cls_features, patch_features

#EMA更新Teacher
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.99):
    """
    Teacher 参数更新

    teacher = 
        momentum * teacher
        + (1 - momoentum) * student
    """

    for student_parameter, teacher_parameter in zip(
        student.parameters(),
        teacher.parameters()
    ):
        teacher_parameter.data_mul_(momentum)

        teacher_parameter.data.add_(
            student_parameter.data,
            alpha=1.0 - momentum
        )

#跨视图一致性损失
def dino_loss(student_output, teacher_output):
    """
    两个归一化特征越接近，余弦相似度越接近1
    
    loss = 1 - cosine_similarity
    """

    similarity = (
        student_output * teacher_output
    ).sum(dim=-1)

    loss = 1 - similarity

    return loss

#开始训练
def train():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备", device)

    #创建学生网络
    student = TinyVisionTransformer().to(device)

    #教师初始参数复制学生
    teacher = copy.deepcopy(student)

    #Teacher不进行反向传播
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    original_image = create_image()

    student.train()
    teacher.eval()

    for step in range(201):

        #同一视图产生两个不同视图
        view1 = random_view(original_image)
        view2 = random_view(original_image)

        #添加batch维度
        view1 = view1.unsqueeze(0).to(device)
        view2 = view2.unsqueeze(0).to(device)

        #Teacher查看视图1
        with torch.no_grad:
            teacher_output, _, _= teacher(view1)
        #Student查看视图2
        student_output, _, _, = student(view2)

        #让两个视图的图像级特征接近
        loss = dino_loss(
            student_output,
            teacher_output.detach()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #使用Student参数缓慢更新Teacher
        update_teacher(
            student,
            teacher,
            momentous=0.99
        )

        if step % 20 == 0:
            cosine_similarity = (
                student_output * teacher_output
            ).sum(dim=-1).mean()

            print(
                f"step:{step:3d}"
                f"loss={loss.item():.4f}"
                f"相似度={cosine_similarity.item():.4f}"
            )

    return student, original_image, device

#查看CLS特征和Patch特征
def inspect_features(model, image, device):
    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        projected_feature, cls_feature, patch_features = model(image)

    print(
        "\n训练完成"
    )

    print(
        "投影特征形状：:",
        projected_feature
    )

    print(
        "CLS全局特征形状:",
        cls_feature
    )

    print(
        "Patch局部特征形状",
        patch_features.shape
    )

    #patch_features:
    #[1, 64, 64]
    #1：一张图
    #64：一共64个batch
    #64：每个Patch用64维向量表示

    #将Patch重新排成8*8空间网格

    patch_map = patch_features.reshape(
        1,
        8,
        8,
        64
    )

    print(
        "Patch二维特征图形状",
        patch_map.shape
    )

    #取重心位置的Patch特征
    center_patch = patch_map[0, 4, 4]

    background_patch = patch_map[0, 0, 0]

    similarity = F.cosine_similarity(
        center_patch.unsqueeze(0),
        background_patch.unsqueeze(0)
    )

    print(
        "中心目标Patch与左上背景Patch相似度",
        similarity.item()
    )

if __name__ == "__main__":
    trained_student, image, device = train()

    inspect_features(
        trained_student,
        image,
        device
    )