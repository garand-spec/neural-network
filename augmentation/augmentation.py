import cv2 # OpenCV (Open Source Computer Vision Library) 的 Python 接口
import numpy as np
import albumentations as A # 导入 Albumentations (一个强大的图像增强库)
import matplotlib.pyplot as plt

# ==========================================
# 1. 准备测试数据 (为了让新手直接运行，我们生成假数据)
# ==========================================
# 生成一张 640x640 的随机彩色图片
img_height, img_width = 640, 640
image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)

# 模拟 YOLO 格式的 Bounding Box (边界框)
# YOLO 格式为: <class_id> <x_center> <y_center> <width> <height> (均为归一化到 0-1 的值)
# 假设我们有两个目标：类别0（苹果）和类别1（香蕉）
bboxes = [
    [0.3, 0.4, 0.2, 0.3], # 类别0，中心点(0.3, 0.4)，宽0.2，高0.3
    [0.7, 0.6, 0.15, 0.25] # 类别1，中心点(0.7, 0.6)，宽0.15，高0.25
]
class_labels = [0, 1]

# 为了可视化，我们在原图上画出原始的框
image_with_orig_boxes = image.copy()
for bbox in bboxes:
    x_c, y_c, w, h = bbox
    # 将归一化坐标转为像素坐标
    x1, y1 = int((x_c - w/2) * img_width), int((y_c - h/2) * img_height)
    x2, y2 = int((x_c + w/2) * img_width), int((y_c + h/2) * img_height)
    cv2.rectangle(image_with_orig_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2) # 画绿框

# ==========================================
# 2. 构建数据增强 Pipeline (数据增强流水线)
# ==========================================
# 思考：为什么我们要把增强分为 BboxSafeTransform 和普通的像素变换？
transform = A.Compose([
    # 几何变换：会改变 Bounding Box (边界框) 的位置和形状
    A.HorizontalFlip(p=0.5),             # 50% 概率水平翻转
    A.VerticalFlip(p=0.3),               # 30% 概率垂直翻转
    A.ShiftScaleRotate(                  # 平移、缩放、旋转
        shift_limit=0.1,                 # 平移比例
        scale_limit=0.2,                 # 缩放比例
        rotate_limit=30,                 # 旋转角度
        p=0.5
    ),
    
    # 像素变换：只改变颜色，不改变 Bounding Box (边界框)
    A.RandomBrightnessContrast(p=0.5),   # 随机亮度和对比度
    A.HueSaturationValue(                # HSV (Hue, Saturation, Value) 变换
        hue_shift_limit=20, 
        sat_shift_limit=30, 
        val_shift_limit=20, 
        p=0.5
    ),
    A.GaussNoise(p=0.3),                 # 添加高斯噪声
], 
# 核心关键：告诉 Albumentations 如何处理 Bounding Box (边界框)
bbox_params=A.BboxParams(
    format='yolo',                       # 指定输入格式为 YOLO 格式
    label_fields=['class_labels']        # 指定类别标签的变量名
))

# ==========================================
# 3. 应用增强并可视化
# ==========================================
# 执行增强
augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
aug_image = augmented['image']
aug_bboxes = augmented['bboxes']
aug_labels = augmented['class_labels']

# 在增强后的图片上画出新的框
image_with_aug_boxes = aug_image.copy()
for bbox, label in zip(aug_bboxes, aug_labels):
    x_c, y_c, w, h = bbox
    x1, y1 = int((x_c - w/2) * aug_image.shape[1]), int((y_c - h/2) * aug_image.shape[0])
    x2, y2 = int((x_c + w/2) * aug_image.shape[1]), int((y_c + h/2) * aug_image.shape[0])
    # 根据类别画不同颜色的框
    color = (255, 0, 0) if label == 0 else (0, 0, 255) 
    cv2.rectangle(image_with_aug_boxes, (x1, y1), (x2, y2), color, 2)

# 显示结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_with_orig_boxes)
axes[0].set_title("Original Image with Bounding Boxes")
axes[0].axis('off')

axes[1].imshow(image_with_aug_boxes)
axes[1].set_title("Augmented Image with Transformed Boxes")
axes[1].axis('off')

plt.tight_layout()
plt.show()