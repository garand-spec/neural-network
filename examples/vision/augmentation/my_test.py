import cv2
import numpy as np
import albumentations as A #图像增强库
import matplotlib.pyplot as plt

# 1.准备数据集（假数据）
img_height, img_width = 640, 640
image = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)

#模拟YOLO格式的Bounding Box（边界框）
#YOLO格式为：<class_id> <x_center> <y_center> <width> <height> (均为归一化到0-1的值)
#假设两个目标 0：苹果； 1：香蕉
bboxes = [
    [0.3, 0.4, 0.2, 0.3],
    [0.7, 0.6, 0.15, 0.25]
]

class_labels = [0, 1]

#为了可视化， 在原图上画出原始框
image_width_orig_boxes = image.copy()
for bbox in bboxes:
    x_c, y_c, w, h = bbox

    #将归一化坐标转换为像素坐标
    x1, y1 = int((x_c - w/2) * img_width), int((y_c - h/2) * img_height)
    x2, y2 = int((x_c + w/2) * img_width), int((y_c + h/2) * img_height)
    cv2.rectangle(image_width_orig_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

#2.构建数据增强 Pipeline(数据增强流水线)
transform = A.Compose([
    #几何变换，会改变Bounding Box(边界框) 的位置和形状
    A.HorizontalFlip(p=0.5), #50% 概率水平翻转
    A.VerticalFlip(p=0.3), #30% 概率垂直翻转
    A.ShiftScaleRotate(
        shift_limit=0.1, #平移缩放
        scale_limit=0.2, #缩放比例
        rotate_limit=30, #旋转角度
        p=0.5
    ),
    #像素变换：只改变颜色， 不改变 Bounding Box(边界框)
    A.RandomBrightnessContrast(p=0.5), #随机亮度和对比度
    A.HueSaturationValue(
        hue_shift_limit=20, 
        sat_shift_limit=30, 
        val_shift_limit=20, 
        p=0.5
    ),
    A.GaussNoise(p=0.3)
])

#核心关键 告诉Albumentations如何处理Bounding Box
bbox_params = A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
)


cv2.imshow("image_width_orig_boxes", image_width_orig_boxes)
cv2.waitKey(0)

