from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import albumentations as A
import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_label(label_path: Path) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    """读取 YOLO txt：class_id, x_center, y_center, width, height。"""
    if not label_path.exists():
        return [], []  # 允许背景图没有 txt 标签文件

    boxes: list[tuple[float, float, float, float]] = []
    class_ids: list[int] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} 必须是 5 列，实际为 {len(parts)} 列")

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # 提前阻止坏标签进入增强流程；否则错误会被放大到整个新数据集。
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and x_center - width / 2 >= 0
            and x_center + width / 2 <= 1
            and y_center - height / 2 >= 0
            and y_center + height / 2 <= 1
        ):
            raise ValueError(f"{label_path}:{line_number} 含非法 YOLO 边界框：{line.strip()}")

        boxes.append((x_center, y_center, width, height))
        class_ids.append(class_id)

    return boxes, class_ids


def write_yolo_label(
    label_path: Path,
    boxes: list[tuple[float, float, float, float]],
    class_ids: list[int],
) -> None:
    """将增强后的边界框重新保存为 YOLO 格式。"""
    lines = [
        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for class_id, (x, y, w, h) in zip(class_ids, boxes)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_transform(image_size: int, seed: int | None) -> A.Compose:
    """
    增强顺序：
    1. 几何变化：模拟相机位置、尺度、姿态变化；
    2. 外观变化：模拟光照和颜色变化；
    3. Resize + Pad：将所有输出统一为 image_size × image_size。
    """
    return A.Compose(
        [
            # 目标左右方向不影响类别时才使用，例如人、车辆、普通商品。
            A.HorizontalFlip(p=0.5),

            # 轻度几何变化；初学阶段不宜设置太激进。
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-8, 8),
                shear=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=114,  # YOLO 常见的中性灰填充
                p=0.7,
            ),

            # 光照和成像差异。
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.3,
            ),

            # 少量模糊，模拟轻微失焦或运动；小目标任务可先关闭。
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                ],
                p=0.10,
            ),

            # 等比例缩放后补边，避免拉伸导致目标形状失真。
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=114,
                position="center",
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],

            # 被裁得只剩很小一部分的框不保留，减少噪声标签。
            min_visibility=0.35,
            min_area=16,
        ),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="为 YOLO 检测数据集生成可追溯的离线增强副本")
    parser.add_argument("--images", type=Path, required=True, help="原训练图片目录")
    parser.add_argument("--labels", type=Path, required=True, help="原训练标签目录")
    parser.add_argument("--out", type=Path, required=True, help="新的输出数据集根目录，必须为空或不存在")
    parser.add_argument("--copies", type=int, default=2, help="每张原图生成多少张增强图")
    parser.add_argument("--size", type=int, default=640, help="输出尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现实验")
    args = parser.parse_args()

    if args.copies < 1:
        raise ValueError("--copies 必须至少为 1")

    out_images = args.out / "images" / "train"
    out_labels = args.out / "labels" / "train"

    # 防止误覆盖已经生成的数据集。
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"输出目录非空，为防止覆盖已停止：{args.out}")

    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    transform = build_transform(args.size, args.seed)
    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到图片：{args.images}")

    manifest_path = args.out / "manifest.jsonl"
    saved_augmented = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV 无法读取图片：{image_path}")

            source_label = args.labels / f"{image_path.stem}.txt"
            boxes, class_ids = read_yolo_label(source_label)

            # 输出一份原图：最终数据集包含“原图 + 增强图”。
            original_image_out = out_images / f"{image_path.stem}_orig{image_path.suffix.lower()}"
            original_label_out = out_labels / f"{image_path.stem}_orig.txt"
            shutil.copy2(image_path, original_image_out)
            write_yolo_label(original_label_out, boxes, class_ids)

            manifest.write(json.dumps({
                "source": str(image_path),
                "output": str(original_image_out),
                "kind": "original",
            }, ensure_ascii=False) + "\n")

            for index in range(args.copies):
                result = transform(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_ids,
                )

                new_boxes = list(result["bboxes"])
                new_class_ids = list(result["class_labels"])

                # 原图有目标，而增强后所有框都消失：不保存，避免假阴性样本。
                if boxes and not new_boxes:
                    skipped += 1
                    continue

                stem = f"{image_path.stem}_aug_{index:02d}"
                image_out = out_images / f"{stem}.jpg"
                label_out = out_labels / f"{stem}.txt"

                if not cv2.imwrite(str(image_out), result["image"]):
                    raise IOError(f"图片写入失败：{image_out}")
                write_yolo_label(label_out, new_boxes, new_class_ids)

                manifest.write(json.dumps({
                    "source": str(image_path),
                    "output": str(image_out),
                    "kind": "augmented",
                    "boxes_before": len(boxes),
                    "boxes_after": len(new_boxes),
                }, ensure_ascii=False) + "\n")
                saved_augmented += 1

    print(f"完成：原图 {len(image_paths)} 张，新增增强图 {saved_augmented} 张，跳过 {skipped} 张。")
    print(f"数据清单：{manifest_path}")


if __name__ == "__main__":
    main()