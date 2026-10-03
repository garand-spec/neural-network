import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import torchvision

def data_load(path):
    project_dir = Path(path)

    print(project_dir)


if __name__ == "__main__":
    data_load(r"D:\BaiduNetdiskDownload\train\20260730-193034\DetectiumFire\real_images\real_images\real_fire\images")