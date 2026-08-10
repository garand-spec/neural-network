import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transforms
        self.samples = [] #清单:[(路径, 标签)]
        self.classes = sorted(os.listdir(root)) #子文件夹名 ＝ 类名
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        for name in self.classes:
            folder = os.path.join(root ,name)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):

