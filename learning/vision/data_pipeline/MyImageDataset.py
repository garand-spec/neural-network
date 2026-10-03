import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = [] #清单:[(路径, 标签)]
        self.classes = sorted(os.listdir(root)) #子文件夹名 ＝ 类名
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        for name in self.classes:
            folder = os.path.join(root ,name)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(folder, fname),
                                         self.class_to_idx[name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB") #读图 PIL
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path          # 路径也返回，随样本走


def collate_with_path(batch):      # 把 (img, label, path) 的列表拼成一批
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)     # [B,3,64,64]
    labels = torch.tensor(labels)       # [B]
    return imgs, labels, paths          # paths 保持字符串，不拼张量


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    root = os.path.join(os.path.dirname(__file__), "mydata")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = MyImageDataset(root, transform=transform)
    print("类别映射:", ds.class_to_idx)
    print("样本总数:", len(ds), "（6 张图，3 张 cat + 3 张 dog）")

    # 返回了 path 字符串后，默认 collate 拼不了，必须自定义 collate
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_with_path)
    print("\n=== 自定义 collate（带路径）===")
    for i, (imgs, labels, paths) in enumerate(dl):
        print(f"batch{i}: img={tuple(imgs.shape)} labels={labels.tolist()}")
        print("         paths:", [os.path.basename(p) for p in paths])