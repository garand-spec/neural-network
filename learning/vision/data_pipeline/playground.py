import torch
from torch.utils.data import Dataset, DataLoader

class TinyDataset(Dataset): #数据仓库
    def __len__(self):  #必须实现：总共几个样本
        return 10

    def __getitem__(self, idx): #必须实现：按编号取一个样本
        return idx * 2 #假装取到数据，返回0， 2， 4， 6...


ds = TinyDataset()
dl = DataLoader(ds, batch_size=3, shuffle=True)

for epoch in range(3):
    batches = []
    for batch in dl:
        batches.append(batch.tolist())

    print(f"epoch:{epoch}: {batches}")