import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
# #1部分：基础自动求导

# #创建标量张量
# x = torch.tensor(1., requires_grad=True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)

# y = w * x + b

# y.backward()

# print(x.grad)    # x.grad = 2 
# print(w.grad)    # w.grad = 1 
# print(b.grad)    # b.grad = 1 

# # 2部分 线性层 损失函数 优化器

# #创建对应数据结构张量
# x = torch.randn(10, 3)
# y = torch.randn(10, 2)

# linear = nn.Linear(3, 2)

# # print('w: ', linear.weight)
# # print('b:', linear.bias)

# # Build loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# #Forward pass
# pred = linear(x)

# #Compute loss
# loss = criterion(pred, y)
# print("loss", loss.item())

# loss.backward()

# print("dL/dw:", linear.weight.grad)
# print("dL/db:", linear.bias.grad)

# optimizer.step()

#3部分 Numpy核Tensor互转

# x = np.array([[1, 2], [3, 4]])

# print(x)

# y = torch.from_numpy(x)

# print(y)

# z = y.numpy()

# print(z)

#4部分 内置数据集合DataLoader

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

image, labels = train_dataset[0]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

data_iter = iter(train_loader)

images, labels = data_iter.next()

for images, labels in train_loader:
    pass

#5. 针对自定义数据集的输入管线
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        #初始化
        #TODO
        pass

    def __getitem__(self, index):
        #TODO
        #返回数据
        pass

    def __len__(self):
        return 0

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

#6.预训练模型

resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


#保存模型
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
