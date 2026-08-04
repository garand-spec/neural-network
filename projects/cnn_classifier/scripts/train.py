import torch
import torch.nn as nn
import torch.optim as optim #优化器
from torch.utils.data import DataLoader #数据加载
from torchvision import datasets, transforms #数据集 和 数据变换
from tqdm import tqdm #训练进度条
import os
from projects.cnn_classifier.model.cnn import simplecnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#对图像做变换

train_transformer = transforms.Compose([
    transforms.Resize((224 , 224)), #将数据重采样缩放224
    transforms.ToTensor(), #吧图片转换为 tensor张量 0-1的像素值
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化 

])

test_transformer = transforms.Compose([
    transforms.Resize((224 , 224)), #将数据重采样缩放224
    transforms.ToTensor(), #吧图片转换为 tensor张量 0-1的像素值
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #标准化 

])

#定义数据及加载类
trainset = datasets.ImageFolder(root=r"D:\BaiduNetdiskDownload\train\train",
                                transform=train_transformer)

testset = datasets.ImageFolder(root="111",
                               transform=test_transformer)

#定义训练集加载器
train_loader = DataLoader(trainset, batch_size=8, num_workers=0, shuffle=True)
#定义测试集加载器
test_loader = DataLoader(testset,batch_size=8, num_workers=0, shuffle=True)


def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"epoch: {epoch+1}/{num_epochs}", unit="batch"): # 训练时可以看到对应的epoch和batch
            inputs, labels = inputs.to(device), labels.to(device) #将数据传到设备上
            #对优化器做梯度清零
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) #用loss乘批次大小， 得到该批次的loss
        epoch_loss = running_loss / len(train_loader.dataset) #总损失除总数据集大小 为我们每轮的损失
        print(f"epoch[{epoch+1} / {num_epochs}, Train_loss{epoch_loss:4f}]")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model saved with best acc", best_acc)


def evaluate(model, test_loader, criterion):
    model.eval() #指定模型为验证模式
    test_loss = 0.0 #初始的测试loss为0
    correct = 0 #初始正确样本为0
    total = 0 #总样本数量为0

    with torch.no_grad(): #在评估模式下不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) #将数据都输送到设备里
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1) #获取模型预测的最大值
            total = total + labels.size(0) #计算总样本的数量
            correct = correct + (predicted == labels).sum().item() #正确样本数累加
    avg_loss = test_loss / len(test_loader.dataset) #计算平均loss
    accuracy = 100.0 * correct / total #计算准确率

    print(f"test Loss:{avg_loss:.4g}, Accuracy:{accuracy:.2f}")
    return accuracy


def save_model(model,save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
            



if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_class = 4
    save_path = "artifacts/cnn_classifier/best.pth"
    model = simplecnn(num_class).to(device) #对模型进行实例化， 并送入gpu或者cpu中
    criterion = nn.CrossEntropyLoss() #指定损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #指定优化器为Adam
    train(model, train_loader, criterion, optimizer, num_epochs) #使用训练集进行训练
    evaluate(model, test_loader, criterion) #使用测试集进行测试
