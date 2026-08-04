import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os

ROOT = './root'

os.makedirs(f"{ROOT}", exist_ok=True)

torchvision.datasets.MNIST.mirrors = [
    'https://storage.googleapis.com/cvdf-datasets/mnist/'
]

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

print(device)

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(
    root=f"{ROOT}",
    transform=transforms.ToTensor(),
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root=f"{ROOT}",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_dataset = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if ((i +1) % 100 == 0):
            print("Epoch[{}/{}], Step[{}/{}], Loss{:.4f}".format(epoch+1, num_epochs, i+1, total_step,loss.item()))
